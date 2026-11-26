"""Frozen-pair heads for the authorized six-hour, seed75 screen.

No assets, filesystem writes, optimizer or CUDA initialization occur on import.
The nonempty conditional distribution is a zero-truncated Bernoulli product.
AMTL formulas/mapping follow the saved public xhm-code/AMTL sources.
"""
from __future__ import annotations

import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

KINDS = ("spu", "atgl", "amtl", "bce", "factor", "set")
AMTL_GROUPS = (
    "P131 P17 P27 P150 P800 P527 P361 P175 P577 P463",
    "P1001 P569 P710 P1344 P570 P495 P264 P161 P571 P30 P3373 P40 P19 "
    "P1441 P26 P607 P156 P155 P69 P50",
    "P22 P400 P206 P108 P170 P102 P178 P127 P54 P674 P6 P1412 P140 P118 "
    "P57 P166 P276 P194 P123 P706 P35 P20 P37 P449 P159 P162 P179 P136 "
    "P58 P355 P749 P31 P580 P112 P937 P403 P137 P585 P241 P576 P1376 "
    "P36 P205 P86 P25 P172",
    "P279 P272 P488 P176 P171 P364 P582 P740 P1366 P1365 P840 P676 P551 "
    "P1056 P1336 P39 P737 P190 P807 P1198",
)


def block_product(vectors, block_size):
    if vectors.ndim != 3 or vectors.shape[1] != 2 or vectors.shape[2] % block_size:
        raise ValueError("Expected [pairs,2,embedding] divisible by block_size")
    h, t = vectors[:, 0], vectors[:, 1]
    h = h.reshape(-1, h.shape[-1] // block_size, block_size)
    t = t.reshape(-1, t.shape[-1] // block_size, block_size)
    return (h.unsqueeze(3) * t.unsqueeze(2)).reshape(len(vectors), -1)


@torch.no_grad()
def extract_vectors(model, inputs):
    """Exactly the pinned model2 TTM eval path up to its bilinear product."""
    if model.training or any(p.requires_grad for p in model.parameters()):
        raise ValueError("Extraction requires the entire source frozen in eval")
    if model.args.model_type != "ttmre":
        raise ValueError("This carrier is the pinned TTM model")
    sequence, attention = model.encode(inputs["input_ids"], inputs["attention_mask"])
    h, r, t = model.get_hrt(sequence, attention, inputs["entity_pos"], inputs["hts"])
    # Pinned model2 intentionally uses head_extractor for BOTH roles.
    h = torch.tanh(model.head_extractor(torch.cat((h, r), dim=1)))
    t = torch.tanh(model.head_extractor(torch.cat((t, r), dim=1)))
    result = []
    for start in range(0, len(h), 256):
        end = start + 256
        encoded = model.mu_encoder(torch.stack((h[start:end], t[start:end]), dim=1).unsqueeze(1))
        vh = encoded[:, 0, 0, :] / 2 + h[start:end] / 2
        vt = encoded[:, 0, 1, :] / 2 + t[start:end] / 2
        result.append(torch.stack((vh, vt), dim=1))
    return torch.cat(result)


def log_nonempty(b):
    """Stable log(1-prod(1-sigmoid(b))) without a probability floor.

    Float64 scalar reductions preserve a float32 head's gradients. In the
    negative tail log(softplus(b)) uses its Taylor series through exp(2*b);
    omitted terms are <1e-26 at the -20 branch, below double precision.
    Unselected branches are finite so torch.where cannot backpropagate NaNs.
    """
    z = b.double()
    safe = z.clamp_min(-20)
    small_exp = torch.exp(z.clamp_max(-20))
    small = z + torch.log1p(-small_exp / 2 + small_exp.square() / 3)
    log_sp = torch.where(z < -20, small, torch.log(F.softplus(safe)))
    log_s = torch.logsumexp(log_sp, dim=-1)
    s = F.softplus(z).sum(-1)
    safe_s = s.clamp_min(math.exp(-20))
    regular = torch.log(-torch.expm1(-safe_s))
    small_s = torch.exp(log_s.clamp_max(-20))
    tail = log_s - small_s / 2 + small_s.square() / 24
    return torch.where(log_s < -20, tail, regular)


def conditional_nll(b, y):
    z = b.double()
    return F.softplus(z).sum(-1) - (z * y).sum(-1) + log_nonempty(z)


def spu_risk(logits, labels, priors, *, e=1., margin=1., beta=0., gamma=1.):
    """Original S-PU over the FULL effective pair batch, including empty means."""
    pos = labels.bool()
    neg = ~pos
    d = logits[:, 1:] - logits[:, :1]
    def average(value, mask):
        return (value * mask).sum(0) / mask.sum(0).clamp_min(1)
    pl = priors.to(device=logits.device)
    po = pl * e
    pu = (po - pl) / (1 - pl)
    first = ((1-po)/(1-pu) * average((d+margin).square(), neg)
             - pu*(1-po)/(1-pu) * average((d+margin).square(), pos))
    second = po * torch.sqrt((1-po)/po) * average((d-margin).square(), pos)
    return torch.where(first < beta, -gamma*first, first+second).sum()


class PairHead(nn.Module):
    def __init__(self, kind, reference_weight, reference_bias, priors, positive_rate,
                 rel2id, block_size=64):
        super().__init__()
        if kind not in KINDS or not 0 < positive_rate < 1:
            raise ValueError("Invalid head kind/presence prior")
        self.kind, self.block_size = kind, block_size
        self.relations = reference_weight.shape[0] - 1
        dim = reference_weight.shape[1]
        self.register_buffer("priors", torch.as_tensor(priors).clone().double())
        if len(priors) != self.relations or torch.any(self.priors <= 0) or torch.any(self.priors >= 1):
            raise ValueError("S-PU priors must be strictly within (0,1)")
        # Construct on CPU with a private RNG; never perturb the shuffle stream.
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(75)
            if kind in ("factor", "set"):
                self.presence = nn.Linear(dim, 1)
                self.conditional = nn.Linear(dim, self.relations)
            else:
                self.linear = nn.Linear(dim, 100 if kind == "amtl" else
                                        self.relations if kind == "bce" else self.relations+1)
        w, bias = reference_weight.detach().cpu(), reference_bias.detach().cpu()
        self.groups = []
        with torch.no_grad():
            if kind in ("factor", "set", "bce"):
                layer = self.linear if kind == "bce" else self.conditional
                layer.weight.copy_(w[1:] - w[:1])
                layer.bias.copy_(bias[1:] - bias[:1])
                if kind != "bce":
                    self.presence.weight.zero_()
                    self.presence.bias.fill_(math.log(positive_rate/(1-positive_rate)))
            elif kind == "amtl":
                names = [r for group in AMTL_GROUPS for r in group.split()]
                if len(names) != 96 or set(names) != set(rel2id)-{"Na"}:
                    raise ValueError("Canonical AMTL mapping requires the exact96 relation IDs")
                cursor = 0
                for group in AMTL_GROUPS:
                    ids = [rel2id[r]-1 for r in group.split()]
                    start = cursor
                    for rid in ids:
                        cursor += 1
                        self.linear.weight[cursor].copy_(w[rid+1])
                        self.linear.bias[cursor].copy_(bias[rid+1])
                    self.groups.append((start, cursor+1, ids))
                    cursor += 1
                # TH rows keep four DISTINCT native Linear random initializations.
            else:
                self.linear.weight.copy_(w)
                self.linear.bias.copy_(bias)

    def forward(self, vectors):
        x = block_product(vectors, self.block_size)
        if self.kind in ("factor", "set"):
            return {"g": self.presence(x).squeeze(-1), "b": self.conditional(x)}
        return self.linear(x)

    def batched_forward(self, vectors, chunk_size=256):
        values = [self(vectors[i:i+chunk_size]) for i in range(0, len(vectors), chunk_size)]
        if isinstance(values[0], dict):
            return {k: torch.cat([v[k] for v in values]) for k in ("g", "b")}
        return torch.cat(values)

    def loss(self, values, labels):
        y = labels.to(dtype=torch.float32)
        if self.kind == "spu":
            return spu_risk(values, y, self.priors)
        if self.kind == "atgl":
            target = torch.cat((torch.ones_like(y[:, :1]), 16.5*y), dim=-1)
            return -(F.log_softmax(values, -1) * target).sum(-1).mean()
        if self.kind == "amtl":
            losses = []
            ths = [start for start, _, _ in self.groups]
            threshold = values[:, ths].sum(-1, keepdim=True)/3.5
            for start, end, ids in self.groups:
                z = torch.cat((threshold, values[:, start+1:end]), dim=-1)
                target = torch.cat((torch.zeros_like(threshold), y[:, ids]), dim=-1)
                th = torch.zeros_like(target)
                th[:, 0] = 1
                loss1 = -(F.log_softmax(z-(1-target-th)*1e30, -1)*target).sum(-1)
                loss2 = -(F.log_softmax(z-target*1e30, -1)*th).sum(-1)
                losses.append(loss1+loss2)
            return torch.stack(losses).mean()
        if self.kind == "bce":
            return F.binary_cross_entropy_with_logits(values, y, reduction="none").sum(-1).mean()
        active = y.any(-1)
        g, b = values["g"], values["b"]
        loss = F.binary_cross_entropy_with_logits(g.double(), active.double(), reduction="sum")
        if active.any():
            if self.kind == "set":
                loss = loss + conditional_nll(b[active], y[active]).sum()
            else:
                loss = loss + F.binary_cross_entropy_with_logits(
                    b[active].double(), y[active].double(), reduction="sum")
        # Without positives the conditional graph is unused, hence grad=None;
        # AdamW must skip it as well, including weight decay.
        return loss/len(y)

    def scores(self, values):
        """Positive score means native relation decision; ordering is native."""
        if self.kind in ("spu", "atgl"):
            return values[:, 1:] - values[:, :1]
        if self.kind == "amtl":
            out = torch.empty(len(values), self.relations, device=values.device, dtype=values.dtype)
            for start, end, ids in self.groups:
                out[:, ids] = values[:, start+1:end]-values[:, start:start+1]
            return out
        if self.kind == "bce":
            return values
        log_p = F.logsigmoid(values["g"].double()).unsqueeze(-1) + F.logsigmoid(values["b"].double())
        if self.kind == "set":
            log_p = log_p-log_nonempty(values["b"]).unsqueeze(-1)
        return log_p - math.log(.5)


def confusion(pred, gold):
    tp = int(np.count_nonzero(pred & gold))
    fp = int(np.count_nonzero(pred & ~gold))
    fn = int(np.count_nonzero(~pred & gold))
    return {"TP": tp, "FP": fp, "FN": fn,
            "F1": 200*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.}


def score_array(pred, gold, ignored_annotated, ignored_distant):
    c = confusion(pred, gold)
    correct, count, total = c["TP"], c["TP"]+c["FP"], c["TP"]+c["FN"]
    recall = correct/total if total else 0.
    precision = correct/count if count else 0.
    def ign(mask):
        removed = int(np.count_nonzero(pred & gold & mask))
        p = (correct-removed)/(count-removed+1e-5)
        return 200*p*recall/(p+recall) if p+recall else 0.
    return {"dev_F1": c["F1"], "dev_precision": precision*100, "dev_recall": recall*100,
            "dev_IgnF1_annotated": ign(ignored_annotated),
            "dev_IgnF1_distant": ign(ignored_distant)}


def diagnostics(scores, gold, buckets, top_k):
    pred = scores > 0
    count = gold.sum(-1)
    subset = {name: confusion(pred[mask], gold[mask]) for name, mask in
              (("na_pairs", count == 0), ("positive_pairs", count > 0),
               ("single_label", count == 1), ("multi_label", count > 1))}
    frequency = {}
    for name in ("rare", "middle", "frequent", "rare_middle"):
        cols = [r-1 for r, bucket in buckets.items()
                if bucket == name or (name == "rare_middle" and bucket in ("rare", "middle"))]
        frequency[name] = confusion(pred[:, cols], gold[:, cols])
    if not 0 < top_k <= scores.size:
        raise ValueError("FixedK outside available relation decisions")
    # Stable canonical flatten order is the tie breaker, never gold labels.
    flat = scores.reshape(-1)
    selected = np.argpartition(flat, flat.size-top_k)[-top_k:]
    cutoff = flat[selected].min()
    strict = np.flatnonzero(flat > cutoff)
    ties = np.flatnonzero(flat == cutoff)[:top_k-len(strict)]
    chosen = np.concatenate((strict, ties))
    top_pred = np.zeros(flat.size, dtype=bool)
    top_pred[chosen] = True
    top = confusion(top_pred.reshape(scores.shape), gold)
    top["K"] = top_k
    return {"subsets": subset, "frequency": frequency, "fixed_K": top}
