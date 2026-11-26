"""N1 training rule: causal dropout weights routed only to shared memory."""
from __future__ import annotations

import math
import ttmre_memory_noise_preflight as preflight


def active(stage, epoch):
    if stage not in ("distant", "human-frozen") or epoch < 1:
        raise ValueError("Invalid stage/epoch")
    return stage == "distant" and epoch == 2


def view_seed(step):
    return 740000 + step  # Separate from the unchanged seed74 student stream.


def labels_tensor(inputs, device):
    import torch
    return torch.tensor([row for doc in inputs["labels"] for row in doc], device=device)


def backward(model, inputs, stage, epoch, step, collect=False, unity=False):
    """No optimizer/zero_grad/step here. Caller owns clearing, global clip and update."""
    import torch
    use_weights = active(stage, epoch)
    labels, weights = None, None
    if use_weights:
        if not model.mu_encoder.memory_tokens.requires_grad:
            raise ValueError("Weighted Distant memory must be trainable")
        labels = labels_tensor(inputs, inputs["input_ids"].device)
        weights = torch.ones_like(labels, dtype=torch.float32) if unity else preflight.two_views(model, inputs, labels, view_seed(step))
    elif unity:
        raise ValueError("Unity routing contract requires active Distant epoch2")
    risks, logits = model(**inputs)
    loss = sum(risks)
    if not torch.isfinite(loss).all() or not torch.isfinite(logits).all():
        raise RuntimeError("Nonfinite N1 forward; no optimizer update allowed")
    routed, branches, original_memory = None, [], None
    weighted_loss = loss
    if use_weights:
        weighted_loss, branches = preflight.weighted_spu(model, logits, labels, weights)
        if not torch.isfinite(weighted_loss).all():
            raise RuntimeError("Nonfinite weighted risk; no optimizer update allowed")
        if unity:
            torch.testing.assert_close(weighted_loss, loss, rtol=2e-6, atol=2e-6)
        routed, = torch.autograd.grad(weighted_loss, model.mu_encoder.memory_tokens, retain_graph=True)
        routed = routed.detach()
        if not torch.isfinite(routed).all():
            raise RuntimeError("Nonfinite routed memory gradient")
    loss.backward()
    if routed is not None:
        if collect:
            original_memory = model.mu_encoder.memory_tokens.grad.detach().clone()
        model.mu_encoder.memory_tokens.grad = routed
    result = {"loss": float(loss.detach()), "weighted_risk": float(weighted_loss.detach()),
              "weighting_active": use_weights, "branch_changes": sum(
                  b["correction_branch"] != b["original_correction_branch"] for b in branches)}
    if use_weights:
        counts = labels[:, 1:].eq(1).sum(0)
        result.update({"positive_terms": int(counts.sum()),
                       "eligible_positive_terms": int(counts[counts >= 2].sum()),
                       "singleton_positive_terms": int(counts[counts == 1].sum())})
    if use_weights and collect:
        old, new = original_memory.double().flatten(), routed.double().flatten()
        norm = float(old.norm())
        if norm == 0 or float(new.norm()) == 0:
            raise RuntimeError("Missing memory gradient in N1 diagnostics")
        result["memory_gradient"] = {"original_norm": norm, "routed_norm": float(new.norm()),
            "relative_change": float((new - old).norm()) / norm,
            "cosine": float(torch.dot(old, new) / (old.norm() * new.norm()))}
        result["weight_groups"] = preflight.weight_summary(weights, labels)
    return result


def add_statistics(summary, record, buckets):
    """All active batches count coverage; detailed weight statistics sampled every100 updates."""
    if not record["weighting_active"]:
        return
    for key in ("positive_terms", "eligible_positive_terms", "singleton_positive_terms", "branch_changes"):
        summary[key] = summary.get(key, 0) + record[key]
    summary["active_batches"] = summary.get("active_batches", 0) + 1
    if "weight_groups" not in record:
        return
    summary["sampled_batches"] = summary.get("sampled_batches", 0) + 1
    for row in record["weight_groups"]:
        bucket = buckets[row["relation_id"]]
        entry = summary.setdefault("sampled_frequency_buckets", {}).setdefault(bucket, {
            "positive_terms": 0, "eligible_positive_terms": 0, "groups": 0, "cv_sum": 0., "minimum_ess_fraction": 1.})
        entry["positive_terms"] += row["positive"]
        entry["eligible_positive_terms"] += row["positive"] if row["eligible"] else 0
        entry["groups"] += 1
        entry["cv_sum"] += row["effective_weight_cv"]
        entry["minimum_ess_fraction"] = min(entry["minimum_ess_fraction"], row["ess_fraction"])


def frequency_buckets(priors):
    if len(priors) != 96 or any(not math.isfinite(x) or x < 0 for x in priors):
        raise ValueError("Expected96 finite training-frequency priors")
    ordered = sorted(range(1, 97), key=lambda r: (priors[r - 1], r))
    return {r: ("rare", "middle", "frequent")[i // 32] for i, r in enumerate(ordered)}
