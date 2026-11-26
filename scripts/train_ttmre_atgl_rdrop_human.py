"""One authorized ATGL + R-Drop Human pilot; all work within six hours.

Default validate only reads metadata and hashes. Use the one-use Linux launcher
for training. Original artifacts and sources remain read-only.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import signal
import sys
import tarfile
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import train_ttmre_atgl_human as base

core, frozen, accepted = base.core, base.frozen, base.accepted
read, digest, emit, write_json = base.read, base.digest, base.emit, base.write_json
check_budget, supervise = frozen.check_budget, frozen.supervise
RUN_ROOT = ROOT / "experiments/2026-10-19-atgl-rdrop-human"
PARENT, PARENT_SHA, CONTROL_ORDER = base.PARENT, base.PARENT_SHA, base.CONTROL_ORDER
REFERENCE_ROOT = base.RUN_ROOT
REFERENCE_SHA = "f795bfc8c51f4b740d55abe992811e728481f36a4167b43036cbb8ca46673c37"
REFERENCE_REPORT_SHA = "6b919e7600a404d9947b437c2786a588894242796a5cc27a5e8c482ad3b82207"
REFERENCE_METRICS = {"dev_F1": 79.86936405697179, "dev_precision": 83.66170414950903,
    "dev_recall": 76.40592455450127, "dev_IgnF1_annotated": 78.83487232479382,
    "dev_IgnF1_distant": 77.86594189912688}
TRAIN_DOCS, DEV_DOCS, EPOCHS = 3053, 500, 30
HARD_SECONDS, SOFT_SECONDS, TERM_SECONDS = 21600, 120, 30
FILES = tuple("scripts/" + n for n in ("train_ttmre_atgl_rdrop_human.py",
    "check_ttmre_atgl_rdrop_human.py", "run_ttmre_atgl_rdrop_human.sh"))
PROGRESS = {"formal_optimizer_updates": 0, "phase": "not-started"}


def protocol():
    p = base.protocol()
    p.update({"version": "atgl-rdrop-human-v1", "epochs": EPOCHS,
        "optimizer_calls": (TRAIN_DOCS // 4) * EPOCHS,
        "warmup_steps": int((TRAIN_DOCS // 4) * EPOCHS * .06),
        "hard_seconds": HARD_SECONDS, "rdrop_lambda": 1.,
        "objective": "mean_two_ATGL + mean_pairs(sum_classes(symmetric_KL(softmax97)))",
        "training_views": "independent backbone dropout; second view private seed750000000+step",
        "gradient_execution": "three grad-mode forwards, two backwards, one optimizer call; one live graph",
        "first_view_replay": "same RNG in private scope; original rtol1e-4/atol1e-4",
        "main_rng": "one ordinary first-view forward per batch; private views leave it unchanged",
        "inference": "single eval forward; original relation>TH decoder",
        "reference_checkpoint_sha256": REFERENCE_SHA,
        "reference_F1": REFERENCE_METRICS["dev_F1"], "screening_delta_pp": .30,
        "both_Ign_deltas_positive_required": True})
    return p


def reference(budget=None):
    claim = read(REFERENCE_ROOT / "atgl-human-once-claim.json")
    workspace = Path(claim["workspace"]).resolve()
    if workspace.name != "workspace" or workspace.parent.parent != REFERENCE_ROOT.resolve():
        raise ValueError("Reference is not the original ATGL workspace")
    report = workspace / "comparison/completion.json"
    checkpoint = workspace / "atgl/human-frozen/artifacts/best_model.pth"
    if digest(report, budget) != REFERENCE_REPORT_SHA or digest(checkpoint, budget) != REFERENCE_SHA:
        raise ValueError("Pinned ATGL reference changed")
    complete = read(report)
    if (not complete.get("passed") or complete.get("marker") != "ATGL_HUMAN_COMPLETE"
            or complete.get("optimizer_steps") != 22890
            or complete.get("test_data_read") is not False
            or complete["protocol"]["parent_sha256"] != PARENT_SHA
            or any(complete["atgl"].get(k) != v for k, v in REFERENCE_METRICS.items())):
        raise ValueError("Incomplete original ATGL reference")
    return {"workspace": str(workspace), "checkpoint": str(checkpoint),
        "checkpoint_sha256": REFERENCE_SHA, "completion": str(report),
        "completion_sha256": REFERENCE_REPORT_SHA, "best": complete["atgl"]}


def validate(budget=None):
    identity = base.validate(budget)
    ref = reference(budget)
    identity.update({"protocol": protocol(), "atgl_reference": ref})
    identity["source_files"].update({n: digest(ROOT / n, budget) for n in FILES})
    identity["input_files"].update({ref["checkpoint"]: REFERENCE_SHA,
                                     ref["completion"]: REFERENCE_REPORT_SHA})
    return identity


def verify_sources(identity, budget):
    if identity["protocol"] != protocol():
        raise ValueError("R-Drop protocol changed")
    for name, sha in identity["source_files"].items():
        if digest(ROOT / name, budget) != sha:
            raise ValueError("Source changed: " + name)


@contextmanager
def private_rng(*, seed=None, state=None):
    import numpy as np
    import torch
    before = core.preflight.rng_state()
    try:
        if state is not None:
            core.preflight.restore_rng(state)
        elif seed is not None:
            core.shared.seed_all(seed, torch, np)
        else:
            raise ValueError("Private RNG requires a seed or a state")
        yield
    finally:
        core.preflight.restore_rng(before)


def symmetric_kl(first, second):
    import torch
    if first.shape != second.shape or first.ndim != 2 or not len(first):
        raise ValueError("Expected matching nonempty pair/class logits")
    logp = torch.nn.functional.log_softmax(first, dim=-1)
    logq = torch.nn.functional.log_softmax(second, dim=-1)
    # Both distributions retain their derivatives, including the target roles.
    return .5 * ((logp.exp() - logq.exp()) * (logp - logq)).sum(-1).mean()


def logits_forward(model, batch, device):
    import torch
    z = model(**core.shared.inputs_from_batch(batch, device, labels=False))
    if z.ndim != 2 or z.shape[1] != 97 or not torch.isfinite(z).all():
        raise RuntimeError("Invalid/nonfinite R-Drop logits; no optimizer call")
    return z


def rdrop_backward(model, batch, device, step, budget=None):
    """Accumulate the exact two-view objective; no zero_grad/clip/optimizer here."""
    import torch
    if not model.training:
        raise ValueError("R-Drop requires training mode")
    if budget is not None:
        check_budget(budget)
    labels = torch.cat([torch.as_tensor(y, device=device) for y in batch[2]], dim=0)
    initial_rng = core.preflight.rng_state()
    # Use grad mode on the discarded first graph too, to avoid no-grad/grad
    # kernel differences on the target runtime. Only detached logits survive.
    z = logits_forward(model, batch, device)
    first = z.detach()
    del z
    model.sims = None
    after_first = core.preflight.rng_state()
    try:
        if budget is not None:
            check_budget(budget)
        with private_rng(seed=750000000 + step):
            z = logits_forward(model, batch, device)
            second = z.detach()
            ce2, kl = base.atgl_loss(z, labels), symmetric_kl(first, z)
            loss = .5 * ce2 + kl
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite second-view objective")
            ce2_value, kl_value = float(ce2.detach()), float(kl.detach())
            difference = float((first - second).abs().max())
            loss.backward()
            del loss, ce2, kl, z
            model.sims = None
        if budget is not None:
            check_budget(budget)
        with private_rng(state=initial_rng):
            z = logits_forward(model, batch, device)
            replay_delta = float((z.detach() - first).abs().max())
            if not torch.allclose(z.detach(), first, rtol=1e-4, atol=1e-4):
                raise RuntimeError("First-view RNG replay differs at original tolerance")
            ce1 = base.atgl_loss(z, labels)
            loss = .5 * ce1 + symmetric_kl(z, second)
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite first-view objective")
            ce1_value = float(ce1.detach())
            loss.backward()
            del loss, ce1, z
            model.sims = None
        core.preflight.assert_rng_equal(after_first, core.preflight.rng_state())
        return {"loss": .5 * (ce1_value + ce2_value) + kl_value,
            "supervised_loss": .5 * (ce1_value + ce2_value), "symmetric_KL": kl_value,
            "view_logits_max_abs_delta": difference, "first_replay_max_abs_delta": replay_delta,
            "main_rng_matches_one_view": True, "forward_calls": 3, "backward_calls": 2,
            "extra_optimizer_updates": 0}
    finally:
        model.sims = None


def probe(model, dataset, args, budget):
    import torch
    from ttmre_memory_fit import reserve_adam_moments
    before, rng = core.preflight.state_digest(model), core.preflight.rng_state()
    modes = [(m, m.training) for m in model.modules()]
    buffers, reservation = reserve_adam_moments(model)
    reports = []
    try:
        for index, (name, indices) in enumerate(dataset.probe_indices().items(), 1):
            check_budget(budget)
            model.train(); model.zero_grad(set_to_none=True)
            record = rdrop_backward(model, core.shared.collate_fn([dataset[i] for i in indices]),
                                    args.device, index, budget)
            gradients = [p.grad for p in model.parameters() if p.grad is not None]
            critical = [model.head_extractor.weight, model.bilinear.weight,
                        model.model.embeddings.word_embeddings.weight]
            if (not gradients or any(not torch.isfinite(g).all() for g in gradients)
                    or any(p.grad is None or not torch.count_nonzero(p.grad) for p in critical)
                    or model.mu_encoder.memory_tokens.grad is not None
                    or record["view_logits_max_abs_delta"] <= 0 or record["symmetric_KL"] <= 0):
                raise RuntimeError("Missing finite gradients, frozen-memory contract, or distinct dropout views")
            reports.append({"probe": name, "indices": indices, "finite_gradients": len(gradients), **record})
            emit("RDROP_ZERO_UPDATE_PROBE", **reports[-1])
            del gradients
    finally:
        model.zero_grad(set_to_none=True); model.sims = None
        del buffers
        for m, flag in modes:
            m.training = flag
        core.preflight.restore_rng(rng)
        if args.device.type == "cuda":
            torch.cuda.empty_cache()
    if core.preflight.state_digest(model) != before:
        raise RuntimeError("Probe changed parameters or buffers")
    core.preflight.assert_rng_equal(rng, core.preflight.rng_state())
    return {"passed": True, "probes": reports, "optimizer_steps": 0,
            "parameters_unchanged": True, "rng_restored": True, "reservation": reservation}


def train_loop(model, loader, dev_loader, args, directory, gold, annotated, distant, budget, optimizer, scheduler):
    import torch
    best, best_epoch, best_step, steps = -math.inf, None, None, 0
    memory = model.mu_encoder.memory_tokens.detach().cpu().clone()
    order = hashlib.sha256()
    learning = None
    kl_sum = loss_sum = 0.
    nonidentical = 0
    with (directory / "train.log").open("x", encoding="utf-8") as log, \
            (directory / "metrics.jsonl").open("x", encoding="utf-8") as metrics_log:
        for epoch in range(1, EPOCHS + 1):
            check_budget(budget)
            started, total_loss = time.perf_counter(), 0.
            emit("epoch_start", epoch=epoch, batches=len(loader), branch="atgl-rdrop", stage="human-frozen")
            for index, batch in enumerate(loader):
                check_budget(budget)
                order.update(batch[0].numpy().tobytes())
                order.update(json.dumps(batch[3:5], separators=(",", ":"), ensure_ascii=False).encode())
                model.train(); model.zero_grad(set_to_none=True)
                record = rdrop_backward(model, batch, args.device, steps + 1, budget)
                if not any(p.grad is not None for p in model.parameters() if p.requires_grad):
                    raise RuntimeError("No R-Drop training gradients")
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
                witness = None
                if learning is None:
                    rates = {id(p): g["lr"] for g in optimizer.param_groups for p in g["params"]}
                    for name, p in sorted(model.named_parameters(), key=lambda x: x[0] != "head_extractor.weight"):
                        if p.requires_grad and p.grad is not None and rates[id(p)] > 0 and torch.count_nonzero(p.grad):
                            witness = (name, p, p.detach().clone(), rates[id(p)])
                            break
                optimizer.step(); scheduler.step(); optimizer.zero_grad(set_to_none=True)
                steps += 1
                PROGRESS["formal_optimizer_updates"] = steps
                model._n1_completed_optimizer_steps = steps
                if witness is not None:
                    name, p, before, rate = witness
                    state_step = int(optimizer.state[p]["step"].item())
                    if torch.equal(before, p.detach()) or state_step != steps:
                        raise RuntimeError("Positive-LR update did not change the witnessed parameter")
                    learning = {"passed": True, "branch": "atgl-rdrop", "stage": "human-frozen",
                        "step": steps, "parameter": name, "lr": rate, "optimizer_state_step": state_step,
                        "parameter_changed": True, "extra_optimizer_updates": 0}
                    write_json(directory / "learning_update_check.json", learning)
                    emit("ATGL_RDROP_LEARNING_UPDATE_VERIFIED", **learning)
                    del before, witness
                total_loss += record["loss"]
                loss_sum += record["supervised_loss"]; kl_sum += record["symmetric_KL"]
                nonidentical += record["view_logits_max_abs_delta"] > 0
                if steps == 1 or steps % 100 == 0:
                    row = {"event": "optimizer_update", "branch": "atgl-rdrop", "stage": "human-frozen",
                           "epoch": epoch, "step": steps, "batch": index + 1, **record}
                    line = json.dumps(row, allow_nan=False)
                    print(line, flush=True); log.write(line + "\n"); log.flush()
                if index + 1 == len(loader) - 1:
                    values = core.evaluate(model, dev_loader, args, gold, annotated, distant)
                    row = core.selection_row(epoch, len(loader), values, best)
                    row.update({"branch": "atgl-rdrop", "stage": "human-frozen"})
                    if row["selected"]:
                        best, best_epoch, best_step = row["dev_F1"], epoch, steps
                        core.atomic_torch_save(core.shared.cpu_snapshot(model), directory / "best_model.pth")
                        write_json(directory / "best_metrics.json", {k: v for k, v in row.items() if k != "selected"})
                    row["train_loss_so_far"] = total_loss / (index + 1)
                    line = json.dumps(row, allow_nan=False)
                    print(line, flush=True)
                    for f in (log, metrics_log):
                        f.write(line + "\n"); f.flush()
            if not torch.equal(memory, model.mu_encoder.memory_tokens.detach().cpu()):
                raise RuntimeError("Frozen memory changed")
            seconds = time.perf_counter() - started
            timing = {"event": "epoch_complete", "epoch": epoch, "step": steps, "seconds": seconds,
                      "train_loss": total_loss / len(loader), "scope": "training plus epoch Dev/save"}
            emit("epoch_complete", **{k: v for k, v in timing.items() if k != "event"})
            log.write(json.dumps(timing, allow_nan=False) + "\n"); log.flush()
            if epoch == 1:
                estimate = seconds * (EPOCHS - 1) + 900
                remaining = budget["deadline_monotonic"] - time.monotonic() - SOFT_SECONDS
                emit("RDROP_REMAINING_PROJECTION", estimate_seconds=estimate, remaining_seconds=remaining)
                if estimate > remaining:
                    raise core.BudgetStop("Measured R-Drop projection exceeds the original six-hour budget")
    if learning is None or nonidentical != steps or kl_sum <= 0:
        raise RuntimeError("No verified learning or active R-Drop views")
    return {"best_epoch": best_epoch, "best_step": best_step, "best_dev_F1": best,
        "optimizer_steps": steps, "memory_unchanged": True, "learning_update_check_passed": True,
        "sample_order_sha256": order.hexdigest(), "final_rng_sha256": core.rng_digest(),
        "mean_supervised_loss": loss_sum / steps, "mean_symmetric_KL": kl_sum / steps,
        "nonidentical_view_batches": nonidentical, "test_data_read": False, "test_evaluated": False}


def export_predictions(model, loader, args, gold, annotated, distant, buckets, best, directory, budget):
    import numpy as np
    import torch
    from evaluation import to_official
    directory.mkdir(parents=True, exist_ok=True)
    predicted, margins, pair_order, offset = [], [], [], 0
    model.eval()
    with torch.no_grad():
        for batch in loader:
            check_budget(budget)
            logits = logits_forward(model, batch, args.device)
            values = logits.cpu().numpy()
            flags = np.zeros_like(values)
            flags[:, 1:] = values[:, 1:] > values[:, :1]
            flags[:, 0] = flags[:, 1:].sum(1) == 0
            count = len(batch[0]); features = loader.dataset[offset:offset + count]
            predicted.extend(to_official(flags, features))
            margins.append(values[:, 1:] - values[:, :1])
            pair_order.extend({"title": f["title"], "hts": f["hts"]} for f in features)
            offset += count
    metrics = core.shared.score_predictions(predicted, gold, annotated, distant)
    if offset != len(gold) or any(not math.isclose(metrics[k], best[k], abs_tol=1e-6, rel_tol=0)
                                   for k in core.baseline.m1.METRIC_KEYS):
        raise RuntimeError("Export does not replay the selected metrics")
    write_json(directory / "predictions.json", predicted)
    write_json(directory / "dev_pair_order.json", pair_order)
    np.save(directory / "dev_margins.npy", np.concatenate(margins), allow_pickle=False)
    rel2id = read(ROOT / "meta/rel2id.json")
    truth = {(d["title"], l["h"], l["t"], l["r"]) for d in gold for l in d["labels"]}
    predictions = {(p["title"], p["h_idx"], p["t_idx"], p["r"]) for p in predicted}
    report = {}
    for name in ("rare", "middle", "frequent"):
        g = {f for f in truth if buckets[rel2id[f[3]]] == name}
        p = {f for f in predictions if buckets[rel2id[f[3]]] == name}
        tp, fp, fn = len(g & p), len(p - g), len(g - p)
        report[name] = {"TP": tp, "FP": fp, "FN": fn, "gold": len(g), "predicted": len(p),
            "F1": 200 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.}
    result = {"passed": True, "metrics": metrics, "buckets": report, "optimizer_steps": 0,
        "test_data_read": False, "predictions_sha256": digest(directory / "predictions.json", budget),
        "margins_sha256": digest(directory / "dev_margins.npy", budget),
        "pair_order_sha256": digest(directory / "dev_pair_order.json", budget)}
    write_json(directory / "frequency_diagnostics.json", result)
    return result, predictions, truth


def screening_gate(best, reference_best):
    keys = core.baseline.m1.METRIC_KEYS
    if not all(math.isfinite(float(x[k])) for x in (best, reference_best) for k in keys):
        raise ValueError("Nonfinite comparison metric")
    delta = {k: best[k] - reference_best[k] for k in keys}
    passed = delta["dev_F1"] >= .30 and all(delta[k] > 0 for k in ("dev_IgnF1_annotated", "dev_IgnF1_distant"))
    return bool(passed), delta


def execute(workspace, identity, budget):
    import numpy as np
    import torch
    from torch.utils.data import DataLoader
    from transformers.optimization import get_linear_schedule_with_warmup
    verify_sources(identity, budget)
    device = core.baseline.control.runtime(identity["environment"])
    args = SimpleNamespace(run_dir=workspace, baseline_workspace=Path(identity["baseline_workspace"]),
        model_dir=Path(identity["model_dir"]), data_dir=Path(identity["data_dir"]), device=device, budget=budget)
    datasets = frozen.original_features(identity)
    human, dev = datasets["train"], datasets["dev"]
    if len(human) != TRAIN_DOCS or len(dev) != DEV_DOCS:
        raise ValueError("Human/Dev counts differ")
    gold = read(args.data_dir / "dev_revised.json")
    annotated = {tuple(x) for x in human.metadata["matching_dev_facts"]}
    distant = {tuple(x) for x in read(args.baseline_workspace / "cache/distant/index.json")["matching_dev_facts"]}
    buckets = core.method.frequency_buckets(human.metadata["priors"])
    model = core.load_generic_model_seed75(args, identity["baseline_identity"])
    model.load_state_dict(torch.load(PARENT, map_location="cpu", weights_only=True), strict=True)
    model.requires_grad_(True); core.baseline.set_stage(model, "human-frozen")
    memory = model.mu_encoder.memory_tokens.detach().cpu().clone()
    directory = workspace / "atgl-rdrop/human-frozen/artifacts"
    directory.mkdir(parents=True, exist_ok=False)
    write_json(directory / "initialization.json", {"parent_sha256": PARENT_SHA,
        "state_sha256": core.preflight.state_digest(model),
        "frozen": [n for n, p in model.named_parameters() if not p.requires_grad]})
    PROGRESS["phase"] = "zero-update-probes"
    write_json(directory / "full_model_check.json", probe(model, human, args, budget))
    optimizer = core.baseline.optimizer_for_stage(model, "human-frozen")
    if optimizer.state:
        raise RuntimeError("Optimizer must start empty")
    core.shared.seed_all(75, torch, np)
    loader = DataLoader(human, batch_size=4, shuffle=True, drop_last=True, collate_fn=core.shared.collate_fn)
    dev_loader = DataLoader(dev, batch_size=4, shuffle=False, collate_fn=core.shared.collate_fn)
    total = len(loader) * EPOCHS
    if total != protocol()["optimizer_calls"]:
        raise ValueError("Unexpected formal optimizer-call count")
    scheduler = get_linear_schedule_with_warmup(optimizer, int(total * .06), total)
    write_json(directory / "manifest.json", {"protocol": protocol(), "identity_sha256": digest(workspace / "identity.json"),
        "optimizer_groups": core.baseline.optimizer_report(model, optimizer),
        "initial_rng_sha256": core.rng_digest(), "total_steps": total})
    PROGRESS["phase"] = "human-training"
    result = train_loop(model, loader, dev_loader, args, directory, gold, annotated, distant, budget, optimizer, scheduler)
    del optimizer, scheduler
    if result["optimizer_steps"] != total or result["sample_order_sha256"] != CONTROL_ORDER:
        raise RuntimeError("Incomplete learning or Human sample order differs from original ATGL/Control75")
    write_json(directory / "training_order.json", {"seed": 75, "sample_order_sha256": result["sample_order_sha256"],
        "formal_optimizer_updates": total, "same_as_original_control_and_ATGL": True})
    PROGRESS["phase"] = "best-replay"
    replay = accepted.replay(model, directory, dev_loader, args, "human-frozen", gold, annotated, distant)
    write_json(directory / "checkpoint_replay.json", replay)
    if not torch.equal(memory, model.mu_encoder.memory_tokens.detach().cpu()):
        raise RuntimeError("Frozen memory differs after best replay")
    best = read(directory / "best_metrics.json")
    frequency, predictions, truth = export_predictions(model, dev_loader, args, gold, annotated, distant,
                                                       buckets, best, directory, budget)
    result.update({"passed": True, "protocol": protocol(), "best": best, "parent_sha256": PARENT_SHA,
        "checkpoint_sha256": digest(directory / "best_model.pth", budget), "checkpoint_replay_passed": True,
        "frequency_diagnostics_passed": frequency["passed"], "historical_control_strict_numeric_passed": False})
    write_json(directory / "completion.json", result)
    PROGRESS["phase"] = "reference-prediction-export"
    ref = identity["atgl_reference"]
    if digest(Path(ref["checkpoint"]), budget) != REFERENCE_SHA:
        raise RuntimeError("ATGL reference changed")
    model.load_state_dict(torch.load(ref["checkpoint"], map_location="cpu", weights_only=True), strict=True)
    reference_frequency, reference_predictions, unused = export_predictions(model, dev_loader, args, gold,
        annotated, distant, buckets, ref["best"], workspace / "reference-atgl", budget)
    del model
    comparison = workspace / "comparison"; comparison.mkdir()
    per_document = []
    for d in gold:
        title = d["title"]
        g = {x for x in truth if x[0] == title}
        a = {x for x in predictions if x[0] == title}
        b = {x for x in reference_predictions if x[0] == title}
        per_document.append({"title": title, "gold": len(g), "rdrop_TP": len(a & g), "rdrop_FP": len(a - g),
            "atgl_TP": len(b & g), "atgl_FP": len(b - g), "added_TP": len((a - b) & g),
            "lost_TP": len((b - a) & g), "added_FP": len((a - b) - g), "removed_FP": len((b - a) - g)})
    write_json(comparison / "paired_documents.json", per_document)
    gate, deltas = screening_gate(best, ref["best"])
    verify_sources(identity, budget)
    if digest(PARENT, budget) != PARENT_SHA:
        raise RuntimeError("Distant parent changed")
    complete = {"passed": True, "marker": "ATGL_RDROP_HUMAN_COMPLETE", "protocol": protocol(),
        "atgl_rdrop": best, "reference_atgl": ref, "delta_percentage_points": deltas,
        "screening_gate_passed": gate, "optimizer_steps": total, "diagnostic_optimizer_steps": 0,
        "reference_export_optimizer_steps": 0, "reference_export": reference_frequency,
        "elapsed_seconds": time.monotonic() - budget["start_monotonic"],
        "automatic_followup": False, "reliable_improvement_established": False, "novelty_established": False,
        "test_data_read": False, "test_evaluated": False}
    check_budget(budget)
    write_json(comparison / "completion.json", complete)
    archive = comparison / "reports.tar.gz"
    with tarfile.open(archive, "x:gz") as target:
        for path in workspace.rglob("*"):
            check_budget(budget)
            if path.is_file() and path.suffix in (".json", ".jsonl", ".log"):
                target.add(path, arcname=str(path.relative_to(workspace)), recursive=False)
    PROGRESS["phase"] = "complete"
    emit("ATGL_RDROP_HUMAN_COMPLETE", F1=best["dev_F1"], delta_F1_pp=deltas["dev_F1"],
         screening_gate_passed=gate, reports=str(archive), automatic_followup=False)


def require_lock():
    if os.name != "posix" or os.environ.get("TTMRE_ATGL_RDROP_LOCK_FD") != "9":
        raise RuntimeError("Use run_ttmre_atgl_rdrop_human.sh")
    import fcntl
    a = os.fstat(9)
    b = (ROOT / "experiments/2026-10-08-baseline-alignment/.training.lock").stat()
    if (a.st_dev, a.st_ino) != (b.st_dev, b.st_ino):
        raise RuntimeError("Wrong shared lock descriptor")
    fcntl.flock(9, fcntl.LOCK_EX | fcntl.LOCK_NB)
    return 9


def verify_context(workspace, budget):
    claim = read(RUN_ROOT / "atgl-rdrop-once-claim.json")
    if (workspace.name != "workspace" or workspace.parent.parent != RUN_ROOT.resolve()
            or budget["hard_seconds"] != HARD_SECONDS
            or budget["deadline_monotonic"] - budget["start_monotonic"] != HARD_SECONDS
            or budget.get("soft_reserve_seconds") != SOFT_SECONDS
            or budget.get("term_reserve_seconds") != TERM_SECONDS
            or any(budget[k] is not False for k in ("automatic_retry", "automatic_resume", "extension"))
            or claim["workspace"] != str(workspace) or claim["protocol"] != protocol()
            or claim["budget_sha256"] != digest(workspace / "budget.json")):
        raise ValueError("Invalid once-claim or immutable six-hour budget")
    check_budget(budget)


def worker(workspace):
    require_lock()
    budget = read(workspace / "budget.json")
    verify_context(workspace, budget)
    def stop(*_):
        raise core.BudgetStop("Worker stopped; retain partials")
    signal.signal(signal.SIGTERM, stop)
    PROGRESS["phase"] = "asset-validation"
    identity = validate(budget)
    identity["budget_sha256"] = digest(workspace / "budget.json")
    write_json(workspace / "identity.json", identity)
    required = max(8 * 1024**3, PARENT.stat().st_size * 3 + 1024**3)
    if shutil.disk_usage(workspace).free < required:
        raise RuntimeError("Require8GiB project-volume free; preserve prior artifacts")
    execute(workspace, identity, budget)


def train(workspace):
    lock = require_lock()
    if workspace.name != "workspace" or workspace.parent.parent != RUN_ROOT.resolve() or workspace.exists():
        raise ValueError("Use a new dated launcher workspace")
    claim = RUN_ROOT / "atgl-rdrop-once-claim.json"
    if claim.exists():
        raise FileExistsError("R-Drop authorization already claimed; no retry")
    workspace.mkdir()
    start = time.monotonic()
    budget = {"start_monotonic": start, "deadline_monotonic": start + HARD_SECONDS,
        "hard_seconds": HARD_SECONDS, "soft_reserve_seconds": SOFT_SECONDS,
        "term_reserve_seconds": TERM_SECONDS, "start_utc": datetime.now(timezone.utc).isoformat(),
        "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
        "automatic_retry": False, "automatic_resume": False, "extension": False}
    write_json(workspace / "budget.json", budget)
    with claim.open("x", encoding="utf-8") as f:
        json.dump({"workspace": str(workspace), "budget_sha256": digest(workspace / "budget.json"),
            "protocol": protocol(), "authorization": "explicit-user-one-ATGL-RDrop-Human-cap6h"}, f, indent=2)
        f.flush(); os.fsync(f.fileno())
    write_json(workspace.parent / "pid.json", {"supervisor_pid": os.getpid()})
    def stop(*_):
        raise core.BudgetStop("Supervisor stopped; no retry/resume/extension")
    signal.signal(signal.SIGTERM, stop)
    try:
        emit("ATGL_RDROP_BUDGET_STARTED", **budget)
        supervise([sys.executable, "-B", str(ROOT / FILES[1])], budget, lock)
        supervise([sys.executable, "-u", "-B", str(Path(__file__).resolve()),
                   "--mode", "worker", "--run-dir", str(workspace)], budget, lock)
    except BaseException as error:
        write_json(workspace / "failure.json", {"error": str(error), "type": type(error).__name__,
            "complete": False, "automatic_retry": False, "resumable": False,
            "elapsed_seconds": time.monotonic() - start})
        raise


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=("validate", "train", "worker"), default="validate")
    p.add_argument("--run-dir", type=Path)
    return p


def main():
    args = parser().parse_args()
    os.chdir(ROOT)
    if args.mode == "validate":
        print(json.dumps(validate(), indent=2))
        return
    if args.run_dir is None:
        raise ValueError("Use the launcher workspace")
    path = args.run_dir.resolve()
    if args.mode == "train":
        train(path)
    else:
        try:
            worker(path)
        except BaseException as error:
            write_json(path / "failure-worker.json", {"error": str(error), "type": type(error).__name__,
                "complete": False, "automatic_retry": False, "resumable": False, **PROGRESS})
            raise


if __name__ == "__main__":
    main()
