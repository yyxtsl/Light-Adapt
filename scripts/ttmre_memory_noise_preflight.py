"""Bounded Distant-train gradient-routing preflight, never an optimizer or training run."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
import math
from pathlib import Path
import random
import statistics
import sys
import time

import train_ttmre_human_distant as baseline

shared, aligned = baseline.shared, baseline.aligned
ROOT = shared.ROOT
REPORT_ROOT = ROOT / "experiments/2026-10-13-memory-noise-preflight"
DEFAULT_WORKSPACE = baseline.RUN_ROOT / "two-stage-paper-freeze-seed74-launch-20261011-190240-rjeB6g/workspace"
DISTANT_SHA = "973d4616a86060e34cd4b654a45185ab9693781dd53f519cf978421de5e19d6f"
MEMORY_NAME = "mu_encoder.memory_tokens"
NEW_FILES = ("scripts/ttmre_memory_noise_preflight.py", "scripts/check_ttmre_memory_noise_preflight.py",
             "scripts/run_ttmre_memory_noise_preflight.sh")
GATES = {"eligible_positives": 64, "eligible_relations": 4, "mean_effective_weight_cv": .03,
         "minimum_ess_fraction": .5, "maximum_batch_gradient_relative_change": .005,
         "median_time_ratio_max": 4.0}
GRADIENT_RTOL, GRADIENT_ABS_FLOOR = 1e-5, 1e-7


def protocol():
    return {"version": "memory-only-weighted-spu-preflight-v1", "seed": 74, "batch_size": 4,
            "stress_batches": 3, "sampled_batches": 5, "documents_max": 32, "precision": "float32",
            "teacher": "two detached dropout views of current model; diagnostic endpoint only",
            "raw_weight": ".25+.75*sqrt(sigmoid(m1)*sigmoid(m2))*exp(-abs(sigmoid(m1)-sigmoid(m2)))",
            "risk": "relation-normalized positive terms only; unchanged unlabeled means/priors",
            "gradient_route": "grad_M(Rw); backward(R); replace M.grad; no optimizer",
            "source_checkpoint_sha256": DISTANT_SHA, "gates": GATES,
            "gradient_equivalence": {"relative_L2_max": GRADIENT_RTOL, "absolute_max_floor": GRADIENT_ABS_FLOOR},
            "cooperative_seconds": 840, "launcher_seconds": 900,
            "interpretation": "engineering/weak-proxy feasibility, no F1/noise accuracy/training permission"}


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--mode", choices=("validate", "diagnose"), default="validate")
    result.add_argument("--baseline-workspace", type=Path, default=DEFAULT_WORKSPACE)
    result.add_argument("--model-dir", type=Path)
    result.add_argument("--output-dir", type=Path)
    result.add_argument("--check-runtime", action="store_true")
    return result


def output_path(path, mode):
    if path is None:
        if mode == "diagnose":
            raise ValueError("Diagnosis requires a new --output-dir")
        return None
    path = path.expanduser().resolve()
    if path == REPORT_ROOT.resolve() or not path.is_relative_to(REPORT_ROOT.resolve()):
        raise ValueError("Output must be below the dated noise-preflight root")
    if path.exists():
        raise FileExistsError(f"Preserve existing output: {path}")
    return path


def assert_hashes(values):
    for name, checksum in values.items():
        if shared.sha256(Path(name)) != checksum:
            raise ValueError(f"Frozen input changed: {name}")


def validate(args):
    """Reuse completed-stage checks, bypass validators that read Dev or archive bytes."""
    args.output_dir = output_path(args.output_dir, args.mode)
    workspace = args.baseline_workspace.expanduser().resolve()
    if not workspace.is_dir() or not workspace.is_relative_to(baseline.RUN_ROOT.resolve()):
        raise ValueError("Expected the completed two-stage workspace")
    identity = shared.read_json(workspace / "identity.json")
    prepared = shared.read_json(workspace / "preparation.json")
    ready = shared.read_json(workspace / "ready.json")
    if (identity.get("protocol") != baseline.protocol() or ready.get("passed") is not True
            or ready.get("optimizer_steps") != 0 or prepared.get("optimizer_steps") != 0
            or ready.get("test_data_read") is not False or prepared.get("test_data_read") is not False
            or ready.get("preparation_sha256") != shared.sha256(workspace / "preparation.json")
            or ready.get("preflight_sha256") != shared.sha256(workspace / "preflight.json")):
        raise ValueError("Incomplete or mismatched preparation")
    hashes = {}
    for name, checksum in identity["source_files"].items():
        path = (ROOT / name).resolve()
        if not path.is_relative_to(ROOT) or not (name.endswith((".py", ".sh")) or name.startswith("meta/")):
            raise ValueError("Unexpected source dependency")
        hashes[str(path)] = checksum
    if any(identity["source_files"].get(n) != s for n, s in shared.SOURCE_HASHES.items()):
        raise ValueError("Upstream source identity differs")
    if identity.get("entrypoint_sha256") != shared.sha256(ROOT / "scripts/train_ttmre_human_distant.py"):
        raise ValueError("Frozen baseline entrypoint changed")
    assert_hashes(hashes)
    args.distant_dir = workspace / "distant/artifacts"
    reference = baseline.verify_completed_stage(args.distant_dir, "distant", identity,
                                                shared.sha256(workspace / "preparation.json"))
    if reference["checkpoint_sha256"] != DISTANT_SHA:
        raise ValueError("Expected the reviewed Distant checkpoint")
    expected = shared.read_json(args.distant_dir / "manifest.json")["environment"]
    baseline.control.verify_environment(expected, expected)
    if ready["environment"] != expected:
        raise ValueError("Prepared and completed environments differ")
    args.metadata = {}
    for split, count in (("human", 3053), ("distant", 101873)):
        directory = workspace / "cache" / split
        entry = prepared["caches"][split]
        if shared.sha256(directory / "index.json") != entry["index_sha256"]:
            raise ValueError(f"Changed {split} index")
        source = shared.SPLITS["train_revised.json"] if split == "human" else entry["source_sha256"]
        metadata = baseline.data.validate_cache(directory, shared.sha256, count, source, True)
        hashes[str(directory / "index.json")] = entry["index_sha256"]
        hashes[str(directory / "features.bin")] = metadata["features_sha256"]
        # Historical indexes contain matching_dev_facts: discard, never use them.
        args.metadata[split] = {key: metadata[key] for key in ("index", "priors", "documents", "source_sha256")}
    priors = args.metadata["human"]["priors"]
    if len(priors) != 96 or any(not math.isfinite(x) or not 0 <= x < 1 for x in priors):
        raise ValueError("Invalid fixed Human-frequency priors")
    args.model_dir = (args.model_dir or Path(identity["model"]["path"])).expanduser().resolve()
    for name, checksum in identity["model"]["sha256"].items():
        if Path(name).is_absolute() or ".." in Path(name).parts:
            raise ValueError("Unexpected generic model member")
        hashes[str(args.model_dir / name)] = checksum
    for name in ("identity.json", "preparation.json", "ready.json", "preflight.json"):
        hashes[str(workspace / name)] = shared.sha256(workspace / name)
    for name in ("manifest.json", "completion.json", "best_metrics.json", "checkpoint_replay.json", "metrics.jsonl", "best_model.pth"):
        hashes[str(args.distant_dir / name)] = shared.sha256(args.distant_dir / name)
    for name in NEW_FILES:
        hashes[str(ROOT / name)] = shared.sha256(ROOT / name)
    assert_hashes(hashes)
    args.baseline_workspace = workspace
    if args.check_runtime or args.mode == "diagnose":
        args.device = baseline.control.runtime(expected)
    return {"protocol": protocol(), "source_identity": identity, "reference": reference,
            "input_sha256": hashes, "environment": expected, "optimizer_steps": 0,
            "dev_files_read": False, "test_data_read": False, "archive_read": False,
            "historical_matching_dev_facts_used": False}


def select_batches(metadata):
    rows = metadata["index"]
    if len(rows) < 32:
        raise ValueError("Insufficient Distant documents")
    selected = [{"probe": "first", "indices": list(range(4))},
                {"probe": "most_pairs", "indices": sorted(range(len(rows)), key=lambda i: (rows[i][3], rows[i][2]), reverse=True)[:4]},
                {"probe": "longest", "indices": sorted(range(len(rows)), key=lambda i: (rows[i][2], rows[i][3]), reverse=True)[:4]}]
    used = {i for batch in selected for i in batch["indices"]}
    samples = random.Random(74).sample([i for i in range(len(rows)) if i not in used], 20)
    selected.extend({"probe": f"sample_{i // 4 + 1}", "indices": samples[i:i + 4]} for i in range(0, 20, 4))
    return selected


def weighted_spu(model, logits, labels, weights=None):
    """Original risk with detached, positive-group-normalized weights; diagnostics are not losses."""
    import torch
    if labels.shape != logits.shape or not torch.isfinite(logits).all() or not ((labels == 0) | (labels == 1)).all():
        raise ValueError("Nonfinite logits or invalid label shape/values")
    if weights is not None:
        weights = weights.detach().to(logits)
        if weights.shape != logits.shape or not torch.isfinite(weights).all() or (weights <= 0).any():
            raise ValueError("Weights must be finite positive values aligned with logits")
    total, details = logits.new_zeros(1), []
    for i in range(model.rels):
        mask = labels[:, i + 1] == 1
        margin = logits[:, i + 1] - logits[:, 0]
        pos, unlabelled = margin[mask], margin[~mask]
        w = None if weights is None else weights[mask, i + 1]
        def positive_loss(target):
            if not len(pos):
                return logits.new_zeros(1)
            values = (pos * target - model.margin).square()
            return values.sum() / len(pos) if w is None else (values * w).sum() / w.sum()
        negative = logits.new_zeros(1) if not len(unlabelled) else (unlabelled * -1. - model.margin).square().sum() / len(unlabelled)
        prior_u = (model.priors_o[i] - model.priors_l[i]) / (1. - model.priors_l[i])
        risk1 = ((1. - model.priors_o[i]) / (1. - prior_u)) * negative - ((prior_u - prior_u * model.priors_o[i]) / (1. - prior_u)) * positive_loss(-1.)
        risk2 = model.priors_o[i] * positive_loss(1.) * model.weight[i]
        corrected = bool(risk1 < model.args.beta)
        if w is None or not len(pos):
            original_corrected = corrected
        else:
            original_positive_negative = (pos.detach() * -1. - model.margin).square().sum() / len(pos)
            original_risk1 = ((1. - model.priors_o[i]) / (1. - prior_u)) * negative.detach() - ((prior_u - prior_u * model.priors_o[i]) / (1. - prior_u)) * original_positive_negative
            original_corrected = bool(original_risk1 < model.args.beta)
        total = total + (-model.args.gamma * risk1 if corrected else risk1 + risk2)
        details.append({"relation_id": i + 1, "positive": len(pos), "unlabeled": len(unlabelled),
                        "correction_branch": corrected, "original_correction_branch": original_corrected})
    return total, details


def reliability_weights(first, second, labels):
    import torch
    if first.shape != second.shape or first.shape != labels.shape or not torch.isfinite(first).all() or not torch.isfinite(second).all():
        raise ValueError("Invalid two-view logits")
    p = torch.sigmoid(first.detach() - first.detach()[:, :1])
    q = torch.sigmoid(second.detach() - second.detach()[:, :1])
    raw = .25 + .75 * torch.sqrt(p * q) * torch.exp(-(p - q).abs())
    weights = torch.where(labels == 1, raw, torch.ones_like(raw))
    weights[:, 0] = 1.
    return weights.detach()


def weight_summary(weights, labels):
    rows = []
    for relation in range(1, labels.shape[1]):
        values = weights[labels[:, relation] == 1, relation].detach().double().cpu()
        n = len(values)
        if not n:
            continue
        effective = values / values.mean()
        rows.append({"relation_id": relation, "positive": n, "eligible": n >= 2,
                     "raw_min": float(values.min()), "raw_max": float(values.max()),
                     "raw_mean": float(values.mean()), "effective_weight_cv": float(effective.std(unbiased=False)),
                     "ess_fraction": float(values.sum().square() / values.square().sum() / n),
                     "maximum_normalized_share": float(values.max() / values.sum())})
    return rows


def state_digest(model):
    import torch
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        digest.update(name.encode())
        value = value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy()
        digest.update(memoryview(value))
    return digest.hexdigest()


def rng_state():
    import numpy as np
    import torch
    return (random.getstate(), np.random.get_state(), torch.get_rng_state(),
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])


def restore_rng(state):
    import numpy as np
    import torch
    random.setstate(state[0]); np.random.set_state(state[1]); torch.set_rng_state(state[2])
    if state[3]:
        torch.cuda.set_rng_state_all(state[3])


def assert_rng_equal(first, second):
    import numpy as np
    import torch
    if (first[0] != second[0] or first[1][0] != second[1][0]
            or not np.array_equal(first[1][1], second[1][1]) or first[1][2:] != second[1][2:]
            or not torch.equal(first[2], second[2]) or len(first[3]) != len(second[3])
            or any(not torch.equal(a, b) for a, b in zip(first[3], second[3]))):
        raise RuntimeError("Gradient routing changed the student RNG stream")


@contextmanager
def preserve_model(model):
    """Restore gradients, mode, transient graph references and RNG on success/failure."""
    original = [(p, p.requires_grad, p.grad, None if p.grad is None else p.grad.detach().clone()) for p in model.parameters()]
    modes = [(m, m.training) for m in model.modules()]
    had_sims, sims = hasattr(model, "sims"), getattr(model, "sims", None)
    had_train_mode, train_mode = hasattr(model, "train_mode"), getattr(model, "train_mode", None)
    before, rng = state_digest(model), rng_state()
    try:
        yield
    finally:
        for p, flag, grad, saved in original:
            p.requires_grad_(flag)
            p.grad = grad
            if grad is not None:
                grad.copy_(saved)
        for m, flag in modes:
            m.training = flag
        if had_sims:
            model.sims = sims
        elif hasattr(model, "sims"):
            del model.sims
        if had_train_mode:
            model.train_mode = train_mode
        elif hasattr(model, "train_mode"):
            del model.train_mode
        restore_rng(rng)
        if state_digest(model) != before:
            raise RuntimeError("Preflight changed model parameters or buffers")


def check_budget(deadline):
    if time.monotonic() >= deadline:
        raise TimeoutError("840s cooperative budget exhausted")


def synchronize(device):
    import torch
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def two_views(model, inputs, labels, seed):
    import numpy as np
    import torch
    state = rng_state()
    try:
        shared.seed_all(seed, torch, np)
        with torch.no_grad():
            first = model(**{k: v for k, v in inputs.items() if k != "labels"})
            second = model(**{k: v for k, v in inputs.items() if k != "labels"})
            return reliability_weights(first, second, labels)
    finally:
        restore_rng(state)


def backward_path(model, inputs, labels, weights=None):
    """One forward, original backward, optional memory-only VJP replacement; never a step."""
    import torch
    model.zero_grad(set_to_none=True)
    risks, logits = model(**inputs)
    loss = sum(risks)
    weighted, branches = loss, []
    memory_gradient = None
    if weights is not None:
        weighted, branches = weighted_spu(model, logits, labels, weights)
        if not torch.isfinite(weighted).all():
            raise RuntimeError("Nonfinite weighted S-PU")
        memory_gradient, = torch.autograd.grad(weighted, model.mu_encoder.memory_tokens, retain_graph=True)
        memory_gradient = memory_gradient.detach()
    loss.backward()
    if memory_gradient is not None:
        model.mu_encoder.memory_tokens.grad = memory_gradient
    if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
        raise RuntimeError("Nonfinite gradient")
    return {"original_loss": float(loss.detach()), "weighted_loss": float(weighted.detach()),
            "branch_changes": sum(b["original_correction_branch"] != b["correction_branch"] for b in branches)}


def gradient_snapshot(model):
    return {name: None if p.grad is None else p.grad.detach().cpu().clone() for name, p in model.named_parameters()}


def compare_gradients(model, reference, include_memory):
    import torch
    compared, largest, largest_relative = 0, 0., 0.
    for name, p in model.named_parameters():
        if not include_memory and name == MEMORY_NAME:
            continue
        old, new = reference[name], p.grad
        if old is None or new is None:
            if not (old is None and new is None):
                raise RuntimeError(f"Gradient connectivity changed: {name}")
            continue
        new = new.detach().cpu()
        delta = (new.double() - old.double())
        maximum, norm = float(delta.abs().max()), float(old.double().norm())
        relative = float(delta.norm()) / max(norm, 1e-30)
        # Repeated unmodified CUDA backward already differs at cancellation
        # entries (embedding scatter reductions). Test the whole tensor's L2
        # error, or an absolute floor for near-zero tensors; record both.
        if maximum > GRADIENT_ABS_FLOOR and relative > GRADIENT_RTOL:
            raise RuntimeError(f"Gradient mismatch {name}: relative_L2={relative}, max_abs={maximum}")
        largest = max(largest, maximum)
        if maximum > GRADIENT_ABS_FLOOR:
            largest_relative = max(largest_relative, relative)
        compared += 1
    return {"compared_tensors": compared, "maximum_absolute_difference": largest,
            "maximum_relative_L2_above_absolute_floor": largest_relative, "passed": True}


def gradient_effect(old, new, other_squared_norm):
    import torch
    old, new = old.double().flatten(), new.double().flatten()
    a, b = float(old.norm()), float(new.norm())
    if a == 0 or b == 0:
        raise RuntimeError("Zero memory gradient; cannot assess routing")
    before = math.sqrt(other_squared_norm + a * a)
    after = math.sqrt(other_squared_norm + b * b)
    c0, c1 = min(1., 1. / (before + 1e-6)), min(1., 1. / (after + 1e-6))
    return {"original_memory_norm": a, "weighted_memory_norm": b,
            "relative_change": float((new - old).norm()) / a,
            "cosine": float(torch.dot(old, new)) / (a * b),
            "original_global_norm": before, "routed_global_norm": after,
            "original_clip_coefficient": c0, "routed_clip_coefficient": c1,
            "clip_coefficient_ratio": c1 / c0, "actual_clipping_performed": False}


def probe_batch(model, features, device, seed, deadline):
    import torch
    inputs = shared.inputs_from_batch(shared.collate_fn(features), device)
    labels = torch.tensor([row for f in features for row in f["labels"]], device=device)
    model.train()
    model.train_mode = "finetune"
    check_budget(deadline)
    peaks = {}
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    synchronize(device); started = time.monotonic()
    weights = two_views(model, inputs, labels, seed)
    synchronize(device); teacher_seconds = time.monotonic() - started
    if device.type == "cuda":
        peaks["two_views"] = {"allocated_GiB": torch.cuda.max_memory_allocated(device) / 1024**3,
                              "reserved_GiB": torch.cuda.max_memory_reserved(device) / 1024**3}
    stats = weight_summary(weights, labels)
    student_rng = rng_state()
    times, records = {}, {}
    reference, after_reference = None, None
    for name in ("original", "unity", "weighted"):
        check_budget(deadline)
        restore_rng(student_rng)
        model.zero_grad(set_to_none=True)
        if device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
        synchronize(device); started = time.monotonic()
        chosen = None if name == "original" else torch.ones_like(weights) if name == "unity" else weights
        records[name] = backward_path(model, inputs, labels, chosen)
        synchronize(device); times[name] = time.monotonic() - started
        if device.type == "cuda":
            peaks[name] = {"allocated_GiB": torch.cuda.max_memory_allocated(device) / 1024**3,
                           "reserved_GiB": torch.cuda.max_memory_reserved(device) / 1024**3}
        after = rng_state()
        if name == "original":
            reference, after_reference = gradient_snapshot(model), after
        else:
            assert_rng_equal(after_reference, after)
            records[name]["gradient_equivalence"] = compare_gradients(model, reference, name == "unity")
            if name == "unity" and not math.isclose(records[name]["weighted_loss"], records[name]["original_loss"], rel_tol=2e-6, abs_tol=2e-6):
                raise RuntimeError("Unity weighted risk does not reproduce upstream S-PU")
    other_squared = sum(float(g.double().square().sum()) for name, g in reference.items() if name != MEMORY_NAME and g is not None)
    effect = gradient_effect(reference[MEMORY_NAME], model.mu_encoder.memory_tokens.grad.detach().cpu(), other_squared)
    model.zero_grad(set_to_none=True)
    if hasattr(model, "sims"):
        model.sims = None
    return {"paths": records, "weight_groups": stats, "memory_gradient": effect,
            "seconds": {**times, "two_views": teacher_seconds}, "peak_memory": peaks,
            "proposed_total_time_ratio": (times["weighted"] + teacher_seconds) / times["original"],
            "student_rng_equal": True, "optimizer_steps": 0}


def decide(records):
    groups = [g for r in records for g in r["weight_groups"] if g["eligible"]]
    counts = sum(g["positive"] for g in groups)
    relations = len({g["relation_id"] for g in groups})
    values = {"eligible_positives": counts, "eligible_relations": relations,
              "mean_effective_weight_cv": statistics.mean(g["effective_weight_cv"] for g in groups) if groups else 0.,
              "minimum_ess_fraction": min((g["ess_fraction"] for g in groups), default=0.),
              "maximum_batch_gradient_relative_change": max(r["memory_gradient"]["relative_change"] for r in records),
              "median_time_ratio": statistics.median(r["proposed_total_time_ratio"] for r in records)}
    if counts < GATES["eligible_positives"] or relations < GATES["eligible_relations"]:
        status = "INCONCLUSIVE"
    elif values["median_time_ratio"] > GATES["median_time_ratio_max"]:
        status = "OVERHEAD_TOO_HIGH"
    elif (values["mean_effective_weight_cv"] < GATES["mean_effective_weight_cv"]
          or values["minimum_ess_fraction"] < GATES["minimum_ess_fraction"]
          or values["maximum_batch_gradient_relative_change"] < GATES["maximum_batch_gradient_relative_change"]):
        status = "NO_WEIGHTING_LEVERAGE"
    else:
        status = "CANDIDATE_ONLY_REVIEW_REQUIRED"
    return {"status": status, "measurements": values, "gates": GATES,
            "automatic_training_authorized": False, "scientific_effect_established": False}


def load_model(args, manifest):
    import torch
    args.run_dir = args.baseline_workspace
    model = baseline.load_model(args, manifest["source_identity"])
    state = torch.load(args.distant_dir / "best_model.pth", map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    del state
    if tuple(model.mu_encoder.memory_tokens.shape) != (200, 768):
        raise ValueError("Unexpected memory shape")
    return model


def write_new(path, value):
    with Path(path).open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def run(args, manifest, deadline):
    args.output_dir.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    try:
        write_new(args.output_dir / "manifest.json", {k: v for k, v in manifest.items() if k != "source_identity"})
        check_budget(deadline)
        selection = select_batches(args.metadata["distant"])
        write_new(args.output_dir / "selection.json", {"batches": selection, "protocol": protocol()})
        import torch
        from ttmre_memory_fit import reserve_adam_moments
        state = rng_state()
        try:
            model = load_model(args, manifest)
            dataset = baseline.data.DiskFeatures(args.baseline_workspace / "cache/distant", args.metadata["distant"])
            records = []
            with preserve_model(model):
                moments, reservation = reserve_adam_moments(model)
                try:
                    with (args.output_dir / "batch_diagnostics.jsonl").open("x", encoding="utf-8") as handle:
                        for i, batch in enumerate(selection):
                            check_budget(deadline)
                            aligned.emit("memory_noise_probe_start", **batch, optimizer_steps=0)
                            features = [dataset[index] for index in batch["indices"]]
                            row = {**batch, "token_lengths": [len(f["input_ids"]) for f in features],
                                   "entity_pairs": [len(f["hts"]) for f in features],
                                   **probe_batch(model, features, args.device, 7400 + i, deadline)}
                            records.append(row)
                            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n"); handle.flush()
                            aligned.emit("memory_noise_probe_complete", probe=batch["probe"],
                                         gradient_change=row["memory_gradient"]["relative_change"],
                                         time_ratio=row["proposed_total_time_ratio"], optimizer_steps=0)
                finally:
                    moments.clear()
            check_budget(deadline)
        finally:
            restore_rng(state)
        assert_hashes(manifest["input_sha256"])
        report = {"execution_passed": True, "decision": decide(records), "batches_completed": len(records),
                  "documents_decoded_unique": len({i for b in selection for i in b["indices"]}),
                  "optimizer_memory_reservation": reservation, "elapsed_seconds": time.monotonic() - started,
                  "parameters_gradients_rng_modes_and_sims_restored": True,
                  "scope": {"optimizer_created": False, "optimizer_steps": 0, "backward_used": True,
                            "distant_training_cache_decoded": True, "human_features_decoded": False,
                            "dev_files_read": False, "test_data_read": False, "archive_read": False,
                            "checkpoint_saved": False, "historical_matching_dev_facts_used": False},
                  "interpretation": "Endpoint weak-proxy/engineering check, not causal teacher validation, F1 or noise accuracy"}
        write_new(args.output_dir / "diagnostic.json", report)
        print(f"MEMORY_NOISE_PREFLIGHT_COMPLETE; status={report['decision']['status']}; optimizer_steps=0; training_started=false", flush=True)
    except Exception as error:
        write_new(args.output_dir / "failure.json", {"error_type": type(error).__name__, "error": str(error),
                  "incomplete": True, "optimizer_steps": 0, "no_automatic_retry": True})
        raise


def main():
    deadline = time.monotonic() + 840
    args = parser().parse_args()
    manifest = validate(args)
    if args.mode == "validate":
        print(json.dumps({k: v for k, v in manifest.items() if k != "source_identity"}, ensure_ascii=False, allow_nan=False))
        print("MEMORY_NOISE_VALIDATE_OK; weights_loaded=false; files_written=false", flush=True)
        return
    sys.path.insert(0, str(ROOT))
    run(args, manifest, deadline)


if __name__ == "__main__":
    main()
