"""Authorized one seed74 N1 Distant2 -> Human30 run; validate is read-only."""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys
import time

import train_ttmre_human_distant as baseline
import ttmre_memory_noise as method

preflight = method.preflight
shared, aligned = baseline.shared, baseline.aligned
ROOT = shared.ROOT
RUN_ROOT = ROOT / "experiments/2026-10-13-memory-noise-n1"
VARIANT = "memory-only-weighted-spu"
NEW_FILES = ("scripts/ttmre_memory_noise.py", "scripts/train_ttmre_memory_noise.py",
             "scripts/check_ttmre_memory_noise.py", "scripts/run_ttmre_memory_noise.sh")
PREFLIGHT_SHA = "cb37ffca662a9d2410d7b4ac82b3271e211efbe26df33564ba10d81db75e368e"
FROZEN_PREFLIGHT = {
    "scripts/ttmre_memory_noise_preflight.py": "2ddb1a0ce5eb002304fb98bc1b97a93db11185a936fe944425806ac1d4e53145",
    "scripts/check_ttmre_memory_noise_preflight.py": "5d2d8545575ed38eca831c6a8b270bd4887c6375f437ccf76ee506d78cebcba0",
    "scripts/run_ttmre_memory_noise_preflight.sh": "bd49cd619e93a569b2ad8ef420c0f2f5d437330c7f47c1961673c6277b6d54c4"}
HUMAN_SHA = "5b68b46354ad2ef54509a6d0fc5dc8081d9e93caf948d51a6f025384be701d68"
PROBE_VERSION = "original-repeatability-key-bias-v2"
ORIGINAL_REPEATS = 3
REPEAT_ENVELOPE = 4.0


def protocol():
    return {**baseline.protocol(), "variant": VARIANT, "method_version": "causal-memory-only-spu-v1",
            "new_parameter_count": 0, "expected_parameters": 381175401,
            "probe_version": PROBE_VERSION,
            "warm_start": "original Distant epoch1; active weighting only epoch2",
            "weight_formula": preflight.protocol()["raw_weight"], "weight_floor": .25,
            "teacher": "current student two no-grad dropout views before its update; no endpoint teacher",
            "view_seed": "740000+one_based_global_stage_step; student RNG restored",
            "gradient_route": "grad_M(Rw); original R.backward(); replace M.grad; global clip1; one AdamW step",
            "human_rule": "new selected Distant checkpoint; only memory_tokens frozen; original S-PU",
            "statistics": "coverage all active batches; detailed groups/gradients each100 updates + active first/last",
            "screening_gate_delta_pp": .5, "screening_reference_F1": 79.17549145362884,
            "interpretation": "single-seed screening; consistency proxy is not calibrated noise truth"}


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--mode", choices=("validate", "probe", "train"), default="validate")
    result.add_argument("--baseline-workspace", type=Path, default=preflight.DEFAULT_WORKSPACE)
    result.add_argument("--preflight-report", type=Path)
    result.add_argument("--run-dir", type=Path)
    result.add_argument("--model-dir", type=Path)
    result.add_argument("--data-dir", type=Path, default=ROOT / "dataset/docred")
    result.add_argument("--check-runtime", action="store_true")
    return result


def reviewed_preflight(path):
    if path is None:
        candidates = sorted(preflight.REPORT_ROOT.glob("run-*/artifacts/diagnostic.json"))
        matches = [p for p in candidates if shared.sha256(p) == PREFLIGHT_SHA]
        if not matches:
            raise FileNotFoundError("Reviewed preflight report not found; pass --preflight-report to its original diagnostic.json")
        path = matches[0]
    path = path.expanduser().resolve()
    if shared.sha256(path) != PREFLIGHT_SHA:
        raise ValueError("Preflight report is not the reviewed completed report")
    report = shared.read_json(path)
    if (report.get("execution_passed") is not True or report.get("batches_completed") != 8
            or report.get("decision", {}).get("status") != "CANDIDATE_ONLY_REVIEW_REQUIRED"
            or report.get("scope", {}).get("optimizer_created") is not False
            or report["scope"].get("optimizer_steps") != 0 or report["scope"].get("test_data_read") is not False):
        raise ValueError("Preflight incomplete or wrong scope")
    return {"path": str(path), "sha256": PREFLIGHT_SHA, "decision": report["decision"]}


def baseline_identity(args):
    workspace = args.baseline_workspace.expanduser().resolve()
    if not workspace.is_dir() or not workspace.is_relative_to(baseline.RUN_ROOT.resolve()):
        raise ValueError("Use the completed baseline workspace")
    recorded = shared.read_json(workspace / "identity.json")
    settings = baseline.parser().parse_args(["--reference-run-dir", recorded["matched_b2_reference"]["path"],
        "--run-dir", str(workspace), "--data-dir", str(args.data_dir)])
    settings.model_dir, settings.check_runtime = args.model_dir, args.check_runtime
    identity = baseline.validate(settings)
    prepared = baseline.verify_preparation(workspace, identity)
    references = {stage: baseline.verify_completed_stage(workspace / stage / "artifacts", stage,
                    identity, shared.sha256(workspace / "preparation.json")) for stage in baseline.STAGES}
    if references["distant"]["checkpoint_sha256"] != preflight.DISTANT_SHA or references["human-frozen"]["checkpoint_sha256"] != HUMAN_SHA:
        raise ValueError("Use the reviewed paired N0 checkpoints as metric references")
    human_saved = shared.read_json(workspace / "human-frozen/artifacts/manifest.json")
    if human_saved.get("distant_initialization_reference") != references["distant"]:
        raise ValueError("N0 Human did not use its paired Distant checkpoint")
    args.baseline_workspace, args.model_dir, args.data_dir = workspace, settings.model_dir, settings.data_dir
    return identity, prepared, references


def validate(args):
    if args.run_dir is not None:
        args.run_dir = args.run_dir.expanduser().resolve()
        if args.run_dir == RUN_ROOT.resolve() or not args.run_dir.is_relative_to(RUN_ROOT.resolve()):
            raise ValueError("Use an exclusive N1 artifacts directory inside the dated root")
        if args.run_dir.exists():
            raise FileExistsError(f"Preserve existing run: {args.run_dir}")
    elif args.mode == "train":
        raise ValueError("Training requires --run-dir")
    for name, checksum in FROZEN_PREFLIGHT.items():
        if shared.sha256(ROOT / name) != checksum:
            raise ValueError(f"Frozen preflight source changed: {name}")
    report = reviewed_preflight(args.preflight_report)
    identity, prepared, references = baseline_identity(args)
    input_hashes = {str(args.baseline_workspace / "identity.json"): shared.sha256(args.baseline_workspace / "identity.json"),
                    str(args.baseline_workspace / "preparation.json"): shared.sha256(args.baseline_workspace / "preparation.json"),
                    report["path"]: report["sha256"]}
    for split in ("human", "distant", "dev"):
        for name, key in (("index.json", "index_sha256"), ("features.bin", "features_sha256")):
            path = args.baseline_workspace / "cache" / split / name
            # preparation keeps index hashes; feature hashes live in each index.
            checksum = prepared["caches"][split].get(key)
            if checksum is None:
                checksum = shared.read_json(path.parent / "index.json")[key]
            input_hashes[str(path)] = checksum
    preflight.assert_hashes(input_hashes)
    return {"protocol": protocol(), "baseline_identity": identity, "baseline_workspace": str(args.baseline_workspace),
            "cache_input_sha256": input_hashes, "baseline_references": references,
            "reviewed_preflight": report, "source_files": {**identity["source_files"], **FROZEN_PREFLIGHT,
                **{n: shared.sha256(ROOT / n) for n in NEW_FILES}},
            "entrypoint_sha256": shared.sha256(Path(__file__)), "model": identity["model"],
            "data": identity["data"], "environment": identity["matched_b2_reference"]["environment"],
            "optimizer_steps": 0, "test_data_read": False}


def cache_dataset(args, split, with_labels=True):
    if split not in ("distant", "human", "dev") or with_labels != (split != "dev"):
        raise ValueError("Only declared train/Dev cache scopes are allowed")
    directory = args.baseline_workspace / "cache" / split
    return baseline.data.DiskFeatures(directory, shared.read_json(directory / "index.json"), with_labels)


def load_generic_model(args, identity):
    # Read the original Human priors through baseline loader; never load a task checkpoint.
    original = args.run_dir
    args.run_dir = args.baseline_workspace
    try:
        return baseline.load_model(args, identity["baseline_identity"])
    finally:
        args.run_dir = original


def attention_key_biases(model):
    # Only standard absolute-position RoBERTa self attention has the required
    # common-key-shift invariance. Never generalize this to arbitrary biases.
    from transformers.models.roberta.modeling_roberta import RobertaSelfAttention
    return {name + ".key.bias" for name, module in model.named_modules()
            if isinstance(module, RobertaSelfAttention)
            and module.position_embedding_type == "absolute"
            and module.key.bias is not None}


def gradient_errors(model, reference, include_memory=True):
    import torch
    if set(reference) != {n for n, _ in model.named_parameters()}:
        raise RuntimeError("Gradient parameter names changed")
    errors = {}
    for name, parameter in model.named_parameters():
        if not include_memory and name == preflight.MEMORY_NAME:
            continue
        old, new = reference[name], parameter.grad
        if old is None or new is None:
            if not (old is None and new is None):
                raise RuntimeError(f"Gradient connectivity changed: {name}")
            continue
        new = new.detach().cpu()
        if old.shape != new.shape or old.dtype != new.dtype:
            raise RuntimeError(f"Gradient shape/dtype changed: {name}")
        if not torch.isfinite(old).all() or not torch.isfinite(new).all():
            raise RuntimeError(f"Nonfinite gradient comparison: {name}")
        a, b = old.double(), new.double()
        difference = b - a
        errors[name] = {"max_abs": float(difference.abs().max()),
                        "delta_L2": float(difference.norm()),
                        "reference_L2": float(a.norm()), "actual_L2": float(b.norm()),
                        "reference_max_abs": float(a.abs().max()),
                        "actual_max_abs": float(b.abs().max()), "elements": old.numel()}
    return errors


def overall_gradient_error(errors):
    change = math.sqrt(sum(e["delta_L2"] ** 2 for e in errors.values()))
    norm = math.sqrt(sum(e["reference_L2"] ** 2 for e in errors.values()))
    return change / max(norm, 1e-30)


def compare_probe_gradients(model, reference, repeat_noise, include_memory=True, path="unknown"):
    errors = gradient_errors(model, reference, include_memory)
    key_biases = attention_key_biases(model)
    calibrated = []
    for name, error in errors.items():
        relative = error["delta_L2"] / max(error["reference_L2"], 1e-30)
        if error["max_abs"] <= preflight.GRADIENT_ABS_FLOOR or relative <= preflight.GRADIENT_RTOL:
            continue
        noise = repeat_noise.get(name) if name in key_biases else None
        if noise is not None and noise["max_abs"] > 0 and noise["delta_L2"] > 0:
            absolute_limit = preflight.GRADIENT_ABS_FLOOR + REPEAT_ENVELOPE * noise["max_abs"]
            l2_limit = preflight.GRADIENT_ABS_FLOOR * math.sqrt(error["elements"]) + REPEAT_ENVELOPE * noise["delta_L2"]
            # All gradients themselves must be numerically null at the measured
            # original-vs-original noise scale, not just close to each other.
            if (max(error["max_abs"], error["reference_max_abs"], error["actual_max_abs"]) <= absolute_limit
                    and max(error["delta_L2"], error["reference_L2"], error["actual_L2"]) <= l2_limit):
                calibrated.append({"parameter": name, **error, "relative_L2": relative,
                                   "original_repeat_noise": noise,
                                   "absolute_limit": absolute_limit, "L2_limit": l2_limit})
                continue
        detail = {"path": path, "parameter": name, **error, "relative_L2": relative,
                  "original_repeat_noise": noise, "probe_version": PROBE_VERSION}
        aligned.emit("n1_gradient_comparison_failed", **detail)
        raise RuntimeError("N1 gradient mismatch: " + json.dumps(detail, allow_nan=False))
    global_relative = overall_gradient_error(errors)
    if calibrated and global_relative > preflight.GRADIENT_RTOL:
        raise RuntimeError(f"Calibrated comparison changes the overall gradient: {path}: {global_relative}")
    return {"passed": True, "path": path, "compared_tensors": len(errors),
            "overall_relative_L2": global_relative,
            "maximum_absolute_difference": max((e["max_abs"] for e in errors.values()), default=0.),
            "repeatability_exceptions": calibrated}


def n1_probe_batch(model, features, device, step):
    """Original repeats calibrate only null key biases; no optimizer or updates."""
    import torch
    inputs = shared.inputs_from_batch(shared.collate_fn(features), device)
    labels = method.labels_tensor(inputs, device)
    model.train()
    model.train_mode = "finetune"
    records, times, peaks = {}, {}, {}

    def begin():
        model.zero_grad(set_to_none=True)
        if hasattr(model, "sims"):
            model.sims = None
        if device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
        preflight.synchronize(device)
        return time.perf_counter()

    def end(name, started):
        preflight.synchronize(device)
        times[name] = time.perf_counter() - started
        if device.type == "cuda":
            peaks[name] = {"allocated_GiB": torch.cuda.max_memory_allocated(device) / 1024**3,
                           "reserved_GiB": torch.cuda.max_memory_reserved(device) / 1024**3}

    started = begin()
    weights = preflight.two_views(model, inputs, labels, method.view_seed(step))
    end("two_views", started)
    student_rng = preflight.rng_state()
    noise = {name: {"max_abs": 0., "delta_L2": 0.} for name in attention_key_biases(model)}
    repeat_errors = []
    for index in range(ORIGINAL_REPEATS):
        preflight.restore_rng(student_rng)
        started = begin()
        key = "original" if index == 0 else f"original_repeat_{index}"
        records[key] = preflight.backward_path(model, inputs, labels)
        end(key, started)
        if index == 0:
            reference = preflight.gradient_snapshot(model)
            after_reference = preflight.rng_state()
        else:
            preflight.assert_rng_equal(after_reference, preflight.rng_state())
            if records[key]["original_loss"] != records["original"]["original_loss"]:
                raise RuntimeError("Original replay forward changed under restored RNG")
            errors = gradient_errors(model, reference)
            if overall_gradient_error(errors) > preflight.GRADIENT_RTOL:
                raise RuntimeError("Original backward global gradient is not repeatable")
            for name in noise:
                if name in errors:
                    for field in noise[name]:
                        noise[name][field] = max(noise[name][field], errors[name][field])
            # Keep only tiny key-bias tensors for delayed calibration; large
            # non-key gradients must satisfy the original fixed tolerance now.
            ordinary = {name: value for name, value in reference.items() if name not in noise}
            for name, error in errors.items():
                if name in ordinary and error["max_abs"] > preflight.GRADIENT_ABS_FLOOR and error["delta_L2"] > preflight.GRADIENT_RTOL * error["reference_L2"]:
                    raise RuntimeError(f"Original backward not repeatable: {name}: {error}")
            repeat_errors.append((key, {name: parameter.grad.detach().cpu().clone()
                                       for name, parameter in model.named_parameters()
                                       if name in noise and parameter.grad is not None}))

    # Establish a bounded baseline before looking at any candidate gradients.
    for key, biases in repeat_errors:
        for name, values in biases.items():
            old = reference[name]
            absolute_limit = preflight.GRADIENT_ABS_FLOOR + REPEAT_ENVELOPE * noise[name]["max_abs"]
            l2_limit = preflight.GRADIENT_ABS_FLOOR * math.sqrt(old.numel()) + REPEAT_ENVELOPE * noise[name]["delta_L2"]
            if (max(float(old.abs().max()), float(values.abs().max())) > absolute_limit
                    or max(float(old.double().norm()), float(values.double().norm())) > l2_limit):
                # A materially nonzero bias is not a calibrated null direction.
                noise[name] = {"max_abs": 0., "delta_L2": 0.}
    aligned.emit("n1_original_repeatability", repeats=ORIGINAL_REPEATS, key_bias_noise=noise, optimizer_steps=0)
    other_squared = sum(float(g.double().square().sum()) for name, g in reference.items()
                        if name != preflight.MEMORY_NAME and g is not None)
    for key in ("unity", "weighted", "training_helper"):
        preflight.restore_rng(student_rng)
        started = begin()
        if key == "training_helper":
            records[key] = method.backward(model, inputs, "distant", 2, step, collect=True)
        else:
            chosen = torch.ones_like(weights) if key == "unity" else weights
            records[key] = preflight.backward_path(model, inputs, labels, chosen)
        end(key, started)
        preflight.assert_rng_equal(after_reference, preflight.rng_state())
        compared_reference = weighted_reference if key == "training_helper" else reference
        records[key]["gradient_equivalence"] = compare_probe_gradients(
            model, compared_reference, noise, key != "weighted", path=key)
        original_loss = records[key]["loss"] if key == "training_helper" else records[key]["original_loss"]
        if original_loss != records["original"]["original_loss"]:
            raise RuntimeError(f"Original loss changed in {key}")
        if key == "unity" and not math.isclose(records[key]["weighted_loss"], original_loss, rel_tol=2e-6, abs_tol=2e-6):
            raise RuntimeError("Unity weighted risk does not reproduce upstream S-PU")
        if key == "weighted":
            effect = preflight.gradient_effect(reference[preflight.MEMORY_NAME],
                        model.mu_encoder.memory_tokens.grad.detach().cpu(), other_squared)
            weighted_reference = preflight.gradient_snapshot(model)
        if key == "training_helper" and records[key]["weighted_risk"] != records["weighted"]["weighted_loss"]:
            raise RuntimeError("Training helper weighted loss mismatch")
    model.zero_grad(set_to_none=True)
    if hasattr(model, "sims"):
        model.sims = None
    return {"paths": records, "weight_groups": preflight.weight_summary(weights, labels),
            "memory_gradient": effect, "seconds": times, "peak_memory": peaks,
            "original_repeatability": {"repeats": ORIGINAL_REPEATS, "key_bias_noise": noise,
                "envelope_factor": REPEAT_ENVELOPE, "probe_version": PROBE_VERSION},
            "training_helper_equivalence": records["training_helper"]["gradient_equivalence"],
            "student_rng_equal": True, "optimizer_steps": 0,
            "proposed_total_time_ratio": (times["weighted"] + times["two_views"]) / times["original"]}


def probe(model, dataset, args, stage):
    import torch
    from ttmre_memory_fit import reserve_adam_moments
    baseline.set_stage(model, stage)
    with preflight.preserve_model(model):
        if stage == "human-frozen":
            return baseline.run_probe(model, dataset, args, stage)
        buffers, reservation = reserve_adam_moments(model)
        try:
            records = []
            for i, (name, indices) in enumerate(dataset.probe_indices().items()):
                aligned.emit("n1_probe_start", stage=stage, probe=name, optimizer_steps=0)
                features = [dataset[j] for j in indices]
                row = n1_probe_batch(model, features, args.device, i + 1)
                if row["paths"]["unity"]["gradient_equivalence"]["compared_tensors"] != 448:
                    raise RuntimeError("Unexpected generic-model gradient connectivity")
                model.zero_grad(set_to_none=True)
                if hasattr(model, "sims"):
                    model.sims = None
                records.append({"probe": name, "document_indices": indices, **row})
                aligned.emit("n1_probe_complete", stage=stage, probe=name, optimizer_steps=0,
                             gradient_change=row["memory_gradient"]["relative_change"])
            return {"passed": True, "stage": stage, "optimizer_steps": 0, "probes": records,
                    "optimizer_memory_reservation": reservation, "state_restored": True}
        finally:
            buffers.clear()
            model.zero_grad(set_to_none=True)
            if args.device.type == "cuda":
                torch.cuda.empty_cache()


def train_loop(model, loader, dev_loader, args, stage, directory, gold, annotated, distant, optimizer, scheduler, buckets):
    import torch
    epochs = baseline.STAGES[stage][1]
    memory_before = model.mu_encoder.memory_tokens.detach().cpu().clone()
    best, best_epoch, best_step, steps = -math.inf, None, None, 0
    summary = {}
    with (directory / "train.log").open("x", encoding="utf-8") as log, \
            (directory / "metrics.jsonl").open("x", encoding="utf-8") as metrics_log, \
            (directory / "weight_diagnostics.jsonl").open("x", encoding="utf-8") as detail:
        for epoch in range(1, epochs + 1):
            total_loss, epoch_start = 0., time.perf_counter()
            aligned.emit("epoch_start", stage=stage, epoch=epoch, batches=len(loader), weighting_active=method.active(stage, epoch))
            for index, batch in enumerate(loader):
                model.train()
                optimizer.zero_grad(set_to_none=True)
                collect = method.active(stage, epoch) and (steps + 1 == len(loader) + 1 or (steps + 1) % 100 == 0 or index + 1 == len(loader))
                record = method.backward(model, shared.inputs_from_batch(batch, args.device), stage, epoch, steps + 1, collect=collect)
                global_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
                optimizer.step(); scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                steps += 1
                total_loss += record["loss"]
                method.add_statistics(summary, record, buckets)
                if collect:
                    detail_row = {"stage": stage, "epoch": epoch, "step": steps, "batch": index + 1,
                                  "global_norm_before_clip": float(global_norm),
                                  "clip_coefficient": min(1., 1. / (float(global_norm) + 1e-6)), **record}
                    detail.write(json.dumps(detail_row, allow_nan=False) + "\n"); detail.flush()
                if steps == 1 or steps % 100 == 0:
                    row = {"event": "optimizer_update", "stage": stage, "epoch": epoch, "step": steps,
                           "batch": index + 1, "loss": record["loss"], "weighted_risk": record["weighted_risk"],
                           "weighting_active": record["weighting_active"]}
                    line = json.dumps(row, allow_nan=False)
                    print(line, flush=True); log.write(line + "\n"); log.flush()
                del record
                if hasattr(model, "sims"):
                    model.sims = None
                if index + 1 == len(loader) - 1:
                    aligned.emit("development_evaluation_start", stage=stage, epoch=epoch, step=steps)
                    values = baseline.evaluate(model, dev_loader, args, gold, annotated, distant)
                    score = values["dev_F1"]
                    if not math.isfinite(score):
                        raise RuntimeError("Nonfinite selection metric")
                    selected = score > best
                    if selected:
                        best, best_epoch, best_step = score, epoch, steps
                        torch.save(shared.cpu_snapshot(model), directory / "best_model.pth")
                        shared.write_json(directory / "best_metrics.json", {"epoch": epoch, "step": steps, "batch": index + 1, **values})
                    row = {"event": "development_evaluation", "stage": stage, "epoch": epoch, "step": steps,
                           "batch": index + 1, "train_loss_so_far": total_loss / (index + 1), "selected": selected, **values}
                    line = json.dumps(row, allow_nan=False)
                    print(line, flush=True)
                    for h in (log, metrics_log):
                        h.write(line + "\n"); h.flush()
            if stage == "human-frozen" and not torch.equal(memory_before, model.mu_encoder.memory_tokens.detach().cpu()):
                raise RuntimeError("Frozen Human memory changed")
            aligned.emit("epoch_complete", stage=stage, epoch=epoch, step=steps, train_loss=total_loss / len(loader),
                         seconds=time.perf_counter() - epoch_start)
    shared.write_json(directory / "weight_summary.json", {"coverage": summary, "frequency_buckets": buckets,
        "details_sampled": True, "interpretation": "proxy weights, not validated noise labels or tail benefit"})
    return {"best_epoch": best_epoch, "best_step": best_step, "best_dev_F1": best, "optimizer_steps": steps,
            "memory_unchanged": bool(torch.equal(memory_before, model.mu_encoder.memory_tokens.detach().cpu())),
            "selected_after_weighting": stage == "distant" and best_epoch == 2,
            "test_data_read": False, "test_evaluated": False}


def verify_stage(directory, stage, identity):
    saved = shared.read_json(directory / "manifest.json")
    for key in ("protocol", "source_files", "entrypoint_sha256", "baseline_workspace", "reviewed_preflight", "baseline_references", "cache_input_sha256"):
        if saved.get(key) != identity[key]:
            raise ValueError(f"N1 completion identity mismatch: {key}")
    complete, best, replay = [shared.read_json(directory / name) for name in ("completion.json", "best_metrics.json", "checkpoint_replay.json")]
    batches, epochs = baseline.STAGES[stage][0] // 4, baseline.STAGES[stage][1]
    if (saved.get("stage") != stage or complete.get("stage") != stage or complete.get("optimizer_steps") != batches * epochs
            or complete.get("checkpoint_replay_passed") is not True or replay.get("passed") is not True
            or replay.get("invariance", {}).get("passed") is not True or replay.get("optimizer_steps") != 0
            or complete.get("test_data_read") is not False or complete.get("test_evaluated") is not False
            or replay.get("test_data_read") is not False or replay.get("test_evaluated") is not False
            or best.get("batch") != batches - 1 or not 1 <= best["epoch"] <= epochs
            or best["step"] != best["epoch"] * batches - 1
            or complete.get("best_epoch") != best["epoch"] or complete.get("best_step") != best["step"]
            or complete.get("best_dev_F1") != best["dev_F1"]
            or replay.get("epoch") != best["epoch"] or replay.get("step") != best["step"]
            or any(replay.get("metrics", {}).get(k) != best.get(k) for k in baseline.m1.METRIC_KEYS)):
        raise ValueError("N1 stage incomplete/selection/replay/scope mismatch")
    with (directory / "metrics.jsonl").open(encoding="utf-8") as h:
        trajectory = [json.loads(line) for line in h]
    if (len(trajectory) != epochs or any(row.get("epoch") != i or row.get("stage") != stage
            or row.get("step") != i * batches - 1 or row.get("batch") != batches - 1
            or not math.isfinite(row["dev_F1"]) for i, row in enumerate(trajectory, 1))
            or max(trajectory, key=lambda row: row["dev_F1"])["epoch"] != best["epoch"]
            or trajectory[best["epoch"] - 1]["dev_F1"] != best["dev_F1"]):
        raise ValueError("N1 trajectory does not select earliest maximum")
    checksum = shared.sha256(directory / "best_model.pth")
    if checksum != saved.get("best_checkpoint_sha256") or checksum != replay.get("checkpoint_sha256"):
        raise ValueError("N1 checkpoint identity mismatch")
    if stage == "human-frozen" and (complete.get("memory_unchanged") is not True
            or complete.get("distant_initialization_reference") != saved.get("distant_initialization_reference")):
        raise ValueError("Human freeze/parent mismatch")
    if stage == "distant" and complete.get("selected_after_weighting") != (best["epoch"] == 2):
        raise ValueError("Incorrect weighting attribution")
    return {"path": str(directory), "checkpoint_sha256": checksum, "best_metrics": best,
            "completion_sha256": shared.sha256(directory / "completion.json"),
            "selected_after_weighting": complete.get("selected_after_weighting")}


def frequency_report(model, loader, args, gold, annotated, distant, buckets, best):
    """Selected checkpoint only, fixed train-frequency relation thirds, no extra selection."""
    import numpy as np
    import torch
    from evaluation import to_official
    predicted, offset = [], 0
    model.eval()
    with torch.no_grad():
        for batch in loader:
            logits = model(**shared.inputs_from_batch(batch, args.device, labels=False))
            if not torch.isfinite(logits).all():
                raise RuntimeError("Nonfinite bucket logits")
            values = logits.cpu().numpy()
            flags = np.zeros_like(values); flags[:, 1:] = values[:, 1:] > values[:, :1]
            flags[:, 0] = flags[:, 1:].sum(1) == 0
            count = batch[0].shape[0]
            predicted.extend(to_official(flags, loader.dataset[offset:offset + count])); offset += count
    if offset != len(gold):
        raise RuntimeError("Bucket coverage mismatch")
    replay = shared.score_predictions(predicted, gold, annotated, distant)
    if any(not math.isclose(replay[k], best[k], abs_tol=1e-6, rel_tol=0) for k in baseline.m1.METRIC_KEYS):
        raise RuntimeError("Bucket inference does not replay selected metrics")
    rel2id = shared.read_json(ROOT / "meta/rel2id.json")
    truth = {(doc["title"], l["h"], l["t"], l["r"]) for doc in gold for l in doc["labels"]}
    predictions = {(p["title"], p["h_idx"], p["t_idx"], p["r"]) for p in predicted}
    report = {}
    for bucket in ("rare", "middle", "frequent"):
        g = {f for f in truth if buckets[rel2id[f[3]]] == bucket}
        p = {f for f in predictions if buckets[rel2id[f[3]]] == bucket}
        tp, fp, fn = len(g & p), len(p - g), len(g - p)
        report[bucket] = {"gold": len(g), "predicted": len(p), "TP": tp, "FP": fp, "FN": fn,
                          "F1": 200 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.,
                          "recall": 100 * tp / len(g) if g else 0., "precision": 100 * tp / len(p) if p else 0.}
    return {"passed": True, "metrics": replay, "buckets": report, "relation_bucket": buckets,
            "interpretation": "train-frequency diagnostics, no historical paired bucket improvement claim"}


def train_stage(model, args, identity, stage, dev_loader, gold, annotated, distant, buckets, parent=None):
    import numpy as np
    import torch
    from torch.utils.data import DataLoader
    from transformers.optimization import get_linear_schedule_with_warmup
    directory = args.run_dir / stage / "artifacts"
    directory.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    baseline.set_stage(model, stage)
    memory_initial = model.mu_encoder.memory_tokens.detach().cpu().clone()
    manifest = {**copy.deepcopy(identity), "stage": stage, "distant_initialization_reference": parent,
                "memory_frozen": stage == "human-frozen", "distant_data_read": True,
                "initial_state_sha256": preflight.state_digest(model)}
    shared.write_json(directory / "manifest.json", manifest)
    try:
        dataset = cache_dataset(args, "distant" if stage == "distant" else "human")
        phase = time.perf_counter()
        probe_result = probe(model, dataset, args, stage)
        probe_seconds = time.perf_counter() - phase
        shared.write_json(directory / "full_model_check.json", probe_result)
        optimizer = baseline.optimizer_for_stage(model, stage)  # Must follow passed real probes.
        groups = baseline.optimizer_report(model, optimizer)
        shared.seed_all(74, torch, np)
        loader = DataLoader(dataset, batch_size=4, shuffle=True, drop_last=True, collate_fn=shared.collate_fn)
        total = len(loader) * baseline.STAGES[stage][1]
        if total != baseline.STAGES[stage][0] // 4 * baseline.STAGES[stage][1]:
            raise ValueError("Incorrect N1 update budget")
        scheduler = get_linear_schedule_with_warmup(optimizer, int(total * .06), total)
        manifest.update({"parameters": sum(p.numel() for p in model.parameters()), "total_steps": total,
                         "warmup_steps": int(total * .06), "batches_per_epoch": len(loader), "optimizer_groups": groups})
        shared.write_json(directory / "manifest.json", manifest)
        shared.write_json(directory / "args.json", vars(args))
        phase = time.perf_counter()
        complete = train_loop(model, loader, dev_loader, args, stage, directory, gold, annotated, distant, optimizer, scheduler, buckets)
        training_seconds = time.perf_counter() - phase
        if complete["optimizer_steps"] != total:
            raise RuntimeError("Incomplete N1 training budget")
        del optimizer, scheduler
        model.zero_grad(set_to_none=True)
        if args.device.type == "cuda":
            torch.cuda.empty_cache()
        phase = time.perf_counter()
        replay = baseline.replay(model, directory, dev_loader, args, stage, gold, annotated, distant)
        replay_seconds = time.perf_counter() - phase
        if stage == "human-frozen" and not torch.equal(memory_initial, model.mu_encoder.memory_tokens.detach().cpu()):
            raise RuntimeError("Selected Human checkpoint altered frozen memory")
        best = shared.read_json(directory / "best_metrics.json")
        phase = time.perf_counter()
        bucket_report = frequency_report(model, dev_loader, args, gold, annotated, distant, buckets, best)
        bucket_seconds = time.perf_counter() - phase
        shared.write_json(directory / "frequency_diagnostics.json", bucket_report)
        shared.write_json(directory / "checkpoint_replay.json", replay)
        manifest["best_checkpoint_sha256"] = replay["checkpoint_sha256"]
        manifest["selected_memory_change_L2"] = float((model.mu_encoder.memory_tokens.detach().cpu() - memory_initial).double().norm())
        preflight.assert_hashes(identity["cache_input_sha256"])
        for name, checksum in identity["source_files"].items():
            if shared.sha256(ROOT / name) != checksum:
                raise RuntimeError(f"Source changed during training: {name}")
        shared.write_json(directory / "manifest.json", manifest)
        shared.write_json(directory / "timing.json", {"elapsed_seconds": time.perf_counter() - started,
            "phases_seconds": {"probes": probe_seconds, "training_including_dev_and_saves": training_seconds,
                               "checkpoint_replay": replay_seconds, "frequency_diagnostics": bucket_seconds}})
        shared.write_json(directory / "completion.json", {"stage": stage, "variant": VARIANT, **complete,
            "checkpoint_replay_passed": True, "distant_initialization_reference": parent,
            "frequency_diagnostics_passed": True,
            "delta_vs_baseline_F1_pp": complete["best_dev_F1"] - identity["baseline_references"][stage]["best_metrics"]["dev_F1"],
            "screening_gate_passed": (complete["best_dev_F1"] >= 79.17549145362884 + .5) if stage == "human-frozen" else None,
            "interpretation": "single-seed N1; Human-derived priors disclosed; no Test"})
        aligned.emit("stage_complete", stage=stage, **complete)
    except Exception as error:
        shared.write_json(directory / "failure.json", {"error_type": type(error).__name__, "error": str(error),
            "time_utc": datetime.now(timezone.utc).isoformat(), "no_automatic_retry": True})
        raise


def run(args, identity):
    import torch
    from torch.utils.data import DataLoader
    baseline.control.runtime(identity["environment"])
    # Reservation outside failure handler: never write to an existing run.
    args.run_dir.mkdir(parents=True, exist_ok=False)
    try:
        shared.write_json(args.run_dir / "identity.json", identity)
        model = load_generic_model(args, identity)
        dev = cache_dataset(args, "dev", with_labels=False)
        loader = DataLoader(dev, batch_size=4, shuffle=False, collate_fn=shared.collate_fn)
        gold = shared.read_json(args.data_dir / "dev_revised.json")
        human, remote = cache_dataset(args, "human"), cache_dataset(args, "distant")
        annotated = {tuple(f) for f in human.metadata["matching_dev_facts"]}
        distant = {tuple(f) for f in remote.metadata["matching_dev_facts"]}
        buckets = method.frequency_buckets(human.metadata["priors"])
        train_stage(model, args, identity, "distant", loader, gold, annotated, distant, buckets)
        parent_dir = args.run_dir / "distant/artifacts"
        parent = verify_stage(parent_dir, "distant", identity)
        model.load_state_dict(torch.load(parent_dir / "best_model.pth", map_location="cpu", weights_only=True), strict=True)
        aligned.emit("human_initialization_loaded", checkpoint_sha256=parent["checkpoint_sha256"],
                     selected_after_weighting=parent["selected_after_weighting"])
        train_stage(model, args, identity, "human-frozen", loader, gold, annotated, distant, buckets, parent)
        verify_stage(args.run_dir / "human-frozen/artifacts", "human-frozen", identity)
        print("MEMORY_NOISE_N1_TRAINING_COMPLETE; optimizer_steps=73826; test_labels_read=false", flush=True)
    except Exception as error:
        shared.write_json(args.run_dir / "failure.json", {"error_type": type(error).__name__, "error": str(error),
            "time_utc": datetime.now(timezone.utc).isoformat(), "no_automatic_retry": True})
        raise


def main():
    args = parser().parse_args()
    identity = validate(args)
    if args.mode == "validate":
        print(json.dumps({"validation": "passed", "identity": identity, "weights_loaded": False,
                          "optimizer_steps": 0, "files_written": False}, ensure_ascii=False, indent=2))
        return
    sys.path.insert(0, str(ROOT))
    if args.mode == "probe":
        model = load_generic_model(args, identity)
        args.device = next(model.parameters()).device
        report = probe(model, cache_dataset(args, "distant"), args, "distant")
        print(json.dumps(report, allow_nan=False))
        print("MEMORY_NOISE_N1_PROBE_OK; optimizer_steps=0; files_written=false", flush=True)
        return
    run(args, identity)


if __name__ == "__main__":
    main()
