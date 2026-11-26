"""Two authorized original S-PU Human seeds76/77 plus fixed soup, total6h.

Default validate is read-only. Launch once with the new Bash entry point.
No Distant/ATGL training, Test, tuning, retry, resume or automatic follow-up.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import shutil
import signal
import statistics
import subprocess
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import evaluate_ttmre_atgl_soup_diagnostic as diagnostic

soup, base = diagnostic.soup, diagnostic.base
core = base.core
read, digest, write_json, emit = diagnostic.read, diagnostic.digest, diagnostic.write_json, diagnostic.emit
RUN_ROOT = ROOT / "experiments/2026-10-21-spu-soup-control"
CLAIM = "spu-soup-control-once-claim.json"
FILES = tuple("scripts/" + n for n in (
    "train_ttmre_spu_soup_control.py", "check_ttmre_spu_soup_control.py", "run_ttmre_spu_soup_control.sh"))
HARD_SECONDS, SOFT_SECONDS, TERM_SECONDS = 21600, 120, 30
TRAIN_DOCS, DEV_DOCS, EPOCHS = 3053, 500, 30
SEEDS, MEMORY, GIB = (76, 77), soup.MEMORY, 1024 ** 3
METRICS = soup.METRICS
SPU_FORWARD = core.training_forward
PROGRESS = {"phase": "not-started", "seed": None, "formal_optimizer_updates": 0}
DIAGNOSTIC_PINS = {
    "identity.json": "53bcb45bbfa2b263c45b8a0977a2050ab37614bfccdff803f055a669d8eebfe0",
    "comparison/completion.json": "e09e707ac1c9c94a9906bebd768a4e24715229bdebf9377596d66311444d4402"}
ATGL_PINS = {
    76: {"manifest.json": "9509c9c3117150660b2028cdce28199782b4ecf083cdd268f4a1b4787bb1c88e",
         "training_order.json": "7188a1cc3dc5711284486f43227d683ef55b7a227998f32c6e696fd718f07fdf"},
    77: {"manifest.json": "857cdc9fc5e8f75e53e59ac9a46cad59713128fc36d181db2425098b499d2856",
         "training_order.json": "3353c4a4dea9d31db33b9e53c0130cdbe9224dbd77a10987eec15197261be8a7"}}


def updates():
    return TRAIN_DOCS // 4 * EPOCHS


def protocol():
    return {"version": "spu-soup-control-v1", "new_Human_seeds": list(SEEDS), "reused_seed": 75,
            "parent_sha256": base.PARENT_SHA, "loss": "original model2 S-PU with unchanged priors and double prior factors",
            "epochs": EPOCHS, "updates_per_arm": updates(), "new_optimizer_updates": 2 * updates(),
            "batch_size": 4, "precision": "float32", "freeze": [MEMORY],
            "backbone_lr": 3e-5, "head_lr": 1e-5, "eps": 1e-6, "weight_decay": .01,
            "warmup_steps": int(updates() * .06), "clip": 1.,
            "selection": "penultimate batch raw Dev F1; earliest tie", "decoder": soup.protocol()["decoder"],
            "soup_seeds": [75, 76, 77], "soup_weights": [1/3] * 3, "soup_accumulation": "CPU float64 then float32",
            "new_soup_checkpoint_saved": False, "ATGL_model_forwards": 0, "test_data_read": False,
            "hard_seconds": HARD_SECONDS, "soft_reserve_seconds": SOFT_SECONDS, "term_reserve_seconds": TERM_SECONDS,
            "ATGL_minus_SPU_soup_F1_investment_min_pp": .30, "both_Ign_nonnegative": True,
            "automatic_retry": False, "automatic_resume": False, "extension": False, "automatic_followup": False}


def check_budget(budget, reserve=SOFT_SECONDS):
    if budget is None:
        return
    if time.monotonic() >= budget["deadline_monotonic"] - reserve:
        raise TimeoutError("Six-hour budget reserve reached; preserve partials, no extension")
    if os.name == "posix" and Path("/proc/sys/kernel/random/boot_id").read_text().strip() != budget["boot_id"]:
        raise TimeoutError("Boot identity changed; do not reset budget")


def verify_files(files, budget):
    for name, sha in files.items():
        check_budget(budget)
        if digest(Path(name), budget) != sha:
            raise ValueError("Pinned source/input missing or changed: " + name)


def validate(budget=None):
    """Pin the completed reports, needed assets and effective training pairing."""
    old = Path(read(diagnostic.RUN_ROOT / diagnostic.CLAIM)["workspace"])
    if old != old.resolve() or old.name != "workspace" or old.parent.parent != diagnostic.RUN_ROOT.resolve():
        raise ValueError("Unexpected completed diagnostic workspace")
    pinned = {str(old / n): sha for n, sha in DIAGNOSTIC_PINS.items()}
    verify_files(pinned, budget)
    prior, done = read(old / "identity.json"), read(old / "comparison/completion.json")
    if not done["passed"] or done["marker"] != "ATGL_SOUP_DIAGNOSTIC_COMPLETE" or done["optimizer_updates"] != 0:
        raise ValueError("Previous diagnostic is not complete")
    origin_root = Path(prior["soup_checkpoint"]).parent
    for n in ("identity.json", "comparison/completion.json"):
        pinned[str(origin_root / n)] = diagnostic.OLD_PINS[n]
    verify_files(pinned, budget)
    origin = read(origin_root / "identity.json")
    parent = origin["parent"]
    if parent != prior["parent"]:
        raise ValueError("Baseline provenance differs")
    model_dir, cache = Path(parent["model_dir"]), Path(parent["baseline_workspace"]) / "cache"
    wanted = [cache / split / n for split in ("human", "dev") for n in ("index.json", "features.bin")]
    wanted += [cache / "distant/index.json"]  # Precomputed matching facts, not Distant features.
    wanted += [Path(parent["data_dir"]) / n for n in ("train_revised.json", "dev_revised.json")]
    wanted += [Path(n) for n in parent["input_files"] if Path(n).is_relative_to(model_dir)]
    for path in wanted:
        if not path.resolve().is_relative_to(ROOT.resolve()):
            raise ValueError("Input outside project")
        pinned[str(path)] = origin["pinned_files"][str(path)]
    distant_checkpoint, control75 = Path(parent["distant_checkpoint"]), Path(parent["checkpoint"])
    pinned[str(distant_checkpoint)] = base.PARENT_SHA
    pinned[str(control75)] = base.frozen.CHECKPOINT_SHA
    for name in ("identity.json", "control/completion.json", "control/human-frozen/artifacts/checkpoint_replay.json"):
        path = base.frozen.DONE / name
        pinned[str(path)] = parent["input_files"][str(path)]
    references, pairing = {}, {}
    for source in origin["sources"]:
        seed = source["seed"]
        prediction = Path(origin["reference_predictions"][str(seed)])
        pinned[str(prediction)] = origin["pinned_files"][str(prediction)]
        references[str(seed)] = {"metrics": {k: source["best"][k] for k in METRICS}, "predictions": str(prediction)}
        if seed not in SEEDS:
            continue
        art = Path(source["checkpoint"]).parent
        for name, expected in ATGL_PINS[seed].items():
            pinned[str(art / name)] = expected
        for name in ("initialization.json", "completion.json"):
            path = art / name
            pinned[str(path)] = origin["pinned_files"][str(path)]
        verify_files({str(art / n): pinned[str(art / n)] for n in (*ATGL_PINS[seed], "initialization.json", "completion.json")}, budget)
        manifest, init, completion, order = (read(art / n) for n in ("manifest.json", "initialization.json", "completion.json", "training_order.json"))
        if (completion["protocol"]["seed"] != seed or not completion["passed"]
                or completion["optimizer_steps"] != updates() or completion["parent_sha256"] != base.PARENT_SHA
                or not completion["memory_unchanged"] or order["sample_order_sha256"] != completion["sample_order_sha256"]):
            raise ValueError("Invalid paired ATGL metadata")
        pairing[str(seed)] = {"initial_state_sha256": init["state_sha256"],
                              "initial_rng_sha256": manifest["initial_rng_sha256"],
                              "sample_order_sha256": completion["sample_order_sha256"],
                              "final_rng_sha256": completion["final_rng_sha256"],
                              "optimizer_groups": manifest["optimizer_groups"]}
    soup_predictions = origin_root / "predictions.json"
    pinned[str(soup_predictions)] = diagnostic.OLD_PINS["predictions.json"]
    verify_files(pinned, budget)
    sources = dict(prior["source_files"])
    verify_files({str(ROOT / n): sha for n, sha in sources.items()}, budget)
    sources.update({n: digest(ROOT / n, budget) for n in FILES})
    return {"protocol": protocol(), "parent": parent, "pinned_files": pinned, "source_files": sources,
            "distant_checkpoint": str(distant_checkpoint), "control75_checkpoint": str(control75),
            "control75_metrics": parent["control_best"], "ATGL_references": references, "pairing": pairing,
            "ATGL_soup": {"metrics": done["native_references"]["soup"], "predictions": str(soup_predictions)},
            "test_data_read": False, "old_model_optimizations": 0}


def resources(path, identity, phase="start"):
    largest = max(Path(identity[k]).stat().st_size for k in ("distant_checkpoint", "control75_checkpoint"))
    pending = {"start": 3, "seed76": 3, "seed77": 2, "finish": 0}[phase]
    required = max(6 * GIB if phase == "start" else GIB, pending * largest + GIB)
    free = shutil.disk_usage(path).free
    if free < required:
        raise RuntimeError("Project disk free %.3fGiB < required %.3fGiB (%s)" % (free/GIB, required/GIB, phase))
    return {"phase": phase, "free_bytes": free, "required_bytes": required,
            "new_soup_checkpoint_saved": False, "recommended_available_RAM_GiB": 32}


def data_context(workspace, identity, budget):
    parent = identity["parent"]
    args = SimpleNamespace(run_dir=workspace, baseline_workspace=Path(parent["baseline_workspace"]),
                           model_dir=Path(parent["model_dir"]), data_dir=Path(parent["data_dir"]), budget=budget,
                           device=core.baseline.control.runtime(parent["environment"]))
    data = base.frozen.original_features(parent)
    human, dev = data["train"], data["dev"]
    if len(human) != TRAIN_DOCS or len(dev) != DEV_DOCS:
        raise ValueError("Human/Dev counts changed")
    gold = read(args.data_dir / "dev_revised.json")
    annotated = {tuple(x) for x in human.metadata["matching_dev_facts"]}
    distant = {tuple(x) for x in read(args.baseline_workspace / "cache/distant/index.json")["matching_dev_facts"]}
    return args, human, dev, gold, annotated, distant


def probe(model, human, args, budget):
    import torch
    from ttmre_memory_fit import reserve_adam_moments
    before, rng = core.preflight.state_digest(model), core.preflight.rng_state()
    flags = [(module, module.training) for module in model.modules()]
    memory = model.mu_encoder.memory_tokens.detach().cpu().clone()
    buffers, reserve = reserve_adam_moments(model)
    rows = []
    try:
        for name, indices in human.probe_indices().items():
            check_budget(budget)
            model.train(); model.zero_grad(set_to_none=True)
            loss, logits = SPU_FORWARD(model, core.shared.collate_fn([human[i] for i in indices]), args.device)
            loss.backward()
            gradients = [p.grad for p in model.parameters() if p.grad is not None]
            critical = (model.head_extractor.weight, model.bilinear.weight, model.model.embeddings.word_embeddings.weight)
            if (not gradients or any(not torch.isfinite(g).all() for g in gradients)
                    or any(p.grad is None or not torch.count_nonzero(p.grad) for p in critical)
                    or model.mu_encoder.memory_tokens.grad is not None):
                raise RuntimeError("Original S-PU probe missing/nonfinite gradients")
            rows.append({"probe": name, "indices": indices, "loss": float(loss.detach()), "finite_gradient_tensors": len(gradients)})
            emit("SPU_SOUP_ZERO_UPDATE_PROBE", **rows[-1], optimizer_updates=0)
            del loss, logits, gradients
    finally:
        model.zero_grad(set_to_none=True)
        model.sims = None
        del buffers
        core.preflight.restore_rng(rng)
        for module, flag in flags:
            module.training = flag
        if args.device.type == "cuda":
            torch.cuda.empty_cache()
    core.preflight.assert_rng_equal(rng, core.preflight.rng_state())
    if before != core.preflight.state_digest(model) or not diagnostic.tensor_bytes_equal(memory, model.mu_encoder.memory_tokens):
        raise RuntimeError("Probe changed state or frozen memory")
    return {"passed": True, "optimizer_updates": 0, "rng_restored": True, "parameters_unchanged": True,
            "moment_reservation": reserve, "probes": rows}


@contextmanager
def training_scope(seed, budget):
    old_forward, old_emit, old_aligned = core.training_forward, core.emit, core.aligned.emit
    if old_forward is not SPU_FORWARD:
        raise ValueError("Original S-PU forward was replaced")
    def forward(*args):
        check_budget(budget)
        return SPU_FORWARD(*args)
    def event(name, **fields):
        if name == "N1_FULL_SEED75_LEARNING_UPDATE_VERIFIED":
            name = "SPU_SOUP_LEARNING_UPDATE_VERIFIED"
        if name in ("SPU_SOUP_LEARNING_UPDATE_VERIFIED", "epoch_complete"):
            PROGRESS["formal_optimizer_updates"] = SEEDS.index(seed) * updates() + fields["step"]
        emit(name, human_seed=seed, **fields)
        if name == "epoch_complete" and fields["epoch"] == 1:
            remaining_epochs = (len(SEEDS) - SEEDS.index(seed)) * EPOCHS - 1
            estimate = fields["seconds"] * remaining_epochs + 900
            available = budget["deadline_monotonic"] - time.monotonic() - SOFT_SECONDS
            emit("SPU_SOUP_REMAINING_PROJECTION", seed=seed, estimate_seconds=estimate, available_seconds=available)
            if estimate > available:
                raise TimeoutError("Measured projection exceeds original six-hour budget")
    core.training_forward, core.emit, core.aligned.emit = forward, event, event
    try:
        yield
    finally:
        core.training_forward, core.emit, core.aligned.emit = old_forward, old_emit, old_aligned


def save_prediction_report(directory, predictions, order, metrics, gold, human):
    write_json(directory / "predictions.json", predictions)
    write_json(directory / "dev_pair_order.json", order)
    write_json(directory / "metrics.json", metrics)
    buckets = core.method.frequency_buckets(human.metadata["priors"])
    write_json(directory / "frequency_diagnostics.json", {
        "passed": True, "metrics": metrics, "buckets": diagnostic.frequency_report(predictions, gold, buckets, read(ROOT / "meta/rel2id.json")),
        "predictions_sha256": digest(directory / "predictions.json"), "optimizer_updates": 0})


def grouped_replay(model, dev, args, budget, best, memory):
    before = core.preflight.state_digest(model)
    strict = core.aligned.grouping_invariance(model, dev, args.device)
    full, predictions = base.accepted.full_dev_grouping(model, dev, args.device, budget)
    valid = (full["passed"] and strict["changed_relation_decisions"] == 0
             and all(strict["checks"][k]["close"] for k in ("ttm_grouped_vs_single", "ttm_permutation"))
             and before == core.preflight.state_digest(model) and diagnostic.tensor_bytes_equal(memory, model.mu_encoder.memory_tokens))
    return {"passed": bool(valid), "original_strict_grouping": strict, "full_dev_grouping": full,
            "parameters_unchanged": before == core.preflight.state_digest(model), "optimizer_updates": 0}, predictions


def check_selection(directory):
    rows = [json.loads(line) for line in (directory / "metrics.jsonl").read_text().splitlines()]
    batches = TRAIN_DOCS // 4
    if len(rows) != EPOCHS or any(row["epoch"] != i or row["step"] != i*batches-1 or row["batch"] != batches-1
                                  for i, row in enumerate(rows, 1)):
        raise ValueError("Wrong epoch count or penultimate selection timing")
    best = read(directory / "best_metrics.json")
    selected = max(rows, key=lambda r: r["dev_F1"])
    if any(selected[k] != best[k] for k in ("epoch", "step", *METRICS)):
        raise ValueError("Best checkpoint is not earliest raw-F1 maximum")


def run_arm(workspace, identity, budget, seed):
    import numpy as np
    import torch
    from torch.utils.data import DataLoader
    from transformers.optimization import get_linear_schedule_with_warmup
    args, human, dev, gold, annotated, distant = data_context(workspace, identity, budget)
    directory = workspace / ("seed" + str(seed))
    directory.mkdir()
    write_json(directory / "resource_check.json", resources(workspace, identity, "seed" + str(seed)))
    model = core.load_generic_model_seed75(args, identity["parent"]["baseline_identity"])
    model.load_state_dict(torch.load(identity["distant_checkpoint"], map_location="cpu", weights_only=True), strict=True)
    model.requires_grad_(True)
    core.baseline.set_stage(model, "human-frozen")
    expected = identity["pairing"][str(seed)]
    initial = core.preflight.state_digest(model)
    frozen = [n for n, p in model.named_parameters() if not p.requires_grad]
    if initial != expected["initial_state_sha256"] or frozen != [MEMORY]:
        raise ValueError("Initial state/freeze differs from same-seed ATGL")
    memory = model.mu_encoder.memory_tokens.detach().cpu().clone()
    write_json(directory / "initialization.json", {"seed": seed, "state_sha256": initial,
               "parent_sha256": base.PARENT_SHA, "frozen": frozen, "same_as_paired_ATGL": True})
    hook = model.register_forward_pre_hook(lambda *_: check_budget(budget))
    try:
        write_json(directory / "full_model_check.json", probe(model, human, args, budget))
        optimizer = core.baseline.optimizer_for_stage(model, "human-frozen")
        groups = core.baseline.optimizer_report(model, optimizer)
        if optimizer.state or groups != expected["optimizer_groups"]:
            raise ValueError("Optimizer is not fresh or differs from fixed matched groups")
        core.shared.seed_all(seed, torch, np)
        loader = DataLoader(human, batch_size=4, shuffle=True, drop_last=True, collate_fn=core.shared.collate_fn)
        dev_loader = DataLoader(dev, batch_size=4, shuffle=False, collate_fn=core.shared.collate_fn)
        total = len(loader) * EPOCHS
        if total != updates():
            raise ValueError("Wrong optimizer update count")
        scheduler = get_linear_schedule_with_warmup(optimizer, int(total*.06), total)
        initial_rng = core.rng_digest()
        if initial_rng != expected["initial_rng_sha256"]:
            raise ValueError("Training RNG initialization differs from paired ATGL")
        write_json(directory / "manifest.json", {"protocol": protocol(), "seed": seed, "optimizer_groups": groups,
                   "initial_rng_sha256": initial_rng, "total_steps": total, "identity_sha256": digest(workspace / "identity.json")})
        emit("SPU_SOUP_HUMAN_STARTED", seed=seed, original_loss=True, parent_sha256=base.PARENT_SHA)
        with training_scope(seed, budget):
            result = core.train_loop(model, loader, dev_loader, args, "spu", "human-frozen", directory,
                        gold, annotated, distant, core.method.frequency_buckets(human.metadata["priors"]), budget, optimizer, scheduler)
        del optimizer, scheduler
        pairing = {"initial_state": initial == expected["initial_state_sha256"],
                   "initial_rng": initial_rng == expected["initial_rng_sha256"],
                   "sample_order": result["sample_order_sha256"] == expected["sample_order_sha256"],
                   "final_rng": result["final_rng_sha256"] == expected["final_rng_sha256"]}
        write_json(directory / "training_order.json", {"seed": seed, "formal_optimizer_updates": result["optimizer_steps"],
                   "sample_order_sha256": result["sample_order_sha256"], "final_rng_sha256": result["final_rng_sha256"],
                   "same_seed_ATGL_pairing": pairing})
        if not all(pairing.values()) or result["optimizer_steps"] != total or not result["learning_update_check_passed"]:
            raise ValueError("Actual S-PU learning/count/order/RNG pairing failed")
        check_selection(directory)
        model.zero_grad(set_to_none=True)
        model.load_state_dict(torch.load(directory / "best_model.pth", map_location="cpu", weights_only=True), strict=True)
        best = read(directory / "best_metrics.json")
        replay, predictions = grouped_replay(model, dev, args, budget, best, memory)
        score = core.shared.score_predictions(predictions, gold, annotated, distant)
        replay["metrics"] = score
        replay["metric_replay_passed"] = all(math.isclose(score[k], best[k], rel_tol=0, abs_tol=1e-6) for k in METRICS)
        replay["passed"] = replay["passed"] and replay["metric_replay_passed"]
        write_json(directory / "checkpoint_replay.json", replay)
        if not replay["passed"]:
            raise RuntimeError("Selected S-PU checkpoint acceptance failed")
        order = [{"title": f["title"], "hts": f["hts"]} for f in dev]
        save_prediction_report(directory, predictions, order, score, gold, human)
        result.update({"passed": True, "seed": seed, "protocol": protocol(), "best": best,
                       "checkpoint_sha256": digest(directory / "best_model.pth", budget), "pairing": pairing,
                       "checkpoint_replay_passed": True, "frequency_diagnostics_passed": True})
        write_json(directory / "completion.json", result)
        emit("SPU_SOUP_HUMAN_COMPLETE", seed=seed, F1=score["dev_F1"], optimizer_updates=total)
    finally:
        PROGRESS["formal_optimizer_updates"] = SEEDS.index(seed) * updates() + int(getattr(model, "_n1_completed_optimizer_steps", 0))
        hook.remove()


def verify_arm(workspace, seed, identity, budget):
    directory = workspace / ("seed" + str(seed))
    result = read(directory / "completion.json")
    replay, learning, order = (read(directory / n) for n in ("checkpoint_replay.json", "learning_update_check.json", "training_order.json"))
    if (not result["passed"] or result["protocol"] != protocol() or result["seed"] != seed
            or result["optimizer_steps"] != updates()
            or set(result["pairing"]) != {"initial_state", "initial_rng", "sample_order", "final_rng"}
            or not all(result["pairing"].values())
            or not result["memory_unchanged"] or not replay["passed"] or not replay["metric_replay_passed"]
            or not learning["passed"] or not learning["parameter_changed"] or learning["lr"] <= 0
            or not 1 <= learning["step"] <= updates()
            or learning["optimizer_state_step"] != learning["step"] or learning["extra_optimizer_updates"] != 0
            or order["seed"] != seed or order["formal_optimizer_updates"] != updates()
            or order["sample_order_sha256"] != identity["pairing"][str(seed)]["sample_order_sha256"]
            or order["final_rng_sha256"] != identity["pairing"][str(seed)]["final_rng_sha256"]
            or any(not math.isclose(replay["metrics"][k], result["best"][k], rel_tol=0, abs_tol=1e-6) for k in METRICS)
            or digest(directory / "best_model.pth", budget) != result["checkpoint_sha256"]):
        raise ValueError("Completed S-PU arm evidence failed")
    check_selection(directory)
    return result


def comparison(spu, atgl, spu_soup, atgl_soup):
    paired = {str(seed): {k: atgl[str(seed)][k] - spu[str(seed)][k] for k in METRICS} for seed in (75, 76, 77)}
    summaries = {}
    for label, rows, combined in (("SPU", spu, spu_soup), ("ATGL", atgl, atgl_soup)):
        means = {k: statistics.mean(rows[str(seed)][k] for seed in (75, 76, 77)) for k in METRICS}
        summaries[label] = {"single_seed_mean": means, "single_seed_sample_sd": {
            k: statistics.stdev(rows[str(seed)][k] for seed in (75, 76, 77)) for k in METRICS},
            "soup": combined, "soup_minus_single_mean_pp": {k: combined[k] - means[k] for k in METRICS},
            "independent_soup_groups": 1}
    delta = {k: atgl_soup[k] - spu_soup[k] for k in METRICS}
    interaction = {k: summaries["ATGL"]["soup_minus_single_mean_pp"][k] - summaries["SPU"]["soup_minus_single_mean_pp"][k] for k in METRICS}
    gate = delta["dev_F1"] >= .30 and all(delta[k] >= 0 for k in ("dev_IgnF1_annotated", "dev_IgnF1_distant"))
    return {"paired_ATGL_minus_SPU_pp": paired, "summaries": summaries, "ATGL_minus_SPU_soup_pp": delta,
            "difference_in_soup_increments_pp": interaction, "candidate_investment_gate_passed": bool(gate),
            "interpretation": "one soup per loss/shared Distant parent; interaction is descriptive, not a novelty/significance claim"}


def finish(workspace, identity, budget):
    import torch
    from torch.utils.data import DataLoader
    results = {str(seed): verify_arm(workspace, seed, identity, budget) for seed in SEEDS}
    args, human, dev, gold, annotated, distant = data_context(workspace, identity, budget)
    write_json(workspace / "resource_finish.json", resources(workspace, identity, "finish"))
    model = core.load_generic_model_seed75(args, identity["parent"]["baseline_identity"])
    model.requires_grad_(False)
    model.eval()
    parent_state = torch.load(identity["distant_checkpoint"], map_location="cpu", weights_only=True)
    memory = parent_state[MEMORY].clone()
    del parent_state
    loader = DataLoader(dev, batch_size=4, shuffle=False, collate_fn=core.shared.collate_fn)
    hook = model.register_forward_pre_hook(lambda *_: check_budget(budget))
    try:
        # Control75 has no prior discrete export; this is one required missing-evidence capture.
        model.load_state_dict(torch.load(identity["control75_checkpoint"], map_location="cpu", weights_only=True), strict=True)
        before = core.preflight.state_digest(model)
        predictions, order = soup.evaluate_predictions(model, loader, args, budget)
        score75 = core.shared.score_predictions(predictions, gold, annotated, distant)
        if (any(not math.isclose(score75[k], identity["control75_metrics"][k], abs_tol=1e-6, rel_tol=0) for k in METRICS)
                or before != core.preflight.state_digest(model) or not diagnostic.tensor_bytes_equal(memory, model.mu_encoder.memory_tokens)):
            raise ValueError("Reused Control75 export changed original metrics/state/memory")
        reference = workspace / "reference-spu75"
        reference.mkdir()
        save_prediction_report(reference, predictions, order, score75, gold, human)
        write_json(reference / "reuse.json", {"checkpoint_sha256": base.frozen.CHECKPOINT_SHA, "optimizer_updates": 0,
                   "reason_for_forward": "original accepted Control75 report did not export discrete predictions", "parameters_unchanged": True})
        paths = [identity["control75_checkpoint"], *(str(workspace / ("seed"+str(seed)) / "best_model.pth") for seed in SEEDS)]
        original_emit = soup.emit
        try:
            soup.emit = lambda _event, **fields: emit("SPU_SOUP_SOURCE_LOADED", **fields)
            state, average = soup.uniform_state(paths, budget)
        finally:
            soup.emit = original_emit
        average.update({"source_checkpoints": [{"path": path, "sha256": digest(Path(path), budget)} for path in paths],
                        "checkpoint_saved": False, "rebuild": "same source order75/76/77; FP64 sum/divide3 -> FP32; preserve frozen memory/nonfloat"})
        expected_sources = [digest(Path(identity["control75_checkpoint"]), budget),
                            *(results[str(seed)]["checkpoint_sha256"] for seed in SEEDS)]
        if [row["sha256"] for row in average["source_checkpoints"]] != expected_sources:
            raise ValueError("Trained source checkpoint changed before averaging")
        write_json(workspace / "averaging.json", average)
        model.load_state_dict(state, strict=True)
        state_sha = core.preflight.state_digest(model)
        predictions, order = soup.evaluate_predictions(model, loader, args, budget)
        soup_score = core.shared.score_predictions(predictions, gold, annotated, distant)
        model.load_state_dict(state, strict=True)  # Strict reload from the immutable, reconstructible in-memory state.
        del state
        replay, repeated = grouped_replay(model, dev, args, budget, soup_score, memory)
        replay["prediction_sets_equal"] = soup.facts(predictions) == soup.facts(repeated)
        replay["passed"] = replay["passed"] and replay["prediction_sets_equal"] and state_sha == core.preflight.state_digest(model)
        directory = workspace / "spu-soup"
        directory.mkdir()
        write_json(directory / "checkpoint_replay.json", replay)
        if not replay["passed"]:
            raise RuntimeError("SPU soup acceptance failed")
        save_prediction_report(directory, predictions, order, soup_score, gold, human)
        write_json(directory / "recipe.json", {"state_sha256": state_sha, "averaging": average,
                   "parameters_unchanged": True, "memory_unchanged": True, "new_full_checkpoint": False})
    finally:
        hook.remove()
    atgl = {}
    for seed, row in identity["ATGL_references"].items():
        score = core.shared.score_predictions(read(Path(row["predictions"])), gold, annotated, distant)
        if any(not math.isclose(score[k], row["metrics"][k], abs_tol=1e-6, rel_tol=0) for k in METRICS):
            raise ValueError("Archived ATGL prediction scores changed")
        atgl[seed] = score
    atgl_soup = core.shared.score_predictions(read(Path(identity["ATGL_soup"]["predictions"])), gold, annotated, distant)
    if any(not math.isclose(atgl_soup[k], identity["ATGL_soup"]["metrics"][k], abs_tol=1e-6, rel_tol=0) for k in METRICS):
        raise ValueError("Archived ATGL soup scores changed")
    spu = {"75": score75, **{seed: {k: row["best"][k] for k in METRICS} for seed, row in results.items()}}
    report = comparison(spu, atgl, soup_score, atgl_soup)
    compare_dir = workspace / "comparison"
    compare_dir.mkdir()
    write_json(compare_dir / "paired_metrics.json", {"SPU": spu, "ATGL": atgl, **report})
    verify_files(identity["pinned_files"], budget)
    verify_files({str(ROOT / n): sha for n, sha in identity["source_files"].items()}, budget)
    complete = {"passed": True, "marker": "SPU_SOUP_CONTROL_COMPLETE", "protocol": protocol(),
                "formal_optimizer_updates_new": 2*updates(), "old_model_optimizer_updates_new": 0,
                "ATGL_model_forwards": 0, "new_soup_checkpoint_saved": False, "soup_metrics": soup_score,
                "ATGL_soup_metrics": atgl_soup, **report, "test_data_read": False, "test_evaluated": False,
                "elapsed_seconds": time.monotonic() - budget["start_monotonic"], "automatic_followup": False,
                "reliable_improvement_established": False, "novelty_established": False}
    write_json(compare_dir / "completion.json", complete)
    soup.archive_reports(workspace, budget)
    emit("SPU_SOUP_CONTROL_COMPLETE", SPU_soup_F1=soup_score["dev_F1"],
         ATGL_minus_SPU_soup_F1_pp=report["ATGL_minus_SPU_soup_pp"]["dev_F1"],
         candidate_investment_gate_passed=report["candidate_investment_gate_passed"], reports=str(compare_dir / "reports.tar.gz"))


def require_lock():
    if os.name != "posix" or os.environ.get("TTMRE_SPU_SOUP_LOCK_FD") != "9":
        raise RuntimeError("Use the new one-use SPU soup launcher")
    import fcntl
    a, b = os.fstat(9), (ROOT / "experiments/2026-10-08-baseline-alignment/.training.lock").stat()
    if (a.st_dev, a.st_ino) != (b.st_dev, b.st_ino):
        raise RuntimeError("Wrong shared training lock")
    fcntl.flock(9, fcntl.LOCK_EX | fcntl.LOCK_NB)
    return 9


def verify_context(workspace, budget):
    claim = read(RUN_ROOT / CLAIM)
    if (workspace != workspace.resolve() or workspace.name != "workspace" or workspace.parent.parent != RUN_ROOT.resolve()
            or budget["hard_seconds"] != HARD_SECONDS or budget["deadline_monotonic"] - budget["start_monotonic"] != HARD_SECONDS
            or any(budget[k] is not False for k in ("automatic_retry", "automatic_resume", "extension"))
            or claim["protocol"] != protocol() or claim["workspace"] != str(workspace)
            or claim["budget_sha256"] != digest(workspace / "budget.json")
            or claim["identity_sha256"] != digest(workspace / "identity.json")):
        raise ValueError("Invalid immutable six-hour workspace/budget/claim")
    check_budget(budget)


def supervise(command, budget, lock):
    check_budget(budget)
    child = subprocess.Popen(command, start_new_session=True, pass_fds=(lock,))
    term_sent = False
    try:
        while True:
            now, status = time.monotonic(), child.poll()
            if now >= budget["deadline_monotonic"]:
                raise TimeoutError("Six-hour hard cap reached; kill worker group")
            if status is not None:
                if status or term_sent:
                    raise RuntimeError("Worker/checker failed exit=" + str(status) + "; no retry")
                return
            check_budget(budget, reserve=0)
            if now >= budget["deadline_monotonic"] - TERM_SECONDS and not term_sent:
                os.killpg(child.pid, signal.SIGTERM)
                term_sent = True
            time.sleep(.2)
    finally:
        if child.poll() is None:
            os.killpg(child.pid, signal.SIGKILL)
            child.wait()


def worker(workspace, phase):
    require_lock()
    budget = read(workspace / "budget.json")
    verify_context(workspace, budget)
    identity = read(workspace / "identity.json")
    PROGRESS.update({"phase": phase, "seed": int(phase[4:]) if phase.startswith("seed") else None,
                     "formal_optimizer_updates": {"seed76": 0, "seed77": updates(), "finish": 2*updates()}[phase]})
    def stop(*_):
        raise TimeoutError("Worker interrupted; preserve reports, no resume")
    signal.signal(signal.SIGTERM, stop)
    try:
        verify_files(identity["pinned_files"], budget)
        verify_files({str(ROOT / n): sha for n, sha in identity["source_files"].items()}, budget)
        if phase == "finish":
            finish(workspace, identity, budget)
        else:
            run_arm(workspace, identity, budget, int(phase[4:]))
    except BaseException as error:
        # The loop's witnessed progress also covers a mid-epoch stop.
        write_json(workspace / ("failure-"+phase+".json"), {"error": str(error), "type": type(error).__name__,
                   "complete": False, "progress": PROGRESS, "automatic_retry": False, "resumable": False})
        try:
            if not (workspace / "comparison/reports.tar.gz").exists():
                soup.archive_reports(workspace, budget)
        except BaseException:
            pass
        raise


def start(workspace):
    lock = require_lock()
    if workspace.name != "workspace" or workspace.parent.parent != RUN_ROOT.resolve() or workspace.exists():
        raise ValueError("Use a new SPU soup workspace")
    claim_path = RUN_ROOT / CLAIM
    if claim_path.exists():
        raise FileExistsError("Already claimed; no retry")
    now = time.monotonic()
    budget = {"start_monotonic": now, "deadline_monotonic": now+HARD_SECONDS, "hard_seconds": HARD_SECONDS,
              "soft_reserve_seconds": SOFT_SECONDS, "term_reserve_seconds": TERM_SECONDS,
              "start_utc": datetime.now(timezone.utc).isoformat(), "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
              "automatic_retry": False, "automatic_resume": False, "extension": False}
    def stop(*_):
        raise TimeoutError("Supervisor interrupted")
    signal.signal(signal.SIGTERM, stop)
    try:
        emit("SPU_SOUP_BUDGET_STARTED", **budget)
        identity = validate(budget)
        emit("SPU_SOUP_SOURCES_VERIFIED", resources=resources(workspace.parent, identity))
        supervise([sys.executable, "-B", str(ROOT / FILES[1])], budget, lock)
        check_budget(budget)
        workspace.mkdir()
        write_json(workspace / "budget.json", budget)
        write_json(workspace / "identity.json", identity)
        with claim_path.open("x", encoding="utf-8") as handle:
            json.dump({"workspace": str(workspace), "protocol": protocol(), "budget_sha256": digest(workspace / "budget.json"),
                       "identity_sha256": digest(workspace / "identity.json"), "authorization": "user confirmed two SPU Human seeds and fixed soup, total6h"}, handle, indent=2)
            handle.flush(); os.fsync(handle.fileno())
        command = [sys.executable, "-u", "-B", str(Path(__file__).resolve()), "--mode", "worker", "--run-dir", str(workspace)]
        for phase in ("seed76", "seed77", "finish"):
            supervise(command + ["--phase", phase], budget, lock)
    except BaseException as error:
        write_json(workspace.parent / "failure-supervisor.json", {"error": str(error), "type": type(error).__name__,
                   "complete": False, "elapsed_seconds": time.monotonic()-now, "automatic_retry": False})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("validate", "start", "worker"), default="validate")
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--phase", choices=("seed76", "seed77", "finish"))
    args = parser.parse_args()
    if args.mode == "validate":
        print(json.dumps(validate(), indent=2))
    elif args.run_dir is None or (args.mode == "worker" and args.phase is None):
        raise ValueError("Use the single Bash launcher")
    elif args.mode == "start":
        start(args.run_dir.resolve())
    else:
        worker(args.run_dir.resolve(), args.phase)


if __name__ == "__main__":
    main()
