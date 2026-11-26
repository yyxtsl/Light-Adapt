"""One authorized three-ATGL uniform weight soup, zero training, total cap1h.

Default validate is read-only. Use the one-use Bash launcher for evaluation.
All input checkpoints/reports stay read-only; no Test or automatic follow-up.
"""
from __future__ import annotations

import argparse
from collections import Counter, OrderedDict
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tarfile
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import train_ttmre_atgl_human as base
import train_ttmre_atgl_rdrop_human as rdrop
import train_ttmre_atgl_view_repeat as repeat

RUN_ROOT = ROOT / "experiments/2026-10-21-atgl-uniform-soup"
HARD_SECONDS, SOFT_SECONDS, TERM_SECONDS = 3600, 120, 30
GIB = 1024 ** 3
MEMORY = "mu_encoder.memory_tokens"
SEEDS = (75, 76, 77)
TRAIN_DOCS, DEV_DOCS = 3053, 500
METRICS = repeat.METRIC_KEYS
CHECKPOINTS = {
    75: "f795bfc8c51f4b740d55abe992811e728481f36a4167b43036cbb8ca46673c37",
    76: "04985ff6a7d1cf904bc1be14f6d94dd2d080996256ee31b850cb70a9253ed6e3",
    77: "28e6f9fd427a8b85df11ac7a9415a99e1b4295fdb6ca57369268ce4a1c309ac0",
}
REPEAT_COMPLETION_SHA = "7df1dba0ef04abe40475a244b4bd7a36ea0ae14eccc3f097b5c455b70a8ff9d9"
FILES = tuple("scripts/" + name for name in (
    "evaluate_ttmre_atgl_soup.py", "check_ttmre_atgl_soup.py", "run_ttmre_atgl_soup.sh"))
# Pinned metadata hashes are populated from the already audited user archives.
REPORT_PINS = {
    "75": {
        "identity.json": "07f10ce850ed99788365c449b6b9ec2b86ab6ccd1b06731b10e3f8a9926eea73",
        "atgl/human-frozen/artifacts/completion.json": "2ddf5c10d65a13c4054d175c937ccbc3ad62cbd4e2e2e98e98f04ce337f8c8e2",
        "atgl/human-frozen/artifacts/initialization.json": "f7cbafa5706033805f955bf75d1237b9de5f74a2bd32ddd400bd3e150aefd4b6",
        "atgl/human-frozen/artifacts/checkpoint_replay.json": "ba4701abc49601c21ea63b52590b7b585dac5261c69efc50d468e754b05375ea",
        "atgl/human-frozen/artifacts/frequency_diagnostics.json": "8adcfe890d8030a26ace890adccdda375dac839d8a6d8ec825a598b3fe50c0d4"
    },
    "76": {
        "identity.json": "e02bf6776011434fe482081157379c59107ead289ffeeb87b331dca6f311933d",
        "atgl/human-frozen/artifacts/completion.json": "8c6e759423c50686759bd9c1d0c9ddb1db39545c495e4e74d9ef93bf5f144d98",
        "atgl/human-frozen/artifacts/initialization.json": "f7cbafa5706033805f955bf75d1237b9de5f74a2bd32ddd400bd3e150aefd4b6",
        "atgl/human-frozen/artifacts/checkpoint_replay.json": "7eec2be8f101096cec19e1d4f0f11eeb16e1038335983ed3492be7a477735006",
        "atgl/human-frozen/artifacts/frequency_diagnostics.json": "55e943227cb7c30d4f87b2d3c241eaf325fb9d6ba96e9ba551235f380bebae5c"
    },
    "77": {
        "identity.json": "f66483cda2f95dc2a6a769f293da82dda7b0b539628316a205f10f2a09ec803d",
        "atgl/human-frozen/artifacts/completion.json": "2e82f59c8136349d50976b3f33df1c208518e4318bfd810c388df052116fb2d7",
        "atgl/human-frozen/artifacts/initialization.json": "f7cbafa5706033805f955bf75d1237b9de5f74a2bd32ddd400bd3e150aefd4b6",
        "atgl/human-frozen/artifacts/checkpoint_replay.json": "06438b5061604062cd9c0dfaebe54f6130f1bc673905034a726229db0b1c97a9",
        "atgl/human-frozen/artifacts/frequency_diagnostics.json": "46f0be105df813bdd31e6883a6f122272f113efaad07e4257cc9394e237275d7"
    }
}
read, write_json, digest = base.read, base.write_json, base.digest


def emit(event, **fields):
    print(json.dumps({"event": event, **fields}, allow_nan=False), flush=True)


def protocol():
    return {"version": "atgl-uniform-soup-v1", "seeds": list(SEEDS),
            "weights": [1 / 3] * 3, "accumulation": "CPU float64 then original float32",
            "preserve_identical": [MEMORY, "all non-floating buffers"],
            "optimizer_updates": 0, "training": False, "test_data_read": False,
            "dev_documents": 500, "dev_pairs": 193232, "dev_batch_size": 4,
            "precision": "float32", "parent_sha256": base.PARENT_SHA,
            "decoder": "all relation logits > TH; no top-k cap",
            "candidate_F1_min": 80.54123723651996,
            "best_parent_seed": 77, "Ign_reference": {"dev_IgnF1_annotated": 79.13881552850721,
                                                       "dev_IgnF1_distant": 78.11483738514427},
            "hard_seconds": HARD_SECONDS, "soft_reserve_seconds": SOFT_SECONDS,
            "term_reserve_seconds": TERM_SECONDS, "automatic_retry": False,
            "automatic_resume": False, "extension": False, "automatic_followup": False}


def check_budget(budget, reserve=SOFT_SECONDS):
    if time.monotonic() >= budget["deadline_monotonic"] - reserve:
        raise TimeoutError("One-hour budget reserve reached; no retry/extension")
    if os.name == "posix" and Path("/proc/sys/kernel/random/boot_id").read_text().strip() != budget["boot_id"]:
        raise TimeoutError("Boot changed; budget cannot restart")


def pinned_read(path, sha, budget):
    if digest(path, budget) != sha:
        raise ValueError("Pinned source report changed: " + str(path))
    return read(path)


def validate(budget=None):
    """Verify completed sources/inputs only; no tensor loading or inference."""
    parent = base.validate(budget)
    source75 = rdrop.reference(budget)
    repeated = Path(read(repeat.RUN_ROOT / "atgl-view-repeat-once-claim.json")["workspace"])
    if repeated != repeated.resolve() or repeated.name != "workspace" or repeated.parent.parent != repeat.RUN_ROOT.resolve():
        raise ValueError("Unexpected paired-repeat workspace")
    completion = pinned_read(repeated / "comparison/completion.json", REPEAT_COMPLETION_SHA, budget)
    if (not completion["passed"] or completion["marker"] != "ATGL_VIEW_REPEAT_COMPLETE"
            or completion["formal_optimizer_updates_new_runs"] != 91560
            or completion["test_data_read"] is not False):
        raise ValueError("Paired-repeat sources are incomplete")
    sources, files = [], dict(parent["input_files"])
    union_sources = dict(parent["source_files"])
    for seed in SEEDS:
        workspace = Path(source75["workspace"]) if seed == 75 else repeated / f"seed{seed}/atgl/workspace"
        pins = REPORT_PINS[str(seed)]
        reports = {name: pinned_read(workspace / name, expected, budget) for name, expected in pins.items()}
        identity = reports["identity.json"]
        art = "atgl/human-frozen/artifacts/"
        stage, init, replay = (reports[art + name] for name in ("completion.json", "initialization.json", "checkpoint_replay.json"))
        p = stage["protocol"]
        if (p["seed"] != seed or p["epochs"] != 30 or p["alpha"] != 16.5
                or p["precision"] != "float32" or p["freeze"] != [MEMORY]
                or p["parent_sha256"] != base.PARENT_SHA or p["train_batch_size"] != 4
                or p["loss"] != base.protocol()["loss"] or p["decoder"] != protocol()["decoder"]
                or not stage["passed"] or stage["optimizer_steps"] != 22890
                or not stage["memory_unchanged"] or not stage["checkpoint_replay_passed"]
                or not replay["passed"] or replay["full_dev_grouping"]["changed_relation_decisions"] != 0
                or init["parent_sha256"] != base.PARENT_SHA or stage["checkpoint_sha256"] != CHECKPOINTS[seed]):
            raise ValueError("Incompatible ATGL source: seed" + str(seed))
        checkpoint = workspace / art / "best_model.pth"
        if digest(checkpoint, budget) != CHECKPOINTS[seed]:
            raise ValueError("ATGL checkpoint bytes changed: " + str(seed))
        files.update({str(workspace / name): expected for name, expected in pins.items()})
        files[str(checkpoint)] = CHECKPOINTS[seed]
        union_sources.update(identity["source_files"])
        sources.append({"seed": seed, "workspace": str(workspace), "checkpoint": str(checkpoint),
                        "checkpoint_sha256": CHECKPOINTS[seed], "best": stage["best"],
                        "initialization_state_sha256": init["state_sha256"]})
    if len({item["initialization_state_sha256"] for item in sources}) != 1:
        raise ValueError("Parent initialization differs between ATGL models")
    for name, expected in union_sources.items():
        if digest(ROOT / name, budget) != expected:
            raise ValueError("Historical source changed: " + name)
    union_sources.update({name: digest(ROOT / name, budget) for name in FILES})
    # seed75 predictions were originally exported in R-Drop's reference pool.
    old_reference = Path(read(rdrop.RUN_ROOT / "atgl-rdrop-once-claim.json")["workspace"]) / "reference-atgl"
    predictions = {75: old_reference / "predictions.json", **{
        seed: Path(item["workspace"]) / "atgl/human-frozen/artifacts/predictions.json"
        for seed, item in zip(SEEDS[1:], sources[1:])}}
    # The seed77 exported prediction hash is pinned by its diagnostic metadata.
    for seed, path in predictions.items():
        if seed != 75:
            diag = read(path.parent / "frequency_diagnostics.json")
            # The frequency diagnostic is included in REPORT_PINS.
            expected = diag["predictions_sha256"]
        else:
            expected = "fef9100cd4dd580f0307c61eab94ebdf18eeb88ebdb50de92f6d7f08f6ee74ad"
        if digest(path, budget) != expected:
            raise ValueError("Reference predictions changed")
        files[str(path)] = expected
    return {"protocol": protocol(), "parent": parent, "sources": sources,
            "source_files": union_sources, "pinned_files": files,
            "reference_predictions": {str(seed): str(path) for seed, path in predictions.items()},
            "repeat_completion_sha256": REPEAT_COMPLETION_SHA,
            "optimizer_updates": 0, "test_data_read": False}


def resources(workspace, identity):
    largest = max(Path(row["checkpoint"]).stat().st_size for row in identity["sources"])
    required = max(4 * GIB, 2 * largest + GIB)
    free = shutil.disk_usage(workspace).free
    if free < required:
        raise RuntimeError("Project disk free %.3fGiB < required %.3fGiB" % (free / GIB, required / GIB))
    return {"free_bytes": free, "required_bytes": required, "recommended_available_RAM_GiB": 32}


def uniform_state(paths, budget):
    """Strict streaming mean; frozen memory/nonfloat must match exactly."""
    import torch
    if len(paths) != 3:
        raise ValueError("Exactly three ATGL checkpoints are required")
    result, schema, preserved, parameter_bytes = OrderedDict(), {}, set(), 0
    for index, path in enumerate(paths):
        check_budget(budget)
        state = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(state, dict) or MEMORY not in state:
            raise ValueError("Expected raw state_dict including frozen memory")
        if index and set(state) != set(schema):
            raise ValueError("Checkpoint state keys differ")
        for name, value in state.items():
            check_budget(budget)
            if not isinstance(value, torch.Tensor) or value.is_complex() or value.layout != torch.strided:
                raise ValueError("Unsupported state tensor: " + name)
            if index == 0:
                schema[name] = (value.shape, value.dtype)
                parameter_bytes += value.numel() * value.element_size()
            elif schema[name] != (value.shape, value.dtype):
                raise ValueError("Tensor shape or dtype changed: " + name)
            if value.is_floating_point():
                if value.dtype != torch.float32 or not torch.isfinite(value).all():
                    raise ValueError("Expected finite FP32 model state")
            keep = name == MEMORY or not value.is_floating_point()
            if keep:
                preserved.add(name)
                if index == 0:
                    result[name] = value.clone()
                elif not torch.equal(result[name], value):
                    raise ValueError("Frozen memory/nonfloating state differs: " + name)
            elif index == 0:
                result[name] = value.to(torch.float64)
            else:
                result[name].add_(value)
        del state
        emit("ATGL_SOUP_SOURCE_LOADED", seed=SEEDS[index], optimizer_updates=0)
    for name in result:
        check_budget(budget)
        if name not in preserved:
            result[name] = result[name].div_(3).to(torch.float32)
            if not torch.isfinite(result[name]).all():
                raise ValueError("Soup is nonfinite")
    return result, {"passed": True, "models": 3, "weights": [1 / 3] * 3,
                    "accumulation": "CPU float64 then float32", "parameter_bytes": parameter_bytes,
                    "preserved_equal_tensors": sorted(preserved), "optimizer_updates": 0}


def save_state(path, state):
    import torch
    partial = path.with_suffix(".partial.pth")
    if path.exists() or partial.exists():
        raise FileExistsError("Preserve existing soup/partial checkpoint")
    with partial.open("xb") as handle:
        torch.save(state, handle)
        handle.flush()
        os.fsync(handle.fileno())
    partial.rename(path)


def evaluate_predictions(model, loader, args, budget):
    import torch
    import numpy as np
    from evaluation import to_official
    predicted, order, offset = [], [], 0
    model.eval()
    with torch.no_grad():
        for batch in loader:
            check_budget(budget)
            logits = model(**base.core.shared.inputs_from_batch(batch, args.device, labels=False))
            if not torch.isfinite(logits).all():
                raise RuntimeError("Nonfinite soup evaluation logits")
            values = logits.cpu().numpy()
            flags = np.zeros_like(values)
            flags[:, 1:] = values[:, 1:] > values[:, :1]
            flags[:, 0] = flags[:, 1:].sum(1) == 0
            count = len(batch[0])
            features = loader.dataset[offset:offset + count]
            predicted.extend(to_official(flags, features))
            order.extend({"title": item["title"], "hts": item["hts"]} for item in features)
            offset += count
    return predicted, order


def facts(predictions):
    return {(x["title"], x["h_idx"], x["t_idx"], x["r"]) for x in predictions}


def candidate_gate(metrics):
    return (metrics["dev_F1"] >= protocol()["candidate_F1_min"]
            and all(metrics[key] >= value for key, value in protocol()["Ign_reference"].items()))


def execute(workspace, identity, budget):
    import torch
    from torch.utils.data import DataLoader
    parent = identity["parent"]
    args = SimpleNamespace(run_dir=workspace, baseline_workspace=Path(parent["baseline_workspace"]),
                           model_dir=Path(parent["model_dir"]), data_dir=Path(parent["data_dir"]), budget=budget)
    resources_row = resources(workspace, identity)
    write_json(workspace / "resource_check.json", resources_row)
    state, average = uniform_state([item["checkpoint"] for item in identity["sources"]], budget)
    checkpoint = workspace / "soup_model.pth"
    memory = state[MEMORY].clone()
    write_json(workspace / "averaging.json", average)
    save_state(checkpoint, state)
    del state
    emit("ATGL_SOUP_AVERAGE_SAVED", checkpoint_sha256=digest(checkpoint, budget), optimizer_updates=0)
    args.device = base.core.baseline.control.runtime(parent["environment"])
    data = base.frozen.original_features(parent)
    human, dev = data["train"], data["dev"]
    if len(human) != TRAIN_DOCS or len(dev) != DEV_DOCS:
        raise ValueError("Human/Dev scope changed")
    gold = read(args.data_dir / "dev_revised.json")
    annotated = {tuple(x) for x in human.metadata["matching_dev_facts"]}
    distant = {tuple(x) for x in read(args.baseline_workspace / "cache/distant/index.json")["matching_dev_facts"]}
    model = base.core.load_generic_model_seed75(args, parent["baseline_identity"])
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True), strict=True)
    model.requires_grad_(False)
    model.eval()
    before = base.core.preflight.state_digest(model)
    loader = DataLoader(dev, batch_size=4, shuffle=False, collate_fn=base.core.shared.collate_fn)
    predictions, pair_order = evaluate_predictions(model, loader, args, budget)
    score = base.core.shared.score_predictions(predictions, gold, annotated, distant)
    write_json(workspace / "predictions.json", predictions)
    write_json(workspace / "dev_pair_order.json", pair_order)
    write_json(workspace / "metrics.json", score)
    emit("ATGL_SOUP_DEV_EVALUATED", **score, optimizer_updates=0)
    # Strict save/load replay then the existing full-Dev decision acceptance.
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True), strict=True)
    strict = base.core.aligned.grouping_invariance(model, dev, args.device)
    full, replay_predictions = base.accepted.full_dev_grouping(model, dev, args.device, budget)
    replay_score = base.core.shared.score_predictions(replay_predictions, gold, annotated, distant)
    ttm_ok = all(strict["checks"][name]["close"] for name in ("ttm_grouped_vs_single", "ttm_permutation"))
    valid = (full["passed"] and ttm_ok and strict["changed_relation_decisions"] == 0
             and facts(predictions) == facts(replay_predictions)
             and all(math.isclose(score[key], replay_score[key], abs_tol=1e-6, rel_tol=0) for key in METRICS))
    replay = {"passed": bool(valid), "original_strict_grouping": strict, "full_dev_grouping": full,
              "metrics": replay_score, "optimizer_updates": 0, "test_data_read": False}
    write_json(workspace / "checkpoint_replay.json", replay)
    if not valid or before != base.core.preflight.state_digest(model) or not torch.equal(memory, model.mu_encoder.memory_tokens.detach().cpu()):
        raise RuntimeError("Soup replay/frozen-state acceptance failed; preserve reports")
    buckets = base.core.method.frequency_buckets(human.metadata["priors"])
    rel2id = read(ROOT / "meta/rel2id.json")
    truth = {(d["title"], label["h"], label["t"], label["r"]) for d in gold for label in d["labels"]}
    predicted = facts(predictions)
    frequency = {}
    for name in ("rare", "middle", "frequent"):
        g = {x for x in truth if buckets[rel2id[x[3]]] == name}
        p = {x for x in predicted if buckets[rel2id[x[3]]] == name}
        tp, fp, fn = len(g & p), len(p - g), len(g - p)
        frequency[name] = {"TP": tp, "FP": fp, "FN": fn, "F1": 200 * tp / (2 * tp + fp + fn) if tp + fp + fn else 0}
    write_json(workspace / "frequency_diagnostics.json", frequency)
    vote_counts = Counter()
    references = []
    for item in identity["sources"]:
        old_predictions = read(Path(identity["reference_predictions"][str(item["seed"])]))
        old_score = base.core.shared.score_predictions(old_predictions, gold, annotated, distant)
        if any(not math.isclose(old_score[key], item["best"][key], abs_tol=1e-6, rel_tol=0) for key in METRICS):
            raise ValueError("Old prediction metrics changed")
        vote_counts.update(facts(old_predictions))
        references.append({"seed": item["seed"], "metrics": old_score,
                           "soup_minus_parent_pp": {key: score[key] - old_score[key] for key in METRICS}})
    votes = [dict(title=x[0], h_idx=x[1], t_idx=x[2], r=x[3]) for x, count in sorted(vote_counts.items()) if count >= 2]
    vote_score = base.core.shared.score_predictions(votes, gold, annotated, distant)
    comparison = workspace / "comparison"
    comparison.mkdir()
    write_json(comparison / "references.json", {"parents": references, "fixed_majority_2of3": vote_score,
                                                "no_parent_model_forwards": True})
    # Hash sources and read-only inputs again; no parent-model replay.
    for path, expected in identity["pinned_files"].items():
        if digest(Path(path), budget) != expected:
            raise ValueError("Read-only source changed during soup evaluation")
    for name, expected in identity["source_files"].items():
        if digest(ROOT / name, budget) != expected:
            raise ValueError("Source code changed during evaluation")
    check_budget(budget)
    completion = {"passed": True, "marker": "ATGL_UNIFORM_SOUP_COMPLETE", "protocol": protocol(),
                  "metrics": score, "candidate_screen_passed": bool(candidate_gate(score)),
                  "checkpoint_sha256": digest(checkpoint, budget), "sources": identity["sources"],
                  "frozen_memory_unchanged": True, "parameters_unchanged_in_evaluation": True,
                  "checkpoint_replay_passed": True, "optimizer_updates": 0,
                  "parent_model_forwards": 0, "test_data_read": False, "test_evaluated": False,
                  "elapsed_seconds": time.monotonic() - budget["start_monotonic"],
                  "automatic_followup": False, "reliable_improvement_established": False,
                  "novelty_established": False}
    write_json(comparison / "completion.json", completion)
    archive_reports(workspace, budget)
    emit("ATGL_UNIFORM_SOUP_COMPLETE", F1=score["dev_F1"], candidate_screen_passed=completion["candidate_screen_passed"],
         reports=str(comparison / "reports.tar.gz"), optimizer_updates=0, automatic_followup=False)


def archive_reports(workspace, budget):
    comparison = workspace / "comparison"
    comparison.mkdir(exist_ok=True)
    with tarfile.open(comparison / "reports.tar.gz", "x:gz") as target:
        for path in workspace.rglob("*"):
            check_budget(budget)
            if path.is_file() and path.suffix in (".json", ".jsonl", ".log"):
                target.add(path, arcname=path.relative_to(workspace).as_posix(), recursive=False)


def require_lock():
    if os.name != "posix" or os.environ.get("TTMRE_ATGL_SOUP_LOCK_FD") != "9":
        raise RuntimeError("Use the one-use soup launcher")
    import fcntl
    actual = os.fstat(9)
    expected = (ROOT / "experiments/2026-10-08-baseline-alignment/.training.lock").stat()
    if (actual.st_dev, actual.st_ino) != (expected.st_dev, expected.st_ino):
        raise RuntimeError("Wrong shared training lock")
    fcntl.flock(9, fcntl.LOCK_EX | fcntl.LOCK_NB)
    return 9


def verify_context(workspace, budget):
    claim = read(RUN_ROOT / "atgl-soup-once-claim.json")
    if (workspace != workspace.resolve() or workspace.name != "workspace" or workspace.parent.parent != RUN_ROOT.resolve()
            or budget["hard_seconds"] != HARD_SECONDS
            or budget["deadline_monotonic"] - budget["start_monotonic"] != HARD_SECONDS
            or any(budget[key] is not False for key in ("automatic_retry", "automatic_resume", "extension"))
            or claim["workspace"] != str(workspace) or claim["protocol"] != protocol()
            or claim["budget_sha256"] != digest(workspace / "budget.json")):
        raise ValueError("Invalid immutable soup workspace/claim/budget")
    check_budget(budget)


def worker(workspace):
    require_lock()
    budget = read(workspace / "budget.json")
    verify_context(workspace, budget)
    def stop(*_):
        raise TimeoutError("Soup worker stopped; preserve partials")
    signal.signal(signal.SIGTERM, stop)
    try:
        identity = validate(budget)
        identity["budget_sha256"] = digest(workspace / "budget.json")
        write_json(workspace / "identity.json", identity)
        execute(workspace, identity, budget)
    except BaseException as error:
        write_json(workspace / "failure-worker.json", {"error": str(error), "type": type(error).__name__,
                   "complete": False, "optimizer_updates": 0, "automatic_retry": False})
        if not (workspace / "comparison/reports.tar.gz").exists():
            try:
                archive_reports(workspace, budget)
            except BaseException:
                pass
        raise


def start(workspace):
    lock = require_lock()
    if workspace.name != "workspace" or workspace.parent.parent != RUN_ROOT.resolve() or workspace.exists():
        raise ValueError("Use a new dated soup workspace")
    claim_path = RUN_ROOT / "atgl-soup-once-claim.json"
    if claim_path.exists():
        raise FileExistsError("Soup run already claimed; no retry")
    start_time = time.monotonic()
    budget = {"start_monotonic": start_time, "deadline_monotonic": start_time + HARD_SECONDS,
              "hard_seconds": HARD_SECONDS, "soft_reserve_seconds": SOFT_SECONDS,
              "term_reserve_seconds": TERM_SECONDS, "start_utc": datetime.now(timezone.utc).isoformat(),
              "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
              "automatic_retry": False, "automatic_resume": False, "extension": False}
    # Validate assets, actual disk and checker before consuming the once claim.
    identity = validate(budget)
    resource = resources(workspace.parent, identity)
    emit("ATGL_SOUP_BUDGET_STARTED", **budget, resources=resource)
    base.supervise([sys.executable, "-B", str(ROOT / FILES[1])], budget, lock)
    workspace.mkdir()
    write_json(workspace / "budget.json", budget)
    with claim_path.open("x", encoding="utf-8") as handle:
        json.dump({"workspace": str(workspace), "protocol": protocol(),
                   "budget_sha256": digest(workspace / "budget.json"),
                   "authorization": "user-confirmed one uniform ATGL soup, no training, cap1h"}, handle, indent=2)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        base.supervise([sys.executable, "-u", "-B", str(Path(__file__).resolve()),
                        "--mode", "worker", "--run-dir", str(workspace)], budget, lock)
    except BaseException as error:
        write_json(workspace / "failure-supervisor.json", {"error": str(error), "type": type(error).__name__,
                   "complete": False, "elapsed_seconds": time.monotonic() - start_time,
                   "optimizer_updates": 0, "automatic_retry": False})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("validate", "start", "worker"), default="validate")
    parser.add_argument("--run-dir", type=Path)
    args = parser.parse_args()
    if args.mode == "validate":
        print(json.dumps(validate(), indent=2))
    elif args.run_dir is None:
        raise ValueError("Use the single launcher")
    elif args.mode == "start":
        start(args.run_dir.resolve())
    else:
        worker(args.run_dir.resolve())


if __name__ == "__main__":
    main()
