"""Paired seed76/77 Human-stage replication: single-view ATGL vs two-view no-KL."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime, timezone
import json
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
import train_ttmre_atgl_human as atgl
import train_ttmre_atgl_nokl_human as nokl
import train_ttmre_atgl_rdrop_human as rdrop

RUN_ROOT = ROOT / "experiments/2026-10-20-atgl-view-repeat"
PARENT_SHA = "4fd5aca22c20584dde442c2d80dba8a72db8fe20a4b9fa920fa7c43a7fe13ce9"
SEEDS = (76, 77)
REFERENCE_SEED = 75
REFERENCE_ORDER_SHA = "5dac02babee79230a2676bee28ae8c2ec9d21ddf7c6bdfa0ca667c1f0045798d"
TRAIN_DOCS, DEV_DOCS, EPOCHS = 3053, 500, 30
UPDATES_PER_ARM = (TRAIN_DOCS // 4) * EPOCHS
HARD_SECONDS, SOFT_SECONDS, TERM_SECONDS = 14 * 3600, 120, 30
GIB = 1024 ** 3
METRIC_KEYS = ("dev_F1", "dev_precision", "dev_recall", "dev_IgnF1_annotated", "dev_IgnF1_distant")
FILES = tuple("scripts/" + name for name in (
    "train_ttmre_atgl_view_repeat.py",
    "check_ttmre_atgl_view_repeat.py",
    "run_ttmre_atgl_view_repeat.sh",
))
PROGRESS = {"phase": "not-started", "seed": None, "method": None, "formal_optimizer_updates": 0}


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def digest(path, budget=None):
    return atgl.digest(path, budget)


def write_json(path, value):
    return atgl.write_json(path, value)


def emit(event, **fields):
    print(json.dumps({"event": event, **fields}, ensure_ascii=False, allow_nan=False), flush=True)


def protocol():
    return {
        "version": "atgl-view-repeat-v1",
        "seeds": list(SEEDS),
        "methods": ["single-view-ATGL", "two-view-ATGL-noKL"],
        "reference_seed": REFERENCE_SEED,
        "parent_sha256": PARENT_SHA,
        "human_epochs": EPOCHS,
        "optimizer_updates_per_arm": UPDATES_PER_ARM,
        "train_batch_size": 4,
        "precision": "float32",
        "alpha": 16.5,
        "freeze": ["mu_encoder.memory_tokens"],
        "decoder": "all relation logits > TH; no top-k cap",
        "primary_comparison": "same-Human-seed two-view-noKL minus single-view-ATGL",
        "reference_seed75_is_screened_not_independent_confirmation": True,
        "test_data_read": False,
        "distant_optimizer_steps": 0,
        "hard_seconds": HARD_SECONDS,
        "soft_reserve_seconds": SOFT_SECONDS,
        "term_reserve_seconds": TERM_SECONDS,
        "automatic_retry": False,
        "automatic_resume": False,
        "extension": False,
        "automatic_followup": False,
        "candidate_screen": {
            "three_seed_mean_F1_delta_pp_min": 0.20,
            "positive_seed_count_min": 2,
            "mean_Ign_deltas_nonnegative": True,
            "status": "predeclared_candidate_investment_screen; not a significance test",
        },
    }


def validate_seed75_reference(budget=None):
    """Read and validate the already completed seed75 ATGL/no-KL pair."""
    claim_path = nokl.RUN_ROOT / "atgl-nokl-once-claim.json"
    if not claim_path.is_file():
        raise FileNotFoundError("Completed seed75 no-KL once-claim is missing")
    claim = read(claim_path)
    workspace = Path(claim["workspace"]).resolve()
    if workspace.name != "workspace" or workspace.parent.parent != nokl.RUN_ROOT.resolve():
        raise ValueError("Seed75 no-KL workspace is outside its fixed experiment root")
    old_budget = read(workspace / "budget.json")
    if (claim.get("protocol") != nokl.protocol()
            or claim.get("budget_sha256") != digest(workspace / "budget.json", budget)
            or old_budget.get("hard_seconds") != nokl.HARD_SECONDS
            or old_budget.get("automatic_retry") is not False
            or old_budget.get("automatic_resume") is not False
            or old_budget.get("extension") is not False):
        raise ValueError("Seed75 no-KL claim or budget does not match the completed run")
    identity = read(workspace / "identity.json")
    nokl.verify_sources(identity, budget)
    complete = read(workspace / "comparison/completion.json")
    stage_dir = workspace / "atgl-nokl/human-frozen/artifacts"
    stage = read(stage_dir / "completion.json")
    order = read(stage_dir / "training_order.json")
    replay = read(stage_dir / "checkpoint_replay.json")
    if (complete.get("passed") is not True
            or complete.get("marker") != "ATGL_NOKL_HUMAN_COMPLETE"
            or complete.get("protocol") != nokl.protocol()
            or complete.get("optimizer_steps") != UPDATES_PER_ARM
            or complete.get("diagnostic_optimizer_steps") != 0
            or complete.get("reference_optimizer_steps") != 0
            or complete.get("reference_model_forwards") != 0
            or complete.get("test_data_read") is not False
            or complete.get("test_evaluated") is not False
            or complete["protocol"].get("parent_sha256") != PARENT_SHA
            or complete["protocol"].get("seed") != REFERENCE_SEED
            or stage.get("optimizer_steps") != UPDATES_PER_ARM
            or stage.get("memory_unchanged") is not True
            or stage.get("checkpoint_replay_passed") is not True
            or stage.get("test_data_read") is not False
            or order.get("seed") != REFERENCE_SEED
            or order.get("formal_optimizer_updates") != UPDATES_PER_ARM
            or order.get("same_as_original_control_and_ATGL") is not True
            or order.get("sample_order_sha256") != atgl.CONTROL_ORDER
            or replay.get("passed") is not True
            or replay.get("full_dev_grouping", {}).get("changed_relation_decisions") != 0):
        raise ValueError("Seed75 completed result failed a required pairing/completion check")
    old_refs = complete.get("completed_references", {})
    old_rdrop = nokl.reference(budget)
    if (old_refs.get("pinned_files") != old_rdrop.get("pinned_files")
            or old_refs.get("rdrop_checkpoint_sha256") != old_rdrop.get("rdrop_checkpoint_sha256")):
        raise ValueError("Seed75 no-KL report does not match its pinned R-Drop reference")
    atgl_metrics = old_refs.get("atgl")
    nokl_metrics = stage.get("best")
    if not isinstance(atgl_metrics, dict) or not isinstance(nokl_metrics, dict):
        raise ValueError("Seed75 paired metric references are missing")
    deltas = {key: nokl_metrics[key] - atgl_metrics[key] for key in METRIC_KEYS}
    if any(not isinstance(x, (int, float)) for x in deltas.values()):
        raise ValueError("Seed75 paired metrics are invalid")
    return {
        "seed": REFERENCE_SEED,
        "atgl": {key: atgl_metrics[key] for key in METRIC_KEYS},
        "noKL": {key: nokl_metrics[key] for key in METRIC_KEYS},
        "noKL_minus_ATGL_percentage_points": deltas,
        "atgl_checkpoint_sha256": old_refs.get("atgl_checkpoint_sha256", old_refs.get("reference_atgl_checkpoint_sha256")),
        "noKL_checkpoint_sha256": stage.get("checkpoint_sha256"),
        "sample_order_sha256": order["sample_order_sha256"],
        "source_workspace": str(workspace),
        "source_completion_sha256": digest(workspace / "comparison/completion.json", budget),
        "rdrop_reference": old_rdrop,
        "screened_before_seed76_77": True,
    }


def validate(budget=None):
    """Read-only validation. Does not load a task checkpoint or run a forward."""
    baseline_identity = atgl.validate(budget)
    nokl_identity = nokl.validate(budget)
    seed75 = validate_seed75_reference(budget)
    if baseline_identity.get("distant_checkpoint_sha256") != PARENT_SHA:
        raise ValueError("Fixed Distant parent differs")
    if nokl_identity["protocol"].get("parent_sha256") != PARENT_SHA:
        raise ValueError("No-KL reference parent differs")
    sources = {name: digest(ROOT / name, budget) for name in FILES if (ROOT / name).is_file()}
    if len(sources) != len(FILES):
        raise FileNotFoundError("Upload all three paired-repeat scripts before validation")
    return {
        "passed": True,
        "protocol": protocol(),
        "distant_parent_sha256": PARENT_SHA,
        "seed75_reference": seed75,
        "rdrop_reference": seed75["rdrop_reference"],
        "environment": baseline_identity.get("environment"),
        "baseline_workspace": baseline_identity.get("baseline_workspace"),
        "model_dir": baseline_identity.get("model_dir"),
        "data_dir": baseline_identity.get("data_dir"),
        "source_sha256": sources,
        "task_checkpoint_loaded": False,
        "optimizer_steps": 0,
        "test_data_read": False,
        "automatic_followup": False,
    }


def paired_metrics(atgl_best, nokl_best):
    if any(key not in system or not isinstance(system[key], (int, float))
           for system in (atgl_best, nokl_best) for key in METRIC_KEYS):
        raise ValueError("Missing or invalid paired metrics")
    return {key: float(nokl_best[key]) - float(atgl_best[key]) for key in METRIC_KEYS}


def candidate_gate(rows):
    if len(rows) != 3 or {row["seed"] for row in rows} != {75, 76, 77}:
        raise ValueError("The candidate gate requires the seed75 reference and both new paired seeds")
    deltas = [row["noKL_minus_ATGL_percentage_points"] for row in rows]
    mean = {key: sum(float(row[key]) for row in deltas) / len(deltas) for key in METRIC_KEYS}
    positive = sum(float(row["dev_F1"]) > 0 for row in deltas)
    passed = (mean["dev_F1"] >= .20 and positive >= 2
              and mean["dev_IgnF1_annotated"] >= 0
              and mean["dev_IgnF1_distant"] >= 0)
    return bool(passed), {"mean_percentage_points": mean, "positive_F1_seed_count": positive,
                          "criteria": protocol()["candidate_screen"]}


def make_reference_pool(workspace, seed, atgl_workspace, atgl_result, seed75_pilot, budget):
    pool = workspace / "reference-pool" / ("seed" + str(seed))
    pool.mkdir(parents=True, exist_ok=False)
    current_atgl = pool / "reference-atgl"
    current_rdrop = pool / "atgl-rdrop/human-frozen/artifacts"
    current_atgl.mkdir(parents=True)
    current_rdrop.mkdir(parents=True)
    source_atgl = atgl_workspace / "atgl/human-frozen/artifacts"
    source_rdrop = Path(seed75_pilot["workspace"]) / "atgl-rdrop/human-frozen/artifacts"
    pinned = {}
    for name in ("predictions.json", "dev_pair_order.json", "frequency_diagnostics.json"):
        src_a = source_atgl / name
        src_r = source_rdrop / name
        # The old R-Drop source was verified by nokl.reference() before training.
        relative_r = "atgl-rdrop/human-frozen/artifacts/" + name
        expected_r = seed75_pilot["pinned_files"][relative_r]
        if digest(src_r, budget) != expected_r:
            raise ValueError("Seed75 R-Drop reference changed: " + name)
        shutil.copyfile(src_a, current_atgl / name)
        shutil.copyfile(src_r, current_rdrop / name)
        rel_a = "reference-atgl/" + name
        rel_r = "atgl-rdrop/human-frozen/artifacts/" + name
        pinned[rel_a] = digest(current_atgl / name, budget)
        pinned[rel_r] = digest(current_rdrop / name, budget)
    atgl_diag = read(current_atgl / "frequency_diagnostics.json")
    rdrop_diag = read(current_rdrop / "frequency_diagnostics.json")
    if (any(atgl_diag["metrics"][key] != atgl_result["best"][key] for key in METRIC_KEYS)
            or rdrop_diag["pair_order_sha256"] != atgl_diag["pair_order_sha256"]):
        raise ValueError("Paired Dev prediction provenance or order differs")
    manifest_path = pool / "reference_manifest.json"
    write_json(manifest_path, {
        "seed": seed,
        "primary_ATGL_reference": "same-seed trained arm",
        "primary_ATGL_checkpoint_sha256": atgl_result["checkpoint_sha256"],
        "primary_ATGL_metrics": atgl_result["best"],
        "archival_RDrop_reference_seed": REFERENCE_SEED,
        "archival_RDrop_metrics": seed75_pilot["rdrop"],
        "archival_RDrop_used_for_paired_gate": False,
        "pinned_files": pinned,
        "test_data_read": False,
    })
    pinned["reference_manifest.json"] = digest(manifest_path, budget)
    return {
        "workspace": str(pool),
        "completion_sha256": digest(manifest_path, budget),
        "pinned_files": pinned,
        "atgl": atgl_result["best"],
        "atgl_checkpoint_sha256": atgl_result["checkpoint_sha256"],
        "rdrop": seed75_pilot["rdrop"],
        "rdrop_checkpoint_sha256": seed75_pilot["rdrop_checkpoint_sha256"],
        "rdrop_reference_seed": REFERENCE_SEED,
        "rdrop_is_archival_only": True,
    }


@contextmanager
def variant_scope(method, seed, order_sha, references=None, group_start=None):
    """Apply seed/order parameters without editing the already accepted scripts."""
    core = atgl.core
    saved = {
        "seed_all": core.shared.seed_all,
        "core_train_loop": core.train_loop,
        "aligned_emit": core.aligned.emit,
        "atgl_protocol": atgl.protocol,
        "atgl_order": atgl.CONTROL_ORDER,
        "atgl_hard": atgl.HARD_SECONDS,
        "atgl_write": atgl.write_json,
        "atgl_emit": atgl.emit,
        "nokl_protocol": nokl.protocol,
        "nokl_order": nokl.CONTROL_ORDER,
        "nokl_hard": nokl.HARD_SECONDS,
        "nokl_write": nokl.write_json,
        "nokl_reference": nokl.reference,
        "nokl_private_rng": nokl.private_rng,
        "nokl_gate": nokl.screening_gate,
        "nokl_emit": nokl.emit,
        "frequency_report": core.n1.frequency_report,
    }
    actual_atgl_protocol = saved["atgl_protocol"]
    actual_nokl_protocol = saved["nokl_protocol"]
    actual_seed_all = saved["seed_all"]
    actual_train_loop = saved["core_train_loop"]
    actual_aligned_emit = saved["aligned_emit"]
    actual_frequency_report = saved["frequency_report"]
    completed_before_branch = 2 * UPDATES_PER_ARM * SEEDS.index(seed)
    branch_start = time.monotonic()

    def seeded(seed_arg, *args, **kwargs):
        return actual_seed_all(seed if int(seed_arg) == REFERENCE_SEED else seed_arg, *args, **kwargs)

    def atgl_run_protocol():
        value = actual_atgl_protocol()
        value.update({"seed": seed, "hard_seconds": HARD_SECONDS,
                      "study_id": "atgl-view-repeat-20261020"})
        return value

    def nokl_run_protocol():
        value = actual_nokl_protocol()
        value.update({
            "version": "atgl-nokl-human-paired-v1",
            "seed": seed,
            "hard_seconds": HARD_SECONDS,
            "study_id": "atgl-view-repeat-20261020",
            "training_views": "independent backbone dropout; second view private seed=seed*10000000+step",
            "private_view_seed_rule": "Human seed*10000000 + 1-based optimizer step",
            "reference_atgl_checkpoint_sha256": references["atgl_checkpoint_sha256"],
            "reference_atgl_F1": references["atgl"]["dev_F1"],
            "reference_rdrop_F1": references["rdrop"]["dev_F1"],
            "reference_rdrop_seed": REFERENCE_SEED,
            "rdrop_reference_is_archival_only": True,
            "screening_delta_pp": None,
            "both_Ign_deltas_positive_required": None,
            "screen_direction": "paired seedwise noKL minus single-view ATGL; R-Drop is not in this study",
        })
        return value

    def adjust_atgl_write(path, value):
        target = Path(path)
        if target.name == "training_order.json":
            value = dict(value)
            order_assertion = value.pop("same_as_original_control", False)
            value.update({"seed": seed, "control_order_digest_assertion_passed": bool(order_assertion),
                          "same_as_seed75_control_order": value.get("sample_order_sha256") == atgl.CONTROL_ORDER
                          if seed == REFERENCE_SEED else False})
        elif target.name == "completion.json" and target.parent.name == "comparison":
            value = dict(value)
            value.update({"paired_seed": seed,
                          "comparison_scope": "historical Control75 comparison is descriptive; paired study compares same-seed ATGL vs no-KL"})
        return saved["atgl_write"](path, value)

    def adjust_nokl_write(path, value):
        target = Path(path)
        if target.name == "training_order.json":
            value = dict(value)
            same_order = value.pop("same_as_original_control_and_ATGL", False)
            value.update({"seed": seed, "same_as_paired_ATGL": bool(same_order),
                          "same_as_seed75_control_order": value.get("sample_order_sha256") == atgl.CONTROL_ORDER
                          if seed == REFERENCE_SEED else False})
        elif target.name == "completion.json" and target.parent.name == "comparison" \
                and value.get("marker") == "ATGL_NOKL_HUMAN_COMPLETE":
            value = dict(value)
            value["rdrop_minus_nokl_percentage_points"] = {key: None for key in METRIC_KEYS}
            value["consistency_contribution_screen_passed"] = None
            value["consistency_contribution_screen_evaluated"] = False
            value["paired_seed"] = seed
            value["comparison_scope"] = "same-seed noKL minus single-view ATGL; seed75 R-Drop is archival only"
        return saved["nokl_write"](path, value)

    def capture_atgl_order(*args, **kwargs):
        result = actual_train_loop(*args, **kwargs)
        order_sha = result.get("sample_order_sha256")
        if (not isinstance(order_sha, str) or len(order_sha) != 64
                or any(char not in "0123456789abcdef" for char in order_sha)):
            raise ValueError("ATGL training did not produce a valid sample-order digest")
        # The actual digest is generated by the formal training loop after its
        # forwards have consumed the real runtime RNG. The no-KL arm will use it.
        atgl.CONTROL_ORDER = order_sha
        return result

    def export_selected_atgl_predictions(model, dev_loader, args, gold, annotated,
                                         distant, buckets, best):
        # atgl.execute calls frequency_report only after replaying its selected
        # checkpoint. Export there so the paired no-KL arm uses this seed's
        # actual selected-ATGL predictions without another checkpoint load.
        original = actual_frequency_report(model, dev_loader, args, gold, annotated,
                                           distant, buckets, best)
        artifact_dir = args.run_dir / "atgl/human-frozen/artifacts"
        exported, _predictions, _truth = nokl.export_predictions(
            model, dev_loader, args, gold, annotated, distant, buckets, best,
            artifact_dir, args.budget)
        if exported.get("passed") is not True:
            raise ValueError("Selected ATGL Dev predictions failed metric replay")
        return {**original, **exported}

    def dynamic_private_rng(*, seed=None, state=None):
        if seed is not None:
            step = int(seed) - (REFERENCE_SEED * 10_000_000)
            if step < 1 or step > UPDATES_PER_ARM:
                raise ValueError("Unexpected no-KL private-view seed")
            seed = (globals()["_ACTIVE_PAIR_SEED"] * 10_000_000) + step
        return saved["nokl_private_rng"](seed=seed, state=state)

    def disabled_unpaired_gate(_best, _reference_best):
        return False, {key: None for key in METRIC_KEYS}

    def paired_emit(event, **fields):
        branch_offset = completed_before_branch + (UPDATES_PER_ARM if method == "nokl" else 0)
        if event in ("ATGL_NOKL_LEARNING_UPDATE_VERIFIED", "epoch_complete") and "step" in fields:
            PROGRESS["formal_optimizer_updates"] = branch_offset + int(fields["step"])
        if event == "ATGL_NOKL_HUMAN_COMPLETE":
            fields["rdrop_minus_nokl_F1_pp"] = None
            fields["consistency_contribution_screen_passed"] = None
            fields["screening_scope"] = "not evaluated across different Human seeds"
        return saved["nokl_emit"](event, **fields)

    def paired_atgl_emit(event, **fields):
        if event in ("ATGL_LEARNING_UPDATE_VERIFIED", "epoch_complete") and "step" in fields:
            PROGRESS["formal_optimizer_updates"] = completed_before_branch + int(fields["step"])
        return saved["atgl_emit"](event, **fields)

    def progress_aligned_emit(event, **fields):
        if method == "atgl" and event == "epoch_complete" and "step" in fields:
            PROGRESS["formal_optimizer_updates"] = completed_before_branch + int(fields["step"])
        return actual_aligned_emit(event, **fields)

    try:
        globals()["_ACTIVE_PAIR_SEED"] = seed
        core.shared.seed_all = seeded
        core.aligned.emit = progress_aligned_emit
        if method == "atgl":
            atgl.protocol = atgl_run_protocol
            atgl.CONTROL_ORDER = order_sha
            atgl.HARD_SECONDS = HARD_SECONDS
            atgl.write_json = adjust_atgl_write
            atgl.emit = paired_atgl_emit
            core.train_loop = capture_atgl_order
            core.n1.frequency_report = export_selected_atgl_predictions
        elif method == "nokl":
            # noKL.protocol() calls the original seed75 ATGL protocol. Do not
            # patch atgl.protocol here: R-Drop validation relies on seed75.
            nokl.protocol = nokl_run_protocol
            nokl.CONTROL_ORDER = order_sha
            nokl.HARD_SECONDS = HARD_SECONDS
            nokl.write_json = adjust_nokl_write
            nokl.reference = lambda budget=None: references
            nokl.private_rng = dynamic_private_rng
            nokl.screening_gate = disabled_unpaired_gate
            nokl.emit = paired_emit
        else:
            raise ValueError("Unknown paired method")
        yield {"seed": seed, "method": method, "expected_order_sha256": order_sha,
               "branch_start_monotonic": branch_start, "group_start_monotonic": group_start}
    finally:
        core.shared.seed_all = saved["seed_all"]
        core.train_loop = saved["core_train_loop"]
        core.aligned.emit = saved["aligned_emit"]
        core.n1.frequency_report = saved["frequency_report"]
        atgl.protocol = saved["atgl_protocol"]
        atgl.CONTROL_ORDER = saved["atgl_order"]
        atgl.HARD_SECONDS = saved["atgl_hard"]
        atgl.write_json = saved["atgl_write"]
        atgl.emit = saved["atgl_emit"]
        nokl.protocol = saved["nokl_protocol"]
        nokl.CONTROL_ORDER = saved["nokl_order"]
        nokl.HARD_SECONDS = saved["nokl_hard"]
        nokl.write_json = saved["nokl_write"]
        nokl.reference = saved["nokl_reference"]
        nokl.private_rng = saved["nokl_private_rng"]
        nokl.screening_gate = saved["nokl_gate"]
        nokl.emit = saved["nokl_emit"]
        globals().pop("_ACTIVE_PAIR_SEED", None)


def branch_identity(template, method, seed, order_sha, references=None, budget=None):
    result = deepcopy(template)
    result["protocol"] = atgl.protocol() if method == "atgl" else nokl.protocol()
    result["pair_study"] = {"study_protocol": protocol(), "seed": seed,
                             "method": method, "expected_order_sha256": order_sha,
                             "order_source": "captured from same-seed ATGL arm" if method == "nokl"
                             else "captured from this ATGL arm during training"}
    result.setdefault("source_files", {}).update({name: digest(ROOT / name, budget) for name in FILES})
    if method == "nokl":
        result["completed_references"] = references
        for name, sha in references["pinned_files"].items():
            result.setdefault("input_files", {})[str(Path(references["workspace"]) / name)] = sha
    return result


def arm_complete(workspace, method, seed, expected_order):
    if method == "atgl":
        root = workspace / "atgl/human-frozen/artifacts"
        comparison = read(workspace / "comparison/completion.json")
        best = comparison["atgl"]
        marker = "ATGL_HUMAN_COMPLETE"
    else:
        root = workspace / "atgl-nokl/human-frozen/artifacts"
        comparison = read(workspace / "comparison/completion.json")
        best = comparison["atgl_nokl"]
        marker = "ATGL_NOKL_HUMAN_COMPLETE"
    stage = read(root / "completion.json")
    order = read(root / "training_order.json")
    replay = read(root / "checkpoint_replay.json")
    order_flag = (order.get("control_order_digest_assertion_passed") is True if method == "atgl"
                  else order.get("same_as_paired_ATGL") is True)
    run_protocol = comparison.get("protocol", {})
    checks = {
        "comparison_passed": comparison.get("passed") is True,
        "comparison_marker": comparison.get("marker") == marker,
        "comparison_seed": run_protocol.get("seed") == seed,
        "protocol_parent": run_protocol.get("parent_sha256") == PARENT_SHA,
        "protocol_epochs": run_protocol.get("epochs") == EPOCHS,
        "protocol_updates": run_protocol.get("optimizer_calls") == UPDATES_PER_ARM,
        "protocol_batch": run_protocol.get("train_batch_size") == 4,
        "protocol_precision": run_protocol.get("precision") == "float32",
        "protocol_alpha": run_protocol.get("alpha") == 16.5,
        "protocol_freeze": run_protocol.get("freeze") == ["mu_encoder.memory_tokens"],
        "comparison_updates": comparison.get("optimizer_steps") == UPDATES_PER_ARM,
        "comparison_no_test_read": comparison.get("test_data_read") is False,
        "comparison_no_test_eval": comparison.get("test_evaluated") is False,
        "stage_updates": stage.get("optimizer_steps") == UPDATES_PER_ARM,
        "memory_unchanged": stage.get("memory_unchanged") is True,
        "effective_learning": stage.get("learning_update_check_passed") is True,
        "checkpoint_replay": stage.get("checkpoint_replay_passed") is True,
        "frequency_replay": stage.get("frequency_diagnostics_passed") is True,
        "stage_no_test_read": stage.get("test_data_read") is False,
        "order_seed": order.get("seed") == seed,
        "order_hash": order.get("sample_order_sha256") == expected_order,
        "order_assertion": order_flag,
        "full_dev_replay": replay.get("passed") is True,
        "full_dev_decisions": replay.get("full_dev_grouping", {}).get("changed_relation_decisions") == 0,
    }
    if not all(checks.values()):
        failed = [name for name, passed in checks.items() if not passed]
        raise ValueError("Paired arm failed checks: " + ", ".join(failed))
    return {"best": best, "checkpoint_sha256": stage.get("checkpoint_sha256"),
            "sample_order_sha256": order["sample_order_sha256"],
            "completion_sha256": digest(root / "completion.json"),
            "replay_decisions_changed": replay["full_dev_grouping"]["changed_relation_decisions"]}


def run_study(workspace, budget, base_atgl_identity, base_nokl_identity, reference_info):
    PROGRESS.update({"phase": "load-human-dev-features", "seed": None, "method": None,
                     "formal_optimizer_updates": 0})
    args = SimpleNamespace(run_dir=workspace,
        baseline_workspace=Path(base_atgl_identity["baseline_workspace"]),
        model_dir=Path(base_atgl_identity["model_dir"]),
        data_dir=Path(base_atgl_identity["data_dir"]))
    datasets = atgl.frozen.original_features(base_atgl_identity)
    human, dev = datasets["train"], datasets["dev"]
    if len(human) != TRAIN_DOCS or len(dev) != DEV_DOCS:
        raise ValueError("Human/Dev counts differ from the approved protocol")
    pair_order = {"passed": False, "seed75_reference_order_sha256": REFERENCE_ORDER_SHA,
                  "seed75_reference_matches_original_control": True,
                  "seed_order_sha256": {}, "same_seed_pair_order_required": True,
                  "test_data_read": False}
    write_json(workspace / "pair_order_validation.json", pair_order)
    seed75 = reference_info["seed75_reference"]
    rows = [seed75]
    study_start = budget["start_monotonic"]
    for seed in SEEDS:
        check_budget = atgl.check_budget
        check_budget(budget)
        PROGRESS.update({"phase": "ATGL", "seed": seed, "method": "single-view-ATGL",
                         "formal_optimizer_updates": 2 * UPDATES_PER_ARM * SEEDS.index(seed)})
        atgl_workspace = workspace / ("seed" + str(seed)) / "atgl" / "workspace"
        atgl_workspace.mkdir(parents=True, exist_ok=False)
        with variant_scope("atgl", seed, None, group_start=study_start):
            atgl_identity = branch_identity(base_atgl_identity, "atgl", seed, None, budget=budget)
            write_json(atgl_workspace / "identity.json", atgl_identity)
            atgl.execute(atgl_workspace, atgl_identity, budget)
        atgl_order = read(atgl_workspace / "atgl/human-frozen/artifacts/training_order.json")
        expected_order = atgl_order.get("sample_order_sha256")
        if (atgl_order.get("seed") != seed
                or atgl_order.get("formal_optimizer_updates") != UPDATES_PER_ARM
                or not isinstance(expected_order, str) or len(expected_order) != 64
                or any(char not in "0123456789abcdef" for char in expected_order)):
            raise ValueError("ATGL arm did not persist a valid actual sample order")
        pair_order["seed_order_sha256"][str(seed)] = {"atgl": expected_order, "noKL": None,
                                                       "matched": False}
        write_json(workspace / "pair_order_validation.json", pair_order)
        atgl_result = arm_complete(atgl_workspace, "atgl", seed, expected_order)
        atgl_result["best"] = read(atgl_workspace / "atgl/human-frozen/artifacts/best_metrics.json")
        atgl_result["checkpoint_sha256"] = read(atgl_workspace / "atgl/human-frozen/artifacts/completion.json")["checkpoint_sha256"]

        reference_object = make_reference_pool(workspace / ("seed" + str(seed)), seed,
            atgl_workspace, atgl_result, reference_info["rdrop_reference"], budget)
        PROGRESS.update({"phase": "noKL", "seed": seed, "method": "two-view-ATGL-noKL",
                         "formal_optimizer_updates": (2 * SEEDS.index(seed) + 1) * UPDATES_PER_ARM})
        nokl_workspace = workspace / ("seed" + str(seed)) / "nokl" / "workspace"
        nokl_workspace.mkdir(parents=True, exist_ok=False)
        with variant_scope("nokl", seed, expected_order, references=reference_object,
                           group_start=study_start):
            nokl_identity = branch_identity(base_nokl_identity, "nokl", seed, expected_order,
                                            references=reference_object, budget=budget)
            nokl_identity["study_seed75_reference"] = seed75
            write_json(nokl_workspace / "identity.json", nokl_identity)
            nokl.execute(nokl_workspace, nokl_identity, budget)
        nokl_result = arm_complete(nokl_workspace, "nokl", seed, expected_order)
        nokl_result["best"] = read(nokl_workspace / "atgl-nokl/human-frozen/artifacts/best_metrics.json")
        nokl_result["checkpoint_sha256"] = read(nokl_workspace / "atgl-nokl/human-frozen/artifacts/completion.json")["checkpoint_sha256"]
        if atgl_result["sample_order_sha256"] != nokl_result["sample_order_sha256"]:
            raise ValueError("ATGL and no-KL pair did not use identical Human batch order")
        pair_order["seed_order_sha256"][str(seed)].update({
            "noKL": nokl_result["sample_order_sha256"], "matched": True})
        write_json(workspace / "pair_order_validation.json", pair_order)
        rows.append({"seed": seed,
            "atgl": {key: atgl_result["best"][key] for key in METRIC_KEYS},
            "noKL": {key: nokl_result["best"][key] for key in METRIC_KEYS},
            "noKL_minus_ATGL_percentage_points": paired_metrics(atgl_result["best"], nokl_result["best"]),
            "atgl_checkpoint_sha256": atgl_result["checkpoint_sha256"],
            "noKL_checkpoint_sha256": nokl_result["checkpoint_sha256"],
            "sample_order_sha256": expected_order,
            "atgl_completion_sha256": atgl_result["completion_sha256"],
            "noKL_completion_sha256": nokl_result["completion_sha256"],
            "screened_before_seed76_77": False})
        PROGRESS.update({"phase": "pair-complete", "seed": seed, "method": None,
                         "formal_optimizer_updates": 2 * UPDATES_PER_ARM * (SEEDS.index(seed) + 1)})
        emit("ATGL_VIEW_REPEAT_PAIR_COMPLETE", seed=seed,
             noKL_minus_ATGL_F1_pp=rows[-1]["noKL_minus_ATGL_percentage_points"]["dev_F1"],
             reports_ready=False, automatic_followup=False)
    pair_order["passed"] = all(v["matched"] for v in pair_order["seed_order_sha256"].values())
    if not pair_order["passed"] or set(pair_order["seed_order_sha256"]) != {"76", "77"}:
        raise ValueError("Both same-seed paired sample orders must be verified")
    write_json(workspace / "pair_order_validation.json", pair_order)
    passed, gate = candidate_gate(rows)
    comparison = workspace / "comparison"
    comparison.mkdir(exist_ok=False)
    write_json(comparison / "paired_seed_metrics.json", {
        "rows": rows,
        "mean_noKL_minus_ATGL_percentage_points": gate["mean_percentage_points"],
        "candidate_screen": gate,
        "candidate_screen_passed": passed,
        "seed75_screened_and_not_independent": True,
        "fixed_Distant_parent_sha256": PARENT_SHA,
        "test_data_read": False,
        "automatic_followup": False,
    })
    completion = {
        "passed": True,
        "marker": "ATGL_VIEW_REPEAT_COMPLETE",
        "protocol": protocol(),
        "paired_seed_metrics": rows,
        "mean_noKL_minus_ATGL_percentage_points": gate["mean_percentage_points"],
        "candidate_screen_passed": passed,
        "all_four_new_human_arms_complete": True,
        "formal_optimizer_updates_new_runs": 4 * UPDATES_PER_ARM,
        "formal_optimizer_updates_reference_seed75": 0,
        "distant_optimizer_updates": 0,
        "diagnostic_optimizer_updates": 0,
        "test_data_read": False,
        "test_evaluated": False,
        "reliable_improvement_established": False,
        "independent_full_two_stage_seed_generalization_established": False,
        "novelty_established": False,
        "automatic_followup": False,
        "elapsed_seconds": time.monotonic() - study_start,
        "interpretation": "paired Human-stage replication under one fixed Distant parent; seed75 is a previously screened point",
    }
    write_json(comparison / "completion.json", completion)
    PROGRESS.update({"phase": "complete", "seed": None, "method": None,
                     "formal_optimizer_updates": 4 * UPDATES_PER_ARM})
    return completion


def archive_small_reports(workspace, budget):
    comparison = workspace / "comparison"
    comparison.mkdir(exist_ok=True)
    archive = comparison / "reports.tar.gz"
    with tarfile.open(archive, "x:gz") as target:
        for path in workspace.rglob("*"):
            atgl.check_budget(budget)
            if path.is_file() and path != archive and path.suffix in (".json", ".jsonl", ".log"):
                target.add(path, arcname=path.relative_to(workspace).as_posix(), recursive=False)
    return archive


def require_lock():
    if os.name != "posix" or os.environ.get("TTMRE_ATGL_VIEW_REPEAT_LOCK_FD") != "9":
        raise RuntimeError("Use run_ttmre_atgl_view_repeat.sh")
    import fcntl
    actual = os.fstat(9)
    expected = (ROOT / "experiments/2026-10-08-baseline-alignment/.training.lock").stat()
    if (actual.st_dev, actual.st_ino) != (expected.st_dev, expected.st_ino):
        raise RuntimeError("Wrong shared training lock descriptor")
    fcntl.flock(9, fcntl.LOCK_EX | fcntl.LOCK_NB)
    return 9


def make_budget():
    start = time.monotonic()
    boot_id_path = Path("/proc/sys/kernel/random/boot_id")
    if not boot_id_path.is_file():
        raise RuntimeError("The 14-hour once-only experiment requires Linux boot identity")
    return {
        "start_monotonic": start,
        "deadline_monotonic": start + HARD_SECONDS,
        "hard_seconds": HARD_SECONDS,
        "soft_reserve_seconds": SOFT_SECONDS,
        "term_reserve_seconds": TERM_SECONDS,
        "start_utc": datetime.now(timezone.utc).isoformat(),
        "boot_id": boot_id_path.read_text(encoding="utf-8").strip(),
        "automatic_retry": False,
        "automatic_resume": False,
        "extension": False,
    }


def verify_context(workspace, budget):
    claim = read(RUN_ROOT / "atgl-view-repeat-once-claim.json")
    if (workspace.name != "workspace" or workspace.parent.parent != RUN_ROOT.resolve()
            or budget.get("hard_seconds") != HARD_SECONDS
            or budget.get("deadline_monotonic") - budget.get("start_monotonic") != HARD_SECONDS
            or budget.get("soft_reserve_seconds") != SOFT_SECONDS
            or budget.get("term_reserve_seconds") != TERM_SECONDS
            or any(budget.get(key) is not False for key in ("automatic_retry", "automatic_resume", "extension"))
            or claim.get("workspace") != str(workspace)
            or claim.get("protocol") != protocol()
            or claim.get("budget_sha256") != digest(workspace / "budget.json")):
        raise ValueError("Invalid paired-seed once-claim or immutable 14-hour budget")
    atgl.check_budget(budget)


def worker(workspace):
    require_lock()
    budget = read(workspace / "budget.json")
    verify_context(workspace, budget)

    def stop(*_):
        raise atgl.core.BudgetStop("Paired-seed worker stopped; preserve partial artifacts")

    signal.signal(signal.SIGTERM, stop)
    PROGRESS.update({"phase": "read-only-validation", "seed": None, "method": None,
                     "formal_optimizer_updates": 0})
    try:
        reference_info = validate(budget)
        required_free = max(12 * GIB, nokl.PARENT.stat().st_size * 5 + 2 * GIB)
        available = shutil.disk_usage(workspace).free
        if available < required_free:
            raise RuntimeError("Require at least12GiB free on the actual project volume for four retained best checkpoints")
        write_json(workspace / "identity.json", reference_info)
        write_json(workspace / "resource_check.json", {
            "project_volume_free_bytes": available,
            "required_free_bytes": required_free,
            "checkpoint_size_bytes": nokl.PARENT.stat().st_size,
            "passed": True,
            "available_host_ram_recommended_GiB": 32,
        })
        base_atgl_identity = atgl.validate(budget)
        base_nokl_identity = nokl.validate(budget)
        completion = run_study(workspace, budget, base_atgl_identity, base_nokl_identity, reference_info)
        archive = archive_small_reports(workspace, budget)
        emit("ATGL_VIEW_REPEAT_COMPLETE", candidate_screen_passed=completion["candidate_screen_passed"],
             elapsed_seconds=completion["elapsed_seconds"], reports=str(archive), automatic_followup=False)
    except BaseException as error:
        write_json(workspace / "failure-worker.json", {
            "error": str(error), "type": type(error).__name__, "complete": False,
            "automatic_retry": False, "resumable": False, **PROGRESS,
        })
        try:
            archive_small_reports(workspace, budget)
        except BaseException as archive_error:
            write_json(workspace / "archive_failure.json", {"error": str(archive_error)})
        raise


def train(workspace):
    lock = require_lock()
    if workspace.name != "workspace" or workspace.parent.parent != RUN_ROOT.resolve() or workspace.exists():
        raise ValueError("Use a new launch workspace under the dated paired-repeat root")
    claim_path = RUN_ROOT / "atgl-view-repeat-once-claim.json"
    if claim_path.exists():
        raise FileExistsError("Paired seed76/77 authorization already claimed; no retry")
    budget = make_budget()
    # Fail missing-reference/source checks before consuming the one-use claim.
    validate(budget)
    workspace.mkdir(parents=True, exist_ok=False)
    write_json(workspace / "budget.json", budget)
    with claim_path.open("x", encoding="utf-8") as handle:
        json.dump({"workspace": str(workspace), "budget_sha256": digest(workspace / "budget.json", budget),
                   "protocol": protocol(), "authorization": "user-confirmed seed76/77 paired Human comparison; total cap14h"},
                  handle, indent=2)
        handle.flush()
        os.fsync(handle.fileno())
    write_json(workspace.parent / "pid.json", {"supervisor_pid": os.getpid()})
    try:
        emit("ATGL_VIEW_REPEAT_BUDGET_STARTED", **budget)
        atgl.supervise([sys.executable, "-B", str(ROOT / FILES[1])], budget, lock)
        atgl.supervise([sys.executable, "-u", "-B", str(Path(__file__).resolve()),
                        "--mode", "worker", "--run-dir", str(workspace)], budget, lock)
    except BaseException as error:
        write_json(workspace / "failure.json", {
            "error": str(error), "type": type(error).__name__, "complete": False,
            "automatic_retry": False, "resumable": False,
            "elapsed_seconds": time.monotonic() - budget["start_monotonic"],
        })
        raise


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--mode", choices=("validate", "train", "worker"), default="validate")
    result.add_argument("--run-dir", type=Path)
    return result


def main():
    args = parser().parse_args()
    os.chdir(ROOT)
    if args.mode == "validate":
        print(json.dumps(validate(), ensure_ascii=False, indent=2))
        return
    if args.run_dir is None:
        raise ValueError("Use the one-time launcher workspace")
    workspace = args.run_dir.resolve()
    if args.mode == "train":
        train(workspace)
    else:
        worker(workspace)


if __name__ == "__main__":
    main()
