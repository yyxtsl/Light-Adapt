# Replacement attempt: pending new budget; all old attempts and151 updates retained.
"""G-only DWIE generic initialization control; default validate is read-only.

Training requires a new pinned approval and one launcher. Admission checks
generic assets, versions, disk and host memory before claim/mkdir/CUDA. The
same 5400-second proposed clock covers admission and all worker work; pending
authorization grants none. Existing DWIE training functions are reused without
editing them. No Distant tensors, downloads, deletion, Test or automatic retry.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import json
import math
import os
from pathlib import Path
import shutil
import sys
import tarfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]
import train_ttmre_dwie_adaptation as base
import ttmre_dwie_runtime_fix as runtime_fix

DOC = ROOT / "refine-logs/DWIE_GENERIC_CONTROL_IMPLEMENTATION_20261025"
AUTHORIZATION = DOC / "BUDGET_REPLACEMENT_AUTHORIZATION_APPROVED_20261025.json"
AUTHORIZATION_SHA = "5a85341e702dd4d6cc59cedf73eb6bcbe80d1d96c5ba5a0676e9563a0861aa72"
RUN_ROOT = ROOT / "experiments/2026-10-25-dwie-generic-control"
CLAIM = RUN_ROOT / "generic-budget-replacement-once-claim.json"
REFERENCE = DOC / "reference-reports.tar.gz"
REFERENCE_SHA = "8c34bb99b488ab8f5e17c810dc209cd0bfa513e6faab1a8a754338dd00b9e7cb"
MODEL_DIR = ROOT / "models/hf-cache/models--FacebookAI--roberta-large/snapshots/722cf37b1afa9454edce342e7895e588b6ff1d59"
GENERIC_FILES = {
    "config.json": "82ba49810e6441735e696033ac2512ee09da555507b2917ac3865b202d592cc3",
    "model.safetensors": "047c85f0b96269cd62e6f732644f067004eebd95af5b5d35965ae2528f13bf38",
    "merges.txt": "1ce1664773c50f3e0cc8842619a93edc4624525b728b188a9e0be33b7726adc5",
    "tokenizer.json": "847bbeab6174d66a88898f729d52fa8d355fafe1bea101cf960dd404581df70e",
    "tokenizer_config.json": "994f46754c5bf4014f1aa92d34b1374319c3a6b3f702105cd5b742beaecd18ce",
    "vocab.json": "9e7f63c2d15d666b52e21d250d2e513b87c9b713cfa6987a82ed89e5e6e50655",
}
SOURCE_PINS = {**base.SOURCE_PINS,
    "scripts/train_ttmre_dwie_adaptation.py": "4b5e37f89e8a1085781a01fa3531cd7c0e41df7f4fe58815ab31d704cb5a10fd",
    "scripts/ttmre_dwie_runtime_fix.py": "f83769948090f04c424329e9f3436ba3527c882b6422e4326444a7737165dc20"}
SOURCE_PINS.update({
    "scripts/train_ttmre_dwie_generic_control.py": "15470cc9874f2ecf5ca1808e3565d295966b31e0bd8920dc7d5a98ed4fae8321",
    "scripts/check_ttmre_dwie_generic_claim_isolation.py": "fdabb0599ec3884369301b76b4c167a7860434cf7be5ea2a332d8870af3f52e0",
    "scripts/recover_ttmre_dwie_generic_contracts.py": "649026e1d2961da53638910be49a398d92cd67618aad9440c96f3bc0dc93a0bf",
    "scripts/check_ttmre_dwie_generic_budget_replacement.py": "05e6b10c6fcd1dc722828cb874a6fc1371d60711c273a68a2414acdd84a6762f"})
CHECKER_SHA = "0b3a7cd6e68cb087dc864a391078097cc868fb69528b79ef553dc829fca5740c"
TARGET_HEAD_SHA = "7f7e2a9b30ac39c4bbb4f90b4d0c05487a90cef024ed6238c96cd1c3bc1b8e07"
# Original 97-class noTTM serialization bounds the otherwise identical 63-class
# G state, including the inert provenance buffer. No old weight is opened.
CHECKPOINT_BOUND = 1453845935
PROPOSED_HARD_SECONDS = 5400


def verify_budget_stop_report(failure):
    if (failure.get("passed") is not False or failure.get("error_type") != "BudgetStop"
            or failure.get("error") != "G measured throughput exceeds approved remaining total cap"
            or failure.get("optimizer_steps") != 151 or failure.get("test_data_read") is not False
            or any(failure.get(k) is not False for k in ("automatic_retry", "resume", "extension"))):
        raise ValueError("Replacement is limited to the exact incomplete151-update budget stop")


def verify_budget_stop_predecessor():
    old = RUN_ROOT / "contract-recovery-20261025-121856-kLdNmg/workspace"
    claim = base.read_json(RUN_ROOT / "contract-isolation-recovery-once-claim.json")
    if Path(claim["workspace"]).resolve() != old.resolve():
        raise ValueError("Prior recovery claim names a different failed run")
    expected = {old / "failure.json": "a86f7e1b2cfbd651d6e18640135f6cea8ede6d78de78f79654cb76c33c701d59",
                old.parent / "console.log": "99ebc07f8e4827023dde7fd3cae0273f0b2772c6d413e14cec4d15c65c53f6c6",
                old.parent / "supervisor-failure.json": "d4ddbb1231346c8f69d32c599a83ecd7ae1fb7631471f657832929a69f6281f2"}
    if any(base.digest(path) != value for path, value in expected.items()):
        raise ValueError("Prior incomplete-run evidence bytes differ")
    if base.digest(old / "budget.json") != claim["budget_sha256"]:
        raise ValueError("Prior budget bytes no longer match the preserved claim")
    verify_budget_stop_report(base.read_json(old / "failure.json"))
    if (old / "completion.json").exists() or (old / "G/best_model.pth").exists():
        raise ValueError("Unexpected completed/saved prior model; inspect rather than overwrite or restart")
    epoch = base.read_json(old / "G/epoch-01.json")
    if epoch.get("epoch") != 1 or epoch.get("formal_optimizer_steps") != 151:
        raise ValueError("Prior partial epoch disagrees with failure accounting")
    return {"workspace": str(old), "prior_failed_optimizer_updates": 151,
            "prior_attempt_retained": True, "exact_resume_available": False,
            "replacement_starts_from_generic_seed75": True}


def protocol():
    value = base.protocol()
    for key in ("source_parents", "C_and_B_each_reload_original_parent", "benchmark"):
        value.pop(key)
    value.update({"version": "dwie-generic-budget-replacement-v1", "arms": ["G"],
        "prior_failed_optimizer_updates": 151, "expected_total_G_updates_across_attempts": 4681,
        "total_optimizer_updates": 4530,
        "initialization": "generic RoBERTa backbone only; every task-specific parameter freshly constructed with private seed75",
        "target_structure": "exact B/D noTTM shared head/tail projector; not a full native ATLOP reproduction",
        "generic_revision": MODEL_DIR.name, "generic_weight_sha256": GENERIC_FILES["model.safetensors"],
        "source_relation_training": False, "additional_benchmark": False,
        "reference_archive_sha256": REFERENCE_SHA})
    return value


def read_authorization(require_approved=False):
    if base.digest(AUTHORIZATION) != AUTHORIZATION_SHA:
        raise ValueError("G authorization SHA mismatch")
    auth = base.read_json(AUTHORIZATION)
    expected = {"arms": ["G"], "seed": 75, "epochs": 30, "optimizer_updates": 4530,
                "automatic_retry": False, "resume": False, "extension": False,
                "test_data_read": False, "new_seeds": False, "model_download": False,
                "deletion": False, "dependency_install": False}
    if any(auth.get(k) != v for k, v in expected.items()):
        raise ValueError("G authorization scope differs from the single fixed control")
    if require_approved and (auth.get("approved") is not True
            or isinstance(auth.get("hard_seconds"), bool)
            or auth.get("hard_seconds") != PROPOSED_HARD_SECONDS
            or not isinstance(auth.get("user_confirmation"), str)
            or not auth["user_confirmation"].strip()):
        raise PermissionError("G cloud budget has not been explicitly approved; no claim, mkdir, CUDA or training")
    return auth


def validate_local():
    auth = read_authorization()
    identity, train, dev, docs, rel2id = base.validate_inputs(cloud=False)
    for name, expected in {**SOURCE_PINS, "scripts/check_ttmre_dwie_generic_control.py": CHECKER_SHA}.items():
        if base.digest(ROOT / name) != expected:
            raise ValueError("G frozen source SHA mismatch: " + name)
        identity["source_files"][name] = expected
    if base.digest(REFERENCE) != REFERENCE_SHA:
        raise ValueError("Fixed C/B/D reference archive changed")
    identity.update({"protocol": protocol(), "entry_sha256": base.digest(Path(__file__)),
        "authorization": auth, "authorization_sha256": AUTHORIZATION_SHA,
        "cloud_assets_verified": False, "formal_training_started": False,
        "reference_archive_sha256": REFERENCE_SHA})
    return identity, train, dev, docs, rel2id


def disk_admission(path=ROOT):
    required = CHECKPOINT_BOUND + base.REPORT_MARGIN
    free = shutil.disk_usage(path).free
    if free < required:
        raise RuntimeError("G disk admission failed: free=%d required=%d shortfall=%d bytes; no deletion authorized" %
                           (free, required, required - free))
    return {"free_bytes": free, "required_bytes": required, "checkpoint_bound_bytes": CHECKPOINT_BOUND,
            "report_margin_bytes": base.REPORT_MARGIN, "checkpoints_to_write": 1, "files_deleted": 0}


def inventory(hash_weights=False, check_fn=None):
    """Current cloud file metadata and storage only; never opens a tensor."""
    rows, missing = {}, []
    for name, expected in GENERIC_FILES.items():
        path = MODEL_DIR / name
        if not path.is_file():
            missing.append(name)
            continue
        resolved = path.resolve(strict=True)
        if not resolved.is_relative_to((ROOT / "models").resolve()):
            raise ValueError("Generic asset resolves outside project models: " + name)
        row = {"path": str(path), "bytes": path.stat().st_size, "expected_sha256": expected,
               "sha256_verified": False}
        if name != "model.safetensors" or hash_weights:
            if base.digest(path, check_fn) != expected:
                raise ValueError("Pinned official generic asset bytes changed: " + name)
            row["sha256_verified"] = True
        rows[name] = row
    free = shutil.disk_usage(ROOT).free
    return {"mode": "read_only", "model_dir": str(MODEL_DIR), "generic_revision": MODEL_DIR.name,
        "files": rows, "missing_files": missing, "all_generic_assets_present": not missing,
        "weight_SHA_verified": not missing and rows["model.safetensors"]["sha256_verified"],
        "disk": {"free_bytes": free, "required_bytes": CHECKPOINT_BOUND + base.REPORT_MARGIN,
                 "sufficient": free >= CHECKPOINT_BOUND + base.REPORT_MARGIN},
        "tensor_loads": 0, "GPU_initialized": False, "files_written": 0, "files_deleted": 0,
        "training_started": False, "test_data_read": False, "downloaded": False}


def budget_from_environment():
    auth = read_authorization(require_approved=True)
    try:
        start = float(os.environ["TTMRE_DWIE_G_START_MONOTONIC"])
        boot = os.environ["TTMRE_DWIE_G_BOOT_ID"]
        utc = os.environ["TTMRE_DWIE_G_START_UTC"]
    except (KeyError, ValueError) as error:
        raise ValueError("Use the G-only launcher; admission must share its clock") from error
    if not math.isfinite(start) or start <= 0 or start > time.monotonic():
        raise ValueError("Invalid G total-clock origin")
    budget = {"start_monotonic": start, "deadline_monotonic": start + auth["hard_seconds"],
              "hard_seconds": auth["hard_seconds"], "boot_id": boot, "start_utc": utc,
              "authorization_sha256": AUTHORIZATION_SHA, "automatic_retry": False,
              "resume": False, "extension": False}
    base.check_budget(budget)
    return budget


def admit(require_lock=False):
    read_authorization(require_approved=True)
    self_sha = base.digest(Path(__file__))
    if os.environ.get("TTMRE_DWIE_G_SELF_SHA256") != self_sha:
        raise ValueError("Use the pinned G-only launcher")
    budget = budget_from_environment()
    if CLAIM.exists():
        raise ValueError("G control already claimed; no retry or resume")
    if require_lock:
        base.require_lock()
    if (ROOT / "experiments").is_symlink() or RUN_ROOT.is_symlink():
        raise ValueError("G output root must be the canonical project experiment tree")
    identity, _, _, _, _ = validate_local()
    identity["prior_attempt"] = verify_budget_stop_predecessor()
    check_fn = lambda: base.check_budget(budget)
    disk = disk_admission()
    assets = inventory(hash_weights=True, check_fn=check_fn)
    if not assets["all_generic_assets_present"] or not assets["weight_SHA_verified"]:
        raise FileNotFoundError("Pinned generic RoBERTa cache incomplete; no download authorized")
    runtime = runtime_fix.runtime_identity()
    memory = base.host_memory_admission({"parent_paths": {"generic": str(MODEL_DIR / "model.safetensors")}})
    external = base.probe.gpu_preflight.inspect_gpu()
    if not external["passed"]:
        raise RuntimeError("GPU occupancy exceeds 500MiB before G claim; no CUDA initialization")
    base.check_budget(budget)
    return {"budget": budget, "identity": identity, "assets": assets, "disk": disk,
            "runtime": runtime, "cpu_memory_admission": memory,
            "external_gpu_preflight": external, "self_sha256": self_sha}


def default_backbone_loader(model_dir, config):
    from transformers import AutoModel
    backbone, info = AutoModel.from_pretrained(str(model_dir), config=config,
        local_files_only=True, use_safetensors=True, output_loading_info=True)
    missing = set(info.get("missing_keys", []))
    # Transformers4.34 can retain the checkpoint's roberta prefix in loading
    # info although the bare RobertaModel has unprefixed parameter names.
    # Only these two unused pooler tensors are allowed in either spelling;
    # no encoder key, mismatch or error is ignored. Preserve raw loading info.
    unused_pooler = {prefix + name for prefix in ("", "roberta.")
                     for name in ("pooler.dense.weight", "pooler.dense.bias")}
    if (missing - unused_pooler
            or info.get("mismatched_keys") or info.get("error_msgs")
            or any(not key.startswith("lm_head.") for key in info.get("unexpected_keys", []))):
        raise ValueError("Generic encoder load differs beyond unused MLM head/pooler: " + repr(info))
    return backbone, {"loading_info": info, "unused_pooler_may_be_fresh": sorted(missing),
                      "generic_weight_sha256": GENERIC_FILES["model.safetensors"]}


def task_state_digest(model):
    return base.probe.tensor_state_sha256({n: v for n, v in model.state_dict().items() if not n.startswith("model.")})


def build_generic(identity, device, check_fn=None, model_factory=None, backbone_loader=None):
    import torch
    base._check(check_fn)
    factory = model_factory or base.probe.build_model
    loader = backbone_loader or default_backbone_loader
    # Model constructor, task projections and any unused provenance buffer are
    # private-seed fresh. No task checkpoint loader is called on any path.
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(75)
        model, execution = factory(identity["model_config_dir"])
        model.requires_grad_(True)
        adapter = base.probe.replacement.install(model, "bypass")
        head, head_sha = base._target_head(model.bilinear.in_features)
        base.probe.set_classifier(model, base.NUM_CLASSES, head)
        fresh_tasks = task_state_digest(model)
        backbone, load = loader(identity["model_config_dir"], model.config)
        model.model = backbone
    if task_state_digest(model) != fresh_tasks:
        raise RuntimeError("Generic encoder loading changed fresh task-specific state")
    if model_factory is None and head_sha != TARGET_HEAD_SHA:
        raise ValueError("G target head does not match the fixed C/B/D initialization")
    parameters = sum(p.numel() for p in model.parameters())
    if model_factory is None and parameters != 361603647:
        raise ValueError("G parameter geometry differs from the fixed B/D target structure")
    model.args.device = device
    model.train_mode = "finetune"
    model = model.to(device)
    gc.collect()
    return model, {"arm": "G", "initialization": "generic encoder only; no relation checkpoint read",
        "generic_load": load, "fresh_task_state_sha256": fresh_tasks,
        "all_task_parameters_fresh": True, "task_constructor_seed": 75,
        "fresh_target_head_sha256": head_sha, "fresh_target_head_seed": 75,
        "target_classes": base.NUM_CLASSES, "replacement": adapter, "execution": execution,
        "parameters": parameters, "source_task_tensors_read": 0}


def run_contracts():
    import unittest
    from unittest.mock import patch
    import torch
    import check_ttmre_dwie_generic_claim_isolation as generic
    import check_ttmre_dwie_runtime_fix as runtime
    import check_ttmre_dwie_adaptation as original
    import check_ttmre_dwie_generic_budget_replacement as replacement_tests
    suite = unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromModule(m)
                               for m in (generic, runtime, original, replacement_tests))
    with patch.object(torch.cuda, "_lazy_init", side_effect=AssertionError("CPU contracts attempted CUDA")):
        result = unittest.TextTestRunner(stream=sys.stdout).run(suite)
    if not result.wasSuccessful():
        raise RuntimeError("G CPU contracts failed before real generic tensor load")
    return {"passed": True, "tests_run": result.testsRun, "GPU_used": False, "real_weights_loaded": False}


def reference_comparison(result):
    if base.digest(REFERENCE) != REFERENCE_SHA:
        raise ValueError("C/B/D reference changed")
    with tarfile.open(REFERENCE, "r:gz") as archive:
        refs = {a: json.load(archive.extractfile("workspace/" + a + "/completion.json"))["best_dev"] for a in ("C", "B", "D")}
    score = result["best_dev"]["primary_full_id"]["f1"]
    return {"G": result["best_dev"], "fixed_CBD": refs,
        "G_minus_reference_F1_percentage_points": {a: 100 * (score - r["primary_full_id"]["f1"]) for a, r in refs.items()},
        "reference_archive_sha256": REFERENCE_SHA, "new_CBD_updates": 0,
        "automatic_next_experiment": False, "test_data_read": False}


def worker(workspace):
    read_authorization(require_approved=True)
    base.require_lock()
    budget = budget_from_environment()
    workspace = Path(workspace).resolve()
    claim = base.read_json(CLAIM)
    if (workspace != Path(claim["workspace"]) or workspace.parent.parent != RUN_ROOT.resolve()
            or base.digest(workspace / "budget.json") != claim["budget_sha256"]
            or base.read_json(workspace / "budget.json") != budget
            or claim["authorization_sha256"] != AUTHORIZATION_SHA
            or base.digest(Path(__file__)) != claim["entry_sha256"]
            or os.environ.get("TTMRE_DWIE_G_SELF_SHA256") != claim["entry_sha256"]):
        raise ValueError("G worker is not bound to its unique clock/source/claim")
    check_fn = lambda: base.check_budget(budget)
    updates = 0
    try:
        contracts = run_contracts()
        identity, train, dev, docs, rel2id = validate_local()
        identity["prior_attempt"] = verify_budget_stop_predecessor()
        assets = inventory(hash_weights=True, check_fn=check_fn)
        if not assets["all_generic_assets_present"] or not assets["weight_SHA_verified"]:
            raise ValueError("Generic cache changed after admission")
        disk = disk_admission()
        runtime = runtime_fix.runtime_identity()
        memory = base.host_memory_admission({"parent_paths": {"generic": str(MODEL_DIR / "model.safetensors")}})
        # Query external occupancy before the first torch CUDA API.
        external = base.probe.gpu_preflight.inspect_gpu()
        if not external["passed"]:
            raise RuntimeError("GPU occupancy exceeds the unchanged 500MiB threshold")
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable; no CPU fallback")
        device = torch.device("cuda", torch.cuda.current_device())
        if torch.cuda.get_device_properties(device).name != external["name"]:
            raise RuntimeError("CUDA device disagrees with external GPU preflight")
        identity.update({"model_config_dir": str(MODEL_DIR), "generic_assets": assets,
                         "disk_start": disk, "runtime": runtime, "cpu_contracts": contracts,
                         "cpu_memory_admission": memory, "external_gpu_preflight": external,
                         "cloud_assets_verified": True})
        base.write_json(workspace / "identity.json", identity)
        base.event(workspace, "G_construction_started", optimizer_updates=0)
        model, construction = build_generic(identity, device, check_fn)
        base.write_json(workspace / "G-construction.json", construction)
        admission = base.resource_admission(model, train, rel2id, device, check_fn=check_fn)
        base.write_json(workspace / "G-resource-admission.json", admission)
        base.event(workspace, "G_resource_admitted", optimizer_updates=0)

        def epoch_hook(row):
            base.event(workspace, "epoch_complete", arm="G", **row)
            if row["epoch"] == 1:
                estimate = row["epoch_seconds"] * 29 * 1.25 + 600
                available = budget["deadline_monotonic"] - time.monotonic() - base.SOFT_RESERVE
                base.event(workspace, "throughput_admission", estimated_remaining_seconds=estimate,
                           available_seconds=available, estimate_is_projection=True)
                if estimate > available:
                    raise base.BudgetStop("G measured throughput exceeds approved remaining total cap")

        def step_hook(step, row):
            nonlocal updates
            updates = step
            if step in (1, 2, 100) or step % 500 == 0:
                base.event(workspace, "formal_training_progress", arm="G", formal_optimizer_step=step,
                           loss=row["loss"], adam_step=row["adam_step"],
                           learning_rates_used=row["learning_rates_used"], witness=row["witness"])

        result = base.train_arm(model, train, dev, docs, rel2id, device, workspace / "G",
            check_fn=check_fn, epoch_hook=epoch_hook, official_source=base.OFFICIAL_METRIC, step_hook=step_hook)
        if not result["production_config"] or result["optimizer_steps"] != 4530:
            raise RuntimeError("Synthetic/incomplete training cannot satisfy G completion")
        complete = {"passed": True, "arm": "G", "protocol": protocol(), "result": result,
                    "optimizer_steps": 4530, "prior_failed_optimizer_steps": 151,
                    "total_G_optimizer_steps_across_attempts": 4681, "comparison": reference_comparison(result),
                    "elapsed_seconds": time.monotonic() - budget["start_monotonic"],
                    "test_data_read": False, "automatic_retry": False, "resume": False, "extension": False}
        check_fn()
        base.write_json(workspace / "completion.json", complete)
        archive = base.pack_reports(workspace, budget)
        base.write_json(workspace.parent / "reports-archive.json", archive)
        base.event(workspace, "DWIE_GENERIC_CONTROL_COMPLETE", passed=True, optimizer_steps=4530,
                   reports_archive_sha256=archive["sha256"], elapsed_seconds=complete["elapsed_seconds"])
        print(json.dumps({"event": "reports_archive_complete", **archive}), flush=True)
    except BaseException as error:
        base.write_json(workspace / "failure.json", {"passed": False, "error": str(error),
            "error_type": type(error).__name__, "optimizer_steps": updates,
            "test_data_read": False, "automatic_retry": False, "resume": False, "extension": False,
            "elapsed_seconds": time.monotonic() - budget["start_monotonic"]})
        base.event(workspace, "DWIE_GENERIC_CONTROL_FAILED", error=str(error), optimizer_steps=updates)
        raise


def start(workspace):
    context = admit(require_lock=True)
    workspace = Path(workspace).resolve()
    if (workspace.name != "workspace" or workspace.parent.parent != RUN_ROOT.resolve()
            or not workspace.parent.name.startswith("generic75-launch-") or workspace.exists()):
        raise ValueError("G requires one fresh canonical workspace")
    workspace.mkdir(parents=True, exist_ok=False)
    base.write_json(workspace / "budget.json", context["budget"])
    base.write_json(workspace / "preclaim-admission.json", context)
    base.write_json(CLAIM, {"workspace": str(workspace), "budget_sha256": base.digest(workspace / "budget.json"),
                          "authorization_sha256": AUTHORIZATION_SHA, "entry_sha256": context["self_sha256"]})
    base.write_json(workspace.parent / "pid.json", {"supervisor_pid": os.getpid()})
    command = [sys.executable, "-u", "-B", str(Path(__file__).resolve()), "--mode", "worker", "--run-dir", str(workspace)]
    try:
        base.supervise(command, context["budget"], base.require_lock())
    except BaseException as error:
        base.write_json(workspace.parent / "supervisor-failure.json", {
            "error": str(error), "type": type(error).__name__, "complete": False, "automatic_retry": False})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("validate", "inventory", "authorize", "clock", "admit", "start", "worker"), default="validate")
    parser.add_argument("--run-dir", type=Path)
    args = parser.parse_args()
    if args.mode == "validate":
        identity, _, _, _, _ = validate_local()
        print(json.dumps({"passed": True, "mode": "read_only", "protocol": protocol(),
                          "train_documents": identity["train_documents"], "dev_documents": identity["dev_documents"],
                          "GPU_used": False, "test_data_read": False}, ensure_ascii=False))
    elif args.mode == "inventory":
        print(json.dumps(inventory(), indent=2))
    elif args.mode == "authorize":
        read_authorization(require_approved=True)
        print(json.dumps({"approved": True, "hard_seconds": PROPOSED_HARD_SECONDS}))
    elif args.mode == "clock":
        read_authorization(require_approved=True)
        print(time.monotonic(), Path("/proc/sys/kernel/random/boot_id").read_text().strip(), datetime.now(timezone.utc).isoformat())
    elif args.mode == "admit":
        context = admit()
        print(json.dumps({"passed": True, "disk": context["disk"], "clock_already_running": True,
                          "CUDA_initialized": False, "new_claim_created": False}))
    else:
        if args.run_dir is None:
            raise ValueError("G launcher must provide a fresh --run-dir")
        (start if args.mode == "start" else worker)(args.run_dir)


if __name__ == "__main__":
    main()
