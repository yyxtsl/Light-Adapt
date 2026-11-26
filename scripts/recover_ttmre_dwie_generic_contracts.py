"""One explicitly approved pre-update G recovery, using the original deadline.

Old G sources, claim, failure, logs and authorization remain byte-identical.
Only the synthetic test suite is replaced with an isolated fixture version.
No method change, new 90-minute clock, deletion, download or automatic retry.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]
import train_ttmre_dwie_generic_control as run

DOC = run.DOC
PERMISSION = DOC / "CONTRACT_RECOVERY_AUTHORIZATION_APPROVED_20261025.json"
PERMISSION_SHA = "56176a7c4641907390d9737adf764e90ec854644b8c2740e0b25d3abb465f170"
ORIGINAL_CLAIM = run.RUN_ROOT / "generic-control-once-claim.json"
RECOVERY_CLAIM = run.RUN_ROOT / "contract-isolation-recovery-once-claim.json"
ORIGINAL_ENTRY_SHA = "15470cc9874f2ecf5ca1808e3565d295966b31e0bd8920dc7d5a98ed4fae8321"
ORIGINAL_CHECKER_SHA = "0b3a7cd6e68cb087dc864a391078097cc868fb69528b79ef553dc829fca5740c"
ORIGINAL_AUTH_SHA = "22ca7e9931c30dda316e5d1554c6e7836070e5f89d2016f9186540557dee845e"
FAILURE_SHA = "d9376b386ca4ba3f26b871fa42a1144b5a3949e4c317f2134edd14de7416d8de"
CONSOLE_SHA = "4f72a6358175af22593c21628f9bff2bf8262648aea4fa5ce7f5104fd1bf02f7"
FIXED_CHECKER = "scripts/check_ttmre_dwie_generic_claim_isolation.py"
FIXED_CHECKER_SHA = "fdabb0599ec3884369301b76b4c167a7860434cf7be5ea2a332d8870af3f52e0"


def read_permission():
    if run.base.digest(PERMISSION) != PERMISSION_SHA:
        raise ValueError("G contract recovery permission SHA mismatch")
    p = run.base.read_json(PERMISSION)
    if p.get("approved") is not True or not isinstance(p.get("user_confirmation"), str) or not p["user_confirmation"].strip():
        raise PermissionError("One manual G pre-update recovery has not been explicitly approved; no launch or writes")
    expected = {"original_authorization_sha256": ORIGINAL_AUTH_SHA, "original_hard_seconds": 5400,
                "new_budget_seconds": 0, "maximum_manual_recoveries": 1, "automatic_retry": False,
                "resume": False, "extension": False, "test_data_read": False, "deletion": False,
                "download": False, "dependency_install": False}
    if any(p.get(k) != v for k, v in expected.items()):
        raise ValueError("G recovery cannot add time, updates, retry or a different scope")
    return p


def verify_failure_budget(failure, budget):
    if (failure.get("passed") is not False or failure.get("optimizer_steps") != 0
            or failure.get("error") != "G CPU contracts failed before real generic tensor load"
            or failure.get("error_type") != "RuntimeError" or failure.get("test_data_read") is not False
            or any(failure.get(k) is not False for k in ("automatic_retry", "resume", "extension"))):
        raise ValueError("Recovery is limited to the exact zero-update CPU-contract failure")
    times = [budget.get(k) for k in ("start_monotonic", "deadline_monotonic")]
    if (any(isinstance(t, bool) or not isinstance(t, (int, float)) or not math.isfinite(t) or t <= 0 for t in times)
            or budget.get("hard_seconds") != 5400 or budget.get("authorization_sha256") != ORIGINAL_AUTH_SHA
            or abs(times[1] - times[0] - 5400) > 1e-6
            or any(budget.get(k) is not False for k in ("automatic_retry", "resume", "extension"))):
        raise ValueError("Original G clock/authorization changed; no fresh 90-minute cap")


def verify_sources():
    pins = {"scripts/train_ttmre_dwie_generic_control.py": ORIGINAL_ENTRY_SHA,
            "scripts/check_ttmre_dwie_generic_control.py": ORIGINAL_CHECKER_SHA,
            FIXED_CHECKER: FIXED_CHECKER_SHA}
    for name, expected in pins.items():
        if run.base.digest(ROOT / name) != expected:
            raise ValueError("G frozen/fixed source mismatch: " + name)
    if run.base.digest(run.AUTHORIZATION) != ORIGINAL_AUTH_SHA:
        raise ValueError("Original G training authorization changed")
    value = run.base.digest(Path(__file__))
    if os.environ.get("TTMRE_G_CONTRACT_RECOVERY_SELF_SHA256") != value:
        raise ValueError("Use the pinned G contract-recovery launcher")
    return value


def verify_original():
    claim = run.base.read_json(ORIGINAL_CLAIM)
    workspace = Path(claim["workspace"])
    if (workspace != workspace.resolve() or workspace.name != "workspace"
            or workspace.parent.parent != run.RUN_ROOT.resolve()
            or not workspace.parent.name.startswith("generic75-launch-")
            or claim.get("authorization_sha256") != ORIGINAL_AUTH_SHA
            or claim.get("entry_sha256") != ORIGINAL_ENTRY_SHA):
        raise ValueError("Original claim is not the frozen G launch")
    raw = (workspace / "budget.json").read_bytes()
    if run.base.digest(workspace / "budget.json") != claim["budget_sha256"]:
        raise ValueError("Original claim/budget SHA differs")
    if (run.base.digest(workspace / "failure.json") != FAILURE_SHA
            or run.base.digest(workspace.parent / "console.log") != CONSOLE_SHA):
        raise ValueError("Original G failure/console differs from supplied evidence")
    budget = json.loads(raw)
    verify_failure_budget(run.base.read_json(workspace / "failure.json"), budget)
    if any((workspace / name).exists() for name in ("identity.json", "G", "G-construction.json", "G-resource-admission.json", "completion.json")):
        raise ValueError("Formal model/training artifacts already exist; recovery refuses")
    run.base.check_budget(budget)
    return budget, raw, claim, workspace


def restore_clock_environment(budget):
    os.environ["TTMRE_DWIE_G_START_MONOTONIC"] = str(budget["start_monotonic"])
    os.environ["TTMRE_DWIE_G_BOOT_ID"] = budget["boot_id"]
    os.environ["TTMRE_DWIE_G_START_UTC"] = budget["start_utc"]
    os.environ["TTMRE_DWIE_G_SELF_SHA256"] = ORIGINAL_ENTRY_SHA


def admit(allow_claimed=False, require_lock=False):
    read_permission()
    self_sha = verify_sources()
    if require_lock:
        run.base.require_lock()
    if not allow_claimed and RECOVERY_CLAIM.exists():
        raise ValueError("G contract recovery already claimed; no repeat")
    budget, raw, claim, old_workspace = verify_original()
    restore_clock_environment(budget)
    run.validate_local()
    disk = run.disk_admission()
    assets = run.inventory(hash_weights=True, check_fn=lambda: run.base.check_budget(budget))
    if not assets["all_generic_assets_present"] or not assets["weight_SHA_verified"]:
        raise ValueError("Generic assets unavailable; no download or alternate initialization")
    runtime = run.runtime_fix.runtime_identity()
    memory = run.base.host_memory_admission({"parent_paths": {"generic": str(run.MODEL_DIR / "model.safetensors")}})
    external = run.base.probe.gpu_preflight.inspect_gpu()
    if not external["passed"]:
        raise RuntimeError("GPU busy before G recovery; no CUDA initialized")
    run.base.check_budget(budget)
    return {"budget": budget, "raw_budget": raw, "original_claim": claim,
            "old_workspace": old_workspace, "self_sha": self_sha, "disk": disk,
            "runtime": runtime, "memory": memory, "external_gpu": external}


def configure_worker(context):
    budget = context["budget"]
    run.CLAIM = RECOVERY_CLAIM
    original_validate = run.validate_local

    def contracts():
        import unittest
        from unittest.mock import patch
        import torch
        import check_ttmre_dwie_generic_claim_isolation as isolated
        import check_ttmre_dwie_runtime_fix as runtime
        import check_ttmre_dwie_adaptation as original
        run.base.check_budget(budget)
        suite = unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromModule(m) for m in (isolated, runtime, original))
        with patch.object(torch.cuda, "_lazy_init", side_effect=AssertionError("CPU contracts attempted CUDA")):
            result = unittest.TextTestRunner(stream=sys.stdout).run(suite)
        if not result.wasSuccessful():
            raise RuntimeError("Isolated G contracts failed; no real generic tensor load")
        run.base.check_budget(budget)
        return {"passed": True, "tests_run": result.testsRun, "GPU_used": False, "real_weights_loaded": False,
                "claim_fixture_isolation_fixed": True}

    def validate():
        identity, train, dev, docs, rel2id = original_validate()
        identity["source_files"][FIXED_CHECKER] = FIXED_CHECKER_SHA
        identity["source_files"][Path(__file__).relative_to(ROOT).as_posix()] = context["self_sha"]
        identity["manual_contract_recovery"] = {"original_workspace": str(context["old_workspace"]),
            "original_failure_sha256": FAILURE_SHA, "original_budget_retained_byte_for_byte": True,
            "original_deadline_monotonic": budget["deadline_monotonic"], "new_budget_seconds": 0,
            "recovery_authorization_sha256": PERMISSION_SHA}
        return identity, train, dev, docs, rel2id

    run.run_contracts = contracts
    run.validate_local = validate


def claim_record(context, workspace):
    return {"workspace": str(workspace), "budget_sha256": context["original_claim"]["budget_sha256"],
            "authorization_sha256": ORIGINAL_AUTH_SHA, "entry_sha256": ORIGINAL_ENTRY_SHA,
            "recovery_authorization_sha256": PERMISSION_SHA, "recovery_source_sha256": context["self_sha"],
            "original_workspace": str(context["old_workspace"]), "original_failure_sha256": FAILURE_SHA,
            "new_budget_seconds": 0, "automatic_retry": False, "extension": False}


def worker(workspace):
    context = admit(allow_claimed=True, require_lock=True)
    workspace = Path(workspace).resolve()
    if run.base.read_json(RECOVERY_CLAIM) != claim_record(context, workspace):
        raise ValueError("G recovery worker claim/source mismatch")
    if (workspace / "budget.json").read_bytes() != context["raw_budget"]:
        raise ValueError("G recovery budget bytes differ from the original")
    configure_worker(context)
    run.worker(workspace)


def start(workspace):
    context = admit(require_lock=True)
    workspace = Path(workspace).resolve()
    if (workspace.name != "workspace" or workspace.parent.parent != run.RUN_ROOT.resolve()
            or not workspace.parent.name.startswith("contract-recovery-") or workspace.exists()):
        raise ValueError("G recovery requires one new canonical workspace")
    workspace.mkdir(parents=True, exist_ok=False)
    with (workspace / "budget.json").open("xb") as f:
        f.write(context["raw_budget"])
        f.flush()
        os.fsync(f.fileno())
    run.base.write_json(RECOVERY_CLAIM, claim_record(context, workspace))
    run.base.write_json(workspace.parent / "pid.json", {"supervisor_pid": os.getpid()})
    command = [sys.executable, "-u", "-B", str(Path(__file__).resolve()), "--mode", "worker", "--run-dir", str(workspace)]
    try:
        run.base.supervise(command, context["budget"], run.base.require_lock())
    except BaseException as error:
        run.base.write_json(workspace.parent / "supervisor-failure.json", {
            "error": str(error), "complete": False, "original_deadline_retained": True,
            "new_budget_seconds": 0, "automatic_retry": False})
        raise


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=("authorize", "admit", "start", "worker"), default="authorize")
    p.add_argument("--run-dir", type=Path)
    a = p.parse_args()
    if a.mode == "authorize":
        read_permission()
        print(json.dumps({"approved": True, "new_budget_seconds": 0}))
    elif a.mode == "admit":
        c = admit()
        print(json.dumps({"passed": True, "original_deadline_monotonic": c["budget"]["deadline_monotonic"],
                          "new_budget_seconds": 0, "disk": c["disk"]}))
    else:
        if a.run_dir is None:
            raise ValueError("Recovery launcher must provide --run-dir")
        (start if a.mode == "start" else worker)(a.run_dir)


if __name__ == "__main__":
    main()
