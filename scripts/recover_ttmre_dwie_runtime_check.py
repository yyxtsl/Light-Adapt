"""One separately authorized pre-update recovery, within the original deadline.

The failed workspace, original claim, runner, authorization and upload ZIP are
read only. A fresh recovery workspace receives the exact original budget bytes.
No clock, boot ID, method, optimizer setting or training update is reset.
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
import train_ttmre_dwie_adaptation as run
import ttmre_dwie_runtime_fix as fixed

DOCROOT = ROOT / "refine-logs/SECOND_DATASET_ADMISSION_20261023"
PERMISSION = DOCROOT / "DWIE_RUNTIME_RECOVERY_AUTHORIZATION_APPROVED_20261024.json"
PERMISSION_SHA256 = "a3993cc7c5719a802abf0a3cd25551018f49ca7ee17f1ac9974886d131ff8a4a"
ORIGINAL_LAUNCH = run.RUN_ROOT / "seed75-launch-20261024-162616-gqgrqM"
ORIGINAL_WORKSPACE = ORIGINAL_LAUNCH / "workspace"
ORIGINAL_ENTRY_SHA256 = "4b5e37f89e8a1085781a01fa3531cd7c0e41df7f4fe58815ab31d704cb5a10fd"
ORIGINAL_AUTH_SHA256 = "174a3276894ed6b9cbbb55b6f8a2d58918c16b9fa79210f31e4e650571dfce54"
FAILURE_SHA256 = "5381f23c09b1bc3c4809db9fca9342af621efc8dd3e1fb50f3a331b996b0135e"
CONSOLE_SHA256 = "a57847a39f10edcc1f981ea257051df397d540be79a64f71bfcd0807430590e8"
RECOVERY_CLAIM = "runtime-audit-recovery-once-claim.json"
FIX_PINS = {
    "scripts/ttmre_dwie_runtime_fix.py": "f83769948090f04c424329e9f3436ba3527c882b6422e4326444a7737165dc20",
    "scripts/check_ttmre_dwie_runtime_fix.py": "197afc107ff1a8fad5db89268cc546124fbdb805ec35edcc3c3bd557cd7eab23",
}


def read_permission():
    if run.digest(PERMISSION) != PERMISSION_SHA256:
        raise ValueError("Recovery authorization SHA mismatch")
    permission = run.read_json(PERMISSION)
    if (permission.get("approved") is not True
            or not isinstance(permission.get("user_confirmation"), str)
            or not permission["user_confirmation"].strip()):
        raise PermissionError("One manual recovery has not been explicitly confirmed; no write or launch")
    expected = {"original_launch": ORIGINAL_LAUNCH.relative_to(ROOT).as_posix(),
                "original_hard_seconds": 28800, "new_budget_seconds": 0,
                "maximum_manual_recoveries": 1, "test_data_read": False,
                "automatic_retry": False, "extension": False, "deletion": False,
                "dependency_install": False}
    if any(permission.get(key) != value for key, value in expected.items()):
        raise ValueError("Recovery scope/budget differs from the fixed original plan")
    return permission


def verify_failure_budget(failure, budget):
    expected_progress = {"current_arm": None, "current_arm_optimizer_steps": 0,
                         "completed_optimizer_steps": 0}
    if (failure.get("error") != "module 'torch.optim.adamw' has no attribute '_single_tensor_adamw'"
            or failure.get("error_type") != "AttributeError" or failure.get("completed_arms") != {}
            or failure.get("progress") != expected_progress or failure.get("test_data_read") is not False):
        raise ValueError("Failure is not the exact zero-update runtime-check failure")
    times = [budget.get("start_monotonic"), budget.get("deadline_monotonic")]
    if (any(isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or value <= 0 for value in times)
            or budget.get("hard_seconds") != 28800 or budget.get("authorization_sha256") != ORIGINAL_AUTH_SHA256
            or budget.get("resume") is not False or budget.get("extension") is not False
            or abs(budget["deadline_monotonic"] - budget["start_monotonic"] - 28800) > 1e-6):
        raise ValueError("Original budget changed; no new eight-hour clock permitted")


def verify_sources():
    if run.digest(ROOT / "scripts/train_ttmre_dwie_adaptation.py") != ORIGINAL_ENTRY_SHA256:
        raise ValueError("Original failed runner must remain unchanged")
    if run.digest(run.AUTHORIZATION) != ORIGINAL_AUTH_SHA256:
        raise ValueError("Original eight-hour authorization changed")
    self_sha = run.digest(Path(__file__))
    if os.environ.get("TTMRE_DWIE_RUNTIME_RECOVERY_SHA256") != self_sha:
        raise ValueError("Use the pinned manual recovery launcher")
    for name, expected in FIX_PINS.items():
        if run.digest(ROOT / name) != expected:
            raise ValueError("Runtime fix source mismatch: " + name)
    return self_sha


def verify_original():
    claim = run.read_json(run.RUN_ROOT / run.CLAIM_NAME)
    if Path(claim["workspace"]).resolve() != ORIGINAL_WORKSPACE.resolve():
        raise ValueError("Original one-use claim is not the reported failed launch")
    budget_file = ORIGINAL_WORKSPACE / "budget.json"
    if run.digest(budget_file) != claim["budget_sha256"]:
        raise ValueError("Original claim/budget SHA mismatch")
    budget = run.read_json(budget_file)
    if run.digest(ORIGINAL_WORKSPACE / "failure.json") != FAILURE_SHA256:
        raise ValueError("Original zero-update failure bytes differ from supplied report")
    if run.digest(ORIGINAL_LAUNCH / "console.log") != CONSOLE_SHA256:
        raise ValueError("Original console bytes differ from supplied log")
    failure = run.read_json(ORIGINAL_WORKSPACE / "failure.json")
    verify_failure_budget(failure, budget)
    run.check_budget(budget)
    forbidden = ("identity.json", "fixed_epoch_groups.json", "completion.json", "C", "B", "D")
    if any((ORIGINAL_WORKSPACE / name).exists() for name in forbidden):
        raise ValueError("Unexpected formal artifacts in original workspace; recovery refuses")
    return budget, budget_file.read_bytes(), claim


def configure_worker(self_sha, budget):
    # Only the diagnostic callback and a new claim binding are substituted;
    # all model/data/optimizer/step/selection functions remain the frozen code.
    run.CLAIM_NAME = RECOVERY_CLAIM
    run.SOURCE_PINS.update(FIX_PINS)
    run.SOURCE_PINS[Path(__file__).relative_to(ROOT).as_posix()] = self_sha
    base_contracts = run.run_synthetic_contracts

    def contracts():
        import unittest
        from unittest.mock import patch
        import torch
        import check_ttmre_dwie_runtime_fix as extra
        run.check_budget(budget)
        with patch.object(torch.cuda, "_lazy_init", side_effect=AssertionError("CPU contracts attempted CUDA")):
            result = unittest.TextTestRunner(stream=sys.stdout).run(unittest.defaultTestLoader.loadTestsFromModule(extra))
        if not result.wasSuccessful():
            raise RuntimeError("Runtime-fix CPU contracts failed")
        base = base_contracts()
        run.check_budget(budget)
        return {**base, "tests_run": base["tests_run"] + result.testsRun,
                "original_training_contracts": base["tests_run"], "runtime_fix_contracts": result.testsRun}

    def runtime():
        return {**fixed.runtime_identity(), "manual_preupdate_recovery": {
            "original_launch": str(ORIGINAL_LAUNCH), "original_failure_sha256": FAILURE_SHA256,
            "recovery_authorization_sha256": PERMISSION_SHA256,
            "original_deadline_monotonic": budget["deadline_monotonic"],
            "original_budget_retained_byte_for_byte": True, "new_budget_seconds": 0}}

    run.run_synthetic_contracts = contracts
    run.runtime_identity = runtime


def worker(workspace):
    read_permission()
    self_sha = verify_sources()
    run.require_lock()
    budget, _, _ = verify_original()
    claim = run.read_json(run.RUN_ROOT / RECOVERY_CLAIM)
    if (Path(claim["workspace"]).resolve() != Path(workspace).resolve()
            or claim.get("recovery_authorization_sha256") != PERMISSION_SHA256
            or claim.get("recovery_source_sha256") != self_sha):
        raise ValueError("Worker does not match the single permitted recovery claim")
    configure_worker(self_sha, budget)
    run.worker(workspace)


def start(workspace):
    read_permission()
    self_sha = verify_sources()
    lock_fd = run.require_lock()
    budget, raw_budget, original_claim = verify_original()
    workspace = Path(workspace).resolve()
    if (workspace.name != "workspace" or workspace.parent.parent != run.RUN_ROOT.resolve()
            or not workspace.parent.name.startswith("runtime-recovery-") or workspace.exists()
            or (run.RUN_ROOT / RECOVERY_CLAIM).exists()):
        raise ValueError("Recovery requires one fresh workspace; never reuse a claim")
    workspace.mkdir(parents=True, exist_ok=False)
    with (workspace / "budget.json").open("xb") as stream:
        stream.write(raw_budget)
        stream.flush()
        os.fsync(stream.fileno())
    run.write_json(run.RUN_ROOT / RECOVERY_CLAIM, {
        "workspace": str(workspace), "budget_sha256": original_claim["budget_sha256"],
        "original_workspace": str(ORIGINAL_WORKSPACE), "original_failure_sha256": FAILURE_SHA256,
        "recovery_authorization_sha256": PERMISSION_SHA256, "recovery_source_sha256": self_sha,
        "new_budget_seconds": 0, "automatic_retry": False, "extension": False})
    run.write_json(workspace.parent / "pid.json", {"supervisor_pid": os.getpid()})
    command = [sys.executable, "-u", "-B", str(Path(__file__).resolve()), "--mode", "worker", "--run-dir", str(workspace)]
    try:
        run.supervise(command, budget, lock_fd)
    except BaseException as error:
        run.write_json(workspace.parent / "supervisor-failure.json", {
            "error": str(error), "type": type(error).__name__, "complete": False,
            "original_deadline_retained": True, "new_budget_seconds": 0, "automatic_retry": False})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("authorize", "start", "worker"), default="authorize")
    parser.add_argument("--run-dir", type=Path)
    args = parser.parse_args()
    if args.mode == "authorize":
        read_permission()
        print(json.dumps({"approved": True, "new_budget_seconds": 0, "original_deadline_retained": True}))
    else:
        if args.run_dir is None:
            raise ValueError("Recovery launcher must provide --run-dir")
        (start if args.mode == "start" else worker)(args.run_dir)


if __name__ == "__main__":
    main()
