"""One explicitly authorized fresh N1 seed75 pair with a new 20-hour budget.

Reuses the corrected full75 training implementation; never restores failed task
weights, optimizer, RNG or dataloader state. Default validation is read-only.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import signal
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]
import train_ttmre_n1_full_seed75 as core

RUN_ROOT = ROOT / "experiments/2026-10-17-n1-independent-seed75-restart20h"
HARD_SECONDS = 20 * 60 * 60
CLAIM_NAME = "n1-full75-restart20h-once-claim.json"
FAILED_WORKSPACE = ROOT / (
    "experiments/2026-10-17-n1-independent-seed75/"
    "n1-independent-seed75-repair2-launch-20261017-155737-F69brC/workspace")
FAILED_HASHES = {
    "identity.json": "1efa9331f6dc220ac7eaa1fcb0986ef9a74745084d6fdc51bd8a04006c5f113d",
    "failure.json": "03cdaac09c06accdf6054f3bde060c05bbc85d6335fae3f2358447f94316557c",
    "failure-control.json": "66228a46306f0a4077c57be157d112af83a4b698dc333a6d8be7d4f3c0f70a87",
    "budget.json": "692781a5a13276db0ab3353442f7228a11a28c2f992242842d4458a7b9172055",
}
FILES = (
    "scripts/train_ttmre_n1_full_seed75_restart20h.py",
    "scripts/check_ttmre_n1_full_seed75_restart20h.py",
    "scripts/run_ttmre_n1_full_seed75_restart20h.sh",
)
_original_protocol = core.protocol
_original_validate = core.validate
_original_lock = core.require_lock


def protocol():
    values = _original_protocol()
    values.update({"version": "n1-independent-full-seed75-restart20h-v2",
        "variant": "n1-independent-full-seed75-restart20h", "hard_seconds": HARD_SECONDS,
        "execution_authorization": "user-approved-fresh-seed75-pair-20h-20261017",
        "failed_checkpoint_reused": False, "gradient_update_witness_required": True})
    return values


def previous_failure_context():
    """Read only pinned failure metadata; never load the failed checkpoint."""
    for name, expected in FAILED_HASHES.items():
        if core.digest(FAILED_WORKSPACE / name) != expected:
            raise ValueError("Historical failure evidence mismatch: " + name)
    identity = core.read(FAILED_WORKSPACE / "identity.json")
    budget = core.read(FAILED_WORKSPACE / "budget.json")
    failed = core.read(FAILED_WORKSPACE / "failure-control.json")
    if (identity["budget_sha256"] != FAILED_HASHES["budget.json"]
            or failed["phase"] != "control" or failed["type"] != "ValueError"
            or failed["error"] != "Invalid control/distant trajectory"):
        raise ValueError("Not the reviewed failed full75 execution")
    return {"workspace": str(FAILED_WORKSPACE), "metadata_sha256": dict(FAILED_HASHES),
        "old_budget": budget, "failed_nominal_step_calls": 50936,
        "failed_effective_learning_updates": 0, "checkpoint_or_optimizer_loaded": False,
        "new_budget_explicitly_authorized": True, "old_deadline_not_used_for_new_run": True}


def validate(args, budget=None):
    identity = _original_validate(args, budget)
    identity["historical_failed_attempt"] = previous_failure_context()
    identity["execution_entrypoint_sha256"] = core.digest(Path(__file__))
    weights = [args.model_dir / n for n in identity["model"]["sha256"]
               if n.endswith((".bin", ".safetensors"))]
    required = max(8 * 1024**3, sum(p.stat().st_size for p in weights) * 5 + 1024**3)
    if shutil.disk_usage(args.run_dir or RUN_ROOT.parent).free < required:
        raise RuntimeError("Insufficient free disk for four checkpoints and one atomic partial")
    identity["restart_required_free_bytes"] = required
    return identity


def require_lock():
    if os.environ.get("TTMRE_N1_FULL75_RESTART20H_LOCK_FD") != "9":
        raise RuntimeError("Use only the new restart20h launcher")
    if os.environ.get("TTMRE_N1_FULL75_RECOVERY_LOCK_FD"):
        raise RuntimeError("Old recovery environment is forbidden")
    return _original_lock()


def configure():
    core.RUN_ROOT = RUN_ROOT
    core.HARD_SECONDS = HARD_SECONDS
    core.VARIANT = "n1-independent-full-seed75-restart20h"
    core.NEW_FILES = tuple(dict.fromkeys((*core.NEW_FILES, *FILES)))
    core.protocol = protocol
    core.validate = validate
    core.require_lock = require_lock


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=("validate", "train", "worker"), default="validate")
    p.add_argument("--phase", choices=("prepare", *core.BRANCHES, "finish"))
    p.add_argument("--run-dir", type=Path)
    p.add_argument("--baseline-workspace", type=Path, default=core.DEFAULT_BASELINE)
    p.add_argument("--preflight-report", type=Path)
    p.add_argument("--model-dir", type=Path)
    p.add_argument("--data-dir", type=Path, default=ROOT / "dataset/docred")
    p.add_argument("--check-runtime", action="store_true")
    return p


def exclusive_json(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n"); stream.flush(); os.fsync(stream.fileno())


def new_budget(begin, boot_id, history):
    return {"start_monotonic": begin, "deadline_monotonic": begin + HARD_SECONDS,
        "hard_seconds": HARD_SECONDS, "start_utc": datetime.now(timezone.utc).isoformat(),
        "boot_id": boot_id, "soft_reserve_seconds": 120, "term_reserve_seconds": 30,
        "automatic_retry": False, "automatic_resume": False, "extension": False,
        "authorization": protocol()["execution_authorization"],
        "historical_failed_attempt": history, "new_fresh_training_not_resume": True}


def train(args):
    if os.name != "posix" or args.run_dir is None:
        raise RuntimeError("Linux unique workspace required; use the new launcher")
    lock_fd = require_lock()
    begin = time.monotonic()
    args.run_dir = args.run_dir.expanduser().resolve()
    if args.run_dir.name != "workspace" or args.run_dir.parent.parent != RUN_ROOT.resolve():
        raise ValueError("Use a fresh launch/workspace under the restart20h root")
    claim = RUN_ROOT / CLAIM_NAME
    if claim.exists() or args.run_dir.exists():
        raise FileExistsError("Preserve one-use claim/workspace; no retry or resume")
    history = previous_failure_context()
    args.run_dir.mkdir(parents=True, exist_ok=False)
    budget = new_budget(begin, Path("/proc/sys/kernel/random/boot_id").read_text().strip(), history)
    core.shared.write_json(args.run_dir / "budget.json", budget)
    # The exclusive claim is never rewritten or removed, including on failure.
    exclusive_json(claim, {"workspace": str(args.run_dir), "protocol": protocol(),
        "budget_sha256": core.digest(args.run_dir / "budget.json"), "automatic_retry": False})
    exclusive_json(args.run_dir.parent / "pid", os.getpid())
    def terminate(*_):
        raise core.BudgetStop("Supervisor interrupted; no retry/resume/extension")
    signal.signal(signal.SIGTERM, terminate)
    command = [sys.executable, "-u", "-B", str(Path(__file__).resolve()),
               "--mode", "worker", "--run-dir", str(args.run_dir)]
    # Preserve explicit input paths for the prepare child.
    for name in ("baseline_workspace", "preflight_report", "model_dir", "data_dir"):
        value = getattr(args, name)
        if value is not None:
            command.extend(["--" + name.replace("_", "-"), str(value)])
    try:
        core.emit("N1_FULL_SEED75_RESTART20H_BUDGET_STARTED", **budget)
        core.supervise([sys.executable, "-B", str(ROOT / FILES[1])], budget, lock_fd)
        for phase in ("prepare", *core.BRANCHES, "finish"):
            core.supervise(command + ["--phase", phase], budget, lock_fd)
    except BaseException as error:
        core.shared.write_json(args.run_dir / "failure.json", {
            "error": str(error), "type": type(error).__name__, "complete": False,
            "resumable": False, "automatic_retry": False,
            "elapsed_seconds": time.monotonic() - begin})
        raise


def main():
    configure()
    args = parser().parse_args()
    os.chdir(ROOT)
    if args.mode == "train":
        train(args)
    elif args.mode == "worker":
        core.worker(args)
    else:
        print(json.dumps({"validation": "passed", "identity": validate(args),
                          "optimizer_steps": 0, "files_written": False}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
