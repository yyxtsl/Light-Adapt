"""Run one independent seed75 Control/Weighted N1 two-stage pair.

The default mode is read-only validation.  ``train`` is reachable only from
the documented launcher and creates one exclusive, non-resumable workspace.
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import train_ttmre_human_distant as baseline
import train_ttmre_memory_noise as n1
import ttmre_memory_noise as method

preflight = method.preflight
shared, aligned = baseline.shared, baseline.aligned

RUN_ROOT = ROOT / "experiments/2026-10-17-n1-independent-seed75"
DEFAULT_BASELINE = ROOT / (
    "experiments/2026-10-11-human-distant-baseline/"
    "two-stage-paper-freeze-seed74-launch-20261011-190240-rjeB6g/workspace"
)
BRANCHES = ("control", "weighted")
SEED = 75
DISTANT_UPDATES = baseline.STAGES["distant"][0] // 4 * baseline.STAGES["distant"][1]
HUMAN_UPDATES = baseline.STAGES["human-frozen"][0] // 4 * baseline.STAGES["human-frozen"][1]
UPDATES_PER_BRANCH = DISTANT_UPDATES + HUMAN_UPDATES
FORMAL_UPDATES = 2 * UPDATES_PER_BRANCH
HARD_SECONDS = 12 * 60 * 60
RESERVE_SECONDS = 120
TERM_SECONDS = 30
N1_VIEW_OFFSET = 750000
VARIANT = "n1-independent-full-seed75"
NEW_FILES = (
    "scripts/ttmre_memory_noise.py",
    "scripts/train_ttmre_memory_noise.py",
    "scripts/check_ttmre_memory_noise.py",
    "scripts/run_ttmre_memory_noise.sh",
    "scripts/train_ttmre_n1_full_seed75.py",
    "scripts/check_ttmre_n1_full_seed75.py",
    "scripts/run_ttmre_n1_full_seed75.sh",
    "scripts/retry_ttmre_n1_full_seed75_preupdate_failure.sh",
    "scripts/retry_ttmre_n1_full_seed75_initial_model_path_failure.sh",
)


class BudgetStop(TimeoutError):
    """Cooperative stop at the declared experiment limit."""


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def emit(event, **values):
    print(json.dumps({"event": event, **values}, ensure_ascii=False, allow_nan=False), flush=True)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=("validate", "train", "worker"), default="validate")
    p.add_argument("--phase", choices=("prepare", *BRANCHES, "finish"))
    p.add_argument("--baseline-workspace", type=Path, default=DEFAULT_BASELINE)
    p.add_argument("--preflight-report", type=Path)
    p.add_argument("--data-dir", type=Path, default=ROOT / "dataset/docred")
    p.add_argument("--model-dir", type=Path)
    p.add_argument("--run-dir", type=Path)
    p.add_argument("--recover-preupdate-failure", action="store_true",
                    help="Create one fresh workspace only after a verified pre-update failure; retain the original deadline")
    p.add_argument("--recover-repair1-initial-model-path-failure", action="store_true",
                    help="Create one fresh workspace only for the recorded missing branch/initial_model.json directory failure")
    p.add_argument("--check-runtime", action="store_true")
    return p


def protocol():
    return {
        "version": "n1-independent-full-seed75-v1",
        "variant": VARIANT,
        "branches": list(BRANCHES),
        "seed": SEED,
        "distant_seed": SEED,
        "human_seed": SEED,
        "distant_epochs": 2,
        "human_epochs": 30,
        "updates_per_branch": UPDATES_PER_BRANCH,
        "distant_updates_per_branch": DISTANT_UPDATES,
        "human_updates_per_branch": HUMAN_UPDATES,
        "formal_optimizer_updates": FORMAL_UPDATES,
        "diagnostic_optimizer_updates": 0,
        "train_batch_size": 4,
        "dev_batch_size": 4,
        "precision": "float32",
        "relation_loss": "original S-PU",
        "auxiliary_loss": None,
        "new_parameters": 0,
        "new_memory_method": False,
        "new_loss": False,
        "control_distant": "original S-PU in epochs 1 and 2",
        "weighted_distant": "original S-PU epoch 1; existing N1 memory-gradient route epoch 2",
        "weight_formula": preflight.protocol()["raw_weight"],
        "weight_floor": 0.25,
        "teacher": "current student two detached dropout views before its update",
        "n1_view_seed": "750000+one_based_global_distant_step",
        "gradient_route": "grad_M(Rw); original R.backward(); replace only M.grad; global clip1; AdamW",
        "human_rule": "selected Distant checkpoint; only mu_encoder.memory_tokens frozen; original S-PU",
        "selection": "raw Dev F1; penultimate batch; strict increase; earliest tie",
        "warmup_fraction": 0.06,
        "max_grad_norm": 1.0,
        "screening_gate": "Human Weighted-Control >= +0.5pp and both Ign-F1 deltas positive",
        "hard_seconds": HARD_SECONDS,
        "soft_reserve_seconds": RESERVE_SECONDS,
        "term_seconds": TERM_SECONDS,
        "independent_full_two_stage_seed": True,
        "test_data_read": False,
        "automatic_retry": False,
        "automatic_resume": False,
        "automatic_extension": False,
        "automatic_followup": False,
    }


def runtime_modules():
    return baseline, shared


def check_budget(budget, reserve=RESERVE_SECONDS):
    if time.monotonic() >= budget["deadline_monotonic"] - reserve:
        raise BudgetStop("Declared budget reserve reached; no retry/resume/extension")
    if os.name == "posix" and Path("/proc/sys/kernel/random/boot_id").read_text().strip() != budget["boot_id"]:
        raise RuntimeError("Boot identity changed")


def supervise(command, budget, lock_fd=9):
    check_budget(budget)
    child = subprocess.Popen(command, start_new_session=True, pass_fds=(lock_fd,))
    term_sent = False
    try:
        while True:
            status = child.poll()
            if status is not None:
                if status != 0 or term_sent:
                    raise RuntimeError(f"Worker failed/stopped exit={status}; no retry")
                return
            now = time.monotonic()
            if now >= budget["deadline_monotonic"]:
                try:
                    os.killpg(child.pid, signal.SIGKILL)
                finally:
                    child.wait()
                raise BudgetStop("Hard budget reached; worker group killed")
            if now >= budget["deadline_monotonic"] - TERM_SECONDS and not term_sent:
                os.killpg(child.pid, signal.SIGTERM)
                term_sent = True
            time.sleep(0.2)
    finally:
        if child.poll() is None:
            os.killpg(child.pid, signal.SIGKILL)
            child.wait()


def require_lock():
    normal = os.environ.get("TTMRE_N1_FULL75_LOCK_FD") == "9"
    recovery = os.environ.get("TTMRE_N1_FULL75_RECOVERY_LOCK_FD") == "9"
    if os.name != "posix" or normal == recovery:
        raise RuntimeError("Use the one-shot N1 seed75 launcher")
    import fcntl
    lock_path = ROOT / "experiments/2026-10-08-baseline-alignment/.training.lock"
    actual, expected = os.fstat(9), lock_path.stat()
    if (actual.st_dev, actual.st_ino) != (expected.st_dev, expected.st_ino):
        raise RuntimeError("Descriptor9 is not the shared GPU lock")
    fcntl.flock(9, fcntl.LOCK_EX | fcntl.LOCK_NB)
    return 9


def assert_preupdate_failure_state(workspace, failure):
    if failure.get("complete") is not False or failure.get("resumable") is not False:
        raise ValueError("Original attempt is not a recorded non-resumable failure")
    if failure.get("automatic_retry") is not False:
        raise ValueError("Original failure does not explicitly forbid automatic retry")
    if any((workspace / name).exists() for name in ("control", "weighted", "comparison")):
        raise ValueError("A branch/artifact directory exists; this is not a pre-update failure")


def recovery_budget(original_budget, now, boot_id):
    if original_budget.get("boot_id") != boot_id:
        raise ValueError("Original budget boot identity differs; do not reset the deadline")
    remaining = float(original_budget["deadline_monotonic"]) - now
    if remaining <= RESERVE_SECONDS + TERM_SECONDS:
        raise BudgetStop("Original 12-hour deadline has insufficient time remaining")
    return remaining


def recovery_context():
    """Authorize one fresh workspace only for the known pre-update crash."""
    original_claim = RUN_ROOT / "n1-independent-seed75-once-claim.json"
    repair_claim = RUN_ROOT / "n1-independent-seed75-repair-once-claim.json"
    if not original_claim.is_file():
        raise FileNotFoundError("Original failed launch claim is missing")
    if repair_claim.exists():
        raise FileExistsError("The one permitted pre-update recovery was already claimed")
    claim = read(original_claim)
    workspace = Path(claim["workspace"]).expanduser().resolve()
    if workspace == RUN_ROOT.resolve() or not workspace.is_relative_to(RUN_ROOT.resolve()):
        raise ValueError("Original claim points outside the dated N1 root")
    budget_path = workspace / "budget.json"
    failure_path = workspace / "failure.json"
    if not budget_path.is_file() or not failure_path.is_file():
        raise ValueError("Original attempt has no complete budget/failure record")
    old_budget = read(budget_path)
    if claim.get("budget_sha256") != digest(budget_path):
        raise ValueError("Original claim/budget identity mismatch")
    failure = read(failure_path)
    assert_preupdate_failure_state(workspace, failure)
    current_boot = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    remaining = recovery_budget(old_budget, time.monotonic(), current_boot)
    return workspace, old_budget, remaining


def assert_initial_model_path_failure(workspace, failure, failure_control, branch="control"):
    assert_preupdate_failure_state(workspace, failure)
    if failure_control.get("phase") != branch or failure_control.get("type") != "FileNotFoundError":
        raise ValueError("Recovery2 only accepts the recorded branch initial_model path failure")
    message = failure_control.get("error", "")
    if f"/{branch}/initial_model.json" not in message.replace("\\", "/"):
        raise ValueError("Failure is not the missing branch initial_model.json directory")


def recovery_context_initial_model_path_failure():
    """One additional bounded restart for the exact post-init/pre-stage path bug."""
    original_claim_path = RUN_ROOT / "n1-independent-seed75-once-claim.json"
    repair1_claim_path = RUN_ROOT / "n1-independent-seed75-repair-once-claim.json"
    repair2_claim_path = RUN_ROOT / "n1-independent-seed75-repair2-once-claim.json"
    if repair2_claim_path.exists():
        raise FileExistsError("The one permitted initial_model path recovery was already claimed")
    if not original_claim_path.is_file() or not repair1_claim_path.is_file():
        raise FileNotFoundError("Original and first repair claims are both required")
    original_claim, repair1_claim = read(original_claim_path), read(repair1_claim_path)
    original_workspace = Path(original_claim["workspace"]).expanduser().resolve()
    failed_workspace = Path(repair1_claim["workspace"]).expanduser().resolve()
    for workspace in (original_workspace, failed_workspace):
        if workspace == RUN_ROOT.resolve() or not workspace.is_relative_to(RUN_ROOT.resolve()):
            raise ValueError("A prior claim points outside the dated N1 root")
    original_budget_path = original_workspace / "budget.json"
    repair1_budget_path = failed_workspace / "budget.json"
    failure_path = failed_workspace / "failure.json"
    worker_failure_path = failed_workspace / "failure-control.json"
    if not all(path.is_file() for path in (original_budget_path, repair1_budget_path, failure_path, worker_failure_path)):
        raise ValueError("Repair1 is missing its budget or failure evidence")
    original_budget, repair1_budget = read(original_budget_path), read(repair1_budget_path)
    if (original_claim.get("budget_sha256") != digest(original_budget_path)
            or repair1_claim.get("budget_sha256") != digest(repair1_budget_path)
            or repair1_claim.get("recovery_of") != str(original_workspace)
            or repair1_claim.get("attempt") != 2
            or repair1_budget.get("recovery_of_workspace") != str(original_workspace)
            or repair1_budget.get("deadline_monotonic") != original_budget.get("deadline_monotonic")
            or repair1_budget.get("boot_id") != original_budget.get("boot_id")):
        raise ValueError("Original/repair1 claim or retained budget identity mismatch")
    failure, worker_failure = read(failure_path), read(worker_failure_path)
    assert_initial_model_path_failure(failed_workspace, failure, worker_failure)
    current_boot = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    remaining = recovery_budget(original_budget, time.monotonic(), current_boot)
    if remaining < 9 * 60 * 60:
        raise BudgetStop("Less than nine hours remain on the original deadline; do not start a partial pair")
    return failed_workspace, original_budget, remaining


def write_initial_model_report(run_dir, branch, seed, state_sha256):
    branch_dir = Path(run_dir) / branch
    branch_dir.mkdir(parents=True, exist_ok=False)
    path = branch_dir / "initial_model.json"
    shared.write_json(path, {"branch": branch, "seed": seed,
        "generic_initialization": True, "state_sha256": state_sha256, "optimizer_updates": 0})
    return path


def _base_validation(args):
    """Reuse the reviewed N1 input validation without old run-root semantics."""
    probe_args = argparse.Namespace(
        mode="validate", baseline_workspace=args.baseline_workspace,
        preflight_report=args.preflight_report, run_dir=None,
        model_dir=args.model_dir, data_dir=args.data_dir, check_runtime=args.check_runtime,
    )
    base = n1.validate(probe_args)
    args.baseline_workspace, args.model_dir, args.data_dir = (
        probe_args.baseline_workspace, probe_args.model_dir, probe_args.data_dir
    )
    return base


def validate(args, budget=None):
    if args.run_dir is not None:
        args.run_dir = args.run_dir.expanduser().resolve()
        if args.run_dir == RUN_ROOT.resolve() or not args.run_dir.is_relative_to(RUN_ROOT.resolve()):
            raise ValueError("Use a unique workspace below experiments/2026-10-17-n1-independent-seed75")
        if args.run_dir.exists():
            raise FileExistsError(f"Preserve existing run: {args.run_dir}")
    elif args.mode == "train":
        raise ValueError("Training requires --run-dir")
    if budget is not None:
        check_budget(budget)
    base = _base_validation(args)
    source_files = dict(base["source_files"])
    for name in NEW_FILES:
        source_files[name] = digest(ROOT / name)
    identity = {
        **base,
        "protocol": protocol(),
        "variant": VARIANT,
        "source_files": source_files,
        "entrypoint_sha256": digest(Path(__file__)),
        "baseline_workspace": str(args.baseline_workspace),
        "data_dir": str(args.data_dir),
        "model_dir": str(args.model_dir),
        "optimizer_steps": 0,
        "test_data_read": False,
    }
    return identity


def verify_workspace(identity, budget):
    for name, expected in identity["source_files"].items():
        check_budget(budget)
        if digest(ROOT / name) != expected:
            raise ValueError("Source changed: " + name)
    for name, expected in identity.get("input_files", {}).items():
        check_budget(budget)
        if digest(name) != expected:
            raise ValueError("Input changed: " + name)


def prepare(args, budget):
    # The supervisor creates the empty workspace before the prepare worker;
    # validate the inputs without treating that empty directory as a retry.
    run_dir = args.run_dir
    args.run_dir = None
    try:
        identity = validate(args, budget)
    finally:
        args.run_dir = run_dir
    identity["input_files"] = {}
    workspace = args.baseline_workspace
    for split in ("human", "dev", "distant"):
        for filename in ("index.json", "features.bin"):
            path = workspace / "cache" / split / filename
            if path.exists():
                identity["input_files"][str(path)] = digest(path)
    for filename in ("train_revised.json", "dev_revised.json"):
        path = args.data_dir / filename
        identity["input_files"][str(path)] = digest(path)
    for member in identity["model"]["sha256"]:
        path = args.model_dir / member
        identity["input_files"][str(path)] = digest(path)
    required = 8 * 1024**3
    if Path(args.run_dir).anchor and __import__("shutil").disk_usage(args.run_dir).free < required:
        raise RuntimeError("At least 8GiB free disk required; no files deleted")
    identity["required_free_GiB"] = required / 1024**3
    identity["budget_sha256"] = digest(args.run_dir / "budget.json")
    shared.write_json(args.run_dir / "identity.json", identity)
    emit("N1_FULL_SEED75_PREPARED", optimizer_updates=0, test_data_read=False,
         required_free_GiB=identity["required_free_GiB"])


def cache_dataset(args, split, with_labels=True):
    if split not in ("distant", "human", "dev") or with_labels != (split != "dev"):
        raise ValueError("Only Human/Distant train and unlabeled Dev caches are permitted")
    directory = args.baseline_workspace / "cache" / split
    return baseline.data.DiskFeatures(directory, shared.read_json(directory / "index.json"), with_labels)


def expected_environment(identity):
    """Accept either the run identity or its nested baseline identity."""
    if "environment" in identity:
        return identity["environment"]
    return identity["matched_b2_reference"]["environment"]


def load_generic_model_seed75(args, identity):
    """Copy the reviewed generic loader with the task seed changed to 75."""
    import numpy as np
    import torch
    from transformers import AutoConfig, AutoModel, AutoTokenizer
    from model2 import DocREModel
    from ttmre_memory_fit import install_unused_write_bypass, install_ttm_checkpoint

    original = getattr(args, "run_dir", None)
    args.run_dir = args.baseline_workspace
    try:
        baseline.control.runtime(expected_environment(identity))
        for key, value in baseline.aligned.protocol("pair-independent").items():
            setattr(args, key, value)
        config = AutoConfig.from_pretrained(str(args.model_dir), num_labels=97, local_files_only=True)
        tokenizer = AutoTokenizer.from_pretrained(str(args.model_dir), local_files_only=True)
        config.cls_token_id, config.sep_token_id, config.transformer_type = tokenizer.cls_token_id, tokenizer.sep_token_id, "roberta"
        priors = cache_dataset(args, "human").metadata["priors"]
        aligned.emit("generic_backbone_load_start", model_dir=str(args.model_dir), seed=SEED)
        backbone = AutoModel.from_pretrained(str(args.model_dir), config=config, local_files_only=True).to(args.device)
        shared.seed_all(SEED, torch, np)
        model = DocREModel(args, config, torch.tensor(np.array(priors) + 1e-9, device=args.device), backbone, tokenizer).to(args.device)
        baseline.aligned.apply_variant(model, "pair-independent")
        baseline.control.assert_original_read(model)
        if sum(p.numel() for p in model.parameters()) != 381175401:
            raise ValueError("Unexpected baseline model parameters")
        install_unused_write_bypass(model)
        install_ttm_checkpoint(model)
        aligned.emit("task_model_ready", parameters=381175401, generic_initialization=True, seed=SEED)
        return model
    finally:
        args.run_dir = original


def datasets(args):
    human = cache_dataset(args, "human")
    dev = cache_dataset(args, "dev", False)
    distant = cache_dataset(args, "distant")
    if len(human) != 3053 or len(dev) != 500 or len(distant) != 101873:
        raise ValueError("Unexpected cached document counts")
    gold = shared.read_json(args.data_dir / "dev_revised.json")
    annotated = {tuple(x) for x in human.metadata["matching_dev_facts"]}
    distant_facts = {tuple(x) for x in distant.metadata["matching_dev_facts"]}
    buckets = method.frequency_buckets(human.metadata["priors"])
    return human, distant, dev, gold, annotated, distant_facts, buckets


def atomic_torch_save(value, path):
    import torch
    path = Path(path)
    partial = path.with_name(path.name + f".partial-{os.getpid()}-{uuid.uuid4().hex}")
    try:
        torch.save(value, partial)
        with partial.open("r+b") as stream:
            os.fsync(stream.fileno())
        os.replace(partial, path)
    except BaseException:
        # Leave the partial marker for post-mortem inspection.
        raise


def training_forward(model, batch, device):
    import torch
    risks, logits = model(**shared.inputs_from_batch(batch, device))
    loss = sum(risks)
    if not torch.isfinite(loss).all() or not torch.isfinite(logits).all():
        raise RuntimeError("Nonfinite original S-PU forward; no update")
    return loss, logits


def rng_digest():
    """Digest Python/NumPy/Torch RNG state without changing any stream."""
    import pickle
    import random
    import numpy as np
    import torch
    h = hashlib.sha256()
    h.update(pickle.dumps(random.getstate(), protocol=4))
    h.update(pickle.dumps(np.random.get_state(), protocol=4))
    h.update(torch.get_rng_state().cpu().numpy().tobytes())
    if torch.cuda.is_available():
        for state in torch.cuda.get_rng_state_all():
            h.update(state.cpu().numpy().tobytes())
    return h.hexdigest()


def with_seed75_view_seed(function, *args, **kwargs):
    old = method.view_seed
    method.view_seed = lambda step: N1_VIEW_OFFSET + step
    try:
        return function(*args, **kwargs)
    finally:
        method.view_seed = old


def integrated_probe(model, dataset, args, branch, stage):
    """The only pre-optimizer probe for a stage; no optimizer or step is made."""
    if branch == "weighted" and stage == "distant":
        return with_seed75_view_seed(n1.probe, model, dataset, args, stage)
    return baseline.run_probe(model, dataset, args, stage)


def evaluate(model, dev_loader, args, gold, annotated, distant):
    return shared.evaluate_dev(model, dev_loader, args.device, gold, annotated, distant)


def selection_row(epoch, batches, metrics, best):
    score = metrics["dev_F1"]
    if not math.isfinite(score):
        raise RuntimeError("Nonfinite selection metric")
    return {"epoch": epoch, "step": epoch * batches - 1, "batch": batches - 1,
            "selected": score > best, **metrics}


def train_loop(model, loader, dev_loader, args, branch, stage, directory, gold, annotated, distant,
               buckets, budget, optimizer, scheduler):
    import torch
    epochs = baseline.STAGES[stage][1]
    model._n1_completed_optimizer_steps = 0
    memory_before = model.mu_encoder.memory_tokens.detach().cpu().clone()
    best, best_epoch, best_step, steps = -math.inf, None, None, 0
    summary = {}
    learning_check = None
    order = hashlib.sha256()
    with (directory / "train.log").open("x", encoding="utf-8") as log, \
            (directory / "metrics.jsonl").open("x", encoding="utf-8") as metrics_log, \
            (directory / "weight_diagnostics.jsonl").open("x", encoding="utf-8") as detail:
        for epoch in range(1, epochs + 1):
            check_budget(budget)
            total_loss, epoch_start = 0.0, time.perf_counter()
            weighting = branch == "weighted" and method.active(stage, epoch)
            aligned.emit("epoch_start", branch=branch, stage=stage, epoch=epoch,
                         batches=len(loader), weighting_active=weighting)
            for index, batch in enumerate(loader):
                check_budget(budget)
                order.update(batch[0].numpy().tobytes())
                order.update(json.dumps(batch[3:5], separators=(",", ":"), ensure_ascii=False).encode())
                model.train(); model.zero_grad(set_to_none=True)
                collect = weighting and (steps + 1 == len(loader) + 1 or (steps + 1) % 100 == 0 or index + 1 == len(loader))
                if weighting:
                    record = with_seed75_view_seed(
                        method.backward, model, shared.inputs_from_batch(batch, args.device),
                        stage, epoch, steps + 1, collect=collect)
                else:
                    loss, logits = training_forward(model, batch, args.device)
                    loss.backward()
                    record = {"loss": float(loss.detach()), "weighted_risk": float(loss.detach()),
                              "weighting_active": False, "branch_changes": 0}
                if not any(p.grad is not None for p in model.parameters() if p.requires_grad):
                    raise RuntimeError("Missing training gradients; no optimizer step")
                global_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
                witness = None
                if learning_check is None:
                    rates = {id(p): group["lr"] for group in optimizer.param_groups for p in group["params"]}
                    # The original scheduler starts at LR zero. Observe the first
                    # positive-LR update without changing RNG or optimization.
                    candidates = sorted(model.named_parameters(), key=lambda item: item[0] != "head_extractor.weight")
                    for name, parameter in candidates:
                        if (parameter.requires_grad and parameter.grad is not None and rates[id(parameter)] > 0
                                and torch.count_nonzero(parameter.grad).item() > 0):
                            witness = (name, parameter, parameter.detach().clone(), rates[id(parameter)])
                            break
                optimizer.step(); scheduler.step(); optimizer.zero_grad(set_to_none=True)
                steps += 1; total_loss += record["loss"]
                model._n1_completed_optimizer_steps = steps
                if witness is not None:
                    name, parameter, before, rate = witness
                    changed = not torch.equal(before, parameter.detach())
                    state_step = int(optimizer.state[parameter]["step"].item())
                    if not changed or state_step != steps:
                        raise RuntimeError("Positive-LR optimizer update did not change the observed parameter")
                    learning_check = {"passed": True, "branch": branch, "stage": stage,
                        "step": steps, "parameter": name, "lr": rate, "optimizer_state_step": state_step,
                        "parameter_changed": changed, "extra_optimizer_updates": 0}
                    shared.write_json(directory / "learning_update_check.json", learning_check)
                    emit("N1_FULL_SEED75_LEARNING_UPDATE_VERIFIED", **learning_check)
                    del before, witness
                if weighting:
                    method.add_statistics(summary, record, buckets)
                if collect:
                    detail.write(json.dumps({"branch": branch, "stage": stage, "epoch": epoch, "step": steps,
                        "batch": index + 1, "global_norm_before_clip": float(global_norm),
                        "clip_coefficient": min(1., 1. / (float(global_norm) + 1e-6)), **record}, allow_nan=False) + "\n")
                    detail.flush()
                if steps == 1 or steps % 100 == 0:
                    row = {"event": "optimizer_update", "branch": branch, "stage": stage,
                           "epoch": epoch, "step": steps, "batch": index + 1,
                           "loss": record["loss"], "weighted_risk": record["weighted_risk"],
                           "weighting_active": record["weighting_active"]}
                    line = json.dumps(row, allow_nan=False); print(line, flush=True)
                    log.write(line + "\n"); log.flush()
                del record
                if not weighting:
                    del loss, logits
                if hasattr(model, "sims"):
                    model.sims = None
                if index + 1 == len(loader) - 1:
                    check_budget(budget)
                    values = evaluate(model, dev_loader, args, gold, annotated, distant)
                    row = selection_row(epoch, len(loader), values, best)
                    row.update({"branch": branch, "stage": stage})
                    if row["selected"]:
                        best, best_epoch, best_step = row["dev_F1"], epoch, steps
                        atomic_torch_save(shared.cpu_snapshot(model), directory / "best_model.pth")
                        shared.write_json(directory / "best_metrics.json", {k: v for k, v in row.items() if k != "selected"})
                    row["train_loss_so_far"] = total_loss / (index + 1)
                    line = json.dumps(row, allow_nan=False); print(line, flush=True)
                    for handle in (log, metrics_log):
                        handle.write(line + "\n"); handle.flush()
            if stage == "human-frozen" and not torch.equal(memory_before, model.mu_encoder.memory_tokens.detach().cpu()):
                raise RuntimeError("Frozen Human memory changed")
            aligned.emit("epoch_complete", branch=branch, stage=stage, epoch=epoch, step=steps,
                         train_loss=total_loss / len(loader), seconds=time.perf_counter() - epoch_start)
    if learning_check is None:
        raise RuntimeError("No verified positive-LR learning update in completed stage")
    shared.write_json(directory / "weight_summary.json", {
        "branch": branch, "stage": stage, "coverage": summary, "frequency_buckets": buckets,
        "weighting_active": branch == "weighted" and stage == "distant" and epochs >= 2,
        "interpretation": "N1 proxy diagnostics; no calibrated noise truth claim",
    })
    return {"best_epoch": best_epoch, "best_step": best_step, "best_dev_F1": best,
            "optimizer_steps": steps, "memory_unchanged": bool(torch.equal(memory_before, model.mu_encoder.memory_tokens.detach().cpu())),
            "selected_after_weighting": bool(branch == "weighted" and stage == "distant" and best_epoch == 2),
            "weighting_enabled": bool(branch == "weighted" and stage == "distant"),
            "learning_update_check_passed": learning_check["passed"],
            "sample_order_sha256": order.hexdigest(), "final_rng_sha256": rng_digest(),
            "test_data_read": False, "test_evaluated": False}


def verify_stage(directory, identity, branch, stage):
    directory = Path(directory)
    manifest = read(directory / "manifest.json")
    complete = read(directory / "completion.json")
    best = read(directory / "best_metrics.json")
    replay = read(directory / "checkpoint_replay.json")
    batches, epochs = baseline.STAGES[stage][0] // 4, baseline.STAGES[stage][1]
    expected_weighting = branch == "weighted" and stage == "distant"
    required = (manifest.get("protocol") == protocol()
        and manifest.get("identity_sha256") == digest(Path(directory).parents[2] / "identity.json")
        and manifest.get("branch") == branch
        and manifest.get("stage") == stage and manifest.get("weighting_enabled") == expected_weighting
        and complete.get("optimizer_steps") == batches * epochs
        and complete.get("checkpoint_replay_passed") is True
        and complete.get("frequency_diagnostics_passed") is True
        and complete.get("learning_update_check_passed") is True
        and complete.get("test_data_read") is False and complete.get("test_evaluated") is False
        and replay.get("passed") is True and replay.get("optimizer_steps") == 0
        and replay.get("test_data_read") is False and replay.get("test_evaluated") is False
        and replay.get("invariance", {}).get("passed") is True
        and best.get("batch") == batches - 1 and 1 <= best.get("epoch", 0) <= epochs
        and best.get("step") == best["epoch"] * batches - 1
        and complete.get("best_epoch") == best["epoch"] and complete.get("best_step") == best["step"]
        and complete.get("best_dev_F1") == best["dev_F1"]
        and replay.get("epoch") == best["epoch"] and replay.get("step") == best["step"]
        and all(replay.get("metrics", {}).get(k) == best.get(k) for k in baseline.m1.METRIC_KEYS))
    if not required:
        raise ValueError(f"Incomplete {branch}/{stage} stage")
    learning = read(directory / "learning_update_check.json")
    if (learning.get("passed") is not True or learning.get("parameter_changed") is not True
            or learning.get("branch") != branch or learning.get("stage") != stage
            or not 1 <= learning.get("step", 0) <= batches * epochs
            or learning.get("optimizer_state_step") != learning["step"]
            or not math.isfinite(learning.get("lr", float("nan"))) or learning["lr"] <= 0
            or learning.get("extra_optimizer_updates") != 0):
        raise ValueError("Invalid positive-LR learning update evidence")
    rows = [json.loads(line) for line in (directory / "metrics.jsonl").read_text(encoding="utf-8").splitlines()]
    if (len(rows) != epochs or any(r.get("branch") != branch or r.get("stage") != stage or r.get("epoch") != i
            or r.get("step") != i * batches - 1 or r.get("batch") != batches - 1
            or not math.isfinite(r["dev_F1"]) for i, r in enumerate(rows, 1))):
        raise ValueError(f"Invalid {branch}/{stage} trajectory")
    best_row = max(rows, key=lambda r: r["dev_F1"])
    if best_row["epoch"] != best["epoch"] or best_row["dev_F1"] != best["dev_F1"]:
        raise ValueError(f"Selection mismatch in {branch}/{stage}")
    if stage == "distant" and complete.get("selected_after_weighting") != (branch == "weighted" and best["epoch"] == 2):
        raise ValueError("Distant weighting attribution mismatch")
    if stage == "human-frozen" and complete.get("memory_unchanged") is not True:
        raise ValueError("Human memory freeze failed")
    if read(directory / "frequency_diagnostics.json").get("passed") is not True:
        raise ValueError("Frequency diagnostics missing")
    order = read(directory / "training_order.json")
    if order.get("formal_optimizer_updates") != batches * epochs or order.get("seed") != SEED:
        raise ValueError("Training-order record is incomplete")
    checksum = digest(directory / "best_model.pth")
    if checksum != manifest.get("best_checkpoint_sha256") or checksum != replay.get("checkpoint_sha256"):
        raise ValueError("Best checkpoint hash mismatch")
    return {"path": str(directory), "checkpoint_sha256": checksum,
            "completion_sha256": digest(directory / "completion.json"),
            "best_metrics": best, "selected_after_weighting": complete.get("selected_after_weighting", False)}


def train_stage(model, args, identity, branch, stage, dev_loader, gold, annotated, distant, buckets, parent, budget):
    import numpy as np
    import torch
    from torch.utils.data import DataLoader
    from transformers.optimization import get_linear_schedule_with_warmup

    model._n1_completed_optimizer_steps = 0
    directory = args.run_dir / branch / stage / "artifacts"
    directory.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    baseline.set_stage(model, stage)
    memory_initial = model.mu_encoder.memory_tokens.detach().cpu().clone()
    manifest = {"protocol": protocol(), "identity_sha256": digest(args.run_dir / "identity.json"),
                "branch": branch, "stage": stage, "weighting_enabled": branch == "weighted" and stage == "distant",
                "memory_frozen": stage == "human-frozen", "distant_initialization_reference": parent,
                "initial_state_sha256": preflight.state_digest(model), "test_data_read": False}
    shared.write_json(directory / "manifest.json", manifest)
    try:
        dataset = cache_dataset(args, "distant" if stage == "distant" else "human")
        probe_started = time.perf_counter()
        probe_result = integrated_probe(model, dataset, args, branch, stage)
        shared.write_json(directory / "full_model_check.json", probe_result)
        if probe_result.get("optimizer_steps") != 0:
            raise RuntimeError("Probe performed optimizer updates")
        probe_seconds = time.perf_counter() - probe_started
        optimizer = baseline.optimizer_for_stage(model, stage)
        if optimizer.state:
            raise RuntimeError("Optimizer must start empty")
        groups = baseline.optimizer_report(model, optimizer)
        shared.seed_all(SEED, torch, np)
        loader = DataLoader(dataset, batch_size=4, shuffle=True, drop_last=True, collate_fn=shared.collate_fn)
        total = len(loader) * baseline.STAGES[stage][1]
        expected_total = DISTANT_UPDATES if stage == "distant" else HUMAN_UPDATES
        if total != expected_total:
            raise ValueError("Unexpected stage update budget")
        scheduler = get_linear_schedule_with_warmup(optimizer, int(total * 0.06), total)
        manifest.update({"parameters": sum(p.numel() for p in model.parameters()), "total_steps": total,
                         "batches_per_epoch": len(loader), "warmup_steps": int(total * 0.06),
                         "optimizer_groups": groups, "initial_rng_sha256": rng_digest()})
        shared.write_json(directory / "manifest.json", manifest)
        complete = train_loop(model, loader, dev_loader, args, branch, stage, directory,
                              gold, annotated, distant, buckets, budget, optimizer, scheduler)
        training_seconds = time.perf_counter() - started - probe_seconds
        shared.write_json(directory / "training_order.json", {
            "sample_order_sha256": complete["sample_order_sha256"],
            "final_rng_sha256": complete["final_rng_sha256"],
            "formal_optimizer_updates": complete["optimizer_steps"],
            "seed": SEED,
        })
        del optimizer, scheduler
        if complete["optimizer_steps"] != total:
            raise RuntimeError("Incomplete stage update budget")
        model.zero_grad(set_to_none=True)
        replay_started = time.perf_counter()
        replay = baseline.replay(model, directory, dev_loader, args, stage, gold, annotated, distant)
        replay_seconds = time.perf_counter() - replay_started
        if stage == "human-frozen" and not torch.equal(memory_initial, model.mu_encoder.memory_tokens.detach().cpu()):
            raise RuntimeError("Frozen Human checkpoint altered memory")
        shared.write_json(directory / "checkpoint_replay.json", replay)
        best = read(directory / "best_metrics.json")
        frequency_started = time.perf_counter()
        frequency = n1.frequency_report(model, dev_loader, args, gold, annotated, distant, buckets, best)
        frequency_seconds = time.perf_counter() - frequency_started
        if frequency.get("passed") is not True:
            raise RuntimeError("Frequency diagnostics failed")
        shared.write_json(directory / "frequency_diagnostics.json", frequency)
        manifest["best_checkpoint_sha256"] = replay["checkpoint_sha256"]
        manifest["selected_after_weighting"] = complete["selected_after_weighting"]
        manifest["frequency_diagnostics_passed"] = True
        shared.write_json(directory / "manifest.json", manifest)
        shared.write_json(directory / "timing.json", {"elapsed_seconds": time.perf_counter() - started,
            "phases_seconds": {"probe": probe_seconds, "training_and_setup": training_seconds,
                               "checkpoint_replay": replay_seconds, "frequency_diagnostics": frequency_seconds}})
        shared.write_json(directory / "completion.json", {"stage": stage, "branch": branch, **complete,
            "checkpoint_replay_passed": True, "distant_initialization_reference": parent,
            "best_checkpoint_sha256": replay["checkpoint_sha256"], "test_data_read": False,
            "test_evaluated": False, "frequency_diagnostics_passed": True,
            "interpretation": "single independent seed75 screening run"})
        aligned.emit("N1_FULL_SEED75_STAGE_COMPLETE", branch=branch, stage=stage,
                     optimizer_updates=complete["optimizer_steps"], best_epoch=best["epoch"],
                     selected_after_weighting=complete["selected_after_weighting"])
    except BaseException as error:
        shared.write_json(directory / "failure.json", {"error_type": type(error).__name__,
            "error": str(error), "time_utc": datetime.now(timezone.utc).isoformat(),
            "optimizer_step_calls_completed": getattr(model, "_n1_completed_optimizer_steps", 0),
            "learning_update_check_present": (directory / "learning_update_check.json").exists(),
            "resumable": False, "automatic_retry": False})
        raise


def run_branch(args, identity, budget, branch):
    import torch
    from torch.utils.data import DataLoader
    started = time.monotonic()
    human, distant, dev, gold, annotated, distant_facts, buckets = datasets(args)
    model = load_generic_model_seed75(args, identity["baseline_identity"])
    args.device = next(model.parameters()).device
    initial_state = preflight.state_digest(model)
    write_initial_model_report(args.run_dir, branch, SEED, initial_state)
    dev_loader = DataLoader(dev, batch_size=4, shuffle=False, collate_fn=shared.collate_fn)
    parent = None
    train_stage(model, args, identity, branch, "distant", dev_loader, gold, annotated, distant_facts, buckets, parent, budget)
    d_dir = args.run_dir / branch / "distant" / "artifacts"
    d_ref = verify_stage(d_dir, identity, branch, "distant")
    check_budget(budget)
    model.load_state_dict(torch.load(d_dir / "best_model.pth", map_location="cpu", weights_only=True), strict=True)
    aligned.emit("human_initialization_loaded", branch=branch, checkpoint_sha256=d_ref["checkpoint_sha256"],
                 selected_after_weighting=d_ref["selected_after_weighting"])
    parent = d_ref
    train_stage(model, args, identity, branch, "human-frozen", dev_loader, gold, annotated, distant_facts, buckets, parent, budget)
    h_dir = args.run_dir / branch / "human-frozen" / "artifacts"
    h_ref = verify_stage(h_dir, identity, branch, "human-frozen")
    branch_result = {"passed": True, "branch": branch, "seed": SEED,
        "formal_optimizer_updates": UPDATES_PER_BRANCH, "diagnostic_optimizer_updates": 0,
        "initial_model_sha256": initial_state, "distant": d_ref, "human": h_ref,
        "distant_best": d_ref["best_metrics"], "human_best": h_ref["best_metrics"],
        "test_data_read": False, "automatic_retry": False,
        "elapsed_seconds": time.monotonic() - started}
    shared.write_json(args.run_dir / branch / "completion.json", branch_result)
    emit("N1_FULL_SEED75_BRANCH_COMPLETE", branch=branch, optimizer_updates=UPDATES_PER_BRANCH,
         distant_F1=d_ref["best_metrics"]["dev_F1"], human_F1=h_ref["best_metrics"]["dev_F1"])


def verify_selection_files(args, identity, budget):
    for branch in BRANCHES:
        check_budget(budget)
        result = read(args.run_dir / branch / "completion.json")
        if not result.get("passed") or result.get("formal_optimizer_updates") != UPDATES_PER_BRANCH:
            raise ValueError("Branch completion missing: " + branch)
        for stage in ("distant", "human-frozen"):
            verify_stage(args.run_dir / branch / stage / "artifacts", identity, branch, stage)


def paired_delta(control, weighted):
    keys = ("dev_F1", "dev_precision", "dev_recall", "dev_IgnF1_annotated", "dev_IgnF1_distant")
    if not all(math.isfinite(float(metrics[key])) for metrics in (control, weighted) for key in keys):
        raise RuntimeError("Nonfinite paired metric")
    return {key: weighted[key] - control[key] for key in keys}


def finish(args, identity, budget):
    import torch
    verify_workspace(identity, budget)
    verify_selection_files(args, identity, budget)
    control = read(args.run_dir / "control" / "completion.json")
    weighted = read(args.run_dir / "weighted" / "completion.json")
    if control["initial_model_sha256"] != weighted["initial_model_sha256"]:
        raise ValueError("Control/Weighted generic initialization differs")
    # Every stage's optimizer and data stream must remain paired.  Weighting's
    # extra views restore the student RNG, so this is a causal route comparison.
    for stage in ("distant", "human-frozen"):
        c_dir = args.run_dir / "control" / stage / "artifacts"
        w_dir = args.run_dir / "weighted" / stage / "artifacts"
        c = read(c_dir / "manifest.json")
        w = read(w_dir / "manifest.json")
        c_order = read(c_dir / "training_order.json")
        w_order = read(w_dir / "training_order.json")
        if (c["optimizer_groups"] != w["optimizer_groups"]
                or c["batches_per_epoch"] != w["batches_per_epoch"]
                or c["initial_rng_sha256"] != w["initial_rng_sha256"]
                or c_order["sample_order_sha256"] != w_order["sample_order_sha256"]
                or c_order["final_rng_sha256"] != w_order["final_rng_sha256"]):
            raise ValueError("Paired optimizer, RNG or sample-order protocol differs")
    d_control, d_weighted = control["distant_best"], weighted["distant_best"]
    h_control, h_weighted = control["human_best"], weighted["human_best"]
    d_delta, h_delta = paired_delta(d_control, d_weighted), paired_delta(h_control, h_weighted)
    screening = h_delta["dev_F1"] >= 0.5 and h_delta["dev_IgnF1_annotated"] > 0 and h_delta["dev_IgnF1_distant"] > 0
    result = {
        "passed": True,
        "marker": "N1_FULL_SEED75_COMPLETE",
        "protocol": protocol(),
        "formal_optimizer_updates": FORMAL_UPDATES,
        "diagnostic_optimizer_updates": 0,
        "metrics": {"distant": {"control": d_control, "weighted": d_weighted},
                    "human": {"control": h_control, "weighted": h_weighted}},
        "delta_pp": {"distant": d_delta, "human": h_delta},
        "positive_human_signal": h_delta["dev_F1"] > 0,
        "screening_gate_passed": screening,
        "reliable_improvement_established": False,
        "independent_full_two_stage_seed": True,
        "selected_after_weighting": weighted["distant"]["selected_after_weighting"],
        "test_evaluated": False,
        "automatic_followup": False,
        "stop_condition": "<=0pp stops N1; 0..0.5pp weak candidate only; >=0.5pp with both Ign-F1 positive requires review before any new run",
        "interpretation": "One independent paired seed; positive result is screening evidence, not reliable multi-seed improvement",
        "elapsed_seconds": time.monotonic() - budget["start_monotonic"],
    }
    shared.write_json(args.run_dir / "comparison" / "completion.json", result)
    emit("N1_FULL_SEED75_COMPLETE", formal_optimizer_updates=FORMAL_UPDATES,
         human_delta_pp=h_delta["dev_F1"], screening_gate_passed=screening,
         selected_after_weighting=result["selected_after_weighting"])


def worker(args):
    require_lock()
    budget = read(args.run_dir / "budget.json")
    check_budget(budget)
    try:
        if args.phase == "prepare":
            prepare(args, budget); return
        identity = read(args.run_dir / "identity.json")
        if identity.get("budget_sha256") != digest(args.run_dir / "budget.json") or identity.get("protocol") != protocol():
            raise ValueError("Run identity/budget changed")
        args.baseline_workspace = Path(identity["baseline_workspace"])
        args.model_dir = Path(identity["model_dir"])
        args.data_dir = Path(identity["data_dir"])
        args.device = baseline.control.runtime(identity["environment"])
        verify_workspace(identity, budget)
        if args.phase == "finish":
            (args.run_dir / "comparison").mkdir(exist_ok=False)
            finish(args, identity, budget)
        elif args.phase in BRANCHES:
            if (args.run_dir / args.phase / "completion.json").exists():
                raise FileExistsError("Preserve completed branch; no resume")
            run_branch(args, identity, budget, args.phase)
        else:
            raise ValueError("Unknown worker phase")
    except BaseException as error:
        shared.write_json(args.run_dir / f"failure-{args.phase}.json", {
            "phase": args.phase, "error": str(error), "type": type(error).__name__,
            "complete": False, "resumable": False, "automatic_retry": False,
        })
        raise


def train(args):
    if os.name != "posix" or args.run_dir is None:
        raise RuntimeError("Linux unique workspace required; use the launcher")
    lock_fd = require_lock()
    args.run_dir = args.run_dir.expanduser().resolve()
    if args.run_dir == RUN_ROOT.resolve() or not args.run_dir.is_relative_to(RUN_ROOT.resolve()):
        raise ValueError("Output must be inside the new dated root")
    recovery_flags = (args.recover_preupdate_failure, args.recover_repair1_initial_model_path_failure)
    if sum(recovery_flags) > 1:
        raise ValueError("Recovery modes are mutually exclusive")
    recovery = None
    if args.recover_preupdate_failure:
        if os.environ.get("TTMRE_N1_FULL75_RECOVERY_LOCK_FD") != "9":
            raise RuntimeError("Recovery mode requires the dedicated recovery launcher")
        recovery = recovery_context()
    elif args.recover_repair1_initial_model_path_failure:
        if os.environ.get("TTMRE_N1_FULL75_RECOVERY_LOCK_FD") != "9":
            raise RuntimeError("Recovery2 requires the dedicated recovery launcher")
        recovery = recovery_context_initial_model_path_failure()
    elif os.environ.get("TTMRE_N1_FULL75_RECOVERY_LOCK_FD") == "9":
        raise RuntimeError("Recovery launcher must select its exact failure mode")
    begin = time.monotonic()
    if not recovery:
        claim_path = RUN_ROOT / "n1-independent-seed75-once-claim.json"
        attempt = 1
    elif args.recover_preupdate_failure:
        claim_path = RUN_ROOT / "n1-independent-seed75-repair-once-claim.json"
        attempt = 2
    else:
        claim_path = RUN_ROOT / "n1-independent-seed75-repair2-once-claim.json"
        attempt = 3
    if recovery:
        shared.write_json(claim_path, {
            "state": "claimed", "workspace": str(args.run_dir),
            "recovery_of": str(recovery[0]), "attempt": attempt,
            "automatic_retry": False,
        })
    args.run_dir.mkdir(parents=True, exist_ok=False)
    if __import__("shutil").disk_usage(args.run_dir).free < 8 * 1024**3:
        raise RuntimeError("At least 8GiB free disk required; no files deleted")
    if recovery is None:
        budget = {
            "start_monotonic": begin, "deadline_monotonic": begin + HARD_SECONDS,
            "hard_seconds": HARD_SECONDS, "start_utc": datetime.now(timezone.utc).isoformat(),
            "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
            "soft_reserve_seconds": RESERVE_SECONDS, "term_reserve_seconds": TERM_SECONDS,
            "automatic_retry": False, "automatic_resume": False, "extension": False,
        }
    else:
        previous_workspace, original_budget, remaining = recovery
        budget = {**original_budget, "recovery_attempt_start_utc": datetime.now(timezone.utc).isoformat(),
                  "recovery_of_workspace": str(previous_workspace),
                  "recovery_remaining_seconds_at_start": remaining,
                  "recovery_attempt": attempt}
    shared.write_json(args.run_dir / "budget.json", budget)
    shared.write_json(claim_path, {
        "workspace": str(args.run_dir), "budget_sha256": digest(args.run_dir / "budget.json"),
        "protocol": protocol(), "automatic_retry": False,
        "attempt": attempt,
        "recovery_of": str(recovery[0]) if recovery else None,
    })
    command = [sys.executable, "-u", "-B", str(Path(__file__).resolve()), "--mode", "worker",
               "--run-dir", str(args.run_dir)]
    def terminate(*_):
        raise BudgetStop("Supervisor interrupted; no retry/resume/extension")
    signal.signal(signal.SIGTERM, terminate)
    try:
        emit("N1_FULL_SEED75_BUDGET_STARTED", **budget)
        supervise([sys.executable, "-B", str(ROOT / "scripts/check_ttmre_n1_full_seed75.py")], budget, lock_fd)
        for phase in ("prepare", *BRANCHES, "finish"):
            supervise(command + ["--phase", phase], budget, lock_fd)
    except BaseException as error:
        shared.write_json(args.run_dir / "failure.json", {
            "error": str(error), "type": type(error).__name__, "complete": False,
            "resumable": False, "automatic_retry": False,
            "elapsed_seconds": time.monotonic() - begin,
        })
        raise


def main():
    args = parser().parse_args()
    os.chdir(ROOT)
    if args.mode == "train":
        train(args)
    elif args.mode == "worker":
        worker(args)
    else:
        identity = validate(args)
        print(json.dumps({"validation": "passed", "identity": identity, "optimizer_steps": 0,
                          "files_written": False, "test_data_read": False}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
