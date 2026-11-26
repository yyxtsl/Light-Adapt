"""One explicitly authorized seed75 ATGL Human run, total cap4h.

Default validate is read-only. Use run_ttmre_atgl_human.sh once for training.
No Distant optimization, Test, resume, retry, extension or automatic follow-up.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
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
sys.path.insert(0, str(ROOT/"scripts"))
import train_ttmre_nonempty_head as frozen
import train_ttmre_n1_full_seed75 as core
import train_ttmre_n1_weighted_completion as accepted

RUN_ROOT = ROOT/"experiments/2026-10-19-atgl-human"
PARENT = frozen.DONE/"control/distant/artifacts/best_model.pth"
PARENT_SHA = "4fd5aca22c20584dde442c2d80dba8a72db8fe20a4b9fa920fa7c43a7fe13ce9"
CONTROL_ORDER = "5dac02babee79230a2676bee28ae8c2ec9d21ddf7c6bdfa0ca667c1f0045798d"
HARD_SECONDS, SOFT_SECONDS, TERM_SECONDS = 14400, 120, 30
TRAIN_DOCS, DEV_DOCS, EPOCHS = 3053, 500, 30
FILES = tuple("scripts/"+n for n in ("train_ttmre_atgl_human.py",
    "check_ttmre_atgl_human.py", "run_ttmre_atgl_human.sh"))
read, digest, emit = frozen.read, frozen.digest, frozen.emit
check_budget, supervise = frozen.check_budget, frozen.supervise
write_json = core.shared.write_json


def protocol():
    return {"version": "atgl-human-v1", "seed": 75, "alpha": 16.5, "epochs": EPOCHS,
        "optimizer_calls": (TRAIN_DOCS//4)*EPOCHS, "train_batch_size": 4, "dev_batch_size": 4,
        "precision": "float32", "parent_sha256": PARENT_SHA,
        "loss": "-mean(sum((TH_mask+16.5*positive_relation_mask)*log_softmax(logits)))",
        "decoder": "all relation logits > TH; no top-k cap",
        "freeze": ["mu_encoder.memory_tokens"], "backbone_lr": 3e-5, "head_lr": 1e-5,
        "eps": 1e-6, "weight_decay": .01, "warmup_steps": int((TRAIN_DOCS//4)*EPOCHS*.06), "clip": 1.,
        "selection": "penultimate batch raw Dev F1; earliest tie",
        "hard_seconds": HARD_SECONDS, "soft_reserve_seconds": SOFT_SECONDS,
        "term_reserve_seconds": TERM_SECONDS, "acceptance": accepted.POLICY,
        "test_data_read": False, "distant_optimizer_steps": 0,
        "retry": False, "resume": False, "extension": False, "automatic_followup": False}


def atgl_loss(logits, labels):
    import torch
    target = labels.to(device=logits.device, dtype=logits.dtype).clone()
    target[:, 0] = 0
    target.mul_(16.5)
    target[:, 0] = 1
    return -(torch.nn.functional.log_softmax(logits, -1)*target).sum(-1).mean()


def training_forward(model, batch, device):
    import torch
    # Labels are withheld from model2: its original S-PU code is not executed.
    logits = model(**core.shared.inputs_from_batch(batch, device, labels=False))
    labels = torch.cat([torch.as_tensor(y, device=device) for y in batch[2]], dim=0)
    loss = atgl_loss(logits, labels)
    if not torch.isfinite(loss) or not torch.isfinite(logits).all():
        raise RuntimeError("Nonfinite ATGL loss/logits; no optimizer call")
    return loss, logits


def require_lock():
    if os.name != "posix" or os.environ.get("TTMRE_ATGL_HUMAN_LOCK_FD") != "9":
        raise RuntimeError("Use run_ttmre_atgl_human.sh")
    import fcntl
    a = os.fstat(9)
    b = (ROOT/"experiments/2026-10-08-baseline-alignment/.training.lock").stat()
    if (a.st_dev, a.st_ino) != (b.st_dev, b.st_ino):
        raise RuntimeError("Wrong shared lock descriptor")
    fcntl.flock(9, fcntl.LOCK_EX | fcntl.LOCK_NB)
    return 9


def validate(budget=None):
    parent = frozen.validate(budget)
    if digest(PARENT, budget) != PARENT_SHA:
        raise ValueError("Original Control75 Distant checkpoint mismatch")
    complete = read(frozen.DONE/"control/completion.json")
    if complete["distant"]["checkpoint_sha256"] != PARENT_SHA:
        raise ValueError("Distant reference differs")
    parent.update({"protocol": protocol(), "distant_checkpoint": str(PARENT),
        "distant_checkpoint_sha256": PARENT_SHA, "control_metrics": parent["control_best"],
        "frozen_head_atgl_F1": 79.10208114617427})
    parent["source_files"].update({n: digest(ROOT/n) for n in FILES})
    parent["input_files"][str(PARENT)] = PARENT_SHA
    return parent


def verify_sources(identity, budget):
    if identity["protocol"] != protocol():
        raise ValueError("ATGL protocol changed")
    for name, sha in identity["source_files"].items():
        if digest(ROOT/name, budget) != sha:
            raise ValueError("Source changed: "+name)


def probe(model, dataset, args, budget):
    import torch
    from ttmre_memory_fit import reserve_adam_moments
    before = core.preflight.state_digest(model)
    rng = core.preflight.rng_state()
    buffers, reservation = reserve_adam_moments(model)
    reports = []
    try:
        for name, indices in dataset.probe_indices().items():
            check_budget(budget)
            model.train(); model.zero_grad(set_to_none=True)
            loss, logits = training_forward(model, core.shared.collate_fn([dataset[i] for i in indices]), args.device)
            loss.backward()
            gradients = [p.grad for p in model.parameters() if p.grad is not None]
            if not gradients or any(not torch.isfinite(g).all() for g in gradients):
                raise RuntimeError("Nonfinite/missing ATGL probe gradients")
            critical = [model.head_extractor.weight, model.bilinear.weight,
                        model.model.embeddings.word_embeddings.weight]
            if any(p.grad is None or not torch.count_nonzero(p.grad) for p in critical):
                raise RuntimeError("Missing backbone/extractor/bilinear ATGL gradient")
            if model.mu_encoder.memory_tokens.grad is not None:
                raise RuntimeError("Frozen memory received gradient")
            reports.append({"probe": name, "indices": indices, "loss": float(loss.detach()),
                            "optimizer_steps": 0, "finite_gradients": len(gradients)})
            emit("ATGL_ZERO_UPDATE_PROBE", **reports[-1])
            del loss, logits, gradients
    finally:
        model.zero_grad(set_to_none=True)
        model.sims = None
        del buffers
        core.preflight.restore_rng(rng)
        if args.device.type == "cuda":
            torch.cuda.empty_cache()
    core.preflight.assert_rng_equal(rng, core.preflight.rng_state())
    if core.preflight.state_digest(model) != before:
        raise RuntimeError("Probe changed parameters/buffers")
    return {"passed": True, "probes": reports, "optimizer_steps": 0,
            "parameters_unchanged": True, "rng_restored": True, "reservation": reservation}


@contextmanager
def training_adapter(budget):
    old_forward, old_emit = core.training_forward, core.aligned.emit
    old_learning_emit = core.emit
    def event(name, **fields):
        old_emit(name, **fields)
        if name == "epoch_complete" and fields["epoch"] == 1:
            estimate = fields["seconds"]*(EPOCHS-1)+600
            remaining = budget["deadline_monotonic"]-time.monotonic()-SOFT_SECONDS
            emit("ATGL_REMAINING_PROJECTION", estimate_seconds=estimate, remaining_seconds=remaining)
            if estimate > remaining:
                raise core.BudgetStop("ATGL measured projection exceeds the four-hour budget")
    def learning(name, **fields):
        emit("ATGL_LEARNING_UPDATE_VERIFIED" if name == "N1_FULL_SEED75_LEARNING_UPDATE_VERIFIED" else name, **fields)
    core.training_forward, core.aligned.emit, core.emit = training_forward, event, learning
    try:
        yield
    finally:
        core.training_forward, core.aligned.emit, core.emit = old_forward, old_emit, old_learning_emit


def execute(workspace, identity, budget):
    import numpy as np
    import torch
    from torch.utils.data import DataLoader
    from transformers.optimization import get_linear_schedule_with_warmup
    verify_sources(identity, budget)
    device = core.baseline.control.runtime(identity["environment"])
    args = SimpleNamespace(run_dir=workspace, baseline_workspace=Path(identity["baseline_workspace"]),
        model_dir=Path(identity["model_dir"]), data_dir=Path(identity["data_dir"]), device=device, budget=budget)
    datasets = frozen.original_features(identity)
    human, dev = datasets["train"], datasets["dev"]
    if len(human) != TRAIN_DOCS or len(dev) != DEV_DOCS:
        raise ValueError("Human/Dev counts differ")
    gold = read(args.data_dir/"dev_revised.json")
    annotated = {tuple(x) for x in human.metadata["matching_dev_facts"]}
    distant = {tuple(x) for x in read(args.baseline_workspace/"cache/distant/index.json")["matching_dev_facts"]}
    buckets = core.method.frequency_buckets(human.metadata["priors"])
    # Original generic constructor is immediately overwritten by strict Distant
    # load; no Human checkpoint is used for any trainable parameter.
    model = core.load_generic_model_seed75(args, identity["baseline_identity"])
    model.load_state_dict(torch.load(PARENT, map_location="cpu", weights_only=True), strict=True)
    initial = core.preflight.state_digest(model)
    model.requires_grad_(True)
    core.baseline.set_stage(model, "human-frozen")
    memory = model.mu_encoder.memory_tokens.detach().cpu().clone()
    directory = workspace/"atgl/human-frozen/artifacts"
    directory.mkdir(parents=True, exist_ok=False)
    write_json(directory/"initialization.json", {"parent_sha256": PARENT_SHA, "state_sha256": initial,
        "frozen": [n for n,p in model.named_parameters() if not p.requires_grad]})
    write_json(directory/"full_model_check.json", probe(model, human, args, budget))
    optimizer = core.baseline.optimizer_for_stage(model, "human-frozen")
    if optimizer.state:
        raise RuntimeError("Optimizer must start empty")
    groups = core.baseline.optimizer_report(model, optimizer)
    core.shared.seed_all(75, torch, np)
    loader = DataLoader(human, batch_size=4, shuffle=True, drop_last=True, collate_fn=core.shared.collate_fn)
    dev_loader = DataLoader(dev, batch_size=4, shuffle=False, collate_fn=core.shared.collate_fn)
    total = len(loader)*EPOCHS
    if total != protocol()["optimizer_calls"]:
        raise ValueError("Expected22890 calls")
    scheduler = get_linear_schedule_with_warmup(optimizer, int(total*.06), total)
    write_json(directory/"manifest.json", {"protocol": protocol(), "identity_sha256": digest(workspace/"identity.json"),
        "optimizer_groups": groups, "initial_rng_sha256": core.rng_digest(), "total_steps": total})
    with training_adapter(budget):
        result = core.train_loop(model, loader, dev_loader, args, "atgl", "human-frozen", directory,
            gold, annotated, distant, buckets, budget, optimizer, scheduler)
    del optimizer, scheduler
    if result["optimizer_steps"] != total or not result["learning_update_check_passed"]:
        raise RuntimeError("Incomplete actual learning")
    write_json(directory/"training_order.json", {"seed": 75,
        "sample_order_sha256": result["sample_order_sha256"], "formal_optimizer_updates": total,
        "same_as_original_control": result["sample_order_sha256"] == CONTROL_ORDER})
    if result["sample_order_sha256"] != CONTROL_ORDER:
        raise RuntimeError("Human sample order differs from the matched original Control75")
    write_json(directory/"weight_summary.json", {"loss": "ATGL alpha16.5",
        "N1_weighting_active": False, "new_Distant_updates": 0})
    # Keep the already approved exact full-Dev decision acceptance, retaining
    # original strict numeric outcomes separately with unchanged tolerances.
    replay = accepted.replay(model, directory, dev_loader, args, "human-frozen", gold, annotated, distant)
    write_json(directory/"checkpoint_replay.json", replay)
    if not torch.equal(memory, model.mu_encoder.memory_tokens.detach().cpu()):
        raise RuntimeError("Frozen memory differs from Distant parent")
    frequency = core.n1.frequency_report(model, dev_loader, args, gold, annotated, distant, buckets,
                                       read(directory/"best_metrics.json"))
    write_json(directory/"frequency_diagnostics.json", frequency)
    if not frequency.get("passed"):
        raise RuntimeError("Frequency/metric replay failed")
    best = read(directory/"best_metrics.json")
    result.update({"passed": True, "protocol": protocol(), "best": best,
        "parent_sha256": PARENT_SHA, "checkpoint_sha256": digest(directory/"best_model.pth", budget),
        "memory_unchanged": True, "checkpoint_replay_passed": True,
        "frequency_diagnostics_passed": True, "historical_control_strict_numeric_passed": False,
        "interpretation": "one Human-stage existing-loss comparison; not full official ATGL reproduction"})
    write_json(directory/"completion.json", result)
    verify_sources(identity, budget)
    if digest(PARENT, budget) != PARENT_SHA:
        raise RuntimeError("Read-only Distant parent changed")
    delta = best["dev_F1"]-identity["control_metrics"]["dev_F1"]
    comparison = workspace/"comparison"; comparison.mkdir()
    write_json(comparison/"completion.json", {"passed": True, "marker": "ATGL_HUMAN_COMPLETE",
        "protocol": protocol(), "atgl": best, "original_control": identity["control_metrics"],
        "delta_F1_pp": delta, "optimizer_steps": total, "diagnostic_optimizer_steps": 0,
        "elapsed_seconds": time.monotonic()-budget["start_monotonic"],
        "automatic_followup": False, "reliable_improvement_established": False,
        "novelty_established": False, "test_data_read": False, "test_evaluated": False})
    archive = comparison/"reports.tar.gz"
    with tarfile.open(archive, "x:gz") as target:
        for path in workspace.rglob("*"):
            check_budget(budget)
            if path.is_file() and path.suffix in (".json", ".jsonl", ".log"):
                target.add(path, arcname=str(path.relative_to(workspace)), recursive=False)
    emit("ATGL_HUMAN_COMPLETE", delta_F1_pp=delta, F1=best["dev_F1"], reports=str(archive), automatic_followup=False)


def verify_context(workspace, budget):
    claim = read(RUN_ROOT/"atgl-human-once-claim.json")
    if (workspace.name != "workspace" or workspace.parent.parent != RUN_ROOT.resolve()
            or budget["hard_seconds"] != HARD_SECONDS
            or budget["deadline_monotonic"]-budget["start_monotonic"] != HARD_SECONDS
            or any(budget[k] is not False for k in ("automatic_retry", "automatic_resume", "extension"))
            or claim["workspace"] != str(workspace) or claim["protocol"] != protocol()
            or claim["budget_sha256"] != digest(workspace/"budget.json")):
        raise ValueError("Invalid once-claim or immutable4h budget")
    check_budget(budget)


def worker(workspace):
    require_lock()
    budget = read(workspace/"budget.json")
    verify_context(workspace, budget)
    def stop(*_):
        raise core.BudgetStop("Worker stopped; retain partials")
    signal.signal(signal.SIGTERM, stop)
    identity = validate(budget)
    identity["budget_sha256"] = digest(workspace/"budget.json")
    write_json(workspace/"identity.json", identity)
    required = max(8*1024**3, PARENT.stat().st_size*3+1024**3)
    if shutil.disk_usage(workspace).free < required:
        raise RuntimeError("Require8GiB project-volume free; preserve all prior artifacts")
    execute(workspace, identity, budget)


def train(workspace):
    lock = require_lock()
    if workspace.name != "workspace" or workspace.parent.parent != RUN_ROOT.resolve() or workspace.exists():
        raise ValueError("Use a new dated launcher workspace")
    claim = RUN_ROOT/"atgl-human-once-claim.json"
    if claim.exists():
        raise FileExistsError("ATGL authorization already claimed; no retry")
    workspace.mkdir()
    start = time.monotonic()
    budget = {"start_monotonic": start, "deadline_monotonic": start+HARD_SECONDS,
        "hard_seconds": HARD_SECONDS, "soft_reserve_seconds": SOFT_SECONDS,
        "term_reserve_seconds": TERM_SECONDS, "start_utc": datetime.now(timezone.utc).isoformat(),
        "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
        "automatic_retry": False, "automatic_resume": False, "extension": False}
    write_json(workspace/"budget.json", budget)
    with claim.open("x", encoding="utf-8") as handle:
        json.dump({"workspace": str(workspace), "budget_sha256": digest(workspace/"budget.json"),
                   "protocol": protocol(), "authorization": "explicit-user-one-ATGL-Human-cap4h"}, handle, indent=2)
        handle.flush(); os.fsync(handle.fileno())
    write_json(workspace.parent/"pid.json", {"supervisor_pid": os.getpid()})
    def stop(*_):
        raise core.BudgetStop("Supervisor stopped; no retry/resume/extension")
    signal.signal(signal.SIGTERM, stop)
    try:
        emit("ATGL_HUMAN_BUDGET_STARTED", **budget)
        supervise([sys.executable, "-B", str(ROOT/FILES[1])], budget, lock)
        supervise([sys.executable, "-u", "-B", str(Path(__file__).resolve()),
                   "--mode", "worker", "--run-dir", str(workspace)], budget, lock)
    except BaseException as error:
        write_json(workspace/"failure.json", {"error": str(error), "type": type(error).__name__,
            "complete": False, "automatic_retry": False, "resumable": False,
            "elapsed_seconds": time.monotonic()-start})
        raise


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=("validate", "train", "worker"), default="validate")
    p.add_argument("--run-dir", type=Path)
    return p


def main():
    args = parser().parse_args()
    os.chdir(ROOT)
    if args.mode == "validate":
        print(json.dumps(validate(), indent=2))
        return
    if args.run_dir is None:
        raise ValueError("Use launcher workspace")
    path = args.run_dir.resolve()
    if args.mode == "train":
        train(path)
    else:
        try:
            worker(path)
        except BaseException as error:
            write_json(path/"failure-worker.json", {"error": str(error), "type": type(error).__name__,
                "complete": False, "automatic_retry": False, "resumable": False})
            raise


if __name__ == "__main__":
    main()
