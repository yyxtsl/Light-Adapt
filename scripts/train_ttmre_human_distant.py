"""Approved seed74 Distant-only -> Human+Distant baseline; validate is read-only."""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import sys
import time
import zipfile

import train_ttmre_b2_5090 as control
import ttmre_human_distant_data as data

aligned, shared, m1 = control.aligned, control.shared, control.m1
ROOT = shared.ROOT
RUN_ROOT = ROOT / "experiments" / "2026-10-11-human-distant-baseline"
ARCHIVE_SHA256 = "24f304c2612a625183b9cf337ddebc1ceb46ce02bbdc548391978c3cde444f47"
STAGES = {"distant": (101873, 2, 5e-5), "human-frozen": (3053, 30, 1e-5)}
NEW_FILES = ("scripts/ttmre_human_distant_data.py", "scripts/train_ttmre_human_distant.py",
             "scripts/check_ttmre_human_distant.py", "scripts/run_ttmre_human_distant.sh")


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--mode", choices=("validate", "prepare", "train", "replay"), default="validate")
    result.add_argument("--reference-run-dir", type=Path, required=True, help="Completed matched B2 artifacts")
    result.add_argument("--model-dir", type=Path)
    result.add_argument("--data-dir", type=Path, default=ROOT / "dataset" / "docred")
    result.add_argument("--archive", type=Path, default=ROOT / "dataset" / "docred.zip")
    result.add_argument("--run-dir", type=Path, help="New prepared workspace, inside the dated two-stage root")
    result.add_argument("--stage", choices=("both", *STAGES), default="both")
    result.add_argument("--check-runtime", action="store_true")
    return result


def protocol():
    return {**aligned.protocol("pair-independent"), "variant": "two-stage-paper-freeze",
            "protocol_version": "human-priors-b2-distant2-human30-v1",
            "num_train_epochs": {"distant": 2, "human-frozen": 30},
            "head_learning_rate": {"distant": 5e-5, "human-frozen": 1e-5},
            "priors_source": "Human train_revised.json label counts / directed pairs, both stages",
            "distant_only_scope": "Distant optimizer documents only; uses Human training label statistics",
            "memory_freeze": "only mu_encoder.memory_tokens during human-frozen",
            "human_initialization": "strict load selected Distant checkpoint; fresh optimizer/scheduler",
            "distant_initialization": "generic pretrained RoBERTa-large; no task checkpoint",
            "runtime_profile": dict(m1.RUNTIME_PROFILE),
            "execution_profile": m1.protocol()["execution_profile"], "expected_parameters": 381175401,
            "test_evaluation": "separate, not implemented by this entry"}


def verify_reference(directory, current):
    directory = Path(directory).expanduser().resolve()
    saved = shared.read_json(directory / "manifest.json")
    best = shared.read_json(directory / "best_metrics.json")
    complete = shared.read_json(directory / "completion.json")
    replay = shared.read_json(directory / "checkpoint_replay.json")
    if (saved.get("protocol") != control.protocol() or saved.get("parameters") != 381175401
            or saved.get("total_steps") != 22890 or saved.get("batches_per_epoch") != 763
            or complete.get("optimizer_steps") != 22890 or complete.get("variant") != "pair-independent"
            or complete.get("checkpoint_replay_passed") is not True
            or complete.get("test_data_read") is not False or complete.get("test_evaluated") is not False
            or replay.get("passed") is not True or replay.get("invariance", {}).get("passed") is not True
            or replay.get("optimizer_steps") != 0 or replay.get("test_data_read") is not False
            or replay.get("test_evaluated") is not False):
        raise ValueError("Reference is not the completed matched B2 control")
    for name, checksum in saved["source_files"].items():
        if shared.sha256(ROOT / name) != checksum:
            raise ValueError(f"Frozen reference source changed: {name}")
    if saved.get("entrypoint_sha256") != shared.sha256(ROOT / "scripts/train_ttmre_b2_5090.py"):
        raise ValueError("Reference entrypoint changed")
    if saved["model"]["sha256"] != current["model"]["sha256"]:
        raise ValueError("Generic backbone differs from matched B2")
    for split in shared.SPLITS:
        if saved["data"][split]["sha256"] != current["data"][split]["sha256"]:
            raise ValueError("Reference Human/Dev split identity mismatch")
    checksum = shared.sha256(directory / "best_model.pth")
    if saved.get("best_checkpoint_sha256") != checksum or replay.get("checkpoint_sha256") != checksum:
        raise ValueError("Reference checkpoint hash mismatch")
    if (not 1 <= best["epoch"] <= 30 or best.get("batch") != 762
            or best["step"] != (best["epoch"] - 1) * 763 + 762
            or complete.get("best_epoch") != best["epoch"] or complete.get("best_step") != best["step"]
            or complete.get("best_dev_F1") != best["dev_F1"]
            or replay.get("epoch") != best["epoch"] or replay.get("step") != best["step"]
            or any(replay.get("metrics", {}).get(k) != best.get(k) for k in m1.METRIC_KEYS)):
        raise ValueError("Reference selection or metrics mismatch")
    control.verify_environment(saved["environment"], saved["environment"])
    return {"path": str(directory), "manifest_sha256": shared.sha256(directory / "manifest.json"),
            "completion_sha256": shared.sha256(directory / "completion.json"),
            "checkpoint_sha256": checksum, "environment": saved["environment"], "best_metrics": best}


def validate(args):
    reference_dir = args.reference_run_dir.expanduser().resolve()
    if args.model_dir is None:
        args.model_dir = Path(shared.read_json(reference_dir / "manifest.json")["model"]["path"])
    settings = shared.parser().parse_args(["--model-dir", str(args.model_dir), "--data-dir", str(args.data_dir)])
    manifest = shared.validate(settings)
    args.model_dir, args.data_dir = settings.model_dir, settings.data_dir
    args.reference_run_dir = reference_dir
    reference = verify_reference(reference_dir, manifest)
    args.archive = args.archive.expanduser().resolve()
    if shared.sha256(args.archive) != ARCHIVE_SHA256:
        raise ValueError("Pinned dataset archive SHA256 mismatch")
    with zipfile.ZipFile(args.archive) as archive:
        entries = [x for x in archive.infolist() if x.filename == "docred/train_distant.json"]
        if len(entries) != 1 or entries[0].file_size != 437046821:
            raise ValueError("Unexpected distant archive member")
    source_files = {**manifest["source_files"], **{name: shared.sha256(ROOT / name) for name in NEW_FILES}}
    # Pin every reference dependency as well as all new entry/data/checker/launcher files.
    source_files.update(shared.read_json(reference_dir / "manifest.json")["source_files"])
    source_files["scripts/train_ttmre_b2_5090.py"] = shared.sha256(ROOT / "scripts/train_ttmre_b2_5090.py")
    manifest.update({"source_files": source_files, "entrypoint_sha256": shared.sha256(Path(__file__)),
                     "protocol": protocol(), "matched_b2_reference": reference,
                     "archive": {"path": str(args.archive), "sha256": ARCHIVE_SHA256,
                                 "member": "docred/train_distant.json", "member_bytes": 437046821},
                     "distant_data_read": False, "test_data_read": False})
    if args.run_dir is not None:
        args.run_dir = args.run_dir.expanduser().resolve()
        if args.run_dir == RUN_ROOT or not args.run_dir.is_relative_to(RUN_ROOT):
            raise ValueError("Use a unique workspace inside the documented two-stage root")
        if args.mode == "prepare" and args.run_dir.exists():
            raise FileExistsError(args.run_dir)
        if args.mode in ("train", "replay") or (args.mode == "validate" and args.run_dir.exists()):
            verify_preparation(args.run_dir, manifest)
    elif args.mode != "validate":
        raise ValueError("Preparation/training/replay requires --run-dir")
    if args.check_runtime:
        control.runtime(reference["environment"])
        aligned.emit("runtime_validation_passed", **m1.environment_report())
    return manifest


def identity_equal(saved, current):
    for key in ("source_files", "entrypoint_sha256", "protocol", "matched_b2_reference", "archive"):
        if saved.get(key) != current.get(key):
            raise ValueError(f"Preparation identity mismatch: {key}")
    if saved["model"]["sha256"] != current["model"]["sha256"]:
        raise ValueError("Prepared backbone changed")
    for split in shared.SPLITS:
        if saved["data"][split]["sha256"] != current["data"][split]["sha256"]:
            raise ValueError("Prepared Human/Dev changed")


def verify_preparation(directory, current):
    directory = Path(directory)
    prepared = shared.read_json(directory / "preparation.json")
    ready = shared.read_json(directory / "ready.json")
    identity_equal(shared.read_json(directory / "identity.json"), current)
    if (ready.get("passed") is not True or ready.get("optimizer_steps") != 0
            or ready.get("preparation_sha256") != shared.sha256(directory / "preparation.json")
            or ready.get("preflight_sha256") != shared.sha256(directory / "preflight.json")
            or ready.get("test_data_read") is not False):
        raise ValueError("Workspace is not fully prepared and probed")
    probe = shared.read_json(directory / "preflight.json")
    if any(probe.get(stage, {}).get("passed") is not True for stage in STAGES):
        raise ValueError("Missing Distant/Human preflight")
    control.verify_environment(ready["environment"], current["matched_b2_reference"]["environment"])
    for split, count, labels in (("human", 3053, True), ("dev", 500, False), ("distant", 101873, True)):
        cache_dir = directory / "cache" / split
        item = prepared["caches"][split]
        if shared.sha256(cache_dir / "index.json") != item["index_sha256"]:
            raise ValueError("Prepared feature index changed")
        data.validate_cache(cache_dir, shared.sha256, count, item["source_sha256"], labels)
    if (prepared.get("optimizer_steps") != 0 or prepared.get("test_data_read") is not False
            or prepared["caches"]["human"]["source_sha256"] != shared.SPLITS["train_revised.json"]
            or prepared["caches"]["dev"]["source_sha256"] != shared.SPLITS["dev_revised.json"]):
        raise ValueError("Prepared source/scope mismatch")
    return prepared


def cache_dataset(args, split, with_labels=True):
    directory = args.run_dir / "cache" / split
    return data.DiskFeatures(directory, shared.read_json(directory / "index.json"), with_labels)


def load_model(args, identity):
    import numpy as np
    import torch
    from transformers import AutoConfig, AutoModel, AutoTokenizer
    from model2 import DocREModel
    from ttmre_memory_fit import install_unused_write_bypass, install_ttm_checkpoint
    args.device = control.runtime(identity["matched_b2_reference"]["environment"])
    for key, value in aligned.protocol("pair-independent").items():
        setattr(args, key, value)
    config = AutoConfig.from_pretrained(str(args.model_dir), num_labels=97, local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(str(args.model_dir), local_files_only=True)
    config.cls_token_id, config.sep_token_id, config.transformer_type = tokenizer.cls_token_id, tokenizer.sep_token_id, "roberta"
    priors = cache_dataset(args, "human").metadata["priors"]
    aligned.emit("generic_backbone_load_start", model_dir=str(args.model_dir))
    backbone = AutoModel.from_pretrained(str(args.model_dir), config=config, local_files_only=True).to(args.device)
    shared.seed_all(74, torch, np)
    model = DocREModel(args, config, torch.tensor(np.array(priors) + 1e-9, device=args.device), backbone, tokenizer).to(args.device)
    aligned.apply_variant(model, "pair-independent")
    control.assert_original_read(model)
    if sum(p.numel() for p in model.parameters()) != 381175401:
        raise ValueError("Unexpected baseline model parameters")
    install_unused_write_bypass(model)
    install_ttm_checkpoint(model)
    aligned.emit("task_model_ready", parameters=381175401, generic_initialization=True)
    return model


def set_stage(model, stage):
    if stage not in STAGES:
        raise ValueError("Unknown stage")
    model.mu_encoder.memory_tokens.requires_grad_(stage == "distant")
    model.mu_encoder.memory_tokens.grad = None
    model.train_mode = "finetune"  # In upstream Distant mode the backbone also trains.
    frozen = [name for name, parameter in model.named_parameters() if not parameter.requires_grad]
    expected = [] if stage == "distant" else ["mu_encoder.memory_tokens"]
    if frozen != expected:
        raise ValueError(f"Unexpected frozen parameter scope: {frozen}")


def optimizer_for_stage(model, stage):
    import torch
    groups = [{"params": [p for n, p in model.named_parameters() if p.requires_grad and
                           not any(k in n for k in ("extractor", "bilinear"))], "lr": 3e-5},
              {"params": [p for n, p in model.named_parameters() if p.requires_grad and
                           any(k in n for k in ("extractor", "bilinear"))], "lr": STAGES[stage][2]}]
    return torch.optim.AdamW(groups, lr=3e-5, eps=1e-6, weight_decay=.01)


def optimizer_report(model, optimizer):
    names = {id(p): name for name, p in model.named_parameters() if p.requires_grad}
    seen, report = [], []
    for group in optimizer.param_groups:
        members = [names[id(p)] for p in group["params"]]
        seen.extend(id(p) for p in group["params"])
        report.append({"lr": group["lr"], "weight_decay": group["weight_decay"],
                       "parameter_tensors": len(members), "parameters": sum(p.numel() for p in group["params"]),
                       "parameter_names": members})
    if len(seen) != len(set(seen)) or set(seen) != set(names):
        raise RuntimeError("Optimizer must contain every trainable parameter exactly once")
    return report


def run_probe(model, dataset, args, stage):
    import torch
    from ttmre_memory_fit import reserve_adam_moments
    set_stage(model, stage)
    before = model.mu_encoder.memory_tokens.detach().cpu().clone()
    buffers, reservation = reserve_adam_moments(model)
    reports = []
    aligned.emit("probe_memory_reservation", stage=stage, **reservation)
    try:
        for name, indices in dataset.probe_indices().items():
            model.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            start = time.perf_counter()
            features = [dataset[i] for i in indices]
            model.train()
            risks, logits = model(**shared.inputs_from_batch(shared.collate_fn(features), args.device))
            loss = sum(risks)
            if not torch.isfinite(loss).item() or not torch.isfinite(logits).all().item():
                raise RuntimeError("Nonfinite preflight forward")
            loss.backward()
            gradients = [p.grad for p in model.parameters() if p.grad is not None]
            if not gradients or not all(torch.isfinite(g).all().item() for g in gradients):
                raise RuntimeError("Missing/nonfinite preflight gradients")
            critical = [model.head_extractor.weight, model.bilinear.weight, model.model.embeddings.word_embeddings.weight]
            if stage == "distant":
                critical.append(model.mu_encoder.memory_tokens)
            elif model.mu_encoder.memory_tokens.grad is not None:
                raise RuntimeError("Frozen memory unexpectedly received a gradient")
            if any(p.grad is None or not p.grad.abs().sum().item() > 0 for p in critical):
                raise RuntimeError("Missing/zero critical gradient")
            torch.cuda.synchronize()
            report = {"probe": name, "document_indices": indices, "titles": [f["title"] for f in features],
                      "token_lengths": [len(f["input_ids"]) for f in features],
                      "entity_pairs": [len(f["hts"]) for f in features], "loss": loss.item(),
                      "finite_gradient_tensors": len(gradients), "seconds": time.perf_counter() - start,
                      "peak_allocated_GiB": torch.cuda.max_memory_allocated() / 1024**3,
                      "peak_reserved_GiB": torch.cuda.max_memory_reserved() / 1024**3, "optimizer_steps": 0}
            reports.append(report)
            aligned.emit("full_model_probe", stage=stage, **report)
            del risks, logits, loss, gradients, features
        if not torch.equal(before, model.mu_encoder.memory_tokens.detach().cpu()):
            raise RuntimeError("Probe changed memory parameters")
    finally:
        model.zero_grad(set_to_none=True)
        del buffers
        torch.cuda.empty_cache()
    return {"passed": True, "stage": stage, "probes": reports, "optimizer_memory_reservation": reservation,
            "optimizer_steps": 0, "parameters_unchanged": True}


def prepare(args, identity):
    from transformers import AutoTokenizer
    started = time.perf_counter()
    control.runtime(identity["matched_b2_reference"]["environment"])
    args.run_dir.mkdir(parents=True, exist_ok=False)
    args._owns_workspace = True
    shared.write_json(args.run_dir / "identity.json", identity)
    tokenizer = AutoTokenizer.from_pretrained(str(args.model_dir), local_files_only=True)
    rel2id = shared.read_json(ROOT / "meta/rel2id.json")
    gold = shared.read_json(args.data_dir / "dev_revised.json")
    candidates = data.relevant_gold_facts(gold)
    caches = {}
    for split, filename, count, labels in (("human", "train_revised.json", 3053, True), ("dev", "dev_revised.json", 500, False)):
        aligned.emit("cache_prepare_start", split=split)
        with (args.data_dir / filename).open("rb") as stream:
            metadata = data.build_cache(stream, args.run_dir / "cache" / split, tokenizer, rel2id,
                count, candidates, labels, progress=lambda **x: aligned.emit("cache_progress", split=split, **x),
                expected_sha256=shared.SPLITS[filename])
        caches[split] = {key: metadata[key] for key in ("documents", "source_sha256", "ordered_pairs", "label_entries")}
    aligned.emit("cache_prepare_start", split="distant", expected_documents=101873)
    with zipfile.ZipFile(args.archive) as archive, archive.open("docred/train_distant.json") as stream:
        metadata = data.build_cache(stream, args.run_dir / "cache" / "distant", tokenizer, rel2id, 101873,
                                   candidates, progress=lambda **x: aligned.emit("cache_progress", split="distant", **x))
    caches["distant"] = {key: metadata[key] for key in ("documents", "source_sha256", "ordered_pairs", "label_entries")}
    for split in caches:
        directory = args.run_dir / "cache" / split
        caches[split].update({"index_sha256": shared.sha256(directory / "index.json"),
                             "compressed_bytes": (directory / "features.bin").stat().st_size})
    shared.write_json(args.run_dir / "preparation.json", {"caches": caches, "optimizer_steps": 0,
        "distant_data_read": True, "test_data_read": False, "model_download": False,
        "cache_seconds": time.perf_counter() - started,
        "metric_reference": "Dev gold fact intersection only; never used by optimizer or training priors"})
    model = load_model(args, identity)
    probes = {stage: run_probe(model, cache_dataset(args, "distant" if stage == "distant" else "human"), args, stage)
              for stage in STAGES}
    shared.write_json(args.run_dir / "preflight.json", probes)
    shared.write_json(args.run_dir / "ready.json", {"passed": True, "optimizer_steps": 0,
        "preparation_sha256": shared.sha256(args.run_dir / "preparation.json"),
        "preflight_sha256": shared.sha256(args.run_dir / "preflight.json"), "test_data_read": False,
        "elapsed_seconds": time.perf_counter() - started, "environment": m1.environment_report()})
    aligned.emit("preparation_complete", run_dir=str(args.run_dir), optimizer_steps=0,
                 cache=caches, elapsed_seconds=time.perf_counter() - started)
    print("HUMAN_DISTANT_READY; optimizer_steps=0; test_labels_read=false; training_started=false", flush=True)


def evaluate(model, dev_loader, args, gold, annotated, distant):
    return shared.evaluate_dev(model, dev_loader, args.device, gold, annotated, distant)


def train_loop(model, loader, dev_loader, args, stage, directory, gold, annotated, distant, optimizer, scheduler):
    import torch
    epochs = STAGES[stage][1]
    if len(loader) < 2:
        raise ValueError("Need at least two batches for penultimate selection")
    memory = model.mu_encoder.memory_tokens.detach().cpu().clone()
    best, best_epoch, best_step, steps = -math.inf, None, None, 0
    with (directory / "train.log").open("x", encoding="utf-8") as log, (directory / "metrics.jsonl").open("x", encoding="utf-8") as metrics_log:
        for epoch in range(1, epochs + 1):
            aligned.emit("epoch_start", stage=stage, epoch=epoch, batches=len(loader))
            total_loss = 0.0
            for index, batch in enumerate(loader):
                model.train()
                optimizer.zero_grad(set_to_none=True)
                risks, logits = model(**shared.inputs_from_batch(batch, args.device))
                loss = sum(risks)
                if not torch.isfinite(loss).item() or not torch.isfinite(logits).all().item():
                    raise RuntimeError(f"Nonfinite {stage} forward before update {steps + 1}")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                steps += 1
                total_loss += loss.item()
                if steps == 1 or steps % 100 == 0:
                    record = {"event": "optimizer_update", "stage": stage, "epoch": epoch, "step": steps,
                              "batch": index + 1, "loss": loss.item()}
                    line = json.dumps(record, allow_nan=False)
                    print(line, flush=True)
                    log.write(line + "\n"); log.flush()
                # Free the just-consumed graph/logits before evaluation and the next forward.
                del risks, logits, loss
                if index + 1 == len(loader) - 1:
                    aligned.emit("development_evaluation_start", stage=stage, epoch=epoch, step=steps)
                    metrics = evaluate(model, dev_loader, args, gold, annotated, distant)
                    score = metrics["dev_F1"]
                    if not math.isfinite(score):
                        raise RuntimeError("Nonfinite selection score")
                    selected = score > best
                    if selected:
                        best, best_epoch, best_step = score, epoch, steps
                        torch.save(shared.cpu_snapshot(model), directory / "best_model.pth")
                        shared.write_json(directory / "best_metrics.json", {"epoch": epoch, "step": steps,
                                          "batch": index + 1, **metrics})
                    record = {"event": "development_evaluation", "stage": stage, "epoch": epoch, "step": steps,
                              "batch": index + 1, "train_loss_so_far": total_loss / (index + 1), "selected": selected, **metrics}
                    line = json.dumps(record, allow_nan=False)
                    print(line, flush=True)
                    for handle in (log, metrics_log):
                        handle.write(line + "\n"); handle.flush()
            if stage == "human-frozen" and not torch.equal(memory, model.mu_encoder.memory_tokens.detach().cpu()):
                raise RuntimeError("Frozen memory changed during Human training")
            aligned.emit("epoch_complete", stage=stage, epoch=epoch, step=steps, train_loss=total_loss / len(loader))
    return {"best_epoch": best_epoch, "best_step": best_step, "best_dev_F1": best,
            "optimizer_steps": steps, "memory_unchanged": bool(torch.equal(memory, model.mu_encoder.memory_tokens.detach().cpu())),
            "test_data_read": False, "test_evaluated": False}


def replay(model, directory, dev_loader, args, stage, gold, annotated, distant):
    import torch
    model.zero_grad(set_to_none=True)
    model.load_state_dict(torch.load(directory / "best_model.pth", map_location="cpu", weights_only=True), strict=True)
    best = shared.read_json(directory / "best_metrics.json")
    metrics = evaluate(model, dev_loader, args, gold, annotated, distant)
    for key in m1.METRIC_KEYS:
        if not math.isclose(metrics[key], best[key], abs_tol=1e-6, rel_tol=0):
            raise RuntimeError(f"{stage} replay metric mismatch: {key}")
    invariance = aligned.grouping_invariance(model, dev_loader.dataset, args.device)
    if not invariance["passed"]:
        raise RuntimeError("Grouping invariance failed")
    return {"passed": True, "stage": stage, "epoch": best["epoch"], "step": best["step"], "metrics": metrics,
            "invariance": invariance, "checkpoint_sha256": shared.sha256(directory / "best_model.pth"),
            "optimizer_steps": 0, "test_data_read": False, "test_evaluated": False}


def verify_completed_stage(directory, stage, identity, preparation_hash):
    saved = shared.read_json(directory / "manifest.json")
    identity_equal(saved, identity)
    if saved.get("preparation_sha256") != preparation_hash or saved.get("stage") != stage:
        raise ValueError("Stage preparation/source mismatch")
    complete = shared.read_json(directory / "completion.json")
    best = shared.read_json(directory / "best_metrics.json")
    replayed = shared.read_json(directory / "checkpoint_replay.json")
    batches, epochs = STAGES[stage][0] // 4, STAGES[stage][1]
    if (complete.get("stage") != stage or complete.get("optimizer_steps") != batches * epochs
            or complete.get("checkpoint_replay_passed") is not True or complete.get("test_data_read") is not False
            or complete.get("test_evaluated") is not False or replayed.get("passed") is not True
            or replayed.get("optimizer_steps") != 0 or replayed.get("test_data_read") is not False
            or replayed.get("test_evaluated") is not False
            or replayed.get("invariance", {}).get("passed") is not True
            or not 1 <= best["epoch"] <= epochs or best.get("batch") != batches - 1
            or best["step"] != (best["epoch"] - 1) * batches + batches - 1
            or complete.get("best_epoch") != best["epoch"] or complete.get("best_step") != best["step"]
            or complete.get("best_dev_F1") != best["dev_F1"]
            or replayed.get("epoch") != best["epoch"] or replayed.get("step") != best["step"]
            or any(replayed.get("metrics", {}).get(k) != best.get(k) for k in m1.METRIC_KEYS)):
        raise ValueError("Incomplete stage or inconsistent metrics/selection")
    with (directory / "metrics.jsonl").open(encoding="utf-8") as handle:
        trajectory = [json.loads(line) for line in handle]
    if (len(trajectory) != epochs or any(item.get("epoch") != i or item.get("stage") != stage
            or item.get("batch") != batches - 1 or item.get("step") != i * batches - 1
            or not math.isfinite(item["dev_F1"]) for i, item in enumerate(trajectory, 1))
            or max(trajectory, key=lambda item: item["dev_F1"])["epoch"] != best["epoch"]
            or trajectory[best["epoch"] - 1]["dev_F1"] != best["dev_F1"]):
        raise ValueError("Best checkpoint is not the earliest trajectory maximum")
    checksum = shared.sha256(directory / "best_model.pth")
    if checksum != saved.get("best_checkpoint_sha256") or checksum != replayed.get("checkpoint_sha256"):
        raise ValueError("Completed stage checkpoint hash mismatch")
    if stage == "human-frozen" and complete.get("memory_unchanged") is not True:
        raise ValueError("Human memory freeze did not pass")
    return {"path": str(directory), "checkpoint_sha256": checksum,
            "completion_sha256": shared.sha256(directory / "completion.json"), "best_metrics": best}


def train_stage(model, args, identity, stage, dev_loader, gold, annotated, distant, parent=None):
    import numpy as np
    import torch
    from torch.utils.data import DataLoader
    from transformers.optimization import get_linear_schedule_with_warmup
    directory = args.run_dir / stage / "artifacts"
    directory.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    set_stage(model, stage)
    args.num_train_epochs = STAGES[stage][1]
    args.head_learning_rate = STAGES[stage][2]
    memory_initial = model.mu_encoder.memory_tokens.detach().cpu().clone()
    manifest = copy.deepcopy(identity)
    manifest.update({"stage": stage, "preparation_sha256": shared.sha256(args.run_dir / "preparation.json"),
                     "distant_initialization_reference": parent, "environment": m1.environment_report(),
                     "distant_data_read": True, "memory_frozen": stage == "human-frozen",
                     "human_prior_index_sha256": shared.sha256(args.run_dir / "cache" / "human" / "index.json")})
    shared.write_json(directory / "manifest.json", manifest)
    try:
        dataset = cache_dataset(args, "distant" if stage == "distant" else "human")
        phase = time.perf_counter()
        probe = run_probe(model, dataset, args, stage)
        probe_seconds = time.perf_counter() - phase
        shared.write_json(directory / "full_model_check.json", probe)
        optimizer = optimizer_for_stage(model, stage)  # Probes must pass before this line.
        groups = optimizer_report(model, optimizer)
        shared.seed_all(74, torch, np)
        loader = DataLoader(dataset, batch_size=4, shuffle=True, drop_last=True, collate_fn=shared.collate_fn)
        total = len(loader) * STAGES[stage][1]
        if total != STAGES[stage][0] // 4 * STAGES[stage][1]:
            raise ValueError("Unexpected stage update budget")
        scheduler = get_linear_schedule_with_warmup(optimizer, int(total * .06), total)
        manifest.update({"batches_per_epoch": len(loader), "total_steps": total, "warmup_steps": int(total * .06),
                         "parameters": sum(p.numel() for p in model.parameters()), "optimizer_groups": groups})
        shared.write_json(directory / "manifest.json", manifest)
        shared.write_json(directory / "args.json", vars(args))
        phase = time.perf_counter()
        complete = train_loop(model, loader, dev_loader, args, stage, directory, gold, annotated, distant, optimizer, scheduler)
        training_seconds = time.perf_counter() - phase
        if complete["optimizer_steps"] != total:
            raise RuntimeError("Stage update budget incomplete")
        del optimizer, scheduler
        model.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
        phase = time.perf_counter()
        replayed = replay(model, directory, dev_loader, args, stage, gold, annotated, distant)
        replay_seconds = time.perf_counter() - phase
        if stage == "human-frozen" and not torch.equal(memory_initial, model.mu_encoder.memory_tokens.detach().cpu()):
            raise RuntimeError("Frozen checkpoint memory differs from Distant initialization")
        shared.write_json(directory / "checkpoint_replay.json", replayed)
        manifest["best_checkpoint_sha256"] = replayed["checkpoint_sha256"]
        shared.write_json(directory / "manifest.json", manifest)
        shared.write_json(directory / "timing.json", {"elapsed_seconds": time.perf_counter() - started,
            "phases_seconds": {"probes": probe_seconds, "training_including_dev_and_saves": training_seconds,
                               "checkpoint_replay": replay_seconds}})
        shared.write_json(directory / "completion.json", {"stage": stage, **complete,
            "checkpoint_replay_passed": True, "distant_initialization_reference": parent,
            "interpretation": "single-seed corrected TTM baseline; Distant uses Human label-frequency priors"})
        aligned.emit("stage_complete", stage=stage, **complete)
    except Exception as error:
        shared.write_json(directory / "failure.json", {"error_type": type(error).__name__, "error": str(error),
            "time_utc": datetime.now(timezone.utc).isoformat()})
        raise


def run(args, identity):
    import torch
    from torch.utils.data import DataLoader
    verify_preparation(args.run_dir, identity)
    control.runtime(identity["matched_b2_reference"]["environment"])
    stages = list(STAGES) if args.stage == "both" else [args.stage]
    if args.mode == "train":
        for stage in stages:
            if (args.run_dir / stage).exists():
                raise FileExistsError(f"Preserve existing stage directory: {stage}")
    model = load_model(args, identity)
    dev = cache_dataset(args, "dev", with_labels=False)
    dev_loader = DataLoader(dev, batch_size=4, shuffle=False, collate_fn=shared.collate_fn)
    gold = shared.read_json(args.data_dir / "dev_revised.json")
    annotated = {tuple(f) for f in cache_dataset(args, "human").metadata["matching_dev_facts"]}
    distant = {tuple(f) for f in cache_dataset(args, "distant").metadata["matching_dev_facts"]}
    preparation_hash = shared.sha256(args.run_dir / "preparation.json")
    for stage in stages:
        parent = None
        if stage == "human-frozen":
            parent_dir = args.run_dir / "distant" / "artifacts"
            parent = verify_completed_stage(parent_dir, "distant", identity, preparation_hash)
            model.load_state_dict(torch.load(parent_dir / "best_model.pth", map_location="cpu", weights_only=True), strict=True)
            aligned.emit("human_initialization_loaded", checkpoint_sha256=parent["checkpoint_sha256"])
        set_stage(model, stage)
        if args.mode == "replay":
            directory = args.run_dir / stage / "artifacts"
            verify_completed_stage(directory, stage, identity, preparation_hash)
            report = replay(model, directory, dev_loader, args, stage, gold, annotated, distant)
            print(json.dumps(report, allow_nan=False), flush=True)
        else:
            train_stage(model, args, identity, stage, dev_loader, gold, annotated, distant, parent)
    if args.mode == "train":
        print("HUMAN_DISTANT_TRAINING_COMPLETE; test_labels_read=false", flush=True)


def main():
    args = parser().parse_args()
    identity = validate(args)
    if args.mode == "validate":
        print(json.dumps({"validation": "passed", "identity": identity, "optimizer_steps": 0,
                          "test_data_read": False, "files_written": False}, indent=2))
        return
    os.chdir(ROOT)
    sys.path.insert(0, str(ROOT))
    args._owns_workspace = False
    try:
        if args.mode == "prepare":
            prepare(args, identity)
        else:
            run(args, identity)
    except Exception as error:
        if args.mode == "prepare" and args._owns_workspace:
            shared.write_json(args.run_dir / "preparation_failure.json", {"error_type": type(error).__name__,
                "error": str(error), "time_utc": datetime.now(timezone.utc).isoformat()})
        raise


if __name__ == "__main__":
    main()
