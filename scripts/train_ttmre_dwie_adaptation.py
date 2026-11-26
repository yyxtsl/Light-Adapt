"""Fixed DWIE C/B/D adaptation; validate is read-only and never starts CUDA.

Formal cloud execution additionally requires a separately confirmed, pinned
authorization with an explicit total deadline. CPU synthetic contracts may
inject tiny models and features; they are never a DWIE training result.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import gc
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import random
import shutil
import signal
import subprocess
import sys
import tarfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]
import ttmre_dwie_input as dwie
import ttmre_dwie_gpu_probe_v4 as probe

RUN_ROOT = ROOT / "experiments/2026-10-24-dwie-adaptation"
CLAIM_NAME = "dwie-adaptation-once-claim.json"
DOCROOT = ROOT / "refine-logs/SECOND_DATASET_ADMISSION_20261023"
AUTHORIZATION = DOCROOT / "DWIE_TRAINING_AUTHORIZATION_APPROVED_20261024.json"
SCORING_METADATA = DOCROOT / "DWIE_DEV_SCORING_METADATA_20261024.json"
OFFICIAL_METRIC = DOCROOT / "DWIE_OFFICIAL_METRICS_20261024.py"
CHECKER = ROOT / "scripts/check_ttmre_dwie_adaptation.py"
AUTHORIZATION_SHA256 = "174a3276894ed6b9cbbb55b6f8a2d58918c16b9fa79210f31e4e650571dfce54"
SCORING_METADATA_SHA256 = "1509e92c7334bb1b6775a32a909dca7e0e923a4b5efc5ae0e9e0fb227f4235d7"
CHECKER_SHA256 = "85e0f7b4f1c93e32614792a5eaeea405e51b6d944d7098734ef263f2b7537908"
OFFICIAL_METRIC_SHA256 = "0a939d851405a619f4af008b2c224899960c3bc16d54ae098ba2fe0181ed2006"
SOURCE_PINS = dict(probe.PINNED_SOURCES)
SOURCE_PINS.update({"scripts/ttmre_dwie_gpu_probe_v4.py": "456009f64d9c5751f5ebbe27873b03b61baea599a8d40a99180fe614eee543b1",
                   "scripts/check_ttmre_dwie_gpu_probe_v4.py": "f34cbe116c1e88ad9ea96cfc45b7446bd4b44453b09a81213206c5f94eb96748",
                   "refine-logs/SECOND_DATASET_ADMISSION_20261023/sources/dwie_official_evaluator.py.txt": "0a939d851405a619f4af008b2c224899960c3bc16d54ae098ba2fe0181ed2006"})
MODEL_CONFIG_SHA256 = "82ba49810e6441735e696033ac2512ee09da555507b2917ac3865b202d592cc3"
TRAIN_FEATURES, TARGET_SPLIT, REL2ID = probe.TRAIN_FEATURES, probe.TARGET_SPLIT, probe.REL2ID
DEV_FEATURES = TRAIN_FEATURES.with_name("dev.jsonl")
DEV_FEATURES_SHA256 = "15dc035924cce4e5cee6d5a5f764c307118a27f26cf1b4514beee0778dd121ef"
ARMS = ("C", "B", "D")
SEED, EPOCHS, NUM_CLASSES, LOGICAL_BATCH = 75, 30, 63, 4
UPDATES_PER_EPOCH, UPDATES_PER_ARM, WARMUP = 151, 4530, 271
GIB, REPORT_MARGIN, SOFT_RESERVE, TERM_RESERVE = 1024**3, 512 * 1024**2, 120, 15


class BudgetStop(TimeoutError):
    pass


def protocol():
    return {"version": "dwie-adaptation-v1", "arms": list(ARMS), "seed": SEED,
            "epochs_per_arm": EPOCHS, "train_documents": 602, "dev_documents": 100,
            "optimizer_updates_per_epoch": UPDATES_PER_EPOCH,
            "optimizer_updates_per_arm": UPDATES_PER_ARM,
            "total_optimizer_updates": UPDATES_PER_ARM * len(ARMS),
            "microbatch_documents": 1, "logical_batch_documents": LOGICAL_BATCH,
            "last_group_documents": 2, "drop_last": False,
            "accumulation": "per-document mean ATGL multiplied by document_pairs / group_pairs",
            "source_parents": {"C": probe.C_SHA256, "B": probe.C_SHA256, "D": probe.D_SHA256},
            "C_and_B_each_reload_original_parent": True,
            "fresh_target_head_seed": SEED, "target_classes": NUM_CLASSES,
            "loss": "ATGL", "alpha": 6.0, "precision": "float32",
            "optimizer": {"name": "torch.optim.AdamW", "backbone_lr": 3e-5,
                          "head_lr": 1e-5, "eps": 1e-6, "weight_decay": .01,
                          "foreach": False, "fused": False, "clip_norm": 1.0,
                          "warmup_steps": WARMUP, "steps": UPDATES_PER_ARM},
            "input_protocol": probe.protocol()["windowing"],
            "candidate_pairs": "all directed concept pairs including self, retain empty mentions",
            "selection": "full Dev full-ID micro F1 after each epoch; ties retain earliest",
            "unknown_dev_gold": "retained as false negatives; excluded only from model targets",
            "decoder": "every non-TH logit strictly greater than TH; no top-k or threshold tuning",
            "checkpoint": "best CPU snapshot, one final save per arm, strict replay",
            "benchmark": "same-run C/B, 3 rotating rounds; window tensor creation and device transfer included",
            "test_data_read": False, "automatic_retry": False, "resume": False,
            "extension": False, "automatic_next_experiment": False}


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def digest(path, check_fn=None):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            if check_fn is not None:
                check_fn()
            value.update(block)
    return value.hexdigest()


def json_sha256(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def check_budget(budget, reserve=SOFT_RESERVE):
    if time.monotonic() >= budget["deadline_monotonic"] - reserve:
        raise BudgetStop("DWIE total deadline reserve reached; no retry, resume or extension")
    if Path("/proc/sys/kernel/random/boot_id").read_text().strip() != budget["boot_id"]:
        raise BudgetStop("Host boot changed; cannot reset the original budget")


def event(workspace, name, **fields):
    row = {"event": name, "utc": datetime.now(timezone.utc).isoformat(), **fields}
    raw = json.dumps(row, ensure_ascii=False, allow_nan=False)
    print(raw, flush=True)
    with (Path(workspace) / "events.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(raw + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def epoch_groups(n, epoch, seed=SEED, logical_batch=LOGICAL_BATCH):
    if not isinstance(n, int) or n <= 0 or epoch < 0 or logical_batch <= 0:
        raise ValueError("Invalid fixed epoch grouping")
    indices = list(range(n))
    random.Random(seed + epoch).shuffle(indices)
    return [indices[start:start + logical_batch] for start in range(0, n, logical_batch)]


def pair_count(feature):
    inputs = feature["inputs"]
    ids, positions = inputs["entity_ids"], inputs["entity_pos"]
    if (ids != sorted(set(ids)) or len(positions) != len(ids)
            or inputs.get("candidate_rule") != "cartesian_including_self" or not ids):
        raise ValueError("Invalid/nonpositive label-independent candidate pair schema")
    return len(ids) ** 2


def _check(check_fn):
    if check_fn is not None:
        check_fn()


def _finite_gradients(model):
    import torch
    grads = [(n, p.grad) for n, p in model.named_parameters() if p.grad is not None]
    if not grads or not all(bool(torch.isfinite(g).all().item()) for _, g in grads):
        raise FloatingPointError("No active gradients or nonfinite gradient encountered")
    return grads


def accumulate_group(model, features, rel2id, device, logits_fn=None, check_fn=None):
    """Accumulate the concatenated-pair mean; never clear/clip/update here."""
    import torch
    logits_fn = logits_fn or probe.feature_logits
    features = list(features)
    if not features:
        raise ValueError("Cannot silently skip an empty logical group")
    counts = [pair_count(feature) for feature in features]
    total = sum(counts)
    weighted_loss, audits = 0., []
    for feature, count in zip(features, counts):
        _check(check_fn)
        logits, labels, audit = logits_fn(model, feature, rel2id, device)
        if logits.ndim != 2 or len(logits) != count or logits.shape != labels.shape:
            raise ValueError("Forward candidate pairs or label shape changed")
        loss = probe.atgl_loss(logits, labels, alpha=6.0)
        if not bool(torch.isfinite(logits).all().item()) or not bool(torch.isfinite(loss).item()):
            raise FloatingPointError("Nonfinite training logits/loss")
        weight = count / total
        (loss * weight).backward()
        weighted_loss += float(loss.detach().cpu()) * weight
        audits.append(audit)
        del logits, labels, loss
    _finite_gradients(model)
    return {"loss": weighted_loss, "pair_counts": counts, "total_pairs": total,
            "documents": [feature["id"] for feature in features], "forward_audits": audits}


def optimizer_for_model(model):
    import torch
    # Preserve the existing TTM adaptation grouping, including bias decay.
    head = lambda name: any(key in name for key in ("extractor", "bilinear"))
    groups = [{"params": [p for n, p in model.named_parameters() if p.requires_grad and not head(n)], "lr": 3e-5},
              {"params": [p for n, p in model.named_parameters() if p.requires_grad and head(n)], "lr": 1e-5}]
    if not any(group["params"] for group in groups):
        raise ValueError("No trainable parameters")
    return torch.optim.AdamW(groups, lr=3e-5, eps=1e-6, weight_decay=.01,
                             foreach=False, fused=False)


def optimizer_report(model, optimizer):
    names = {id(p): n for n, p in model.named_parameters()}
    return [{"lr": group.get("initial_lr", group["lr"]), "eps": group["eps"],
             "weight_decay": group["weight_decay"], "foreach": group["foreach"],
             "fused": group["fused"], "parameter_names": [names[id(p)] for p in group["params"]],
             "parameter_elements": sum(p.numel() for p in group["params"])}
            for group in optimizer.param_groups]


def make_scheduler(optimizer, total_steps=UPDATES_PER_ARM, warmup=WARMUP):
    import torch
    if total_steps <= 0 or not 0 <= warmup < total_steps:
        raise ValueError("Invalid linear warmup schedule")
    def scale(step):
        if step < warmup:
            return float(step) / max(1, warmup)
        return max(0., float(total_steps - step) / max(1, total_steps - warmup))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, scale)


def _witness_parameter(model):
    import torch
    named = dict(model.named_parameters())
    priority = ["bilinear.weight", "head_extractor.weight", "model.embeddings.word_embeddings.weight"]
    priority += [name for name in named if name not in priority]
    for name in priority:
        p = named[name]
        if p.grad is not None:
            flattened = p.grad.detach().reshape(-1)
            index = int(flattened.abs().argmax().item())
            if bool(flattened[index].ne(0).item()):
                return name, p, index, float(p.detach().reshape(-1)[index].cpu())
    raise RuntimeError("No nonzero gradient parameter for learning witness")


def train_step(model, features, rel2id, device, optimizer, scheduler, logits_fn=None, check_fn=None):
    import torch
    _check(check_fn)
    model.train()
    optimizer.zero_grad(set_to_none=True)
    report = accumulate_group(model, features, rel2id, device, logits_fn, check_fn)
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1., error_if_nonfinite=True)
    name, parameter, index, before = _witness_parameter(model)
    active = [(n, p) for n, p in model.named_parameters() if p.grad is not None]
    lrs = [group["lr"] for group in optimizer.param_groups]
    _check(check_fn)
    optimizer.step()
    after = float(parameter.detach().reshape(-1)[index].cpu())
    if not all(bool(torch.isfinite(p).all().item()) for _, p in active):
        raise FloatingPointError("Nonfinite parameter after formal AdamW step")
    steps = {int(optimizer.state[p]["step"].item()) for _, p in active}
    if len(steps) != 1:
        raise RuntimeError("Active Adam step counters disagree")
    scheduler.step()
    report.update({"gradient_norm": float(grad_norm.detach().cpu()), "learning_rates_used": lrs,
                   "adam_step": steps.pop(), "witness": {"parameter": name, "flat_index": index,
                   "before": before, "after": after, "changed": before != after},
                   "clip_calls": 1, "optimizer_step_calls": 1, "scheduler_step_calls": 1})
    return report


def active_grad_parameters(model):
    return [(name, parameter) for name, parameter in model.named_parameters() if parameter.grad is not None]


def reserve_active_gradients(model):
    """Retain only real backward buffers, never create grad for unused writers."""
    active = active_grad_parameters(model)
    if not active:
        raise ValueError("Identify connected gradients with a real zero-step backward first")
    for _, parameter in active:
        parameter.grad.zero_()
    return active


@contextmanager
def preserved_rng(device):
    import torch
    import numpy as np
    py, numpy = random.getstate(), np.random.get_state()
    cpu = torch.random.get_rng_state().clone()
    cuda = torch.cuda.get_rng_state(device).clone() if device.type == "cuda" else None
    try:
        yield
    finally:
        random.setstate(py)
        np.random.set_state(numpy)
        torch.random.set_rng_state(cpu)
        if cuda is not None:
            torch.cuda.set_rng_state(cuda, device)


def admission_groups(features):
    if not features:
        raise ValueError("No train features for resource admission")
    complexity = lambda feature: (len(dwie.window_starts(len(feature["inputs"]["content_ids"]))),
                                  pair_count(feature), len(feature["inputs"]["content_ids"]))
    order = sorted(range(len(features)), key=lambda index: complexity(features[index]))
    seeds = [order[0], order[len(order) // 2], order[-1],
             max(order, key=lambda index: len(features[index]["inputs"]["content_ids"])),
             max(order, key=lambda index: pair_count(features[index]))]
    groups, seen = [], set()
    for selected in seeds:
        following = sorted(order, key=lambda index: (abs(order.index(index) - order.index(selected)), index))
        group = tuple(following[:min(LOGICAL_BATCH, len(features))])
        if group not in seen:
            seen.add(group)
            groups.append(list(group))
    return groups


def resource_admission(model, train_features, rel2id, device, logits_fn=None, check_fn=None):
    """Zero updates, real active grads + two moments + scalar-AdamW scratch."""
    import torch
    versions = tuple(parameter._version for parameter in model.parameters())
    groups = admission_groups(train_features)
    reports, reservation = [], []
    active_names = set()
    was_training = model.training
    try:
        with preserved_rng(device):
            model.train()
            model.zero_grad(set_to_none=True)
            # Longest and most-pair documents are covered by the declared groups.
            largest = max(range(len(train_features)), key=lambda i: len(train_features[i]["inputs"]["content_ids"]))
            accumulate_group(model, [train_features[largest]], rel2id, device, logits_fn, check_fn)
            active = reserve_active_gradients(model)
            active_names = {name for name, _ in active}
            moment_bytes = sum(p.numel() * p.element_size() * 2 for _, p in active)
            largest_elements = max(p.numel() for _, p in active)
            element_size = max(p.element_size() for _, p in active)
            # foreach=False/fused=False keeps per-tensor temporaries; reserve
            # three largest tensors in addition to both resident moment arrays.
            for _, p in active:
                reservation.extend((torch.empty_like(p), torch.empty_like(p)))
            reservation.append(torch.empty(3 * largest_elements, dtype=active[0][1].dtype, device=device))
            scratch_bytes = 3 * largest_elements * element_size
            for indices in groups:
                _check(check_fn)
                for _, p in active:
                    p.grad.zero_()
                if device.type == "cuda":
                    torch.cuda.reset_peak_memory_stats(device)
                row = accumulate_group(model, [train_features[index] for index in indices],
                                       rel2id, device, logits_fn, check_fn)
                actual_active = {name for name, _ in active_grad_parameters(model)}
                if actual_active != active_names:
                    raise RuntimeError("Connected parameter set changed across admitted real groups")
                row.update({"peak_allocated_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
                            "peak_reserved_bytes": torch.cuda.max_memory_reserved(device) if device.type == "cuda" else None})
                reports.append(row)
            if versions != tuple(parameter._version for parameter in model.parameters()):
                raise RuntimeError("Resource admission changed model parameters")
        return {"passed": True, "optimizer_steps": 0, "parameters_unchanged": True,
                "RNG_restored": True, "active_parameter_names": sorted(active_names),
                "inactive_parameters_retained_grad_none": [n for n, p in model.named_parameters() if p.grad is None],
                "reserved_moment_bytes": moment_bytes, "reserved_scratch_bytes": scratch_bytes,
                "scratch_policy": "3 * largest active parameter; nonforeach/nonfused scalar AdamW",
                "representative_logical_groups": reports}
    finally:
        reservation.clear()
        model.zero_grad(set_to_none=True)
        model.train(was_training)
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()


def _target_head(in_features):
    import torch
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(SEED)
        layer = torch.nn.Linear(in_features, NUM_CLASSES, dtype=torch.float32, device="cpu")
    state = {name: value.detach().clone() for name, value in layer.state_dict().items()}
    return state, probe.tensor_state_sha256(state)


def _load_parent(model, path, expected_sha, check_fn, source_name):
    import torch
    _check(check_fn)
    if digest(path, check_fn) != expected_sha:
        raise ValueError("Pinned source parent SHA mismatch before load: " + source_name)
    state = torch.load(path, map_location="cpu", weights_only=True)
    report = probe.strict_load(model, state, source_name)
    report.update({"checkpoint_sha256": expected_sha, "checkpoint_bytes": Path(path).stat().st_size})
    del state
    gc.collect()
    return report


def build_arm(arm, identity, device, check_fn=None, model_factory=None, state_loader=None):
    import torch
    if arm not in ARMS:
        raise ValueError("Unknown fixed arm")
    _check(check_fn)
    factory, loader = model_factory or probe.build_model, state_loader or _load_parent
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(SEED)
        model, execution = factory(identity["model_config_dir"])
    model.requires_grad_(True)
    adapter = None
    if arm == "D":
        adapter = probe.replacement.install(model, "bypass")
    source_name = "D" if arm == "D" else "C_B"
    expected_sha = probe.D_SHA256 if arm == "D" else probe.C_SHA256
    load = loader(model, identity["parent_paths"][source_name], expected_sha, check_fn,
                  arm + " original source parent")
    if arm == "B":
        adapter = probe.replacement.install(model, "bypass")
    if arm == "C":
        model.mu_encoder.memory_tokens.requires_grad_(False)
        model.mu_encoder.memory_tokens.grad = None
    head, head_sha = _target_head(model.bilinear.in_features)
    probe.set_classifier(model, NUM_CLASSES, head)
    model.args.device = device
    model = model.to(device)
    model.train_mode = "finetune"
    gc.collect()
    return model, {"arm": arm, "source_parent": source_name, "source_sha256": expected_sha,
                   "strict_source_load": load, "execution": execution, "replacement": adapter,
                   "fresh_target_head_sha256": head_sha, "fresh_target_head_seed": SEED,
                   "target_classes": NUM_CLASSES, "fresh_model_and_original_parent_reload": True,
                   "parameters": sum(p.numel() for p in model.parameters()),
                   "frozen_parameter_names": [n for n, p in model.named_parameters() if not p.requires_grad]}


def predict_documents(model, features, rel2id, device, logits_fn=None, check_fn=None):
    import torch
    logits_fn = logits_fn or probe.feature_logits
    by_class = {index: relation for relation, index in rel2id.items() if index > 0}
    predictions = []
    was_training = model.training
    model.eval()
    try:
        with torch.inference_mode():
            for feature in features:
                _check(check_fn)
                # Frozen forward materializes target_labels. Remove scoring gold
                # before it, keeping unknown Dev relations exclusively in scorer.
                public = {"id": feature["id"], "inputs": feature["inputs"], "gold": []}
                logits, _, _ = logits_fn(model, public, rel2id, device)
                if len(logits) != pair_count(feature) or logits.shape[1] != len(rel2id):
                    raise ValueError("Prediction tensor disagrees with fixed IDs/target classes")
                if not bool(torch.isfinite(logits).all().item()):
                    raise FloatingPointError("Nonfinite Dev logits")
                ids = feature["inputs"]["entity_ids"]
                pairs, classes = torch.nonzero(logits[:, 1:] > logits[:, :1], as_tuple=True)
                for pair, relation in zip(pairs.cpu().tolist(), classes.cpu().tolist()):
                    predictions.append((feature["id"], ids[pair // len(ids)], by_class[relation + 1], ids[pair % len(ids)]))
                del logits
    finally:
        model.train(was_training)
    return sorted(predictions)


def evaluate(model, features, documents, rel2id, device, logits_fn=None, check_fn=None, official_source=None):
    predictions = predict_documents(model, features, rel2id, device, logits_fn, check_fn)
    score = dwie.score_documents(documents, predictions,
                                 [name for name, index in rel2id.items() if index > 0], official_source)
    return score, predictions


def save_state_once(path, state):
    import torch
    # Exclusive open refuses accidental overwrite; each arm writes one best.
    with Path(path).open("xb") as stream:
        torch.save(state, stream)
        stream.flush()
        os.fsync(stream.fileno())


def load_saved_state(path):
    import torch
    return torch.load(path, map_location="cpu", weights_only=True)


def _cpu_snapshot(model):
    return {name: value.detach().to("cpu", copy=True) for name, value in model.state_dict().items()}


def _memory_digest(model):
    value = model.state_dict().get("mu_encoder.memory_tokens")
    return probe.tensor_state_sha256({"mu_encoder.memory_tokens": value}) if value is not None else None


def train_arm(model, train_features, dev_features, documents, rel2id, device, output_dir,
              epochs=EPOCHS, seed=SEED, logical_batch=LOGICAL_BATCH, logits_fn=None,
              check_fn=None, epoch_hook=None, official_source=None, step_hook=None):
    import torch
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    production = (len(train_features) == 602 and len(dev_features) == 100 and epochs == EPOCHS
                  and seed == SEED and logical_batch == LOGICAL_BATCH and len(rel2id) == 63)
    total_steps = math.ceil(len(train_features) / logical_batch) * epochs
    if epochs <= 0 or total_steps <= 0:
        raise ValueError("Training requires positive complete epochs")
    warmup = math.floor(.06 * total_steps)
    optimizer = optimizer_for_model(model)
    scheduler = make_scheduler(optimizer, total_steps=total_steps, warmup=warmup)
    groups_report = optimizer_report(model, optimizer)
    initial_memory = _memory_digest(model)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed(seed)
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    initial_rng = hashlib.sha256(torch.random.get_rng_state().numpy().tobytes()).hexdigest()
    best_state, best_score, best_predictions = None, None, None
    best_epoch, step, learning_witness, history = 0, 0, None, []
    first_step = None
    started = time.perf_counter()
    for epoch in range(epochs):
        epoch_started = time.perf_counter()
        loss_pairs, pairs = 0., 0
        order = epoch_groups(len(train_features), epoch, seed, logical_batch)
        for indices in order:
            report = train_step(model, [train_features[index] for index in indices], rel2id,
                                device, optimizer, scheduler, logits_fn, check_fn)
            step += 1
            if report["adam_step"] != step:
                raise RuntimeError("Formal optimizer update accounting changed")
            if first_step is None:
                first_step = report
            if any(lr > 0 for lr in report["learning_rates_used"]) and learning_witness is None:
                if not report["witness"]["changed"]:
                    raise RuntimeError("First positive-LR formal update did not change witnessed parameter")
                learning_witness = {"formal_step": step, **report}
            if step_hook is not None:
                step_hook(step, report)
            loss_pairs += report["loss"] * report["total_pairs"]
            pairs += report["total_pairs"]
        optimizer.zero_grad(set_to_none=True)
        train_seconds = time.perf_counter() - epoch_started
        dev_started = time.perf_counter()
        score, predictions = evaluate(model, dev_features, documents, rel2id, device,
                                      logits_fn, check_fn, official_source)
        dev_seconds = time.perf_counter() - dev_started
        if not math.isfinite(score["primary_full_id"]["f1"]):
            raise FloatingPointError("Nonfinite Dev score")
        if best_score is None or score["primary_full_id"]["f1"] > best_score["primary_full_id"]["f1"]:
            best_state = _cpu_snapshot(model)
            best_score, best_predictions, best_epoch = score, predictions, epoch + 1
        row = {"epoch": epoch + 1, "formal_optimizer_steps": step,
               "train_seconds": train_seconds, "dev_seconds": dev_seconds,
               "epoch_seconds": time.perf_counter() - epoch_started,
               "loss_pair_weighted": loss_pairs / pairs, "train_pairs_seen": pairs,
               "order_sha256": json_sha256([[train_features[i]["id"] for i in group] for group in order]),
               "dev": score, "selected_best_epoch": best_epoch}
        history.append(row)
        write_json(output_dir / ("epoch-%02d.json" % (epoch + 1)), row)
        if epoch_hook is not None:
            epoch_hook(row)
    if step != total_steps or learning_witness is None or best_state is None:
        raise RuntimeError("Formal training incomplete or actual learning unverified")
    if initial_memory != _memory_digest(model):
        raise RuntimeError("Frozen/provenance memory changed during training")
    _check(check_fn)
    optimizer.zero_grad(set_to_none=True)
    del optimizer, scheduler
    gc.collect()
    checkpoint = output_dir / "best_model.pth"
    best_state_sha = probe.tensor_state_sha256(best_state)
    save_state_once(checkpoint, best_state)
    model.load_state_dict(load_saved_state(checkpoint), strict=True)
    if probe.tensor_state_sha256(model.state_dict()) != best_state_sha:
        raise RuntimeError("Strict saved best checkpoint did not reproduce selected state")
    replay_score, replay_predictions = evaluate(model, dev_features, documents, rel2id, device,
                                               logits_fn, check_fn, official_source)
    if replay_score != best_score or replay_predictions != best_predictions:
        raise RuntimeError("Saved best Dev replay differs from selected predictions/score")
    write_json(output_dir / "best_dev_predictions.json", [list(row) for row in replay_predictions])
    result = {"passed": True, "production_config": production, "synthetic_config": not production,
              "optimizer_steps": step, "expected_optimizer_steps": total_steps,
              "complete_epochs": epochs, "best_epoch": best_epoch, "best_dev": best_score,
              "strict_best_replay": True, "checkpoint_saves": 1,
              "checkpoint_state_sha256": best_state_sha, "checkpoint_sha256": digest(checkpoint, check_fn),
              "first_formal_step": first_step, "positive_lr_learning_witness": learning_witness,
              "optimizer_groups": groups_report, "warmup_steps": warmup,
              "frozen_memory_unchanged": True, "initial_cpu_rng_sha256": initial_rng,
              "sample_order_sha256": json_sha256([row["order_sha256"] for row in history]),
              "elapsed_seconds": time.perf_counter() - started, "test_data_read": False}
    write_json(output_dir / "completion.json", result)
    del best_state
    return result


def _decode_logits(logits, feature, rel2id):
    import torch
    ids = feature["inputs"]["entity_ids"]
    names = {index: relation for relation, index in rel2id.items() if index > 0}
    pairs, classes = torch.nonzero(logits[:, 1:] > logits[:, :1], as_tuple=True)
    return [(feature["id"], ids[pair // len(ids)], names[relation + 1], ids[pair % len(ids)])
            for pair, relation in zip(pairs.tolist(), classes.tolist())]


def _benchmark_sequence(model, arm, features, rel2id, device, round_index, reference_logits,
                        logits_fn=None, check_fn=None, expected_predictions=None):
    import torch
    logits_fn = logits_fn or probe.feature_logits
    model.eval()
    synchronize = lambda: torch.cuda.synchronize(device) if device.type == "cuda" else None
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    forward_seconds, numeric, decisions, max_absolute, predictions = 0., 0, 0, 0., []
    started = time.perf_counter()
    with torch.inference_mode():
        for feature in features:
            _check(check_fn)
            public = {"id": feature["id"], "inputs": feature["inputs"], "gold": []}
            synchronize()
            before = time.perf_counter()
            logits, _, _ = logits_fn(model, public, rel2id, device)
            synchronize()
            forward_seconds += time.perf_counter() - before
            # Output transfer, decoding and repeat diagnostics are outside the
            # forward timer; the enclosing e2e timer includes these operations.
            current = logits.detach().cpu()
            del logits
            if current.shape != (pair_count(feature), len(rel2id)) or not bool(torch.isfinite(current).all().item()):
                raise FloatingPointError("Nonfinite/invalid benchmark logits")
            key = (arm, feature["id"])
            if key not in reference_logits:
                reference_logits[key] = current.clone()
            else:
                reference = reference_logits[key]
                delta = (current - reference).abs()
                numeric += int((delta > (1e-4 + 1e-4 * reference.abs())).sum().item())
                max_absolute = max(max_absolute, float(delta.max().item()))
                decisions += int(((current[:, 1:] > current[:, :1]) !=
                                  (reference[:, 1:] > reference[:, :1])).sum().item())
            predictions.extend(_decode_logits(current, feature, rel2id))
    differences = (len(set(predictions) ^ {tuple(row) for row in expected_predictions})
                   if expected_predictions is not None else None)
    return {"round": round_index, "arm": arm, "seconds": forward_seconds,
            "forward_seconds": forward_seconds, "e2e_seconds": time.perf_counter() - started,
            "documents": len(features), "pairs": sum(pair_count(f) for f in features),
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(device) if device.type == "cuda" else None,
            "raw_numeric_violations": numeric, "max_absolute_logit_difference": max_absolute,
            "decision_flips_vs_first_round": decisions, "prediction_difference_vs_best": differences,
            "numeric_tolerances": {"atol": 1e-4, "rtol": 1e-4},
            "all_original_differences_retained": True}


def _benchmark_report(rows):
    means = {arm: sum(row["forward_seconds"] for row in rows if row["arm"] == arm) / 3 for arm in ("C", "B")}
    return {"rounds": rows, "mean_seconds": means,
            "B_relative_time_reduction": 1 - means["B"] / means["C"],
            "timing_scope": "frozen feature_logits including window preparation/tensor creation/input transfer, encoder, merge and pair head; excludes output transfer, decoder/scoring and repeat diagnostics",
            "e2e_scope": "per-arm sequence including forward, output transfer, decoding, repeat diagnostics; excludes model loading",
            "distinct_from_ReDocRED_pure_forward_timing": True, "test_data_read": False}


def benchmark_arms(c_model, b_model, features, rel2id, device, logits_fn=None, check_fn=None, rounds=3):
    if rounds != 3:
        raise ValueError("Fixed benchmark requires exactly three rotating rounds")
    logits_fn = logits_fn or probe.feature_logits
    models, rows, cache = {"C": c_model, "B": b_model}, [], {}
    original = {arm: model.training for arm, model in models.items()}
    try:
        for arm in ("C", "B"):
            predict_documents(models[arm], features[:1], rel2id, device, logits_fn, check_fn)
        for index in range(rounds):
            for arm in (("C", "B") if index % 2 == 0 else ("B", "C")):
                rows.append(_benchmark_sequence(models[arm], arm, features, rel2id, device,
                                                index, cache, logits_fn, check_fn))
    finally:
        for arm, model in models.items():
            model.train(original[arm])
    return _benchmark_report(rows)


def _read_features(path, expected_ids, rel2id, is_train, check_fn=None):
    features = []
    with Path(path).open("r", encoding="utf-8") as stream:
        for line in stream:
            _check(check_fn)
            feature = json.loads(line)
            pair_count(feature)
            if not isinstance(feature["inputs"].get("content_ids"), list):
                raise ValueError("Missing content token list")
            valid = set(feature["inputs"]["entity_ids"])
            for relation in feature["gold"]:
                if relation["h"] not in valid or relation["t"] not in valid:
                    raise ValueError("Gold endpoint absent from full concept IDs")
                if is_train and (relation["r"] not in rel2id or rel2id[relation["r"]] == 0):
                    raise ValueError("Unknown/TH gold in fixed training data")
            features.append(feature)
    if [feature["id"] for feature in features] != expected_ids:
        raise ValueError("Feature order/IDs disagree with the frozen train/Dev split")
    return features


def read_authorization(require_approved=False):
    if digest(AUTHORIZATION) != AUTHORIZATION_SHA256:
        raise ValueError("Training authorization file SHA mismatch")
    value = read_json(AUTHORIZATION)
    fixed = {"arms": list(ARMS), "seed": SEED, "epochs": EPOCHS,
             "optimizer_updates_per_arm": UPDATES_PER_ARM, "total_optimizer_updates": 13590,
             "test_data_read": False, "automatic_retry": False, "resume": False, "extension": False}
    if any(value.get(name) != expected for name, expected in fixed.items()):
        raise ValueError("Training authorization protocol differs from fixed plan")
    if require_approved:
        seconds = value.get("hard_seconds")
        if (value.get("approved") is not True or isinstance(seconds, bool)
                or not isinstance(seconds, (int, float)) or not math.isfinite(seconds) or seconds <= SOFT_RESERVE + TERM_RESERVE
                or not isinstance(value.get("user_confirmation"), str) or not value["user_confirmation"].strip()):
            raise PermissionError("Formal GPU training and total hard cap have not been explicitly confirmed; no claim or run directory created")
    return value


def validate_inputs(cloud=False, check_fn=None):
    sources = {}
    extra = {str(CHECKER.relative_to(ROOT)): CHECKER_SHA256,
             str(SCORING_METADATA.relative_to(ROOT)): SCORING_METADATA_SHA256,
             str(OFFICIAL_METRIC.relative_to(ROOT)): OFFICIAL_METRIC_SHA256}
    for name, expected in {**SOURCE_PINS, **extra}.items():
        actual = digest(ROOT / name, check_fn)
        if actual != expected:
            raise ValueError("Pinned source/metadata mismatch: " + name)
        sources[name] = actual
    if digest(Path(__file__)) != os.environ.get("TTMRE_DWIE_ADAPTATION_SELF_SHA256", digest(Path(__file__))):
        raise ValueError("Entry point differs from launcher-pinned SHA")
    files = ((TRAIN_FEATURES, probe.TRAIN_FEATURES_SHA256), (DEV_FEATURES, DEV_FEATURES_SHA256),
             (TARGET_SPLIT, probe.TARGET_SPLIT_SHA256), (REL2ID, probe.REL2ID_SHA256))
    for path, expected in files:
        if digest(path, check_fn) != expected:
            raise ValueError("Frozen DWIE input SHA mismatch: " + str(path))
    split, rel2id = read_json(TARGET_SPLIT), read_json(REL2ID)
    if (set(split) != {"train", "dev"} or len(split["train"]) != 602 or len(split["dev"]) != 100
            or len(set(split["train"])) != 602 or len(set(split["dev"])) != 100
            or set(split["train"]) & set(split["dev"])):
        raise ValueError("Frozen DWIE 602/100 split changed")
    if len(rel2id) != NUM_CLASSES or rel2id.get("TH") != 0 or sorted(rel2id.values()) != list(range(NUM_CLASSES)):
        raise ValueError("Frozen target classes changed")
    train = _read_features(TRAIN_FEATURES, split["train"], rel2id, True, check_fn)
    dev = _read_features(DEV_FEATURES, split["dev"], rel2id, False, check_fn)
    metadata = read_json(SCORING_METADATA)
    documents = metadata["documents"]
    if metadata.get("dev_features_sha256") != DEV_FEATURES_SHA256 or [d["id"] for d in documents] != split["dev"]:
        raise ValueError("Dev character-scoring metadata provenance differs from fixed features")
    for feature, document in zip(dev, documents):
        if ([e["id"] for e in document["entities"]] != feature["inputs"]["entity_ids"]
                or sorted((r["h"], r["r"], r["t"]) for r in document["relations"])
                != sorted((r["h"], r["r"], r["t"]) for r in feature["gold"])):
            raise ValueError("Dev original-character scoring metadata no longer matches full-ID gold")
        if any(len(e["mentions"]) != len(pos) for e, pos in zip(document["entities"], feature["inputs"]["entity_pos"])):
            raise ValueError("Dev original-character mentions disagree with feature entity clusters")
    dwie.official_metric_types(OFFICIAL_METRIC)
    authorization = read_authorization(require_approved=cloud)
    identity = {"protocol": protocol(), "source_files": sources, "entry_sha256": digest(Path(__file__)),
                "authorization": authorization, "authorization_sha256": AUTHORIZATION_SHA256,
                "train_features_sha256": probe.TRAIN_FEATURES_SHA256,
                "dev_features_sha256": DEV_FEATURES_SHA256, "target_split_sha256": probe.TARGET_SPLIT_SHA256,
                "rel2id_sha256": probe.REL2ID_SHA256, "dev_scoring_metadata_sha256": SCORING_METADATA_SHA256,
                "official_metric_sha256": OFFICIAL_METRIC_SHA256,
                "train_documents": len(train), "dev_documents": len(dev), "test_data_read": False,
                "cloud_assets_verified": False, "formal_training_started": False}
    if cloud:
        if os.name != "posix":
            raise RuntimeError("Cloud training requires Linux")
        d_parent, model_dir = probe.locate_d_parent(None)
        parents = {"C_B": probe.C_CHECKPOINT, "D": d_parent}
        for name, path in parents.items():
            expected = probe.C_SHA256 if name == "C_B" else probe.D_SHA256
            if digest(path, check_fn) != expected:
                raise ValueError("Cloud source parent digest changed: " + name)
        config_path = model_dir / "config.json"
        if digest(config_path, check_fn) != MODEL_CONFIG_SHA256:
            raise ValueError("Pinned complete RoBERTa configuration digest changed")
        config = read_json(config_path)
        expected_config = {"model_type": "roberta", "hidden_size": 1024,
                           "num_hidden_layers": 24, "num_attention_heads": 16,
                           "vocab_size": 50265, "max_position_embeddings": 514}
        if any(config.get(name) != value for name, value in expected_config.items()):
            raise ValueError("Cloud RoBERTa-large config changed")
        identity.update({"parent_paths": {name: str(path) for name, path in parents.items()},
                         "parent_sha256": {"C_B": probe.C_SHA256, "D": probe.D_SHA256},
                         "model_config_dir": str(model_dir), "model_config_sha256": digest(model_dir / "config.json", check_fn),
                         "cloud_assets_verified": True})
    return identity, train, dev, documents, rel2id


def disk_admission(path, identity, remaining_arms):
    # Target head is smaller than the parent head; using full parent sizes
    # conservatively covers tensor serialization and immutable provenance memory.
    pending_bytes = sum(Path(identity["parent_paths"]["C_B" if arm == "C" else "D"]).stat().st_size
                        for arm in remaining_arms)
    required = max(5 * GIB if len(remaining_arms) == 3 else REPORT_MARGIN,
                   pending_bytes + REPORT_MARGIN)
    free = shutil.disk_usage(path).free
    if free < required:
        raise RuntimeError("Disk admission failed: free=%d required=%d shortfall=%d bytes; no deletion authorized" %
                           (free, required, required - free))
    return {"free_bytes": free, "required_bytes": required, "pending_parent_size_bound_bytes": pending_bytes,
            "remaining_arms": list(remaining_arms), "old_files_deleted": False}


def runtime_identity():
    import torch
    import numpy as np
    import transformers
    if torch.__version__ != "2.7.1+cu128":
        raise RuntimeError("Expected verified cloud torch 2.7.1+cu128; no install/version fallback")
    if np.__version__ != "1.24.3" or transformers.__version__ != "4.34.0":
        raise RuntimeError("Expected verified cloud numpy 1.24.3 / transformers 4.34.0; no install/version fallback")
    limits = {}
    for name in ("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory.current",
                 "/sys/fs/cgroup/memory/memory.limit_in_bytes", "/sys/fs/cgroup/memory/memory.usage_in_bytes"):
        path = Path(name)
        if path.is_file():
            limits[name] = path.read_text().strip()
    import torch.optim.adamw as adamw_module
    implementation = inspect.getsource(adamw_module._single_tensor_adamw)
    return {"python": sys.version, "torch": torch.__version__, "numpy": np.__version__,
            "transformers": transformers.__version__, "cuda_runtime": torch.version.cuda,
            "cgroup_memory": limits, "host_meminfo_is_not_container_limit": True,
            "AdamW_single_tensor_source_sha256": hashlib.sha256(implementation.encode()).hexdigest(),
            "foreach": False, "fused": False}


def host_memory_admission(identity):
    """Use remaining container quota as well as host MemAvailable."""
    meminfo = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        name, value = line.split(":", 1)
        if name in ("MemAvailable", "MemFree"):
            meminfo[name] = int(value.strip().split()[0]) * 1024
    if "MemAvailable" not in meminfo:
        raise RuntimeError("Host MemAvailable unavailable; cannot establish CPU snapshot/load headroom")
    candidates, limits = [meminfo["MemAvailable"]], []
    for maximum, current in (("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory.current"),
                             ("/sys/fs/cgroup/memory/memory.limit_in_bytes", "/sys/fs/cgroup/memory/memory.usage_in_bytes")):
        if Path(maximum).is_file() and Path(current).is_file():
            cap, used = Path(maximum).read_text().strip(), Path(current).read_text().strip()
            if cap != "max":
                remaining = max(0, int(cap) - int(used))
                # v1 unlimited sentinel is larger than physical address space;
                # min with host availability safely treats it as non-binding.
                candidates.append(remaining)
                limits.append({"path": maximum, "limit_bytes": int(cap), "current_bytes": int(used),
                               "remaining_bytes": remaining})
    available = min(candidates)
    parent_bound = max(Path(path).stat().st_size for path in identity["parent_paths"].values())
    required = 4 * parent_bound + REPORT_MARGIN
    if available < required:
        raise RuntimeError("Effective CPU memory admission failed: available=%d required=%d bytes" % (available, required))
    return {"effective_available_bytes": available, "host_MemAvailable_bytes": meminfo["MemAvailable"],
            "cgroup_constraints": limits, "required_bytes": required,
            "bound": "four largest parent serializations plus 512MiB for construction/load, best/replay CPU state and data/score caches",
            "host_total_is_not_container_quota": True}


def run_synthetic_contracts():
    import unittest
    from unittest.mock import patch
    import torch
    import check_ttmre_dwie_adaptation as contract_tests
    suite = unittest.defaultTestLoader.loadTestsFromModule(contract_tests)
    with patch.object(torch.cuda, "_lazy_init", side_effect=AssertionError("CPU contracts attempted CUDA")):
        result = unittest.TextTestRunner(stream=sys.stdout, verbosity=1).run(suite)
    if not result.wasSuccessful():
        raise RuntimeError("DWIE CPU contracts failed before real parent load")
    return {"passed": True, "tests_run": result.testsRun, "failures": 0, "errors": 0,
            "gpu_used": False, "real_weights_loaded": False}


def require_lock():
    if os.name != "posix" or os.environ.get("TTMRE_DWIE_ADAPTATION_LOCK_FD") != "9":
        raise RuntimeError("Use the unique run_ttmre_dwie_adaptation.sh launcher")
    import fcntl
    actual = os.fstat(9)
    expected = (ROOT / "experiments/2026-10-08-baseline-alignment/.training.lock").stat()
    if (actual.st_dev, actual.st_ino) != (expected.st_dev, expected.st_ino):
        raise RuntimeError("Worker does not hold the shared training lock")
    fcntl.flock(9, fcntl.LOCK_EX | fcntl.LOCK_NB)
    return 9


def pack_reports(workspace, budget):
    """Archive only this launch's small evidence; never weights or old runs."""
    workspace = Path(workspace).resolve()
    destination = workspace.parent / "reports.tar.gz"
    check_budget(budget, reserve=0)
    members = sorted(path for path in workspace.rglob("*")
                     if path.is_file() and (path.suffix == ".json" or path.name == "events.jsonl"))
    with destination.open("xb") as stream:
        with tarfile.open(fileobj=stream, mode="w:gz") as archive:
            for path in members:
                check_budget(budget, reserve=0)
                if path.is_symlink() or not path.resolve().is_relative_to(workspace):
                    raise ValueError("Report path escapes the current workspace")
                archive.add(path, arcname="workspace/" + path.relative_to(workspace).as_posix(), recursive=False)
        stream.flush()
        os.fsync(stream.fileno())
    check_budget(budget, reserve=0)
    return {"path": str(destination), "bytes": destination.stat().st_size,
            "sha256": digest(destination, lambda: check_budget(budget, reserve=0)),
            "members": len(members), "checkpoint_weights_included": False}


def worker(workspace):
    # Independent refusal even if invoked outside the supervisor.
    read_authorization(require_approved=True)
    require_lock()
    workspace = Path(workspace).resolve()
    claim = read_json(RUN_ROOT / CLAIM_NAME)
    if workspace != Path(claim["workspace"]) or workspace.parent.parent != RUN_ROOT.resolve():
        raise ValueError("Worker workspace is not the one-use claim")
    budget = read_json(workspace / "budget.json")
    if digest(workspace / "budget.json") != claim["budget_sha256"]:
        raise ValueError("Claim budget digest changed")
    check_fn = lambda: check_budget(budget)
    results, progress = {}, {"current_arm": None, "current_arm_optimizer_steps": 0, "completed_optimizer_steps": 0}
    try:
        _check(check_fn)
        contracts = run_synthetic_contracts()
        identity, train, dev, documents, rel2id = validate_inputs(cloud=True, check_fn=check_fn)
        identity["runtime"] = runtime_identity()
        identity["cpu_memory_admission"] = host_memory_admission(identity)
        # External query precedes all torch CUDA APIs.
        external = probe.gpu_preflight.inspect_gpu()
        if not external["passed"]:
            raise RuntimeError("External GPU occupancy exceeds fixed 500MiB threshold")
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable; no CPU training fallback")
        device = torch.device("cuda", torch.cuda.current_device())
        if torch.cuda.get_device_properties(device).name != external["name"]:
            raise RuntimeError("CUDA device and external preflight disagree")
        identity["external_gpu_preflight"] = external
        identity["cpu_contracts"] = contracts
        identity["disk_start"] = disk_admission(workspace, identity, ARMS)
        write_json(workspace / "identity.json", identity)
        orders = [[ [train[index]["id"] for index in group]
                    for group in epoch_groups(len(train), epoch)] for epoch in range(EPOCHS)]
        write_json(workspace / "fixed_epoch_groups.json", {"seed": SEED, "epochs": orders,
                   "sha256": json_sha256(orders), "independent_of_model_RNG": True})
        for arm_index, arm in enumerate(ARMS):
            _check(check_fn)
            resources = disk_admission(workspace, identity, ARMS[arm_index:])
            event(workspace, "arm_started", arm=arm, disk=resources)
            progress.update({"current_arm": arm, "current_arm_optimizer_steps": 0})
            model, construction = build_arm(arm, identity, device, check_fn)
            write_json(workspace / (arm + "-construction.json"), construction)
            admission = resource_admission(model, train, rel2id, device, check_fn=check_fn)
            write_json(workspace / (arm + "-resource-admission.json"), admission)
            event(workspace, "arm_resource_admitted", arm=arm, optimizer_steps=0,
                  reserved_moment_bytes=admission["reserved_moment_bytes"], reserved_scratch_bytes=admission["reserved_scratch_bytes"])
            def epoch_hook(row, current_arm=arm, index=arm_index):
                event(workspace, "epoch_complete", arm=current_arm, **row)
                if row["epoch"] == 1:
                    remaining_epochs = (len(ARMS) - index) * EPOCHS - 1
                    # Include observed epoch Dev cost and conservative headroom
                    # for remaining loads, admissions, saves/replays/benchmark.
                    estimate = row["epoch_seconds"] * remaining_epochs * 1.25 + 600
                    available = budget["deadline_monotonic"] - time.monotonic() - SOFT_RESERVE
                    event(workspace, "throughput_admission", arm=current_arm,
                          estimated_remaining_seconds=estimate, available_seconds=available,
                          estimate_is_projection=True)
                    if estimate > available:
                        raise BudgetStop("Measured first-epoch projection exceeds approved remaining total cap")
            def step_hook(step, row, current_arm=arm):
                progress["current_arm_optimizer_steps"] = step
                if step in (1, 2, 100) or step % 500 == 0:
                    event(workspace, "formal_training_progress", arm=current_arm,
                          formal_optimizer_step=step, loss=row["loss"], adam_step=row["adam_step"],
                          learning_rates_used=row["learning_rates_used"], witness=row["witness"])
            results[arm] = train_arm(model, train, dev, documents, rel2id, device, workspace / arm,
                                     check_fn=check_fn, epoch_hook=epoch_hook, official_source=OFFICIAL_METRIC,
                                     step_hook=step_hook)
            if not results[arm]["production_config"] or results[arm]["optimizer_steps"] != UPDATES_PER_ARM:
                raise RuntimeError("A synthetic/incomplete arm cannot satisfy production completion")
            progress["completed_optimizer_steps"] += results[arm]["optimizer_steps"]
            del model
            gc.collect()
            torch.cuda.empty_cache()
        # Sequential model residence prevents benchmark from requiring two
        # large models concurrently on the already tight training device.
        benchmark = benchmark_checkpoints(workspace, identity, dev, rel2id, device, check_fn)
        write_json(workspace / "same-run-C-B-benchmark.json", benchmark)
        complete = {"passed": True, "arms": results, "optimizer_steps": sum(r["optimizer_steps"] for r in results.values()),
                    "expected_optimizer_steps": 13590, "benchmark": benchmark,
                    "elapsed_seconds": time.monotonic() - budget["start_monotonic"],
                    "test_data_read": False, "automatic_retry": False, "resume": False, "extension": False}
        _check(check_fn)
        write_json(workspace / "completion.json", complete)
        archive = pack_reports(workspace, budget)
        write_json(workspace.parent / "reports-archive.json", archive)
        event(workspace, "DWIE_ADAPTATION_COMPLETE", passed=True,
              optimizer_steps=complete["optimizer_steps"],
              elapsed_seconds=time.monotonic() - budget["start_monotonic"],
              reports_archive_sha256=archive["sha256"])
        print(json.dumps({"event": "reports_archive_complete", **archive}), flush=True)
    except BaseException as error:
        failure = {"passed": False, "error": str(error), "error_type": type(error).__name__,
                   "completed_arms": results, "progress": progress, "test_data_read": False, "automatic_retry": False,
                   "resume": False, "extension": False,
                   "elapsed_seconds": time.monotonic() - budget["start_monotonic"]}
        try:
            write_json(workspace / "failure.json", failure)
        except FileExistsError:
            pass
        event(workspace, "DWIE_ADAPTATION_FAILED", error=str(error), error_type=type(error).__name__)
        if not (workspace.parent / "reports.tar.gz").exists():
            try:
                archive = pack_reports(workspace, budget)
                write_json(workspace.parent / "reports-archive.json", archive)
            except (OSError, ValueError, TimeoutError):
                # Deadline/failure never grants more time or a new process.
                pass
        raise


def benchmark_checkpoints(workspace, identity, features, rel2id, device, check_fn=None,
                          model_builder=None, logits_fn=None, expected_predictions=None):
    """Rotate C/B with one resident model; exclude loading from timed sections."""
    import torch
    builder = model_builder or build_arm
    logits_fn = logits_fn or probe.feature_logits
    expected_predictions = expected_predictions or {
        arm: read_json(Path(workspace) / arm / "best_dev_predictions.json") for arm in ("C", "B")}
    rows, cache = [], {}
    for round_index in range(3):
        for arm in (("C", "B") if round_index % 2 == 0 else ("B", "C")):
            model, _ = builder(arm, identity, device, check_fn)
            model.load_state_dict(load_saved_state(Path(workspace) / arm / "best_model.pth"), strict=True)
            model.eval()
            predict_documents(model, features[:1], rel2id, device, logits_fn, check_fn)
            rows.append(_benchmark_sequence(model, arm, features, rel2id, device, round_index,
                                            cache, logits_fn, check_fn, expected_predictions[arm]))
            del model
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
    return {**_benchmark_report(rows), "single_resident_model": True, "model_load_excluded": True}


def supervise(command, budget, lock_fd):
    check_budget(budget)
    child = subprocess.Popen(command, start_new_session=True, pass_fds=(lock_fd,))
    term_sent = False
    try:
        while True:
            status = child.poll()
            if status is not None:
                if status != 0 or term_sent:
                    raise RuntimeError("Worker failed/stopped; one-use claim retained, no automatic retry")
                check_budget(budget, reserve=0)
                return
            now = time.monotonic()
            if now >= budget["deadline_monotonic"]:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
                raise BudgetStop("Approved hard deadline reached; worker process group killed")
            if now >= budget["deadline_monotonic"] - TERM_RESERVE and not term_sent:
                os.killpg(child.pid, signal.SIGTERM)
                term_sent = True
            time.sleep(.2)
    finally:
        if child.poll() is None:
            os.killpg(child.pid, signal.SIGKILL)
            child.wait()


def start(workspace):
    # Explicit approval and its total hard cap precede locks, mkdir and claim.
    authorization = read_authorization(require_approved=True)
    lock_fd = require_lock()
    workspace = Path(workspace).resolve()
    if (workspace.name != "workspace" or workspace.parent.parent != RUN_ROOT.resolve()
            or workspace.exists() or (RUN_ROOT / CLAIM_NAME).exists()):
        raise ValueError("Use one fresh canonical launch workspace; retries/resume are disabled")
    now = time.monotonic()
    budget = {"start_monotonic": now, "deadline_monotonic": now + authorization["hard_seconds"],
              "hard_seconds": authorization["hard_seconds"], "soft_reserve_seconds": SOFT_RESERVE,
              "term_seconds": TERM_RESERVE, "start_utc": datetime.now(timezone.utc).isoformat(),
              "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
              "authorization_sha256": AUTHORIZATION_SHA256, "automatic_retry": False,
              "resume": False, "extension": False}
    workspace.mkdir(parents=True, exist_ok=False)
    write_json(workspace / "budget.json", budget)
    write_json(RUN_ROOT / CLAIM_NAME, {"workspace": str(workspace), "protocol": protocol(),
               "budget_sha256": digest(workspace / "budget.json"), "authorization_sha256": AUTHORIZATION_SHA256})
    write_json(workspace.parent / "pid.json", {"supervisor_pid": os.getpid()})
    command = [sys.executable, "-u", "-B", str(Path(__file__).resolve()), "--mode", "worker", "--run-dir", str(workspace)]
    try:
        supervise(command, budget, lock_fd)
    except BaseException as error:
        try:
            write_json(workspace.parent / "supervisor-failure.json", {"error": str(error),
                       "error_type": type(error).__name__, "complete": False, "automatic_retry": False,
                       "elapsed_seconds": time.monotonic() - now})
        except FileExistsError:
            pass
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("validate", "authorize", "start", "worker"), default="validate")
    parser.add_argument("--run-dir", type=Path)
    args = parser.parse_args()
    if args.mode == "authorize":
        # Pending approval refuses before any lock, claim or directory write.
        read_authorization(require_approved=True)
        print(json.dumps({"approved": True, "authorization_sha256": AUTHORIZATION_SHA256}))
    elif args.mode == "validate":
        identity, _, _, _, _ = validate_inputs(cloud=False)
        print(json.dumps(identity, indent=2, ensure_ascii=False, allow_nan=False))
    else:
        if args.run_dir is None:
            raise ValueError("The unique cloud launcher must provide --run-dir")
        (start if args.mode == "start" else worker)(args.run_dir)


if __name__ == "__main__":
    main()
