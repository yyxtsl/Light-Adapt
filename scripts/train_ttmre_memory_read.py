"""Offline M1 seed-74 experiment; default validation is read-only."""

from __future__ import annotations

import argparse
from collections import Counter
import copy
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import sys
import time

import train_ttmre_baseline_aligned as aligned

shared = aligned.shared
ROOT = shared.ROOT
RUN_ROOT = ROOT / "experiments" / "2026-10-09-memory-read"
VARIANT = "pair-slot-read"
RUNTIME_PROFILE = {"torch": "2.7.1", "cuda": "12.8", "transformers": "4.34.0", "numpy": "1.24.3"}
PINNED_BASELINE_FILES = {
    "scripts/train_ttmre_baseline_aligned.py": "8436a02d3addd6b2847961872ff8d562281ae0631a65c19e759f6ae23e89d655",
    "scripts/check_ttmre_baseline_aligned.py": "7426ae6cabf04653f095a5b3d69e218206d81faf339779c6822044840a47c1e2",
    "scripts/run_ttmre_baseline_aligned.sh": "98f88ae1ed73565f581bb8c2c61cbd9c494707f41c2bd45916ec500eea267aeb",
}
METHOD_FILES = ("scripts/ttmre_pair_read.py", "scripts/check_ttmre_memory_read.py",
                "scripts/run_ttmre_memory_read.sh", "scripts/ttmre_memory_fit.py")
METRIC_KEYS = ("dev_F1", "dev_precision", "dev_recall", "dev_IgnF1_annotated", "dev_IgnF1_distant")


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--mode", choices=("validate", "probe", "train", "replay"), default="validate")
    result.add_argument("--model-dir", type=Path)
    result.add_argument("--data-dir", type=Path, default=ROOT / "dataset" / "docred")
    result.add_argument("--run-dir", type=Path)
    result.add_argument("--b2-run-dir", type=Path, required=True, help="Completed B2 artifacts; read only")
    result.add_argument("--check-runtime", action="store_true", help="Also verify the CUDA runtime during validation")
    return result


def protocol():
    return {**aligned.protocol("pair-independent"), "variant": VARIANT,
            "method_version": "mass-preserving-pair-read-v1", "read_rank": 64,
            "alpha_init": 0.01, "new_matrix_learning_rate": 3e-5,
            "gate_weight_decay": 0.0, "new_parameter_count": 147457,
            "runtime_profile": dict(RUNTIME_PROFILE),
            "execution_profile": "single-step-unused-write-bypass-ttm-checkpoint-v2; original parameters retained",
            "method_initialization": "install after complete B2 task-model construction; no task checkpoint load"}


def runtime():
    import numpy as np
    import torch
    import transformers
    actual = {"torch": torch.__version__.split("+")[0], "cuda": torch.version.cuda,
              "transformers": transformers.__version__, "numpy": np.__version__}
    if actual != RUNTIME_PROFILE:
        raise RuntimeError(f"M1 requires the Blackwell runtime {RUNTIME_PROFILE}; found {actual}")
    if not torch.cuda.is_available():
        raise RuntimeError("Upstream S-PU requires CUDA")
    capability = torch.cuda.get_device_capability(0)
    architecture = f"sm_{capability[0]}{capability[1]}"
    if architecture not in torch.cuda.get_arch_list():
        raise RuntimeError(f"CUDA build does not contain the GPU architecture {architecture}")
    return torch.device("cuda:0")


def environment_report():
    import numpy as np
    import torch
    import transformers
    return {"versions": {"torch": torch.__version__, "cuda": torch.version.cuda,
                         "transformers": transformers.__version__, "numpy": np.__version__},
            "gpu": torch.cuda.get_device_name(0), "capability": list(torch.cuda.get_device_capability(0)),
            "build_architectures": torch.cuda.get_arch_list(),
            "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "float32_matmul_precision": torch.get_float32_matmul_precision(),
            "cuda_allocator_config": os.environ.get("PYTORCH_CUDA_ALLOC_CONF")}


def validate(args):
    if args.check_runtime:
        runtime()
        environment = environment_report()
        aligned.emit("runtime_validation_passed", **environment)
    for name, expected in PINNED_BASELINE_FILES.items():
        if shared.sha256(ROOT / name) != expected:
            raise ValueError(f"Frozen B1/B2 helper identity mismatch: {name}")
    baseline_args = aligned.parser().parse_args(["--variant", "pair-independent",
                                                 "--data-dir", str(args.data_dir)])
    baseline_args.model_dir = args.model_dir
    baseline = aligned.validate(baseline_args)
    args.model_dir, args.data_dir = baseline_args.model_dir, baseline_args.data_dir
    args.b2_run_dir = args.b2_run_dir.expanduser().resolve()
    reference = aligned.verify_completed_run(args.b2_run_dir, "pair-independent", baseline)
    if args.run_dir is not None:
        args.run_dir = args.run_dir.expanduser().resolve()
        if args.run_dir == RUN_ROOT or not args.run_dir.is_relative_to(RUN_ROOT):
            raise ValueError("Run directory must be inside the documented memory-read directory")
        if args.mode == "replay":
            if not args.run_dir.is_dir():
                raise FileNotFoundError(args.run_dir)
        elif args.run_dir.exists():
            raise FileExistsError(f"Run directory already exists: {args.run_dir}")
    if args.mode in ("train", "replay") and args.run_dir is None:
        raise ValueError(f"{args.mode} requires --run-dir")
    manifest = copy.deepcopy(baseline)
    manifest["baseline_identity"] = {key: copy.deepcopy(baseline[key]) for key in ("source_files", "entrypoint_sha256")}
    manifest["source_files"].update({name: shared.sha256(ROOT / name) for name in METHOD_FILES})
    manifest["entrypoint_sha256"] = shared.sha256(Path(__file__))
    manifest["protocol"] = protocol()
    manifest["b2_reference"] = {"path": str(args.b2_run_dir), **reference,
                                "best_metrics": shared.read_json(args.b2_run_dir / "best_metrics.json")}
    baseline_manifest = shared.read_json(args.b2_run_dir / "manifest.json")
    manifest["b2_reference"]["environment"] = {"versions": baseline_manifest.get("versions", {}),
                                               "gpu": baseline_manifest.get("gpu")}
    if args.check_runtime:
        manifest["environment"] = environment
    if args.mode == "replay":
        verify_method_run(args.run_dir, manifest)
    return manifest


def verify_method_run(run_dir, current):
    saved = shared.read_json(run_dir / "manifest.json")
    for key in ("protocol", "source_files", "entrypoint_sha256", "baseline_identity", "b2_reference"):
        if saved.get(key) != current[key]:
            raise ValueError(f"Method replay identity mismatch: {key}")
    for split in shared.SPLITS:
        if saved["data"][split]["sha256"] != current["data"][split]["sha256"]:
            raise ValueError("Method data identity mismatch")
    if saved["model"]["sha256"] != current["model"]["sha256"]:
        raise ValueError("Method generic model identity mismatch")
    complete = shared.read_json(run_dir / "completion.json")
    replay = shared.read_json(run_dir / "checkpoint_replay.json")
    best = shared.read_json(run_dir / "best_metrics.json")
    if (complete.get("variant") != VARIANT or complete.get("optimizer_steps") != 22890
            or complete.get("test_data_read") is not False or complete.get("test_evaluated") is not False
            or complete.get("checkpoint_replay_passed") is not True or replay.get("passed") is not True
            or replay.get("invariance", {}).get("passed") is not True):
        raise ValueError("Method run is incomplete or has not passed replay/invariance")
    if (not 1 <= best["epoch"] <= 30 or best["step"] != (best["epoch"] - 1) * 763 + 762
            or best.get("batch") != 762 or complete.get("best_epoch") != best["epoch"]
            or complete.get("best_step") != best["step"] or complete.get("best_dev_F1") != best["dev_F1"]
            or replay.get("epoch") != best["epoch"] or replay.get("step") != best["step"]):
        raise ValueError("Method checkpoint selection mismatch")
    if any(replay.get("metrics", {}).get(key) != best.get(key) for key in METRIC_KEYS):
        raise ValueError("Method saved metric mismatch")
    checkpoint_hash = shared.sha256(run_dir / "best_model.pth")
    if saved.get("best_checkpoint_sha256") != checkpoint_hash or replay.get("checkpoint_sha256") != checkpoint_hash:
        raise ValueError("Method checkpoint hash mismatch")


def load_baseline_model(args):
    import numpy as np
    import torch
    from transformers import AutoConfig, AutoModel, AutoTokenizer
    from model2 import DocREModel
    from prepro import read_docred

    args.device = runtime()
    for key, value in protocol().items():
        setattr(args, key, value)
    config = AutoConfig.from_pretrained(str(args.model_dir), num_labels=97, local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(str(args.model_dir), local_files_only=True)
    config.cls_token_id, config.sep_token_id = tokenizer.cls_token_id, tokenizer.sep_token_id
    config.transformer_type = "roberta"
    aligned.emit("preprocessing_train_start", variant=VARIANT)
    training, priors = read_docred(args, str(args.data_dir / "train_revised.json"), tokenizer, 1024)
    if len(training) != 3053:
        raise ValueError("Expected all 3053 training documents")
    training = aligned.canonicalize_features(training, with_labels=True)
    aligned.emit("preprocessing_train_complete", documents=len(training))
    if args.mode == "probe":
        development, gold, annotated = [], [], set()
    else:
        aligned.emit("preprocessing_dev_start")
        development, _ = read_docred(args, str(args.data_dir / "dev_revised.json"), tokenizer, 1024)
        if len(development) != 500:
            raise ValueError("Expected all 500 development documents")
        development = aligned.canonicalize_features(development, with_labels=False)
        gold = shared.read_json(args.data_dir / "dev_revised.json")
        annotated = shared.train_facts(shared.read_json(args.data_dir / "train_revised.json"))
        aligned.emit("preprocessing_dev_complete", documents=len(development))
    aligned.emit("generic_backbone_load_start", model_dir=str(args.model_dir))
    backbone = AutoModel.from_pretrained(str(args.model_dir), config=config, local_files_only=True).to(args.device)
    # Preserve B2's seed timing and complete construction before installing M1.
    shared.seed_all(args.seed, torch, np)
    model = DocREModel(args, config, torch.tensor(priors + 1e-9, device=args.device),
                       backbone, tokenizer).to(args.device)
    aligned.apply_variant(model, "pair-independent")
    aligned.emit("task_model_ready", variant=VARIANT,
                 parameters=sum(parameter.numel() for parameter in model.parameters()))
    return model, training, development, gold, annotated


def load_experiment(args):
    from ttmre_pair_read import install_pair_read
    from ttmre_memory_fit import install_unused_write_bypass, install_ttm_checkpoint
    model, training, development, gold, annotated = load_baseline_model(args)
    installation = install_pair_read(model, args.read_rank, args.alpha_init)
    if installation["new_parameter_count"] != args.new_parameter_count:
        raise RuntimeError("Unexpected new parameter count")
    aligned.emit("pair_read_installed", variant=VARIANT, **installation,
                 parameters=sum(parameter.numel() for parameter in model.parameters()))
    aligned.emit("memory_fit_installed", **install_unused_write_bypass(model))
    aligned.emit("ttm_checkpoint_installed", **install_ttm_checkpoint(model))
    return model, training, development, gold, annotated


def build_optimizer(model, args):
    import torch
    gate = model.mu_encoder.token_turing_machine_unit.token_learner_model.eta
    base, heads = [], []
    for name, parameter in model.named_parameters():
        if parameter is gate:
            continue
        (heads if any(key in name for key in ("extractor", "bilinear")) else base).append(parameter)
    groups = [{"params": base, "lr": 3e-5, "weight_decay": 0.01},
              {"params": heads, "lr": 1e-5, "weight_decay": 0.01},
              {"params": [gate], "lr": 3e-5, "weight_decay": 0.0}]
    return torch.optim.AdamW(groups, lr=3e-5, eps=1e-6, weight_decay=0.01)


def optimizer_report(model, optimizer):
    names = {id(parameter): name for name, parameter in model.named_parameters()}
    seen, report = [], []
    for group in optimizer.param_groups:
        members = [names[id(parameter)] for parameter in group["params"]]
        seen.extend(id(parameter) for parameter in group["params"])
        report.append({"lr": group["lr"], "weight_decay": group["weight_decay"],
                       "parameter_tensors": len(members), "parameters": sum(p.numel() for p in group["params"]),
                       "new_parameters": [name for name in members if any(key in name for key in
                                          ("query_self", "query_other", "memory_key", ".eta"))]})
    if len(seen) != len(set(seen)) or set(seen) != set(names):
        raise RuntimeError("Optimizer does not contain each parameter exactly once")
    return report


def run_probe(model, training, args):
    import torch
    from ttmre_full_model_check import finite_backward, select_batches
    from ttmre_pair_read import new_gradient_report
    from ttmre_memory_fit import reserve_adam_moments
    moment_buffers, reservation = reserve_adam_moments(model)
    aligned.emit("probe_memory_reservation", **reservation)
    reports = []
    for name, indices in select_batches(training, args.train_batch_size).items():
        model.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        loss, gradients = finite_backward(model, shared.collate_fn([training[i] for i in indices]), args.device, torch)
        new_gradients = new_gradient_report(model)
        torch.cuda.synchronize()
        report = {"probe": name, "document_indices": indices,
                  "token_lengths": [len(training[i]["input_ids"]) for i in indices],
                  "entity_pairs": [len(training[i]["hts"]) for i in indices],
                  "loss": loss, "finite_gradient_tensors": gradients, "new_gradients": new_gradients,
                  "seconds": time.perf_counter() - start,
                  "peak_allocated_GiB": torch.cuda.max_memory_allocated() / 1024**3,
                  "peak_reserved_GiB": torch.cuda.max_memory_reserved() / 1024**3,
                  "optimizer_steps": 0, **reservation}
        aligned.emit("memory_read_probe", **report)
        reports.append(report)
    model.zero_grad(set_to_none=True)
    del moment_buffers
    torch.cuda.empty_cache()
    return {"passed": True, "variant": VARIANT, "probes": reports, "optimizer_steps": 0,
            "optimizer_memory_reservation": reservation}


def read_diagnostics(model, features, device):
    import torch
    reader = model.mu_encoder.token_turing_machine_unit.token_learner_model
    captured = []
    def capture(module, inputs):
        remaining = 64 - sum(len(value) for value in captured)
        if remaining > 0:
            captured.append(inputs[0][:remaining].detach().clone())
    handle = reader.register_forward_pre_hook(capture)
    model.eval()
    try:
        with torch.no_grad():
            model(**shared.inputs_from_batch(shared.collate_fn(features[:4]), device, labels=False))
            if not captured:
                raise RuntimeError("No read inputs captured")
            report = reader.diagnostics(torch.cat(captured))
    finally:
        handle.remove()
    if report["memory_mass_max_delta"] > 1e-6 or report["entity_weights_max_delta"] != 0:
        raise RuntimeError("Read no longer preserves the original weight balance")
    return {"scope": "first 64 canonical pairs in first four dev documents; not full-dev coverage",
            "sample_titles": [feature["title"] for feature in features[:4]], **report,
            "optimizer_steps": 0, "test_labels_read": False}


def error_buckets(predictions, gold_documents, features, training_documents, relation_ids):
    predicted = {(item["title"], item["h_idx"], item["t_idx"], item["r"]) for item in predictions}
    gold = {(doc["title"], label["h"], label["t"], label["r"])
            for doc in gold_documents for label in doc["labels"]}
    counts = Counter(label["r"] for doc in training_documents for label in doc["labels"])
    classes = sorted((relation for relation, index in relation_ids.items() if index != 0),
                     key=lambda relation: (counts[relation], relation_ids[relation]))
    if len(classes) != 96:
        raise ValueError("Expected all 96 positive relation classes for frequency buckets")
    frequency = {relation: ("rare", "middle", "frequent")[index * 3 // len(classes)]
                 for index, relation in enumerate(classes)}
    lengths = {item["title"]: len(item["input_ids"]) for item in features}
    if len(lengths) != len(features) or set(lengths) != {doc["title"] for doc in gold_documents}:
        raise ValueError("Error bucket document identities mismatch")
    def summarize(p, g):
        tp = len(p & g)
        precision = tp / len(p) if p else 0.0
        recall = tp / len(g) if g else 0.0
        return {"TP": tp, "FP": len(p - g), "FN": len(g - p),
                "predicted": len(p), "gold": len(g), "precision": 100 * precision,
                "recall": 100 * recall,
                "F1": 200 * precision * recall / (precision + recall) if precision + recall else 0.0}
    relation_report = {}
    for bucket in ("rare", "middle", "frequent"):
        members = {relation for relation in classes if frequency[relation] == bucket}
        relation_report[bucket] = {"relations": sorted(members, key=relation_ids.get),
                                  "train_label_count_min": min(counts[relation] for relation in members),
                                  "train_label_count_max": max(counts[relation] for relation in members),
                                  **summarize({p for p in predicted if p[3] in members},
                                              {g for g in gold if g[3] in members})}
    length_report = {}
    for bucket, long in (("at_most_512", False), ("over_512", True)):
        titles = {title for title, length in lengths.items() if (length > 512) == long}
        length_report[bucket] = {"documents": len(titles),
                                 **summarize({p for p in predicted if p[0] in titles},
                                             {g for g in gold if g[0] in titles})}
    return {"scope": "dev only; frequency buckets fixed by train labels; lengths use pipeline input tokens",
            "frequency_rule": "96 classes sorted by (train label count, relation id); three equal class groups",
            "train_relation_label_counts": dict(sorted(counts.items())),
            "overall": summarize(predicted, gold), "relation_frequency": relation_report,
            "length": length_report, "test_labels_read": False}


def replay_checkpoint(model, args, dev_loader, gold, annotated):
    import numpy as np
    import torch
    from evaluation import to_official
    model.zero_grad(set_to_none=True)
    aligned.emit("checkpoint_replay_start", run_dir=str(args.run_dir))
    model.load_state_dict(torch.load(args.run_dir / "best_model.pth", map_location="cpu", weights_only=True), strict=True)
    expected = shared.read_json(args.run_dir / "best_metrics.json")
    decisions = []
    def capture_decisions(module, inputs, logits):
        with torch.no_grad():
            predicted = torch.zeros_like(logits, dtype=torch.int8)
            predicted[:, 1:] = logits[:, 1:] > logits[:, :1]
            predicted[:, 0] = predicted[:, 1:].sum(1) == 0
            decisions.append(predicted.cpu().numpy())
    handle = model.register_forward_hook(capture_decisions)
    try:
        actual = shared.evaluate_dev(model, dev_loader, args.device, gold, annotated)
    finally:
        handle.remove()
    for key in METRIC_KEYS:
        value = actual[key]
        if (value is None and expected.get(key) is not None or value is not None and
                not math.isclose(value, expected[key], rel_tol=0, abs_tol=1e-6)):
            raise RuntimeError(f"Checkpoint metric replay mismatch: {key}")
    invariance = aligned.grouping_invariance(model, dev_loader.dataset, args.device)
    if not invariance["passed"]:
        raise RuntimeError(f"M1 grouping invariance failed: {invariance}")
    predictions = to_official(np.concatenate(decisions), dev_loader.dataset)
    buckets = error_buckets(predictions, gold, dev_loader.dataset,
                           shared.read_json(args.data_dir / "train_revised.json"),
                           shared.read_json(ROOT / "meta" / "rel2id.json"))
    if not math.isclose(buckets["overall"]["F1"], actual["dev_F1"], rel_tol=0, abs_tol=1e-6):
        raise RuntimeError("Error bucket predictions differ from replay metrics")
    report = {"passed": True, "variant": VARIANT, "epoch": expected["epoch"], "step": expected["step"],
              "metrics": actual, "invariance": invariance, "error_buckets": buckets,
              "checkpoint_sha256": shared.sha256(args.run_dir / "best_model.pth"),
              "optimizer_steps": 0, "test_data_read": False, "test_evaluated": False}
    aligned.emit("checkpoint_replay_passed", **report)
    return report


def run(args, manifest):
    import numpy as np
    import torch
    import transformers
    from torch.utils.data import DataLoader
    from transformers.optimization import get_linear_schedule_with_warmup
    if args.mode == "train":
        args.run_dir.mkdir(parents=True, exist_ok=False)
        args._owns_run_dir = True
        manifest.update({"created_utc": datetime.now(timezone.utc).isoformat(),
                         "scope": {"test_data_read": False, "distant_data_read": False, "model_download": False},
                         "deviations": ["B2 canonical pairs and corrected TTM axes", "M1 slot-conditional memory read",
                                        "gate without weight decay", "offline generic backbone", "test isolation",
                                        "single-step unused write bypass; parameters retained",
                                        "non-reentrant TTM activation checkpointing; RNG preserved"]})
        shared.write_json(args.run_dir / "manifest.json", manifest)
    model, training, development, gold, annotated = load_experiment(args)
    manifest["environment"] = environment_report()
    aligned.emit("experiment_environment", **manifest["environment"])
    if args.mode == "train":
        shared.write_json(args.run_dir / "manifest.json", manifest)
    if args.mode == "probe":
        run_probe(model, training, args)
        print("TTMRE_MEMORY_READ_PROBE_OK; optimizer_steps=0; test_labels_read=false; model_download=false; artifacts_saved=false", flush=True)
        return
    dev_loader = DataLoader(development, batch_size=4, shuffle=False, collate_fn=shared.collate_fn)
    if args.mode == "replay":
        replay = replay_checkpoint(model, args, dev_loader, gold, annotated)
        print(json.dumps({"replay": replay, "read_diagnostics": read_diagnostics(model, development, args.device)}, allow_nan=False), flush=True)
        return
    probe = run_probe(model, training, args)
    shared.write_json(args.run_dir / "full_model_check.json", probe)
    initial_diagnostics = read_diagnostics(model, development, args.device)
    shared.write_json(args.run_dir / "initial_read_diagnostics.json", initial_diagnostics)
    optimizer = build_optimizer(model, args)
    groups = optimizer_report(model, optimizer)
    shared.seed_all(args.seed, torch, np)
    loader = DataLoader(training, batch_size=4, shuffle=True, drop_last=True, collate_fn=shared.collate_fn)
    total = len(loader) * args.num_train_epochs
    if len(loader) != 763 or total != 22890:
        raise ValueError("M1 requires 763 batches/epoch and 22890 updates")
    scheduler = get_linear_schedule_with_warmup(optimizer, int(total * 0.06), total)
    manifest.update({"versions": {"torch": torch.__version__, "cuda": torch.version.cuda,
                                  "transformers": transformers.__version__, "numpy": np.__version__},
                     "gpu": torch.cuda.get_device_name(0), "total_steps": total, "warmup_steps": int(total * 0.06),
                     "batches_per_epoch": len(loader), "parameters": sum(p.numel() for p in model.parameters()),
                     "optimizer_groups": groups})
    shared.write_json(args.run_dir / "manifest.json", manifest)
    shared.write_json(args.run_dir / "args.json", vars(args))
    with (args.run_dir / "train.log").open("x", encoding="utf-8") as train_log, \
            (args.run_dir / "metrics.jsonl").open("x", encoding="utf-8") as metrics_log:
        complete = aligned.train_loop(args, model, loader, dev_loader, gold, annotated,
                                      optimizer, scheduler, train_log, metrics_log)
    if complete["optimizer_steps"] != total:
        raise RuntimeError("Training did not complete the fixed budget")
    del optimizer, scheduler
    model.zero_grad(set_to_none=True)
    torch.cuda.empty_cache()
    replay = replay_checkpoint(model, args, dev_loader, gold, annotated)
    diagnostics = read_diagnostics(model, development, args.device)
    shared.write_json(args.run_dir / "checkpoint_replay.json", replay)
    shared.write_json(args.run_dir / "read_diagnostics.json", diagnostics)
    manifest["best_checkpoint_sha256"] = replay["checkpoint_sha256"]
    shared.write_json(args.run_dir / "manifest.json", manifest)
    baseline_f1 = manifest["b2_reference"]["best_metrics"]["dev_F1"]
    shared.write_json(args.run_dir / "completion.json", {"variant": VARIANT, **complete,
        "checkpoint_replay_passed": True, "b2_dev_F1": baseline_f1,
        "dev_F1_delta_vs_b2": complete["best_dev_F1"] - baseline_f1,
        "comparison_scope": "cross_environment_screening",
        "interpretation": "single-seed dev screening across environments; method effect requires a same-environment B2 control"})
    aligned.emit("training_complete", variant=VARIANT, **complete)
    print("TTMRE_MEMORY_READ_TRAINING_COMPLETE; checkpoint_replay=passed; test_labels_read=false", flush=True)


def main():
    args = parser().parse_args()
    aligned.emit("identity_validation_start", mode=args.mode, variant=VARIANT)
    manifest = validate(args)
    if args.mode == "validate":
        print(json.dumps({"validation": "passed", "manifest": manifest,
                          "optimizer_steps": 0, "model_download": False, "files_written": False}, indent=2))
        return
    os.chdir(ROOT)
    sys.path.insert(0, str(ROOT))
    args._owns_run_dir = False
    try:
        run(args, manifest)
    except Exception as error:
        if args.mode == "train" and args._owns_run_dir:
            shared.write_json(args.run_dir / "failure.json", {"error_type": type(error).__name__, "error": str(error),
                              "time_utc": datetime.now(timezone.utc).isoformat()})
        raise


if __name__ == "__main__":
    main()
