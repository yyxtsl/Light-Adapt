"""Offline B1/B2 controls; validation is read-only and training uses a new run directory."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import sys
import time

import train_ttmre_human_only as shared

ROOT = shared.ROOT
RUN_ROOT = ROOT / "experiments" / "2026-10-08-baseline-alignment"
VARIANTS = ("upstream-compatible", "pair-independent")
EXTRA_SOURCE_HASHES = {
    "train2.py": "ca8307ea44e6e6b5cfc84081906a78fadcde1969d6b058b24582d2f1439e69ff",
    "evaluation.py": "ac760f79d39957c7ab6ffb8a33784330677d83482f44acec92bee25d280e52d2",
    "scripts/train_ttmre_human_only.py": "4ef536052f5896beaa611fcfb1e86aeafa37f2d9b5a868c80ca239dc80117ecf",
    "scripts/ttmre_full_model_check.py": "5f54e5cff0cf8f8035aef97cec0160ba0c1ed299ce2552f83eb9d1b2864e64f5",
}


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--mode", choices=("validate", "probe", "train", "replay"), default="validate")
    result.add_argument("--variant", choices=VARIANTS, required=True)
    result.add_argument("--model-dir", type=Path)
    result.add_argument("--data-dir", type=Path, default=ROOT / "dataset" / "docred")
    result.add_argument("--run-dir", type=Path)
    result.add_argument("--b1-run-dir", type=Path, help="Completed B1 artifacts; required for B2 training")
    return result


def protocol(variant):
    return {
        "variant": variant, "seed": 74, "num_train_epochs": 30,
        "train_batch_size": 4, "dev_batch_size": 4, "precision": "float32",
        "model_type": "ttmre", "m_tag": "S-PU", "isrank": 1,
        "num_class": 97, "num_layers": 4, "memory_size": 200,
        "e": 1.0, "m": 1.0, "beta": 0.0, "gamma": 1.0,
        "max_seq_length": 1024, "learning_rate": 3e-5,
        "head_learning_rate": 1e-5, "adam_epsilon": 1e-6, "weight_decay": 0.01,
        "gradient_accumulation_steps": 1, "max_grad_norm": 1.0, "warmup_ratio": 0.06,
        "seed_timing": "after generic backbone load, before DocREModel; reset before train loader",
        "dev_timing": "after penultimate training batch update in each epoch",
        "selection": "dev_F1; earliest on ties",
        "pair_order": "upstream positive-first" if variant == VARIANTS[0] else "entity-index canonical",
        "ttm_axes": "upstream sequence-first" if variant == VARIANTS[0] else "transpose before/after processing Transformer",
        "initialization": "generic RoBERTa-large; no task checkpoint; upstream Xavier memory",
    }


def validate(args):
    if args.model_dir is None:
        source = shared.read_json(ROOT / "models" / "roberta-large-source.json")
        args.model_dir = (ROOT / "models" / "hf-cache" / "models--FacebookAI--roberta-large"
                          / "snapshots" / source["revision"])
    settings = shared.parser().parse_args(["--model-dir", str(args.model_dir),
                                          "--data-dir", str(args.data_dir)])
    manifest = shared.validate(settings)
    args.model_dir, args.data_dir = settings.model_dir, settings.data_dir
    for name, expected in EXTRA_SOURCE_HASHES.items():
        actual = shared.sha256(ROOT / name)
        if actual != expected:
            raise ValueError(f"Pinned source/helper identity mismatch: {name}")
        manifest["source_files"][name] = actual
    for name in ("scripts/check_ttmre_baseline_aligned.py", "scripts/run_ttmre_baseline_aligned.sh"):
        manifest["source_files"][name] = shared.sha256(ROOT / name)
    manifest["entrypoint_sha256"] = shared.sha256(Path(__file__))
    manifest["protocol"] = protocol(args.variant)
    manifest["distant_data_read"] = False
    if args.run_dir is not None:
        args.run_dir = args.run_dir.expanduser().resolve()
        if not args.run_dir.is_relative_to(RUN_ROOT) or args.run_dir == RUN_ROOT:
            raise ValueError("Run directory must be inside the documented baseline-alignment directory")
        if args.mode == "replay":
            if not args.run_dir.is_dir():
                raise FileNotFoundError(args.run_dir)
        elif args.run_dir.exists():
            raise FileExistsError(f"Run directory already exists: {args.run_dir}")
    if args.mode in ("train", "replay") and args.run_dir is None:
        raise ValueError(f"{args.mode} requires --run-dir")
    if args.variant == VARIANTS[1] and (args.mode == "train" or args.b1_run_dir is not None):
        if args.b1_run_dir is None:
            raise ValueError("Finish and replay B1 before B2; supply --b1-run-dir")
        args.b1_run_dir = args.b1_run_dir.expanduser().resolve()
        reference = verify_completed_run(args.b1_run_dir, VARIANTS[0], manifest)
        manifest["b1_reference"] = {"path": str(args.b1_run_dir), **reference}
    if args.mode == "replay":
        verify_completed_run(args.run_dir, args.variant, manifest)
    return manifest


def verify_completed_run(run_dir, variant, current):
    saved = shared.read_json(run_dir / "manifest.json")
    complete = shared.read_json(run_dir / "completion.json")
    replay = shared.read_json(run_dir / "checkpoint_replay.json")
    best = shared.read_json(run_dir / "best_metrics.json")
    if saved.get("protocol") != protocol(variant):
        raise ValueError("Reference run has a different baseline protocol")
    for key in ("source_files", "entrypoint_sha256"):
        if saved.get(key) != current[key]:
            raise ValueError(f"Reference run source mismatch: {key}")
    for name in shared.SPLITS:
        if saved["data"][name]["sha256"] != current["data"][name]["sha256"]:
            raise ValueError("Reference data identity mismatch")
    if saved["model"]["sha256"] != current["model"]["sha256"]:
        raise ValueError("Reference generic model identity mismatch")
    if (complete.get("optimizer_steps") != 22890 or complete.get("test_data_read") is not False
            or complete.get("test_evaluated") is not False or replay.get("passed") is not True
            or complete.get("checkpoint_replay_passed") is not True
            or complete.get("best_epoch") != best.get("epoch")
            or complete.get("best_step") != best.get("step")
            or complete.get("best_dev_F1") != best.get("dev_F1")):
        raise ValueError("Reference run is incomplete or has not passed checkpoint replay")
    if not 1 <= best["epoch"] <= 30 or replay.get("epoch") != best["epoch"] or replay.get("step") != best["step"]:
        raise ValueError("Reference replay checkpoint selection mismatch")
    for key in ("dev_F1", "dev_precision", "dev_recall", "dev_IgnF1_annotated", "dev_IgnF1_distant"):
        if replay.get("metrics", {}).get(key) != best.get(key):
            raise ValueError(f"Reference replay metrics mismatch: {key}")
    if best["step"] != (best["epoch"] - 1) * 763 + 762:
        raise ValueError("Reference checkpoint was not selected at the penultimate batch")
    checkpoint_hash = shared.sha256(run_dir / "best_model.pth")
    if saved.get("best_checkpoint_sha256") != checkpoint_hash or replay.get("checkpoint_sha256") != checkpoint_hash:
        raise ValueError("Reference checkpoint identity mismatch")
    if variant == VARIANTS[1] and replay.get("invariance", {}).get("passed") is not True:
        raise ValueError("B2 checkpoint has not passed grouping invariance checks")
    return {"checkpoint_sha256": checkpoint_hash,
            "completion_sha256": shared.sha256(run_dir / "completion.json")}


def canonicalize_features(features, with_labels):
    result = []
    for feature in features:
        count = len(feature["entity_pos"])
        ordered = [(head, tail) for head in range(count) for tail in range(count) if head != tail]
        pairs = [tuple(pair) for pair in feature["hts"]]
        if len(pairs) != len(ordered) or len(set(pairs)) != len(pairs) or set(pairs) != set(ordered):
            raise ValueError("Each directed non-self pair must occur exactly once")
        item = {key: value for key, value in feature.items() if key != "labels"}
        item["hts"] = [list(pair) for pair in ordered]
        item["labels"] = []  # The shared collator expects this key, even during inference.
        if with_labels:
            labels = feature["labels"]
            if len(labels) != len(pairs) or any(len(row) != 97 for row in labels):
                raise ValueError("Training pair/label count or class dimension mismatch")
            positions = {pair: index for index, pair in enumerate(pairs)}
            item["labels"] = [list(labels[positions[pair]]) for pair in ordered]
        result.append(item)
    return result


def _sequence_first_input(module, inputs):
    return (inputs[0].transpose(0, 1),)


def _batch_first_output(module, inputs, output):
    return output.transpose(0, 1)


def apply_variant(model, variant):
    if variant not in VARIANTS:
        raise ValueError(f"Unknown variant: {variant}")
    if variant == VARIANTS[1]:
        encoder = model.mu_encoder.token_turing_machine_unit.vit_encoder
        if getattr(encoder, "_ttmre_pair_independent", False):
            raise ValueError("Axis correction already installed")
        if any(layer.self_attn.batch_first for layer in encoder.layers):
            raise ValueError("Expected the pinned sequence-first Transformer")
        # Hooks preserve all original parameter names and state_dict keys.
        encoder.register_forward_pre_hook(_sequence_first_input)
        encoder.register_forward_hook(_batch_first_output)
        encoder._ttmre_pair_independent = True
    return model


def runtime():
    import numpy as np
    import torch
    import transformers
    if (torch.__version__.split("+")[0] != "2.0.1" or torch.version.cuda != "11.8"
            or transformers.__version__ != "4.34.0" or np.__version__ != "1.24.3"):
        raise RuntimeError("Use the verified cloud versions: torch 2.0.1/cu118, transformers 4.34.0, numpy 1.24.3")
    if not torch.cuda.is_available():
        raise RuntimeError("Upstream S-PU requires CUDA")
    return torch.device("cuda:0")


def emit(event, **values):
    print(json.dumps({"event": event, **values}, allow_nan=False), flush=True)


def load_experiment(args):
    import numpy as np
    import torch
    from transformers import AutoConfig, AutoModel, AutoTokenizer
    from model2 import DocREModel
    from prepro import read_docred

    args.device = runtime()
    for key, value in protocol(args.variant).items():
        setattr(args, key, value)
    config = AutoConfig.from_pretrained(str(args.model_dir), num_labels=97, local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(str(args.model_dir), local_files_only=True)
    config.cls_token_id, config.sep_token_id = tokenizer.cls_token_id, tokenizer.sep_token_id
    config.transformer_type = "roberta"
    emit("preprocessing_train_start", variant=args.variant)
    training, priors = read_docred(args, str(args.data_dir / "train_revised.json"), tokenizer, 1024)
    if len(training) != 3053:
        raise ValueError("Expected all 3053 training documents")
    if args.variant == VARIANTS[1]:
        training = canonicalize_features(training, with_labels=True)
    emit("preprocessing_train_complete", documents=len(training))
    if args.mode == "probe":
        development, gold, annotated = [], [], set()
    else:
        emit("preprocessing_dev_start")
        development, _ = read_docred(args, str(args.data_dir / "dev_revised.json"), tokenizer, 1024)
        if len(development) != 500:
            raise ValueError("Expected all 500 development documents")
        if args.variant == VARIANTS[1]:
            development = canonicalize_features(development, with_labels=False)
        gold = shared.read_json(args.data_dir / "dev_revised.json")
        annotated = shared.train_facts(shared.read_json(args.data_dir / "train_revised.json"))
        emit("preprocessing_dev_complete", documents=len(development))
    emit("generic_backbone_load_start", model_dir=str(args.model_dir))
    backbone = AutoModel.from_pretrained(str(args.model_dir), config=config, local_files_only=True).to(args.device)
    # Official main sets the task-model seed after loading the generic backbone.
    shared.seed_all(args.seed, torch, np)
    model = DocREModel(args, config, torch.tensor(priors + 1e-9, device=args.device),
                       backbone, tokenizer).to(args.device)
    apply_variant(model, args.variant)
    emit("task_model_ready", variant=args.variant,
         parameters=sum(parameter.numel() for parameter in model.parameters()))
    return model, training, development, gold, annotated


def run_probe(model, training, args):
    import torch
    from ttmre_full_model_check import finite_backward, select_batches

    reports = []
    parameter_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    emit("full_model_probe_start", variant=args.variant, parameter_GiB=parameter_bytes / 1024**3,
         adam_moments_upper_estimate_GiB=2 * parameter_bytes / 1024**3)
    for name, indices in select_batches(training, args.train_batch_size).items():
        model.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        free, total = torch.cuda.mem_get_info()
        start = time.perf_counter()
        loss, gradients = finite_backward(model, shared.collate_fn([training[i] for i in indices]), args.device, torch)
        torch.cuda.synchronize()
        report = {"probe": name, "document_indices": indices,
                  "titles": [training[i]["title"] for i in indices],
                  "token_lengths": [len(training[i]["input_ids"]) for i in indices],
                  "entity_pairs": [len(training[i]["hts"]) for i in indices],
                  "loss": loss, "finite_gradient_tensors": gradients,
                  "seconds": time.perf_counter() - start,
                  "peak_allocated_GiB": torch.cuda.max_memory_allocated() / 1024**3,
                  "peak_reserved_GiB": torch.cuda.max_memory_reserved() / 1024**3,
                  "free_before_probe_GiB": free / 1024**3, "total_GiB": total / 1024**3,
                  "optimizer_steps": 0}
        emit("full_model_probe", **report)
        reports.append(report)
    model.zero_grad(set_to_none=True)
    torch.cuda.empty_cache()
    return {"passed": True, "variant": args.variant, "probes": reports, "optimizer_steps": 0}


def grouping_invariance(model, features, device):
    import torch

    model.eval()
    with torch.no_grad():
        sample = features[:4]
        grouped = model(**shared.inputs_from_batch(shared.collate_fn(sample), device, labels=False)).cpu()
        separate = torch.cat([model(**shared.inputs_from_batch(shared.collate_fn([item]), device,
                                                               labels=False)).cpu() for item in sample])
        generator = torch.Generator().manual_seed(74)
        tokens = torch.randn(32, 1, 2, model.emb_size, generator=generator).to(device)
        whole = model.mu_encoder(tokens).cpu()
        single = torch.cat([model.mu_encoder(item.unsqueeze(0)).cpu() for item in tokens])
        perm = torch.randperm(len(tokens), generator=generator)
        restored = model.mu_encoder(tokens[perm.to(device)]).cpu()[torch.argsort(perm)]
        checks = {"documents_grouped_vs_separate": (grouped, separate),
                  "ttm_grouped_vs_single": (whole, single), "ttm_permutation": (whole, restored)}
        report = {name: {"max_abs_delta": (a - b).abs().max().item(),
                         "close": bool(torch.allclose(a, b, rtol=1e-4, atol=1e-4))}
                  for name, (a, b) in checks.items()}
        decisions = ((grouped[:, 1:] > grouped[:, :1]) != (separate[:, 1:] > separate[:, :1])).sum().item()
        passed = all(item["close"] for item in report.values()) and decisions == 0
    return {"passed": passed, "sample_documents": len(sample), "sample_ttm_pairs": 32,
            "changed_relation_decisions": decisions, "tolerances": {"rtol": 1e-4, "atol": 1e-4},
            "checks": report}


def replay_checkpoint(model, args, dev_loader, gold, annotated):
    import torch

    model.zero_grad(set_to_none=True)
    emit("checkpoint_replay_start", run_dir=str(args.run_dir))
    model.load_state_dict(torch.load(args.run_dir / "best_model.pth", map_location="cpu", weights_only=True), strict=True)
    expected = shared.read_json(args.run_dir / "best_metrics.json")
    actual = shared.evaluate_dev(model, dev_loader, args.device, gold, annotated)
    for key, value in actual.items():
        if value is None:
            if expected.get(key) is not None:
                raise RuntimeError(f"Checkpoint metric replay mismatch: {key}")
        elif not math.isclose(value, expected[key], rel_tol=0, abs_tol=1e-6):
            raise RuntimeError(f"Checkpoint metric replay mismatch: {key}: {value} vs {expected[key]}")
    report = {"passed": True, "variant": args.variant, "epoch": expected["epoch"],
              "step": expected["step"], "metrics": actual,
              "checkpoint_sha256": shared.sha256(args.run_dir / "best_model.pth"),
              "optimizer_steps": 0, "test_data_read": False, "test_evaluated": False}
    if args.variant == VARIANTS[1]:
        report["invariance"] = grouping_invariance(model, dev_loader.dataset, args.device)
        if not report["invariance"]["passed"]:
            raise RuntimeError(f"B2 grouping invariance failed: {report['invariance']}")
    emit("checkpoint_replay_passed", **report)
    return report


def train_loop(args, model, loader, dev_loader, gold, annotated, optimizer, scheduler, train_log, metrics_log):
    import torch

    if len(loader) < 2:
        raise ValueError("Penultimate-batch evaluation requires at least two training batches")
    best_score, best_epoch, best_step, steps = -math.inf, None, None, 0
    for epoch in range(1, args.num_train_epochs + 1):
        total_loss = 0.0
        emit("epoch_start", epoch=epoch, batches=len(loader))
        for index, batch in enumerate(loader):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            risks, logits = model(**shared.inputs_from_batch(batch, args.device))
            loss = sum(risks)
            if not torch.isfinite(loss).item() or not torch.isfinite(logits).all().item():
                raise RuntimeError(f"Nonfinite training forward at update {steps + 1}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            steps += 1
            total_loss += loss.detach().item()
            if steps == 1 or steps % 100 == 0:
                record = {"event": "optimizer_update", "epoch": epoch, "step": steps,
                          "batch": index + 1, "loss": loss.detach().item()}
                line = json.dumps(record, allow_nan=False)
                print(line, flush=True)
                train_log.write(line + "\n")
                train_log.flush()
            if index + 1 == len(loader) - 1:
                emit("development_evaluation_start", epoch=epoch, step=steps, batch=index + 1)
                metrics = shared.evaluate_dev(model, dev_loader, args.device, gold, annotated)
                best_score, selected = shared.record_best(model, metrics["dev_F1"], best_score, epoch, args.run_dir)
                if selected is not None:
                    best_epoch, best_step = selected, steps
                    shared.write_json(args.run_dir / "best_metrics.json",
                                      {"epoch": epoch, "step": steps, "batch": index + 1, **metrics})
                record = {"event": "development_evaluation", "epoch": epoch, "step": steps,
                          "batch": index + 1, "train_loss_so_far": total_loss / (index + 1),
                          "selected": selected is not None, **metrics}
                line = json.dumps(record, allow_nan=False)
                print(line, flush=True)
                for handle in (train_log, metrics_log):
                    handle.write(line + "\n")
                    handle.flush()
        emit("epoch_complete", epoch=epoch, step=steps, train_loss=total_loss / len(loader))
    return {"best_epoch": best_epoch, "best_step": best_step,
            "best_dev_F1": best_score, "optimizer_steps": steps,
            "test_data_read": False, "test_evaluated": False}


def run(args, manifest):
    import numpy as np
    import torch
    import transformers
    from torch.utils.data import DataLoader
    from transformers.optimization import get_linear_schedule_with_warmup

    # Reserve the new artifacts directory before loading data or weights.
    if args.mode == "train":
        args.run_dir.mkdir(parents=True, exist_ok=False)
        args._owns_run_dir = True
        manifest.update({"created_utc": datetime.now(timezone.utc).isoformat(),
                         "deviations": ["test isolation (including test-loader RNG consumption)",
                                        "offline generic backbone load", "finite-value checks",
                                        "empty predictions score zero", "CPU checkpoint snapshots are cloned"],
                         "scope": {"test_data_read": False, "distant_data_read": False, "model_download": False}})
        if args.variant == VARIANTS[1]:
            manifest["deviations"] += ["TTM batch axis correction", "canonical entity-pair order"]
        shared.write_json(args.run_dir / "manifest.json", manifest)
        shared.write_json(args.run_dir / "args.json", vars(args))
    model, training, development, gold, annotated = load_experiment(args)
    if args.mode == "probe":
        run_probe(model, training, args)
        print("TTMRE_ALIGNED_PROBE_OK; optimizer_steps=0; test_labels_read=false; model_download=false; artifacts_saved=false", flush=True)
        return
    dev_loader = DataLoader(development, batch_size=args.dev_batch_size, shuffle=False,
                            collate_fn=shared.collate_fn, drop_last=False)
    if args.mode == "replay":
        replay_checkpoint(model, args, dev_loader, gold, annotated)
        print("TTMRE_ALIGNED_REPLAY_OK; optimizer_steps=0; test_labels_read=false; files_written=false", flush=True)
        return
    if args.variant == VARIANTS[1]:
        probe = run_probe(model, training, args)
        shared.write_json(args.run_dir / "full_model_check.json", probe)
    optimizer = shared.build_optimizer(model, args)
    shared.seed_all(args.seed, torch, np)
    loader = DataLoader(training, batch_size=args.train_batch_size, shuffle=True,
                        collate_fn=shared.collate_fn, drop_last=True)
    total = len(loader) * args.num_train_epochs
    if len(loader) != 763 or total != 22890:
        raise ValueError("Aligned baseline requires 763 batches/epoch and 22890 updates")
    scheduler = get_linear_schedule_with_warmup(optimizer, int(total * 0.06), total)
    manifest.update({"versions": {"torch": torch.__version__, "cuda": torch.version.cuda,
                                  "transformers": transformers.__version__, "numpy": np.__version__},
                     "gpu": torch.cuda.get_device_name(0), "total_steps": total,
                     "warmup_steps": int(total * 0.06), "batches_per_epoch": len(loader),
                     "parameters": sum(p.numel() for p in model.parameters())})
    shared.write_json(args.run_dir / "manifest.json", manifest)
    shared.write_json(args.run_dir / "args.json", vars(args))
    with (args.run_dir / "train.log").open("x", encoding="utf-8") as train_log, \
            (args.run_dir / "metrics.jsonl").open("x", encoding="utf-8") as metrics_log:
        complete = train_loop(args, model, loader, dev_loader, gold, annotated,
                              optimizer, scheduler, train_log, metrics_log)
    # Release optimizer moments before loading/replaying the selected checkpoint.
    del optimizer, scheduler
    model.zero_grad(set_to_none=True)
    torch.cuda.empty_cache()
    replay = replay_checkpoint(model, args, dev_loader, gold, annotated)
    shared.write_json(args.run_dir / "checkpoint_replay.json", replay)
    manifest["best_checkpoint_sha256"] = replay["checkpoint_sha256"]
    shared.write_json(args.run_dir / "manifest.json", manifest)
    shared.write_json(args.run_dir / "completion.json", {"variant": args.variant, **complete,
                                                         "checkpoint_replay_passed": True})
    emit("training_complete", variant=args.variant, **complete)
    print("TTMRE_ALIGNED_TRAINING_COMPLETE; checkpoint_replay=passed; test_labels_read=false", flush=True)


def main():
    args = parser().parse_args()
    emit("identity_validation_start", mode=args.mode, variant=args.variant)
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
            shared.write_json(args.run_dir / "failure.json",
                              {"error_type": type(error).__name__, "error": str(error),
                               "time_utc": datetime.now(timezone.utc).isoformat()})
        raise


if __name__ == "__main__":
    main()
