"""One matched RTX 5090 B2 seed-74 control; default validation is read-only."""

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

import train_ttmre_memory_read as m1

aligned, shared = m1.aligned, m1.shared
ROOT = shared.ROOT
RUN_ROOT = ROOT / "experiments" / "2026-10-10-b2-5090-control"
VARIANT = "pair-independent"
PINNED_M1_FILES = {
    "scripts/train_ttmre_memory_read.py": "79d4f07da6d3bee77ed4db426d6fecf167db68f0e657437f01a0437ef7bd7568",
    "scripts/ttmre_pair_read.py": "530ee978e19e536accadf5aa2f02e23678dabdf8fb98b970a493a59abcd787d0",
    "scripts/ttmre_memory_fit.py": "f1e2e41e6e7368d6f11be4a3333a1febb237c2fe6300f48b6ed1eba09650a1da",
    "scripts/check_ttmre_memory_read.py": "b41f2f6f6600e83cb755a9aa18a357cb9dd2b82237c04e923faf2135bb6ec1fc",
    "scripts/run_ttmre_memory_read.sh": "583f1469d69f51c8de18c90137681741805269c39bed5c59f4250754faf8c838",
}
CONTROL_FILES = ("scripts/check_ttmre_b2_5090.py", "scripts/run_ttmre_b2_5090.sh")


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--mode", choices=("validate", "probe", "train", "replay"), default="validate")
    result.add_argument("--model-dir", type=Path)
    result.add_argument("--data-dir", type=Path, default=ROOT / "dataset" / "docred")
    result.add_argument("--run-dir", type=Path)
    result.add_argument("--b2-run-dir", type=Path, required=True, help="Completed historical B2 artifacts; read only")
    result.add_argument("--m1-run-dir", type=Path, required=True, help="Completed RTX 5090 M1 artifacts; read only")
    result.add_argument("--check-runtime", action="store_true")
    return result


def protocol():
    return {**aligned.protocol(VARIANT), "runtime_profile": dict(m1.RUNTIME_PROFILE),
            "execution_profile": m1.protocol()["execution_profile"],
            "control_version": "matched-b2-5090-v1", "expected_parameters": 381175401}


def verify_environment(actual, expected):
    required = {"versions", "gpu", "capability", "build_architectures", "matmul_allow_tf32",
                "cudnn_allow_tf32", "cudnn_benchmark", "cudnn_deterministic",
                "float32_matmul_precision", "cuda_allocator_config"}
    if not isinstance(expected, dict) or not required.issubset(expected):
        raise ValueError("Completed M1 is missing the recorded runtime or math settings")
    versions = expected["versions"]
    if (not isinstance(versions, dict) or versions != {**m1.RUNTIME_PROFILE, "torch": "2.7.1+cu128"}
            or expected["capability"] != [12, 0] or "sm_120" not in expected["build_architectures"]
            or any(type(expected[key]) is not bool for key in
                   ("matmul_allow_tf32", "cudnn_allow_tf32", "cudnn_benchmark", "cudnn_deterministic"))):
        raise ValueError("Completed M1 has an invalid Blackwell environment record")
    if actual != expected:
        differences = {key: {"actual": actual.get(key), "expected": expected.get(key)}
                       for key in set(actual) | set(expected) if actual.get(key) != expected.get(key)}
        raise ValueError(f"Environment differs from completed M1: {differences}")
    if actual.get("gpu") != "NVIDIA GeForce RTX 5090":
        raise ValueError("This control requires the matched RTX 5090 GPU")
    if actual.get("cuda_allocator_config") != "expandable_segments:True":
        raise ValueError("Use PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True")


def runtime(expected):
    device = m1.runtime()
    verify_environment(m1.environment_report(), expected)
    return device


def validate(args):
    for name, expected in PINNED_M1_FILES.items():
        if shared.sha256(ROOT / name) != expected:
            raise ValueError(f"Frozen M1 helper identity mismatch: {name}")
    settings = m1.parser().parse_args(["--b2-run-dir", str(args.b2_run_dir),
                                      "--data-dir", str(args.data_dir)])
    settings.model_dir = args.model_dir
    reference = m1.validate(settings)
    args.model_dir, args.data_dir, args.b2_run_dir = settings.model_dir, settings.data_dir, settings.b2_run_dir
    args.m1_run_dir = args.m1_run_dir.expanduser().resolve()
    m1.verify_method_run(args.m1_run_dir, reference)
    saved = shared.read_json(args.m1_run_dir / "manifest.json")
    # Validate the environment even without CUDA initialization in read-only mode.
    expected_env = saved.get("environment", {})
    verify_environment(expected_env, expected_env)
    if saved.get("total_steps") != 22890 or saved.get("parameters") != 381322858:
        raise ValueError("Completed M1 has an unexpected training budget or parameter count")
    if args.run_dir is not None:
        args.run_dir = args.run_dir.expanduser().resolve()
        if args.run_dir == RUN_ROOT or not args.run_dir.is_relative_to(RUN_ROOT):
            raise ValueError("Run directory must be inside the documented B2-5090 control root")
        if args.mode == "replay":
            if not args.run_dir.is_dir():
                raise FileNotFoundError(args.run_dir)
        elif args.run_dir.exists():
            raise FileExistsError(f"Run directory already exists: {args.run_dir}")
    if args.mode in ("train", "replay") and args.run_dir is None:
        raise ValueError(f"{args.mode} requires --run-dir")
    manifest = copy.deepcopy(reference)
    manifest["m1_reference"] = {
        "path": str(args.m1_run_dir), "manifest_sha256": shared.sha256(args.m1_run_dir / "manifest.json"),
        "completion_sha256": shared.sha256(args.m1_run_dir / "completion.json"),
        "checkpoint_sha256": saved["best_checkpoint_sha256"],
        "best_metrics": shared.read_json(args.m1_run_dir / "best_metrics.json"),
        "environment": expected_env}
    manifest["source_files"]["scripts/train_ttmre_memory_read.py"] = PINNED_M1_FILES["scripts/train_ttmre_memory_read.py"]
    manifest["source_files"].update({name: shared.sha256(ROOT / name) for name in CONTROL_FILES})
    manifest["entrypoint_sha256"] = shared.sha256(Path(__file__))
    manifest["protocol"] = protocol()
    if args.check_runtime:
        runtime(expected_env)
        aligned.emit("matched_environment_validation_passed", **m1.environment_report())
    if args.mode == "replay":
        verify_completed_control(args.run_dir, manifest)
    return manifest


def verify_completed_control(run_dir, current):
    saved = shared.read_json(run_dir / "manifest.json")
    for key in ("source_files", "entrypoint_sha256", "protocol", "baseline_identity", "b2_reference", "m1_reference"):
        if saved.get(key) != current[key]:
            raise ValueError(f"Control replay identity mismatch: {key}")
    for split in shared.SPLITS:
        if saved["data"][split]["sha256"] != current["data"][split]["sha256"]:
            raise ValueError("Control replay data mismatch")
    if saved["model"]["sha256"] != current["model"]["sha256"]:
        raise ValueError("Control replay model mismatch")
    verify_environment(saved["environment"], current["m1_reference"]["environment"])
    best = shared.read_json(run_dir / "best_metrics.json")
    complete = shared.read_json(run_dir / "completion.json")
    replay = shared.read_json(run_dir / "checkpoint_replay.json")
    if (complete.get("variant") != VARIANT or complete.get("optimizer_steps") != 22890
            or complete.get("checkpoint_replay_passed") is not True
            or complete.get("test_data_read") is not False or complete.get("test_evaluated") is not False
            or replay.get("passed") is not True or replay.get("invariance", {}).get("passed") is not True
            or replay.get("variant") != VARIANT or replay.get("optimizer_steps") != 0
            or replay.get("test_data_read") is not False or replay.get("test_evaluated") is not False
            or saved.get("parameters") != 381175401 or saved.get("total_steps") != 22890
            or saved.get("batches_per_epoch") != 763 or saved.get("warmup_steps") != 1373):
        raise ValueError("Control is incomplete or failed replay/invariance")
    if (not 1 <= best["epoch"] <= 30 or best.get("batch") != 762
            or best["step"] != (best["epoch"] - 1) * 763 + 762
            or complete.get("best_epoch") != best["epoch"] or complete.get("best_step") != best["step"]
            or replay.get("epoch") != best["epoch"] or replay.get("step") != best["step"]
            or complete.get("best_dev_F1") != best["dev_F1"]):
        raise ValueError("Control selection mismatch")
    if any(replay.get("metrics", {}).get(key) != best.get(key) for key in m1.METRIC_KEYS):
        raise ValueError("Control saved metrics mismatch")
    checksum = shared.sha256(run_dir / "best_model.pth")
    if saved.get("best_checkpoint_sha256") != checksum or replay.get("checkpoint_sha256") != checksum:
        raise ValueError("Control checkpoint mismatch")


def assert_original_read(model):
    from ttm import TokenLearnerModuleV11
    reader = model.mu_encoder.token_turing_machine_unit.token_learner_model
    if type(reader) is not TokenLearnerModuleV11:
        raise ValueError("B2 requires the original TokenLearner, not a disabled M1 branch")
    if any(any(key in name for key in ("query_self", "query_other", "memory_key", ".eta"))
           for name, _ in model.named_parameters()):
        raise ValueError("M1 parameters found in B2")


def load_experiment(args, manifest):
    import numpy as np
    import torch
    from transformers import AutoConfig, AutoModel, AutoTokenizer
    from model2 import DocREModel
    from prepro import read_docred
    from ttmre_memory_fit import install_unused_write_bypass, install_ttm_checkpoint

    args.device = runtime(manifest["m1_reference"]["environment"])
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
    shared.seed_all(args.seed, torch, np)
    model = DocREModel(args, config, torch.tensor(priors + 1e-9, device=args.device), backbone, tokenizer).to(args.device)
    aligned.apply_variant(model, VARIANT)
    assert_original_read(model)
    count = sum(parameter.numel() for parameter in model.parameters())
    if count != args.expected_parameters:
        raise ValueError(f"Unexpected B2 parameter count: {count}")
    aligned.emit("task_model_ready", variant=VARIANT, parameters=count, original_token_learner=True)
    aligned.emit("memory_fit_installed", **install_unused_write_bypass(model))
    aligned.emit("ttm_checkpoint_installed", **install_ttm_checkpoint(model))
    return model, training, development, gold, annotated


def run_probe(model, training, args):
    from ttmre_memory_fit import reserve_adam_moments
    buffers, reservation = reserve_adam_moments(model)
    aligned.emit("probe_memory_reservation", **reservation)
    try:
        report = aligned.run_probe(model, training, args)
        return {**report, "optimizer_memory_reservation": reservation}
    finally:
        model.zero_grad(set_to_none=True)
        del buffers
        import torch
        torch.cuda.empty_cache()


def replay_checkpoint(model, args, dev_loader, gold, annotated):
    import numpy as np
    import torch
    from evaluation import to_official
    model.zero_grad(set_to_none=True)
    model.load_state_dict(torch.load(args.run_dir / "best_model.pth", map_location="cpu", weights_only=True), strict=True)
    expected = shared.read_json(args.run_dir / "best_metrics.json")
    decisions = []
    def capture(module, inputs, logits):
        with torch.no_grad():
            predicted = torch.zeros_like(logits, dtype=torch.int8)
            predicted[:, 1:] = logits[:, 1:] > logits[:, :1]
            predicted[:, 0] = predicted[:, 1:].sum(1) == 0
            decisions.append(predicted.cpu().numpy())
    aligned.emit("checkpoint_replay_start", variant=VARIANT)
    handle = model.register_forward_hook(capture)
    try:
        metrics = shared.evaluate_dev(model, dev_loader, args.device, gold, annotated)
    finally:
        handle.remove()
    for key in m1.METRIC_KEYS:
        actual, saved = metrics[key], expected.get(key)
        if actual is None:
            if saved is not None:
                raise RuntimeError(f"Replay metric mismatch: {key}")
        elif saved is None or not math.isclose(actual, saved, abs_tol=1e-6, rel_tol=0):
            raise RuntimeError(f"Replay metric mismatch: {key}")
    invariance = aligned.grouping_invariance(model, dev_loader.dataset, args.device)
    if not invariance["passed"]:
        raise RuntimeError("B2-5090 grouping invariance failed")
    buckets = m1.error_buckets(to_official(np.concatenate(decisions), dev_loader.dataset), gold, dev_loader.dataset,
                              shared.read_json(args.data_dir / "train_revised.json"),
                              shared.read_json(ROOT / "meta" / "rel2id.json"))
    if not math.isclose(buckets["overall"]["F1"], metrics["dev_F1"], abs_tol=1e-6, rel_tol=0):
        raise RuntimeError("Error buckets differ from replay metrics")
    report = {"passed": True, "variant": VARIANT, "epoch": expected["epoch"], "step": expected["step"],
              "metrics": metrics, "invariance": invariance, "error_buckets": buckets,
              "checkpoint_sha256": shared.sha256(args.run_dir / "best_model.pth"),
              "optimizer_steps": 0, "test_data_read": False, "test_evaluated": False}
    aligned.emit("checkpoint_replay_passed", **report)
    return report


def run(args, manifest):
    import numpy as np
    import torch
    from torch.utils.data import DataLoader
    from transformers.optimization import get_linear_schedule_with_warmup
    started = time.perf_counter()
    timing = {"started_utc": datetime.now(timezone.utc).isoformat(), "phases_seconds": {}}
    if args.mode == "train":
        args.run_dir.mkdir(parents=True, exist_ok=False)
        args._owns_run_dir = True
        manifest.update({"created_utc": timing["started_utc"],
                         "scope": {"test_data_read": False, "distant_data_read": False, "model_download": False}})
        shared.write_json(args.run_dir / "manifest.json", manifest)
    phase = time.perf_counter()
    model, training, development, gold, annotated = load_experiment(args, manifest)
    timing["phases_seconds"]["data_and_model"] = time.perf_counter() - phase
    manifest["environment"] = m1.environment_report()
    verify_environment(manifest["environment"], manifest["m1_reference"]["environment"])
    aligned.emit("experiment_environment", **manifest["environment"])
    if args.mode == "probe":
        run_probe(model, training, args)
        print("B2_5090_PROBE_OK; optimizer_steps=0; test_labels_read=false; artifacts_saved=false", flush=True)
        return
    dev_loader = DataLoader(development, batch_size=4, shuffle=False, collate_fn=shared.collate_fn)
    if args.mode == "replay":
        replay_checkpoint(model, args, dev_loader, gold, annotated)
        print("B2_5090_REPLAY_OK; optimizer_steps=0; test_labels_read=false; files_written=false", flush=True)
        return
    phase = time.perf_counter()
    probe = run_probe(model, training, args)
    timing["phases_seconds"]["probes"] = time.perf_counter() - phase
    shared.write_json(args.run_dir / "full_model_check.json", probe)
    optimizer = shared.build_optimizer(model, args)
    groups = m1.optimizer_report(model, optimizer)
    shared.seed_all(args.seed, torch, np)
    loader = DataLoader(training, batch_size=4, shuffle=True, drop_last=True, collate_fn=shared.collate_fn)
    total = len(loader) * args.num_train_epochs
    if len(loader) != 763 or total != 22890:
        raise ValueError("B2-5090 requires 763 batches/epoch and 22890 updates")
    scheduler = get_linear_schedule_with_warmup(optimizer, int(total * .06), total)
    manifest.update({"versions": manifest["environment"]["versions"], "gpu": manifest["environment"]["gpu"],
                     "total_steps": total, "warmup_steps": int(total * .06), "batches_per_epoch": len(loader),
                     "parameters": sum(p.numel() for p in model.parameters()), "optimizer_groups": groups})
    shared.write_json(args.run_dir / "manifest.json", manifest)
    shared.write_json(args.run_dir / "args.json", vars(args))
    phase = time.perf_counter()
    with (args.run_dir / "train.log").open("x", encoding="utf-8") as log, \
            (args.run_dir / "metrics.jsonl").open("x", encoding="utf-8") as metrics_log:
        complete = aligned.train_loop(args, model, loader, dev_loader, gold, annotated,
                                      optimizer, scheduler, log, metrics_log)
    timing["phases_seconds"]["training_including_dev_and_saves"] = time.perf_counter() - phase
    if complete["optimizer_steps"] != total:
        raise RuntimeError("Control did not complete the fixed update budget")
    del optimizer, scheduler
    model.zero_grad(set_to_none=True)
    torch.cuda.empty_cache()
    phase = time.perf_counter()
    replay = replay_checkpoint(model, args, dev_loader, gold, annotated)
    timing["phases_seconds"]["checkpoint_replay_and_buckets"] = time.perf_counter() - phase
    shared.write_json(args.run_dir / "checkpoint_replay.json", replay)
    manifest["best_checkpoint_sha256"] = replay["checkpoint_sha256"]
    shared.write_json(args.run_dir / "manifest.json", manifest)
    timing.update({"finished_utc": datetime.now(timezone.utc).isoformat(),
                   "elapsed_seconds": time.perf_counter() - started,
                   "scope": "run after identity validation; training includes dev and checkpoint saves"})
    shared.write_json(args.run_dir / "timing.json", timing)
    m1_f1 = manifest["m1_reference"]["best_metrics"]["dev_F1"]
    shared.write_json(args.run_dir / "completion.json", {"variant": VARIANT, **complete,
        "checkpoint_replay_passed": True, "comparison_scope": "matched_environment_single_seed_screening",
        "m1_dev_F1": m1_f1, "m1_dev_F1_delta_vs_matched_b2": m1_f1 - complete["best_dev_F1"],
        "interpretation": "matched environment and budget; one seed does not establish significance"})
    aligned.emit("training_complete", variant=VARIANT, **complete, elapsed_seconds=timing["elapsed_seconds"])
    print("B2_5090_TRAINING_COMPLETE; checkpoint_replay=passed; test_labels_read=false", flush=True)


def main():
    args = parser().parse_args()
    aligned.emit("identity_validation_start", variant=VARIANT, mode=args.mode)
    manifest = validate(args)
    if args.mode == "validate":
        print(json.dumps({"validation": "passed", "manifest": manifest, "optimizer_steps": 0,
                          "model_download": False, "files_written": False}, indent=2))
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
