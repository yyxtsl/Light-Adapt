"""One-use, zero-update DWIE GPU admission probe for the approved C/B/D arms."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]
import ttmre_dwie_input as dwie
import ttmre_nvidia_smi_preflight as gpu_preflight
import ttmre_pair_replacement as replacement

RUN_ROOT = ROOT / "experiments/2026-10-23-dwie-gpu-probe-v4"
CLAIM_NAME = "dwie-gpu-probe-v4-once-claim.json"
# Preserve the original 15-minute aggregate cap across the three pre-weight
# failures (4.608s + 4.120s + 4.051s); round the remaining budget down to 885s.
HARD_SECONDS, SOFT_RESERVE_SECONDS, TERM_SECONDS = 885, 20, 15
SEED, ALPHA, NUM_CLASSES, SOURCE_CLASSES = 75, 6.0, 63, 97
DATA_ROOT = ROOT / "dataset/dwie-adaptation-20261023-recovery2"
TRAIN_FEATURES = DATA_ROOT / "features-v1/train.jsonl"
TARGET_SPLIT = DATA_ROOT / "manifests/target_split.json"
REL2ID = DATA_ROOT / "manifests/rel2id.json"
TRAIN_FEATURES_SHA256 = "40ae45a5384ae2a01cf1f967ff4c37423a3b2d253d9448bfad110c7acb8bc723"
TARGET_SPLIT_SHA256 = "bec1c252fc7ed0e5e262c2a02b17dc236deb82431c8fbf36bfbad56e0d9a272e"
REL2ID_SHA256 = "91fac5948767ed92c0c745c4a0dd4761e3ac19e36b19bea7826ee485c5da1c22"
CHECKER_SHA256 = "f34cbe116c1e88ad9ea96cfc45b7446bd4b44453b09a81213206c5f94eb96748"
NVIDIA_PREFLIGHT_TEST_SHA256 = "971997551dec48ff1292c0c03e68b6a9a03ff7e8a50802fa7075d0ce0b7aeb65"
C_SHA256 = "4fd5aca22c20584dde442c2d80dba8a72db8fe20a4b9fa920fa7c43a7fe13ce9"
D_SHA256 = "086c57e6baa5de13ab166bf03babb7794d2591eecfcca37cc611443cae0e8c01"
C_CHECKPOINT = ROOT / (
    "experiments/2026-10-18-n1-weighted-completion/"
    "weighted-completion-launch-20261018-121010-EkYgSZ/workspace/"
    "control/distant/artifacts/best_model.pth"
)
D_RUN_ROOT = ROOT / "experiments/2026-10-22-bypass-source-control"
D_CLAIM = D_RUN_ROOT / "bypass-source-control-once-claim.json"
D_UPDATES = 50936
SAMPLES = (
    {"id": "DW_15768955", "line": 46, "content_tokens": 2441, "entities": 88,
     "gold_relations": 172, "pairs": 7744, "windows": 9},
    {"id": "DW_18036260", "line": 186, "content_tokens": 2175, "entities": 95,
     "gold_relations": 27, "pairs": 9025, "windows": 8},
)
PINNED_SOURCES = {
    "model2.py": "9c9ef5c15ab585f8477d9d7534cf1615348e9c8f8e968de20179a97a53dbdc11",
    "ttm.py": "c9677d704b94f9e8a48e7707c806f2531acfa2efbd88a4017fc78d2a8c74decc",
    "scripts/ttmre_dwie_input.py": "d4cab72ebbd468d90e7a29160954d6592a0ac6ed48803be9a4c8a35c5dbef2f2",
    "scripts/ttmre_pair_replacement.py": "7ded529adafe3bf30363333287cedf4f3e607d106fd9805a4a786ac4cd6be70f",
    "scripts/ttmre_nvidia_smi_preflight.py": "f152891288df0cf095ab5f926a56297d41287068fb370e9b1cd0d07ceb294633",
    "scripts/train_ttmre_baseline_aligned.py": "8436a02d3addd6b2847961872ff8d562281ae0631a65c19e759f6ae23e89d655",
    "scripts/train_ttmre_human_only.py": "4ef536052f5896beaa611fcfb1e86aeafa37f2d9b5a868c80ca239dc80117ecf",
    "scripts/ttmre_memory_fit.py": "f1e2e41e6e7368d6f11be4a3333a1febb237c2fe6300f48b6ed1eba09650a1da",
}


class BudgetStop(TimeoutError):
    pass


def protocol():
    return {
        "version": "dwie-gpu-admission-probe-v4",
        "arms": {"C": {"parent_sha256": C_SHA256, "ttm": True},
                 "B": {"parent_sha256": C_SHA256, "ttm": False},
                 "D": {"parent_sha256": D_SHA256, "ttm": False}},
        "samples": [row["id"] for row in SAMPLES],
        "sample_scope": "two fixed documents from target train only; no Dev/Test rows parsed",
        "windowing": {"payload_tokens": 510, "stride": 256, "tail_aligned": True,
                      "max_total_tokens": 512, "merge": "mean hidden states; sum and row-normalize final-layer attention"},
        "candidate_pairs": "all directed entity pairs including self pairs; retain empty-mention entities",
        "target_head": "same deterministic fresh 63-class head in all arms; source 97-class head excluded",
        "target_head_seed": SEED,
        "loss": "ATGL alpha=6.0: -mean(sum(target * log_softmax(logits))); TH column=1, positive relation columns=6",
        "ttm": {"layers": 4, "pair_chunk": 256, "frozen": ["mu_encoder.memory_tokens"]},
        "precision": "float32",
        "forward_backward_runs": 6,
        "synthetic_cpu_contracts_before_asset_load": True,
        "gpu_preflight": "external nvidia-smi before any torch CUDA query; used memory < 500 MiB",
        "c_execution": "pair-independent axis correction; single-step write bypass; RNG-preserving nonreentrant TTM checkpoint",
        "optimizer_created": False,
        "optimizer_updates": 0,
        "checkpoint_saves": 0,
        "model_download": False,
        "test_data_read": False,
        "automatic_retry": False,
        "resume": False,
        "extension": False,
        "hard_seconds": HARD_SECONDS,
        "soft_reserve_seconds": SOFT_RESERVE_SECONDS,
        "gpu_preflight": "external nvidia-smi query before any torch CUDA initialization; 500MiB strict limit",
    }


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


def check_budget(budget, reserve=SOFT_RESERVE_SECONDS):
    if time.monotonic() >= budget["deadline_monotonic"] - reserve:
        raise BudgetStop("DWIE GPU probe budget reserve reached; no retry or extension")
    if Path("/proc/sys/kernel/random/boot_id").read_text().strip() != budget["boot_id"]:
        raise BudgetStop("Host boot changed; do not reset the probe budget")


def digest(path, budget=None):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            if budget is not None:
                check_budget(budget, reserve=0)
            result.update(block)
    return result.hexdigest()


def event(workspace, name, **fields):
    row = {"event": name, "utc": datetime.now(timezone.utc).isoformat(), **fields}
    raw = json.dumps(row, ensure_ascii=False, allow_nan=False)
    print(raw, flush=True)
    with (Path(workspace) / "events.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(raw + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def require_lock():
    if os.name != "posix" or os.environ.get("TTMRE_DWIE_GPU_PROBE_LOCK_FD") != "9":
        raise RuntimeError("Use scripts/run_ttmre_dwie_gpu_probe_v4.sh exactly once")
    import fcntl
    actual = os.fstat(9)
    lock = (ROOT / "experiments/2026-10-08-baseline-alignment/.training.lock").stat()
    if (actual.st_dev, actual.st_ino) != (lock.st_dev, lock.st_ino):
        raise RuntimeError("Probe does not hold the shared training lock")
    fcntl.flock(9, fcntl.LOCK_EX | fcntl.LOCK_NB)
    return 9


def check_source_inputs(budget):
    checked = {}
    for name, expected in PINNED_SOURCES.items():
        if budget is not None:
            check_budget(budget, reserve=0)
        path = ROOT / name
        actual = digest(path, budget)
        if actual != expected:
            raise ValueError("Pinned source mismatch: %s expected=%s actual=%s" % (name, expected, actual))
        checked[name] = actual
    return checked


def _find_model_dirs(value):
    found = []
    if isinstance(value, dict):
        for key, child in value.items():
            if key == "model_dir" and isinstance(child, str):
                found.append(Path(child))
            found.extend(_find_model_dirs(child))
    elif isinstance(value, list):
        for child in value:
            found.extend(_find_model_dirs(child))
    return found


def locate_d_parent(budget):
    claim = read_json(D_CLAIM)
    workspace = Path(claim["workspace"]).resolve()
    if (workspace != Path(claim["workspace"]) or workspace.name != "workspace"
            or workspace.parent.parent != D_RUN_ROOT.resolve()):
        raise ValueError("No canonical completed D source workspace")
    checkpoint = workspace / "distant/best_model.pth"
    completion = read_json(workspace / "distant/completion.json")
    if (completion.get("passed") is not True or completion.get("optimizer_steps") != D_UPDATES
            or completion.get("checkpoint_sha256") != D_SHA256):
        raise ValueError("Fixed D parent completion record does not match the approved source")
    source_identity = read_json(workspace / "identity.json")
    model_dirs = {p.resolve() for p in _find_model_dirs(source_identity)
                  if (p / "config.json").is_file()}
    if len(model_dirs) != 1:
        raise ValueError("Could not resolve one pinned local RoBERTa config directory from D identity")
    model_dir = model_dirs.pop()
    if not model_dir.is_relative_to(ROOT.resolve()):
        raise ValueError("Model config must be inside the project workspace")
    return checkpoint, model_dir


def _read_probe_features(split, rel2id, budget):
    positions = {row["line"]: row for row in SAMPLES}
    wanted = {}
    with TRAIN_FEATURES.open("r", encoding="utf-8") as stream:
        for index, line in enumerate(stream):
            if index in positions:
                row = positions[index]
                feature = json.loads(line)
                if feature.get("id") != row["id"]:
                    raise ValueError("Fixed train feature line no longer has its pinned document ID")
                wanted[row["id"]] = feature
            if index >= max(positions):
                break
            if budget is not None and index % 32 == 0:
                check_budget(budget, reserve=0)
    if set(wanted) != {row["id"] for row in SAMPLES}:
        raise ValueError("One or more selected train features are absent")
    target_train = split["train"]
    if any(target_train[row["line"]] != row["id"] for row in SAMPLES):
        raise ValueError("Probe sample is not at its predeclared train split position")
    summaries = []
    for spec in SAMPLES:
        feature = wanted[spec["id"]]
        inputs = feature.get("inputs", {})
        ids, positions = inputs.get("entity_ids"), inputs.get("entity_pos")
        content = inputs.get("content_ids")
        if (not isinstance(content, list) or not isinstance(ids, list) or not isinstance(positions, list)
                or inputs.get("candidate_rule") != "cartesian_including_self"
                or len(ids) != len(positions) or ids != sorted(set(ids))):
            raise ValueError("Probe feature input schema/order mismatch")
        mappings, coverage = dwie.window_layout(len(content))
        pairs = len(ids) ** 2
        if (len(content) != spec["content_tokens"] or len(ids) != spec["entities"]
                or len(feature.get("gold", [])) != spec["gold_relations"] or pairs != spec["pairs"]
                or len(mappings) != spec["windows"] or min(coverage) < 1):
            raise ValueError("Probe feature counts differ from the approved samples")
        valid_ids = set(ids)
        for relation in feature["gold"]:
            if (relation["h"] not in valid_ids or relation["t"] not in valid_ids
                    or relation["r"] not in rel2id or rel2id[relation["r"]] == 0):
                raise ValueError("Selected train gold cannot be represented by the frozen target labels")
        summaries.append({"id": spec["id"], "line": spec["line"], "content_tokens": len(content),
                          "total_tokens": len(content) + 2, "windows": len(mappings),
                          "entities": len(ids), "candidate_pairs_including_self": pairs,
                          "gold_relations": len(feature["gold"])})
    return wanted, summaries


def validate_inputs(budget=None):
    if os.name != "posix":
        raise RuntimeError("The approved cloud launcher requires Linux")
    sources = check_source_inputs(budget)
    for path, expected in ((TRAIN_FEATURES, TRAIN_FEATURES_SHA256),
                           (TARGET_SPLIT, TARGET_SPLIT_SHA256), (REL2ID, REL2ID_SHA256)):
        if digest(path, budget) != expected:
            raise ValueError("DWIE train input digest mismatch: " + str(path))
    split = read_json(TARGET_SPLIT)
    if (set(split) != {"train", "dev"} or len(split["train"]) != 602 or len(split["dev"]) != 100
            or len(set(split["train"])) != 602 or set(split["train"]) & set(split["dev"])):
        raise ValueError("Fixed target train/Dev split manifest mismatch")
    rel2id = read_json(REL2ID)
    if (len(rel2id) != NUM_CLASSES or rel2id.get("TH") != 0
            or sorted(v for k, v in rel2id.items() if k != "TH") != list(range(1, NUM_CLASSES))):
        raise ValueError("Expected frozen 63-class target relation mapping")
    features, samples = _read_probe_features(split, rel2id, budget)
    d_checkpoint, model_dir = locate_d_parent(budget)
    if not C_CHECKPOINT.is_file() or not d_checkpoint.is_file():
        raise FileNotFoundError("One or both pinned source checkpoints are missing")
    parent_hashes = {"C_B": digest(C_CHECKPOINT, budget), "D": digest(d_checkpoint, budget)}
    if parent_hashes != {"C_B": C_SHA256, "D": D_SHA256}:
        raise ValueError("One or both fixed parent checkpoint SHA256 values mismatch")
    config_data = read_json(model_dir / "config.json")
    expected_config = {"model_type": "roberta", "hidden_size": 1024,
                       "num_hidden_layers": 24, "num_attention_heads": 16, "vocab_size": 50265,
                       "max_position_embeddings": 514, "bos_token_id": 0, "eos_token_id": 2}
    if any(config_data.get(key) != value for key, value in expected_config.items()):
        raise ValueError("Local model config is not the expected RoBERTa-large architecture")
    return ({"protocol": protocol(), "source_files": sources, "parent_paths": {"C_B": str(C_CHECKPOINT),
             "D": str(d_checkpoint)}, "parent_sha256": parent_hashes,
             "model_config_dir": str(model_dir), "model_config_sha256": digest(model_dir / "config.json", budget),
             "train_features": str(TRAIN_FEATURES), "train_features_sha256": TRAIN_FEATURES_SHA256,
             "target_split_sha256": TARGET_SPLIT_SHA256, "rel2id_sha256": REL2ID_SHA256,
             "target_train_documents": 602, "target_dev_documents": 100,
             "target_classes": NUM_CLASSES, "selected_train_samples": samples,
             "dev_feature_file_opened": False, "test_files_opened": False}, features)


def target_labels(feature, rel2id, device, num_classes=NUM_CLASSES):
    import torch
    entity_ids = feature["inputs"]["entity_ids"]
    entity_index = {ident: index for index, ident in enumerate(entity_ids)}
    count = len(entity_ids)
    labels = torch.zeros((count * count, num_classes), dtype=torch.float32, device=device)
    for relation in feature["gold"]:
        if relation["r"] not in rel2id:
            raise ValueError("Unknown train relation cannot be converted to the target head")
        if not 0 < rel2id[relation["r"]] < num_classes:
            raise ValueError("Relation class ID is outside the target head")
        pair = entity_index[relation["h"]] * count + entity_index[relation["t"]]
        labels[pair, rel2id[relation["r"]]] = 1.0
    return labels


def atgl_loss(logits, labels, alpha=ALPHA):
    import torch.nn.functional as functional
    target = labels.to(device=logits.device, dtype=logits.dtype).clone()
    target[:, 0] = 0
    target.mul_(alpha)
    target[:, 0] = 1
    return -(functional.log_softmax(logits, dim=-1) * target).sum(-1).mean()


def feature_logits(model, feature, rel2id, device):
    import torch
    windows, _ = dwie.window_inputs(feature["inputs"]["content_ids"])
    mappings = [window["global_indices"] for window in windows]
    sequences, attentions = [], []
    for window in windows:
        input_ids = torch.tensor([window["input_ids"]], dtype=torch.long, device=device)
        attention_mask = torch.tensor([window["attention_mask"]], dtype=torch.long, device=device)
        output = model.model(input_ids=input_ids, attention_mask=attention_mask,
                             output_attentions=True, return_dict=True)
        if not output.attentions:
            raise RuntimeError("RoBERTa did not return final-layer attention")
        sequences.append(output.last_hidden_state[0])
        attentions.append(output.attentions[-1][0])
    global_size = len(feature["inputs"]["content_ids"]) + 2
    sequence, attention = dwie.merge_window_outputs(sequences, attentions, mappings, global_size)
    hs, context, ts, zero_context = dwie.pair_representations(sequence, attention, feature["inputs"]["entity_pos"])
    hs = torch.tanh(model.head_extractor(torch.cat((hs, context), dim=-1)))
    ts = torch.tanh(model.head_extractor(torch.cat((ts, context), dim=-1)))
    pair_input = torch.stack((hs, ts), dim=1).unsqueeze(1)
    enhanced_head, enhanced_tail = [], []
    for offset in range(0, len(pair_input), 256):
        output = model.mu_encoder(pair_input[offset:offset + 256], train=False)
        enhanced_head.append(output[:, 0, 0, :])
        enhanced_tail.append(output[:, 0, 1, :])
    head = torch.cat(enhanced_head, dim=0)
    tail = torch.cat(enhanced_tail, dim=0)
    head = (head / 2 + hs / 2).view(-1, model.emb_size // model.block_size, model.block_size)
    tail = (tail / 2 + ts / 2).view(-1, model.emb_size // model.block_size, model.block_size)
    bilinear = (head.unsqueeze(3) * tail.unsqueeze(2)).reshape(-1, model.emb_size * model.block_size)
    logits = model.bilinear(bilinear)
    labels = target_labels(feature, rel2id, device, model.bilinear.out_features)
    if logits.shape != labels.shape:
        raise ValueError("Target logit and label shapes disagree")
    return logits, labels, {"candidate_pairs_including_self": len(pair_input),
                            "zero_attention_context_pairs": zero_context, "windows": len(windows)}


def tensor_state_sha256(state):
    digest_value = hashlib.sha256()
    for name in sorted(state):
        tensor = state[name].detach().cpu().contiguous()
        digest_value.update(name.encode("utf-8"))
        digest_value.update(str(tensor.dtype).encode("ascii"))
        digest_value.update(tensor.numpy().tobytes())
    return digest_value.hexdigest()


def fixed_target_head(in_features):
    import torch
    from torch import nn
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(SEED)
        layer = nn.Linear(in_features, NUM_CLASSES)
    state = {name: value.detach().clone() for name, value in layer.state_dict().items()}
    return state, tensor_state_sha256(state)


def set_classifier(model, classes, state=None):
    import torch
    from torch import nn
    layer = nn.Linear(model.bilinear.in_features, classes, device="cpu", dtype=torch.float32)
    if state is not None:
        layer.load_state_dict(state, strict=True)
    model.bilinear = layer
    model.config.num_labels = classes
    model.args.num_class = classes
    model.rels = classes - 1


def strict_load(model, state, source_name):
    expected = model.state_dict()
    actual_keys = set(state)
    expected_keys = set(expected)
    missing, unexpected = sorted(expected_keys - actual_keys), sorted(actual_keys - expected_keys)
    shapes = sorted((name, tuple(expected[name].shape), tuple(state[name].shape))
                    for name in expected_keys & actual_keys if expected[name].shape != state[name].shape)
    if missing or unexpected or shapes:
        raise RuntimeError("Strict %s checkpoint mismatch: missing=%s unexpected=%s shapes=%s" %
                           (source_name, missing, unexpected, shapes))
    non_fp32 = [name for name, value in state.items()
                if value.is_floating_point() and value.dtype != __import__("torch").float32]
    if non_fp32:
        raise RuntimeError("Expected FP32 parent tensors; non-FP32 keys=" + repr(sorted(non_fp32)))
    model.load_state_dict(state, strict=True)
    return {"strict": True, "source_keys": len(state), "loaded_keys": len(expected),
            "missing_keys": [], "unexpected_keys": [], "shape_mismatches": [],
            "source_name": source_name}


def load_checkpoint(model, path, expected_sha, budget, source_name):
    import torch
    check_budget(budget)
    actual_sha = digest(path, budget)
    if actual_sha != expected_sha:
        raise ValueError("Parent SHA256 changed before tensor load: " + source_name)
    state = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(state, dict) or not state or not all(torch.is_tensor(value) for value in state.values()):
        raise TypeError("Parent checkpoint is not a plain tensor state dict: " + source_name)
    report = strict_load(model, state, source_name)
    report["checkpoint_sha256"] = actual_sha
    report["checkpoint_bytes"] = Path(path).stat().st_size
    del state
    gc.collect()
    return report


def build_model(config_dir):
    import torch
    from transformers import AutoConfig, AutoModel
    from model2 import DocREModel
    import train_ttmre_baseline_aligned as aligned
    from ttmre_memory_fit import install_ttm_checkpoint, install_unused_write_bypass
    config = AutoConfig.from_pretrained(str(config_dir), local_files_only=True)
    config.num_labels = SOURCE_CLASSES
    config.output_attentions = True
    config.transformer_type = "roberta"
    config.cls_token_id, config.sep_token_id = 0, 2
    if hasattr(config, "_attn_implementation"):
        config._attn_implementation = "eager"
    args = SimpleNamespace(model_type="ttmre", num_class=SOURCE_CLASSES, isrank=True,
                           num_layers=4, e=1.0, m=1.0, m_tag="ATLoss", device=torch.device("cpu"))
    priors = torch.full((SOURCE_CLASSES - 1,), 0.01, dtype=torch.float32)
    backbone = AutoModel.from_config(config)
    model = DocREModel(args, config, priors, backbone, tokenizer=None, emb_size=768, block_size=64)
    model.requires_grad_(True)
    aligned.apply_variant(model, "pair-independent")
    bypass_report = install_unused_write_bypass(model)
    checkpoint_report = install_ttm_checkpoint(model)
    return model, {"pair_independent_ttm_axis_correction": True,
                   "unused_writer_bypassed": bypass_report,
                   "ttm_activation_checkpointing": checkpoint_report}


def _synchronize(device):
    import torch
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def probe_document(model, feature, rel2id, device, seed, budget):
    import torch
    check_budget(budget)
    model.train()
    model.zero_grad(set_to_none=True)
    with torch.random.fork_rng(devices=[device.index] if device.type == "cuda" else []):
        torch.manual_seed(seed)
        if device.type == "cuda":
            torch.cuda.manual_seed(seed)
        try:
            versions = tuple(parameter._version for parameter in model.parameters())
            base_allocated = torch.cuda.memory_allocated(device) if device.type == "cuda" else 0
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            _synchronize(device)
            started = time.perf_counter()
            logits, labels, audit = feature_logits(model, feature, rel2id, device)
            loss = atgl_loss(logits, labels)
            _synchronize(device)
            forward_seconds = time.perf_counter() - started
            if not torch.isfinite(logits).all().item() or not torch.isfinite(loss).item():
                raise RuntimeError("Nonfinite logits or ATGL loss")
            backward_started = time.perf_counter()
            loss.backward()
            _synchronize(device)
            backward_seconds = time.perf_counter() - backward_started
            gradients = [(name, parameter.grad) for name, parameter in model.named_parameters()
                         if parameter.grad is not None]
            if not gradients:
                raise RuntimeError("Backward produced no gradients")
            finite_flags = torch.stack([torch.isfinite(gradient).all() for _, gradient in gradients])
            nonzero_flags = torch.stack([torch.count_nonzero(gradient) > 0 for _, gradient in gradients])
            gradients_finite = bool(finite_flags.all().item())
            nonzero_gradient_tensors = int(nonzero_flags.sum().item())
            named = dict(model.named_parameters())
            critical = ("bilinear.weight", "head_extractor.weight", "model.embeddings.word_embeddings.weight")
            critical_nonzero = {}
            for name in critical:
                gradient = named[name].grad if name in named else None
                critical_nonzero[name] = bool(gradient is not None and torch.count_nonzero(gradient).item() > 0)
            if not gradients_finite or not all(critical_nonzero.values()):
                raise RuntimeError("Gradient check failed: finite=%s critical=%s" %
                                   (gradients_finite, critical_nonzero))
            parameter_versions_unchanged = versions == tuple(parameter._version for parameter in model.parameters())
            if not parameter_versions_unchanged:
                raise RuntimeError("A parameter value changed during the zero-update probe")
            memory_parameter = dict(model.named_parameters()).get("mu_encoder.memory_tokens")
            frozen_memory_gradient_none = memory_parameter is None or memory_parameter.grad is None
            peak_allocated = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
            peak_reserved = torch.cuda.max_memory_reserved(device) if device.type == "cuda" else None
            return {"document_id": feature["id"], "loss": float(loss.detach().cpu()),
                    "logits_finite": True, "loss_finite": True,
                    "forward_seconds": forward_seconds, "backward_seconds": backward_seconds,
                    "total_forward_backward_seconds": forward_seconds + backward_seconds,
                    "gradient_tensor_count": len(gradients), "nonzero_gradient_tensor_count": nonzero_gradient_tensors,
                    "all_gradients_finite": gradients_finite, "critical_gradient_nonzero": critical_nonzero,
                    "parameters_unchanged": True, "optimizer_updates": 0,
                    "frozen_memory_gradient_none": frozen_memory_gradient_none,
                    "base_allocated_bytes": base_allocated, "peak_allocated_bytes": peak_allocated,
                    "peak_reserved_bytes": peak_reserved,
                    "incremental_peak_allocated_bytes": (peak_allocated - base_allocated) if peak_allocated is not None else None,
                    **audit}
        finally:
            model.zero_grad(set_to_none=True)


def _load_source_state(path, expected_sha, budget):
    import torch
    check_budget(budget)
    if digest(path, budget) != expected_sha:
        raise ValueError("Parent SHA mismatch before load: " + str(path))
    state = torch.load(path, map_location="cpu", weights_only=True)
    return state


def run_synthetic_contracts():
    import unittest
    from unittest.mock import patch
    import torch
    import check_ttmre_dwie_gpu_probe_v4 as contract_tests
    import check_ttmre_nvidia_smi_preflight as gpu_contract_tests
    if digest(ROOT / "scripts/check_ttmre_dwie_gpu_probe_v4.py") != CHECKER_SHA256:
        raise ValueError("V4 synthetic contract source hash mismatch")
    if digest(ROOT / "scripts/check_ttmre_nvidia_smi_preflight.py") != NVIDIA_PREFLIGHT_TEST_SHA256:
        raise ValueError("Standard-library GPU preflight contract source hash mismatch")
    gpu_suite = unittest.defaultTestLoader.loadTestsFromTestCase(gpu_contract_tests.NvidiaSmiPreflightContracts)
    gpu_result = unittest.TextTestRunner(stream=sys.stdout, verbosity=1).run(gpu_suite)
    if not gpu_result.wasSuccessful():
        raise RuntimeError("nvidia-smi standard-library contracts failed; stop before assets/GPU")
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(contract_tests.Contracts)
    with patch.object(torch.cuda, "_lazy_init", side_effect=AssertionError("Synthetic contract attempted CUDA")), \
         patch.object(torch.optim, "AdamW", side_effect=AssertionError("Synthetic contract created an optimizer")), \
         patch.object(torch, "save", side_effect=AssertionError("Synthetic contract saved a checkpoint")):
        result = unittest.TextTestRunner(stream=sys.stdout, verbosity=1).run(suite)
    if not result.wasSuccessful():
        raise RuntimeError("Synthetic CPU contracts failed; stop before real data/weights/GPU")
    return {"passed": True, "tests_run": result.testsRun + gpu_result.testsRun,
            "torch_contract_tests": result.testsRun, "stdlib_gpu_preflight_tests": gpu_result.testsRun,
            "failures": 0, "errors": 0,
            "gpu_used": False, "real_data_read": False, "real_weights_loaded": False}


def run_probe(workspace, identity, features, budget):
    # This subprocess reads the device before torch.cuda.current_device(),
    # get_device_properties(), or mem_get_info() can initialize a CUDA context.
    gpu_preflight_report = gpu_preflight.inspect_gpu()
    if not gpu_preflight_report["passed"]:
        used = gpu_preflight_report["memory_used_bytes"] / 1024**2
        raise RuntimeError("GPU preflight found %.1fMiB externally used / %.1fMiB total (500MiB limit); no parent tensor load" %
                           (used, gpu_preflight_report["memory_total_bytes"] / 1024**2))
    import torch
    from ttmre_pair_replacement import install
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; no CPU fallback is permitted")
    device = torch.device("cuda", torch.cuda.current_device())
    properties = torch.cuda.get_device_properties(device)
    if properties.name != gpu_preflight_report["name"]:
        raise RuntimeError("CUDA logical device does not match the nvidia-smi preflight device")
    rel2id = read_json(REL2ID)
    head_state, head_sha = fixed_target_head(768 * 64)
    model, c_execution_adaptations = build_model(identity["model_config_dir"])
    results = {"C": [], "B": [], "D": []}
    install_report = {}
    c_load = load_checkpoint(model, C_CHECKPOINT, C_SHA256, budget, "C/B shared TTM parent")
    c_load["source_output_classes"] = SOURCE_CLASSES
    install_report["C_parent_load"] = c_load
    install_report["C_execution_adaptations"] = c_execution_adaptations
    model.mu_encoder.memory_tokens.requires_grad_(False)
    set_classifier(model, NUM_CLASSES, head_state)
    install_report["fresh_target_head"] = {"source_output_classes": SOURCE_CLASSES,
        "target_output_classes": NUM_CLASSES, "reconstructed_keys": ["bilinear.weight", "bilinear.bias"],
        "seed": SEED, "sha256": head_sha, "identical_across_C_B_D": True}
    for spec in SAMPLES:
        check_budget(budget)
        event(workspace, "probe_started", arm="C", document_id=spec["id"])
        row = probe_document(model.to(device), features[spec["id"]], rel2id, device,
                             SEED + spec["line"], budget)
        row["arm"] = "C"
        results["C"].append(row)
        event(workspace, "probe_complete", **row)
    model.zero_grad(set_to_none=True)
    model.to("cpu")
    gc.collect()
    torch.cuda.empty_cache()

    before_keys = set(model.state_dict())
    # C freezes only the inherited TTM memory parameter. The B structure has
    # no trainable TTM parameters, so restore common parameters to trainable
    # before the replacement installer checks its optimizer-facing scope.
    model.requires_grad_(True)
    install_report["B_adapter"] = install(model, "bypass")
    after_keys = set(model.state_dict())
    install_report["B_adapter"]["removed_state_keys"] = sorted(before_keys - after_keys)
    install_report["B_adapter"]["retained_state_keys"] = sorted(before_keys & after_keys)
    model.requires_grad_(True)
    set_classifier(model, NUM_CLASSES, head_state)
    for spec in SAMPLES:
        check_budget(budget)
        event(workspace, "probe_started", arm="B", document_id=spec["id"])
        row = probe_document(model.to(device), features[spec["id"]], rel2id, device,
                             SEED + spec["line"], budget)
        row["arm"] = "B"
        results["B"].append(row)
        event(workspace, "probe_complete", **row)
    model.zero_grad(set_to_none=True)
    model.to("cpu")
    gc.collect()
    torch.cuda.empty_cache()

    set_classifier(model, SOURCE_CLASSES)
    d_state = _load_source_state(Path(identity["parent_paths"]["D"]), D_SHA256, budget)
    install_report["D_parent_load"] = strict_load(model, d_state, "D no-TTM source parent")
    install_report["D_parent_load"].update({"checkpoint_sha256": D_SHA256,
        "checkpoint_bytes": Path(identity["parent_paths"]["D"]).stat().st_size,
        "source_output_classes": SOURCE_CLASSES,
        "reconstructed_keys": ["bilinear.weight", "bilinear.bias"],
        "loaded_no_ttm_state_keys": sorted(name for name in model.state_dict()
                                             if name.startswith("mu_encoder."))})
    del d_state
    gc.collect()
    set_classifier(model, NUM_CLASSES, head_state)
    for spec in SAMPLES:
        check_budget(budget)
        event(workspace, "probe_started", arm="D", document_id=spec["id"])
        row = probe_document(model.to(device), features[spec["id"]], rel2id, device,
                             SEED + spec["line"], budget)
        row["arm"] = "D"
        results["D"].append(row)
        event(workspace, "probe_complete", **row)
    model.zero_grad(set_to_none=True)
    model.to("cpu")
    gc.collect()
    torch.cuda.empty_cache()
    check_budget(budget, reserve=0)
    result = {"passed": all(len(results[arm]) == len(SAMPLES) for arm in ("C", "B", "D")),
        "protocol": protocol(), "gpu": {"name": properties.name, "total_memory_bytes": properties.total_memory,
        "compute_capability": list(properties.major_minor) if hasattr(properties, "major_minor") else
        [properties.major, properties.minor], "torch_version": torch.__version__},
        "gpu_preflight": gpu_preflight_report, "install_and_load_audit": install_report, "arms": results,
        "optimizer_created": False, "optimizer_updates": 0, "checkpoint_saves": 0,
        "model_download": False, "test_data_read": False,
        "interpretation_limit": "single-document backward admission only; does not test batch=4, model quality, or global attention across windows",
        "elapsed_seconds": time.monotonic() - budget["start_monotonic"]}
    write_json(Path(workspace) / "completion.json", result)
    return result


def verify_context(workspace, budget):
    workspace = Path(workspace).resolve()
    claim = read_json(RUN_ROOT / CLAIM_NAME)
    if (workspace != Path(claim["workspace"]) or workspace.name != "workspace"
            or workspace.parent.parent != RUN_ROOT.resolve() or claim["protocol"] != protocol()
            or claim["budget_sha256"] != digest(workspace / "budget.json")
            or budget["hard_seconds"] != HARD_SECONDS
            or not math.isclose(budget["deadline_monotonic"] - budget["start_monotonic"],
                                HARD_SECONDS, abs_tol=1e-6, rel_tol=0)):
        raise ValueError("Immutable one-use probe context changed")
    check_budget(budget)


def worker(workspace):
    require_lock()
    workspace = Path(workspace).resolve()
    budget = read_json(workspace / "budget.json")
    verify_context(workspace, budget)
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(BudgetStop("Probe worker interrupted")))
    try:
        expected_runner = os.environ.get("TTMRE_DWIE_GPU_PROBE_ENTRY_SHA256")
        if not expected_runner or digest(Path(__file__), budget) != expected_runner:
            raise ValueError("Probe entrypoint differs from the approved launcher pin")
        contract_result = run_synthetic_contracts()
        identity, features = validate_inputs(budget)
        identity["synthetic_cpu_contracts"] = contract_result
        identity["source_files"].update({
            "scripts/ttmre_dwie_gpu_probe_v4.py": expected_runner,
            "scripts/check_ttmre_dwie_gpu_probe_v4.py": digest(ROOT / "scripts/check_ttmre_dwie_gpu_probe_v4.py", budget),
            "scripts/check_ttmre_nvidia_smi_preflight.py": digest(ROOT / "scripts/check_ttmre_nvidia_smi_preflight.py", budget),
            "scripts/run_ttmre_dwie_gpu_probe_v4.sh": digest(ROOT / "scripts/run_ttmre_dwie_gpu_probe_v4.sh", budget),
        })
        write_json(workspace / "identity.json", identity)
        write_json(workspace / "manifest.json", {"protocol": protocol(),
            "identity_sha256": digest(workspace / "identity.json"), "budget_sha256": digest(workspace / "budget.json"),
        "report_scope": "manifest, events, completion/failure only; no checkpoint or prediction writes"})
        event(workspace, "assets_verified", samples=identity["selected_train_samples"],
              parent_sha256=identity["parent_sha256"])
        result = run_probe(workspace, identity, features, budget)
        event(workspace, "probe_complete_all_arms", passed=result["passed"],
              elapsed_seconds=result["elapsed_seconds"])
    except BaseException as error:
        failure = {"passed": False, "error": str(error), "type": type(error).__name__,
                   "optimizer_created": False, "optimizer_updates": 0,
                   "checkpoint_saves": 0, "automatic_retry": False,
                   "elapsed_seconds": time.monotonic() - budget["start_monotonic"]}
        try:
            write_json(workspace / "failure.json", failure)
        except FileExistsError:
            pass
        event(workspace, "probe_failed", error=str(error), error_type=type(error).__name__)
        raise


def supervise(command, budget, lock_fd):
    check_budget(budget)
    child = subprocess.Popen(command, start_new_session=True, pass_fds=(lock_fd,))
    term_sent = False
    try:
        while True:
            status = child.poll()
            if status is not None:
                if status != 0 or term_sent:
                    raise RuntimeError("Probe worker failed/stopped; no retry")
                check_budget(budget, reserve=0)
                return
            now = time.monotonic()
            if now >= budget["deadline_monotonic"]:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
                raise BudgetStop("Hard 15-minute deadline reached; worker group killed")
            if now >= budget["deadline_monotonic"] - TERM_SECONDS and not term_sent:
                os.killpg(child.pid, signal.SIGTERM)
                term_sent = True
            time.sleep(0.2)
    finally:
        if child.poll() is None:
            os.killpg(child.pid, signal.SIGKILL)
            child.wait()


def start(workspace):
    lock_fd = require_lock()
    workspace = Path(workspace).resolve()
    if (workspace.name != "workspace" or workspace.parent.parent != RUN_ROOT.resolve()
            or workspace.exists() or (RUN_ROOT / CLAIM_NAME).exists()):
        raise ValueError("Use the one fresh cloud launch directory; retries are disabled")
    now = time.monotonic()
    budget = {"start_monotonic": now, "deadline_monotonic": now + HARD_SECONDS,
        "hard_seconds": HARD_SECONDS, "soft_reserve_seconds": SOFT_RESERVE_SECONDS,
        "term_seconds": TERM_SECONDS, "start_utc": datetime.now(timezone.utc).isoformat(),
        "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
        "automatic_retry": False, "resume": False, "extension": False}
    workspace.mkdir(parents=True)
    write_json(workspace / "budget.json", budget)
    write_json(RUN_ROOT / CLAIM_NAME, {"workspace": str(workspace), "protocol": protocol(),
        "budget_sha256": digest(workspace / "budget.json"), "authorization": "one approved 15-minute C/B/D zero-update GPU probe"})
    (workspace.parent / "pid.json").write_text(json.dumps({"supervisor_pid": os.getpid()}) + "\n", encoding="utf-8")
    command = [sys.executable, "-u", "-B", str(Path(__file__).resolve()),
               "--mode", "worker", "--run-dir", str(workspace)]
    try:
        supervise(command, budget, lock_fd)
    except BaseException as error:
        failure = {"error": str(error), "type": type(error).__name__, "complete": False,
                   "automatic_retry": False, "elapsed_seconds": time.monotonic() - now}
        try:
            write_json(workspace.parent / "supervisor-failure.json", failure)
        except FileExistsError:
            pass
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("validate", "start", "worker"), default="validate")
    parser.add_argument("--run-dir", type=Path)
    args = parser.parse_args()
    if args.mode == "validate":
        identity, _ = validate_inputs()
        print(json.dumps(identity, indent=2, ensure_ascii=False, allow_nan=False))
    elif args.mode == "start":
        if args.run_dir is None:
            raise ValueError("The one-use launcher must provide --run-dir")
        start(args.run_dir)
    else:
        if args.run_dir is None:
            raise ValueError("Worker requires --run-dir")
        worker(args.run_dir)


if __name__ == "__main__":
    main()
