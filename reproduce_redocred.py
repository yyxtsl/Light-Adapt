"""Portable entry point for the recorded Re-DocRED target-stage recipes and controls.

Run without arguments for usage. ``validate`` reads file metadata and the local
RoBERTa config only. Training, evaluation, and averaging require explicit modes.
The original dated execution scripts remain available for provenance.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parent
MEMORY = "mu_encoder.memory_tokens"
TRAIN_DOCUMENTS = 3053
EVALUATION_DOCUMENTS = 500
EPOCHS = 30
BATCH_SIZE = 4
EXPECTED_PARAMETERS = {"C": 381175401, "B": 363274849, "C-off": 363274849,
                       "D": 363274849, "G": 363274849, "U": 363274849}
TRAIN_SHA256 = "c40a137a1c57f3e2daaf5ebf511eaa1518f9305ffaa3052d46271921a91826f2"
DEV_SHA256 = "051ee1d057204a5d08ef5502beacdadf191245b5eaf0e29ec2c607cf002c016f"
TEST_SHA256 = "ea673b7ef91e16510d7eb88863d3903e47318dc5653ef361266f27ae3b84723c"


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--mode", required=True,
                        choices=("validate", "train", "evaluate", "average"))
    result.add_argument("--validate-for", choices=("train", "evaluate", "average"),
                        default="evaluate", help="Asset checks performed in validate mode")
    result.add_argument("--variant", choices=("C", "B", "C-off", "D", "G", "U"), default="C")
    result.add_argument("--model-dir", type=Path, help="Local RoBERTa-large model/tokenizer directory")
    result.add_argument("--data-dir", type=Path,
                        help="Directory containing train_revised.json and the requested split")
    result.add_argument("--checkpoint", type=Path,
                        help="Stage-1 source for train, or a target checkpoint for evaluate")
    result.add_argument("--checkpoints", type=Path, nargs=3,
                        help="Three target checkpoints for uniform weight averaging")
    result.add_argument("--output-dir", type=Path, help="New directory for this operation")
    result.add_argument("--split", choices=("dev", "test"), default="dev")
    result.add_argument("--seed", type=int, default=75, help="Target-stage seed")
    result.add_argument("--device", default="cuda:0")
    result.add_argument("--export-logits", action="store_true",
                        help="Save evaluation logits and label-independent pair order for diagnostics")
    result.add_argument("--expected-checkpoint-sha256",
                        help="Optional expected digest of --checkpoint")
    return result


def read_json(path):
    with Path(path).open(encoding="utf-8") as stream:
        return json.load(stream)


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            result.update(block)
    return result.hexdigest()


def emit(event, **values):
    print(json.dumps({"event": event, **values}, allow_nan=False), flush=True)


def normalized_args(arguments):
    for name in ("model_dir", "data_dir", "checkpoint", "output_dir"):
        value = getattr(arguments, name)
        if value is not None:
            setattr(arguments, name, value.expanduser().resolve())
    if arguments.checkpoints is not None:
        arguments.checkpoints = [path.expanduser().resolve() for path in arguments.checkpoints]
    return arguments


def protocol(arguments):
    return {
        "dataset": "Re-DocRED", "variant": arguments.variant,
        "source_checkpoint": str(arguments.checkpoint) if arguments.checkpoint else None,
        "target_seed": arguments.seed, "precision": "float32", "epochs": EPOCHS,
        "batch_size": BATCH_SIZE, "drop_last": True, "alpha": 16.5,
        "backbone_lr": 3e-5, "task_head_lr": 1e-5,
        "adam_epsilon": 1e-6, "weight_decay": 0.01,
        "warmup_ratio": 0.06, "gradient_clip_norm": 1.0,
        "selection": "Dev micro-F1 after each epoch's penultimate batch update; earliest tie",
        "pair_order": "all directed non-self pairs in head/tail entity-index order",
        "ttm": "pair-independent axis correction, unused single-step write bypass",
        "memory": "frozen parameter in C; unread provenance buffer in bypass variants",
        "initialization": ("generic encoder and fresh task parameters" if arguments.variant == "G"
                           else "no-TTM Stage-1 source" if arguments.variant == "D"
                           else "shared Stage-1 source"),
        "pair_projections": "independent, head-to-tail initialization" if arguments.variant == "U" else "tied",
        "decoder": "all relation logits strictly greater than TH; no top-k cap",
        "train_test_reads": False,
    }


def required_inputs(arguments, operation):
    if arguments.output_dir is not None and arguments.output_dir.exists():
        raise FileExistsError("Use a new output directory: " + str(arguments.output_dir))
    if operation == "average":
        if arguments.checkpoints is None or len(set(arguments.checkpoints)) != 3:
            raise ValueError("Average requires three distinct --checkpoints")
        if arguments.checkpoint is not None or arguments.expected_checkpoint_sha256 is not None:
            raise ValueError("Average uses --checkpoints instead of a single --checkpoint")
        return list(arguments.checkpoints)
    if arguments.checkpoints is not None:
        raise ValueError("Use --checkpoint for training or evaluation")
    generic_training = operation == "train" and arguments.variant == "G"
    if arguments.model_dir is None or arguments.data_dir is None or (
            arguments.checkpoint is None and not generic_training):
        raise ValueError("Supply --model-dir, --data-dir and --checkpoint")
    if generic_training and (arguments.checkpoint is not None or
                             arguments.expected_checkpoint_sha256 is not None):
        raise ValueError("Generic training initializes from --model-dir only")
    if operation == "train" and arguments.variant == "C-off":
        raise ValueError("C-off evaluates a trained C checkpoint; use C or B for training")
    if operation == "train" and arguments.split != "dev":
        raise ValueError("Training selects checkpoints on Dev; use --split dev")
    config = arguments.model_dir / "config.json"
    if not config.is_file():
        raise FileNotFoundError(config)
    model_config = read_json(config)
    if model_config.get("model_type") != "roberta" or model_config.get("hidden_size") != 1024:
        raise ValueError("The recorded recipe requires a local RoBERTa-large configuration")
    files = [config, arguments.model_dir / "vocab.json", arguments.model_dir / "merges.txt",
             arguments.model_dir / "tokenizer.json", arguments.model_dir / "tokenizer_config.json",
             arguments.data_dir / "train_revised.json"]
    if arguments.checkpoint is not None:
        files.append(arguments.checkpoint)
    weights = arguments.model_dir / "model.safetensors"
    if not weights.is_file():
        weights = arguments.model_dir / "pytorch_model.bin"
    files.append(weights)
    split = "dev" if operation == "train" else arguments.split
    files.append(arguments.data_dir / (split + "_revised.json"))
    return files


def validate(arguments):
    operation = arguments.validate_for if arguments.mode == "validate" else arguments.mode
    files = required_inputs(arguments, operation)
    metadata = []
    for path in files:
        if not path.is_file():
            raise FileNotFoundError(path)
        metadata.append({"path": str(path), "bytes": path.stat().st_size})
    if arguments.expected_checkpoint_sha256 is not None:
        value = arguments.expected_checkpoint_sha256.lower()
        if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
            raise ValueError("Expected checkpoint digest must contain 64 hexadecimal characters")
    if arguments.mode != "validate" and arguments.output_dir is None:
        raise ValueError("Supply a new --output-dir")
    return {"operation": operation, "files": metadata,
            "checkpoint_contents_read": False, "dataset_contents_read": False,
            "model_execution": False, "dependencies_imported": False,
            "checkpoint_hash_checked": False}


@contextmanager
def project_imports():
    previous_directory = Path.cwd()
    previous_path = list(sys.path)
    os.chdir(ROOT)
    sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]
    try:
        yield
    finally:
        os.chdir(previous_directory)
        sys.path[:] = previous_path


def runtime():
    import numpy as np
    import torch
    import transformers
    return {"python": sys.version, "torch": torch.__version__,
            "numpy": np.__version__, "transformers": transformers.__version__,
            "cuda_runtime": torch.version.cuda,
            "float32_matmul_precision": torch.get_float32_matmul_precision(),
            "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "cuda_allocator_config": os.environ.get("PYTORCH_CUDA_ALLOC_CONF")}


def check_documents(documents, count, rel2id, data):
    if not isinstance(documents, list) or len(documents) != count:
        raise ValueError("Unexpected complete Re-DocRED split size")
    if len({document["title"] for document in documents}) != count:
        raise ValueError("Duplicate document titles")
    for document in documents:
        data.check_document(document, rel2id)


def read_data(arguments, split):
    import ttmre_human_distant_data as data
    rel2id = read_json(ROOT / "meta/rel2id.json")
    if sorted(rel2id.values()) != list(range(97)):
        raise ValueError("Expected the original 96 relations plus TH mapping")
    train_path = arguments.data_dir / "train_revised.json"
    split_path = arguments.data_dir / (split + "_revised.json")
    actual_train = digest(train_path)
    actual_split = digest(split_path)
    expected_split = DEV_SHA256 if split == "dev" else TEST_SHA256
    if actual_train != TRAIN_SHA256 or actual_split != expected_split:
        raise ValueError("Expected the exact benchmark split bytes recorded for this study")
    training = read_json(train_path)
    documents = read_json(split_path)
    check_documents(training, TRAIN_DOCUMENTS, rel2id, data)
    check_documents(documents, EVALUATION_DOCUMENTS, rel2id, data)
    priors = [0] * 96
    pair_count = 0
    for document in training:
        entities = len(document["vertexSet"])
        pair_count += entities * (entities - 1)
        for label in document["labels"]:
            priors[rel2id[label["r"]] - 1] += 1
    priors = [value / pair_count for value in priors]
    return training, documents, rel2id, priors, {
        "train_sha256": actual_train, "evaluation_sha256": actual_split,
        "split": split, "training_documents": len(training),
        "evaluation_documents": len(documents), "training_pairs": pair_count,
    }


def load_model(arguments, priors, target_checkpoint):
    import numpy as np
    import torch
    from transformers import AutoConfig, AutoModel, AutoTokenizer
    from model2 import DocREModel
    import train_ttmre_baseline_aligned as aligned
    import train_ttmre_human_only as shared
    from ttmre_memory_fit import install_unused_write_bypass, install_ttm_checkpoint
    from ttmre_pair_replacement import install

    device = torch.device(arguments.device)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = False
    config = AutoConfig.from_pretrained(str(arguments.model_dir), num_labels=97, local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(str(arguments.model_dir), local_files_only=True)
    config.cls_token_id = tokenizer.cls_token_id
    config.sep_token_id = tokenizer.sep_token_id
    config.transformer_type = "roberta"
    settings = SimpleNamespace(**aligned.protocol("pair-independent"))
    settings.device = device
    backbone = AutoModel.from_pretrained(str(arguments.model_dir), config=config,
                                         local_files_only=True).to(device)
    # Checkpoint loading replaces every task parameter. Reset the target stream
    # immediately before the training loaders, as in the original C/B runners.
    shared.seed_all(75, torch, np)
    model = DocREModel(settings, config, torch.tensor(np.array(priors) + 1e-9, device=device),
                       backbone, tokenizer).to(device)
    aligned.apply_variant(model, "pair-independent")
    install_unused_write_bypass(model)
    install_ttm_checkpoint(model)
    model.requires_grad_(True)
    # D's source was pretrained without TTM. Other adapted lightweight states
    # likewise require the replacement geometry before strict target loading.
    before_load_bypass = arguments.variant == "D" or (
        target_checkpoint and arguments.variant in {"B", "G", "U"})
    if before_load_bypass:
        install(model, "bypass")
    if arguments.checkpoint is not None:
        state = torch.load(arguments.checkpoint, map_location="cpu", weights_only=True)
        if not isinstance(state, dict) or MEMORY not in state:
            raise ValueError("Expected an original raw model state dictionary")
        model.load_state_dict(state, strict=True)
        del state
    if (not target_checkpoint and arguments.variant in {"B", "G", "U"}) or arguments.variant == "C-off":
        install(model, "bypass")
        gc.collect()
    if arguments.variant == "U":
        if not target_checkpoint:
            model.tail_extractor.load_state_dict(model.head_extractor.state_dict(), strict=True)
        model.args.model_type = "ATLOP"
    if arguments.variant == "C":
        model.mu_encoder.memory_tokens.requires_grad_(False)
    count = sum(parameter.numel() for parameter in model.parameters())
    if count != EXPECTED_PARAMETERS[arguments.variant]:
        raise ValueError("Model geometry differs from the recorded configuration")
    model.train_mode = "finetune"
    return model, tokenizer, device


def checkpoint_identity(arguments):
    if arguments.checkpoint is None:
        return {"initialization": "generic RoBERTa-large with fresh task parameters",
                "task_constructor_seed": 75, "source_task_checkpoint": None}
    checksum = digest(arguments.checkpoint)
    if arguments.expected_checkpoint_sha256 is not None:
        if checksum != arguments.expected_checkpoint_sha256.lower():
            raise ValueError("Checkpoint digest differs from --expected-checkpoint-sha256")
    return {"path": str(arguments.checkpoint), "sha256": checksum}


def save_checkpoint(state, destination):
    import torch
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(destination)
    partial = destination.with_name(destination.name + ".partial")
    with partial.open("xb") as stream:
        torch.save(state, stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(partial, destination)


def selection_due(completed_batches, total_batches):
    return completed_batches == total_batches - 1


def train(arguments):
    import numpy as np
    import torch
    from torch.utils.data import DataLoader
    from transformers.optimization import get_linear_schedule_with_warmup
    import train_ttmre_atgl_human as atgl
    import train_ttmre_human_only as shared
    import ttmre_human_distant_data as data

    output = arguments.output_dir
    output.mkdir(parents=True, exist_ok=False)
    source = checkpoint_identity(arguments)
    training, development, rel2id, priors, data_identity = read_data(arguments, "dev")
    model, tokenizer, device = load_model(arguments, priors, target_checkpoint=False)
    candidates = data.relevant_gold_facts(development)
    caches = output / "cache"
    datasets = {}
    metadata = {}
    for split, labels, count in (("train", True, TRAIN_DOCUMENTS), ("dev", False, EVALUATION_DOCUMENTS)):
        path = arguments.data_dir / (split + "_revised.json")
        with path.open("rb") as stream:
            metadata[split] = data.build_cache(stream, caches / split, tokenizer, rel2id,
                                               count, candidates, with_labels=labels)
        datasets[split] = data.DiskFeatures(caches / split, metadata[split], with_labels=labels)
    del training
    annotated = {tuple(value) for value in metadata["train"]["matching_dev_facts"]}
    memory = model.mu_encoder.memory_tokens.detach().cpu().clone()
    groups = [
        {"params": [parameter for name, parameter in model.named_parameters()
                    if parameter.requires_grad and not any(key in name for key in ("extractor", "bilinear"))],
         "lr": 3e-5},
        {"params": [parameter for name, parameter in model.named_parameters()
                    if parameter.requires_grad and any(key in name for key in ("extractor", "bilinear"))],
         "lr": 1e-5},
    ]
    optimizer = torch.optim.AdamW(groups, lr=3e-5, eps=1e-6, weight_decay=0.01)
    shared.seed_all(arguments.seed, torch, np)
    loader = DataLoader(datasets["train"], batch_size=4, shuffle=True, drop_last=True,
                        collate_fn=shared.collate_fn, num_workers=0)
    dev_loader = DataLoader(datasets["dev"], batch_size=4, shuffle=False,
                            collate_fn=shared.collate_fn, num_workers=0)
    total = len(loader) * EPOCHS
    if total != 22890:
        raise ValueError("Expected 22,890 target optimizer updates")
    scheduler = get_linear_schedule_with_warmup(optimizer, int(total * 0.06), total)
    write_json(output / "manifest.json", {"protocol": protocol(arguments), "runtime": runtime(),
               "source": source, "data": data_identity, "total_optimizer_updates": total,
               "test_data_read": False, "created_utc": datetime.now(timezone.utc).isoformat()})
    best_score = -math.inf
    best = None
    best_state = None
    steps = 0
    order = hashlib.sha256()
    with (output / "metrics.jsonl").open("x", encoding="utf-8") as metrics_log:
        for epoch in range(1, EPOCHS + 1):
            started = time.perf_counter()
            for index, batch in enumerate(loader, 1):
                order.update(batch[0].numpy().tobytes())
                order.update(json.dumps(batch[3:5], separators=(",", ":"), ensure_ascii=False).encode())
                model.train()
                model.zero_grad(set_to_none=True)
                logits = model(**shared.inputs_from_batch(batch, device, labels=False))
                labels = torch.cat([torch.as_tensor(value, device=device) for value in batch[2]], dim=0)
                loss = atgl.atgl_loss(logits, labels)
                if not torch.isfinite(loss) or not torch.isfinite(logits).all():
                    raise RuntimeError("Nonfinite target-stage loss or logits")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                steps += 1
                if steps == 1 or steps % 100 == 0:
                    emit("optimizer_update", epoch=epoch, step=steps, loss=float(loss.detach()))
                del loss, logits, labels
                if hasattr(model, "sims"):
                    model.sims = None
                if selection_due(index, len(loader)):
                    metrics = shared.evaluate_dev(model, dev_loader, device, development, annotated)
                    row = {"epoch": epoch, "step": steps, "batch": index, **metrics}
                    row["selected"] = metrics["dev_F1"] > best_score
                    if row["selected"]:
                        best_score = metrics["dev_F1"]
                        best = dict(row)
                        best_state = shared.cpu_snapshot(model)
                    metrics_log.write(json.dumps(row, allow_nan=False) + "\n")
                    metrics_log.flush()
                    emit("development_evaluation", **row)
            if not torch.equal(memory, model.mu_encoder.memory_tokens.detach().cpu()):
                raise RuntimeError("Frozen memory/provenance buffer changed")
            emit("epoch_complete", epoch=epoch, step=steps, seconds=time.perf_counter() - started)
    if steps != total or best_state is None:
        raise RuntimeError("Training did not complete the recorded target-stage recipe")
    destination = output / "best_model.pth"
    save_checkpoint(best_state, destination)
    model.load_state_dict(best_state, strict=True)
    del best_state, optimizer, scheduler
    replay = shared.evaluate_dev(model, dev_loader, device, development, annotated)
    for key in ("dev_F1", "dev_precision", "dev_recall", "dev_IgnF1_annotated"):
        if not math.isclose(replay[key], best[key], abs_tol=1e-6, rel_tol=0):
            raise RuntimeError("Selected checkpoint Dev replay differs")
    write_json(output / "completion.json", {"complete": True, "best": best, "Dev_replay": replay,
               "optimizer_updates": steps, "checkpoint_sha256": digest(destination),
               "sample_order_sha256": order.hexdigest(), "test_data_read": False})
    emit("training_complete", checkpoint=str(destination), best_dev_F1=best_score)


def predict(model, features, device, shared, logit_output=None):
    import numpy as np
    import torch
    from evaluation import to_official
    predictions = []
    saved_logits = []
    model.requires_grad_(False)
    model.eval()
    with torch.no_grad():
        for offset in range(0, len(features), BATCH_SIZE):
            batch_features = features[offset:offset + BATCH_SIZE]
            inputs = shared.inputs_from_batch(shared.collate_fn(batch_features), device, labels=False)
            logits = model(**inputs)
            if logits.dtype != torch.float32 or not torch.isfinite(logits).all():
                raise RuntimeError("Nonfinite or non-FP32 evaluation logits")
            values = logits.detach().cpu().numpy()
            if logit_output is not None:
                saved_logits.append(values.copy())
            flags = np.zeros_like(values)
            flags[:, 1:] = values[:, 1:] > values[:, :1]
            flags[:, 0] = flags[:, 1:].sum(axis=1) == 0
            predictions.extend(to_official(flags, batch_features))
    if logit_output is not None:
        with Path(logit_output).open('xb') as stream:
            np.save(stream, np.concatenate(saved_logits), allow_pickle=False)
    return predictions


def evaluate(arguments):
    import train_ttmre_human_only as shared
    import ttmre_human_distant_data as data
    output = arguments.output_dir
    output.mkdir(parents=True, exist_ok=False)
    checkpoint = checkpoint_identity(arguments)
    training, documents, rel2id, priors, data_identity = read_data(arguments, arguments.split)
    model, tokenizer, device = load_model(arguments, priors, target_checkpoint=True)
    features = []
    for document in documents:
        model_document = {"title": document["title"], "sents": document["sents"],
                          "vertexSet": document["vertexSet"], "labels": []}
        sparse = data.sparse_feature(model_document, tokenizer, rel2id, with_labels=False)
        features.append(data.expand_feature(sparse, with_labels=False))
    predictions = predict(model, features, device, shared,
                          output / 'logits.npy' if arguments.export_logits else None)
    if arguments.export_logits:
        write_json(output / 'pair_order.json', [
            {'title': feature['title'], 'hts': feature['hts']} for feature in features])
    prediction_path = output / "predictions.json"
    write_json(prediction_path, predictions)
    annotated = shared.train_facts(training)
    scores = shared.score_predictions(predictions, documents, annotated)
    scores = {key.replace("dev_", arguments.split + "_", 1): value for key, value in scores.items()}
    write_json(output / "metrics.json", {"variant": arguments.variant, "split": arguments.split,
               "metrics": scores, "checkpoint": checkpoint, "data": data_identity,
               "runtime": runtime(), "predictions_sha256": digest(prediction_path),
               "model_inputs_have_labels": False, "optimizer_updates": 0,
               "IgnD": None, "IgnD_reason": "Distant matching facts were not loaded"})
    emit("evaluation_complete", **scores)


def average(arguments):
    import torch
    output = arguments.output_dir
    output.mkdir(parents=True, exist_ok=False)
    result = {}
    schema = {}
    preserved = set()
    sources = []
    for index, path in enumerate(arguments.checkpoints):
        sources.append({"path": str(path), "sha256": digest(path)})
        state = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(state, dict) or MEMORY not in state:
            raise ValueError("Expected raw model state dictionaries with memory provenance")
        if index and state.keys() != schema.keys():
            raise ValueError("Checkpoint parameter keys differ")
        for name, value in state.items():
            if not isinstance(value, torch.Tensor) or value.is_complex() or value.layout != torch.strided:
                raise ValueError("Unsupported checkpoint tensor: " + name)
            geometry = (tuple(value.shape), str(value.dtype))
            if not index:
                schema[name] = geometry
            elif schema[name] != geometry:
                raise ValueError("Checkpoint shapes or dtypes differ: " + name)
            if value.is_floating_point():
                if value.dtype != torch.float32 or not torch.isfinite(value).all():
                    raise ValueError("Expected finite FP32 checkpoint tensors: " + name)
            keep = name == MEMORY or not value.is_floating_point()
            if keep:
                preserved.add(name)
                if not index:
                    result[name] = value.clone()
                elif not torch.equal(result[name], value):
                    raise ValueError("Preserved checkpoint buffer differs: " + name)
            elif not index:
                result[name] = value.to(torch.float64)
            else:
                result[name].add_(value)
        del state
        emit("checkpoint_accumulated", source=index + 1)
    for name in result:
        if name not in preserved:
            result[name] = result[name].div_(3).to(torch.float32)
        if result[name].is_floating_point() and not torch.isfinite(result[name]).all():
            raise ValueError("Nonfinite averaged tensor: " + name)
    destination = output / "averaged_model.pth"
    save_checkpoint(result, destination)
    write_json(output / "recipe.json", {"sources": sources, "coefficients": [1 / 3] * 3,
               "accumulation": "CPU FP64 then FP32", "preserved_equal_buffers": sorted(preserved),
               "checkpoint_sha256": digest(destination), "optimizer_updates": 0,
               "model_execution": False, "test_data_read": False,
               "torch": torch.__version__})
    emit("average_complete", checkpoint=str(destination))


def main(argv=None):
    command = parser()
    values = sys.argv[1:] if argv is None else argv
    if not values:
        command.print_help()
        return 0
    arguments = normalized_args(command.parse_args(values))
    validation = validate(arguments)
    if arguments.mode == "validate":
        print(json.dumps(validation, ensure_ascii=False, indent=2))
        return 0
    with project_imports():
        if arguments.mode == "train":
            train(arguments)
        elif arguments.mode == "evaluate":
            evaluate(arguments)
        else:
            average(arguments)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
