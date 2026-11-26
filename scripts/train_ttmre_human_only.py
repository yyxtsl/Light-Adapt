"""Offline, single-seed human-only baseline. Default mode is read-only validation."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[1]
REVISION = "6c86abdfbe38d1e12e991ea54cbeac8d057c2219"
SPLITS = {
    "train_revised.json": "c40a137a1c57f3e2daaf5ebf511eaa1518f9305ffaa3052d46271921a91826f2",
    "dev_revised.json": "051ee1d057204a5d08ef5502beacdadf191245b5eaf0e29ec2c607cf002c016f",
}
SOURCE_HASHES = {
    "model2.py": "9c9ef5c15ab585f8477d9d7534cf1615348e9c8f8e968de20179a97a53dbdc11",
    "ttm.py": "c9677d704b94f9e8a48e7707c806f2531acfa2efbd88a4017fc78d2a8c74decc",
    "prepro.py": "e05cd6e6b68ec3a81bbe1f4e1f4735660f4607cd485e36c46c1737ef58bfbead",
    "meta/rel2id.json": "b671eee2c3edb7bd44bede98ce6e292cbd154bbf9202c939b816c5a79063b5e3",
    "meta/relation_map.json": "ce15fe45203f4513192b8f5712909353bbd5ad7652badb6d61d8f9e1103fcd5d",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path):
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--mode", choices=("validate", "train"), default="validate")
    result.add_argument("--model-dir", type=Path, required=True)
    result.add_argument("--data-dir", type=Path, default=ROOT / "dataset" / "docred")
    result.add_argument("--run-dir", type=Path)
    result.add_argument("--distant-facts", type=Path,
                        help="Optional precomputed JSON list of [head_name, tail_name, relation]; evaluation only")
    result.add_argument("--seed", type=int, default=74)
    result.add_argument("--num-train-epochs", type=int, default=30)
    result.add_argument("--train-batch-size", type=int, default=4)
    result.add_argument("--dev-batch-size", type=int, default=4)
    return result


def validate(args) -> dict:
    source_hashes = {name: sha256(ROOT / name) for name in SOURCE_HASHES}
    for name, expected in SOURCE_HASHES.items():
        if source_hashes[name] != expected:
            raise ValueError(f"Pinned source identity mismatch: {name}")
    if min(args.num_train_epochs, args.train_batch_size, args.dev_batch_size) < 1:
        raise ValueError("Epochs and batch sizes must be positive")
    args.data_dir = args.data_dir.expanduser().resolve()
    args.model_dir = args.model_dir.expanduser().resolve()
    if args.run_dir is not None:
        args.run_dir = args.run_dir.expanduser().resolve()
        if not args.run_dir.is_relative_to(ROOT / "experiments" / "2026-10-07-human-only-seed74"):
            raise ValueError("Run directory must follow the documented dated experiment structure")
        if args.run_dir.exists():
            raise FileExistsError(f"Run directory already exists: {args.run_dir}")
    if args.mode == "train" and args.run_dir is None:
        raise ValueError("Train mode requires a new --run-dir")
    if args.seed != 74:
        raise ValueError("The approved first baseline uses seed 74 only")
    data = {}
    for name, expected in SPLITS.items():
        path = args.data_dir / name
        actual = sha256(path)
        if actual != expected:
            raise ValueError(f"Split identity mismatch: {path}")
        data[name] = {"path": str(path), "sha256": actual}
    config = read_json(args.model_dir / "config.json")
    for key, expected in (("model_type", "roberta"), ("hidden_size", 1024),
                          ("num_hidden_layers", 24), ("num_attention_heads", 16),
                          ("vocab_size", 50265), ("max_position_embeddings", 514)):
        if config.get(key) != expected:
            raise ValueError(f"Not the expected RoBERTa-large config: {key}")
    # Hash only files actually used by the offline loader, including sharded weights.
    files = {"config.json"}
    for name in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
                 "vocab.json", "merges.txt", "added_tokens.json"):
        if (args.model_dir / name).is_file():
            files.add(name)
    if "tokenizer.json" not in files and not {"vocab.json", "merges.txt"} <= files:
        raise FileNotFoundError("RoBERTa tokenizer files are missing")
    for name in ("model.safetensors", "model.safetensors.index.json",
                 "pytorch_model.bin", "pytorch_model.bin.index.json"):
        if (args.model_dir / name).is_file():
            files.add(name)
            if name.endswith("index.json"):
                files.update(read_json(args.model_dir / name)["weight_map"].values())
            break
    else:
        raise FileNotFoundError("No supported Hugging Face weights found")
    model_hashes = {}
    for name in sorted(files):
        path = (args.model_dir / name).resolve()
        if not path.is_relative_to(args.model_dir):
            # HF snapshots commonly use symlinks to their repository's blobs.
            if not path.is_relative_to(args.model_dir.parent.parent / "blobs"):
                raise ValueError(f"Model file resolves outside its directory/cache: {name}")
        model_hashes[name] = sha256(path)
    distant = None
    if args.distant_facts:
        args.distant_facts = args.distant_facts.expanduser().resolve()
        if args.distant_facts.name == "train_distant.json":
            raise ValueError("Supply a precomputed fact list, not distant training documents")
        load_distant_facts(args.distant_facts)
        distant = {"path": str(args.distant_facts), "sha256": sha256(args.distant_facts)}
    source_hashes.update({name: sha256(ROOT / name) for name in ("evaluation.py", "train2.py")})
    return {"source_revision_reference": REVISION,
            "source_files": source_hashes,
            "entrypoint_sha256": sha256(Path(__file__)), "data": data,
            "model": {"path": str(args.model_dir), "sha256": model_hashes,
                      "provenance": "User must supply the original generic pretrained RoBERTa-large"},
            "distant_reference": distant, "test_data_read": False}


def train_facts(documents: list) -> set:
    return {(head["name"], tail["name"], label["r"])
            for doc in documents for label in doc["labels"]
            for head in doc["vertexSet"][label["h"]]
            for tail in doc["vertexSet"][label["t"]]}


def load_distant_facts(path: Path | None) -> set | None:
    if path is None:
        return None
    values = read_json(path)
    if not isinstance(values, list) or any(
        not isinstance(value, list) or len(value) != 3
        or not all(isinstance(item, str) for item in value) for value in values
    ):
        raise ValueError("Distant reference must be a JSON list of three-string facts")
    return {tuple(value) for value in values}


def score_predictions(predictions: list, documents: list, annotated: set,
                      distant: set | None = None) -> dict:
    gold = {(doc["title"], label["h"], label["t"], label["r"])
            for doc in documents for label in doc["labels"]}
    entities = {doc["title"]: doc["vertexSet"] for doc in documents}
    predicted = {(pred["title"], pred["h_idx"], pred["t_idx"], pred["r"])
                 for pred in predictions}
    correct = predicted & gold
    precision = len(correct) / len(predicted) if predicted else 0.0
    recall = len(correct) / len(gold) if gold else 0.0

    def f1(p):
        return 200 * p * recall / (p + recall) if p + recall else 0.0

    def ignored(reference):
        count = sum(any((h["name"], t["name"], relation) in reference
                        for h in entities[title][head] for t in entities[title][tail])
                    for title, head, tail, relation in correct)
        p = (len(correct) - count) / (len(predicted) - count + 1e-5)
        return f1(p)

    return {"dev_F1": f1(precision), "dev_precision": precision * 100,
            "dev_recall": recall * 100, "dev_IgnF1_annotated": ignored(annotated),
            "dev_IgnF1_distant": ignored(distant) if distant is not None else None}


def seed_all(seed: int, torch, np) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def collate_fn(batch):
    import torch
    max_len = max(len(item["input_ids"]) for item in batch)
    ids = [item["input_ids"] + [0] * (max_len - len(item["input_ids"])) for item in batch]
    mask = [[1.0] * len(item["input_ids"]) + [0.0] * (max_len - len(item["input_ids"]))
            for item in batch]
    return (torch.tensor(ids, dtype=torch.long), torch.tensor(mask, dtype=torch.float),
            [item["labels"] for item in batch], [item["entity_pos"] for item in batch],
            [item["hts"] for item in batch])


def inputs_from_batch(batch, device, labels=True) -> dict:
    inputs = {"input_ids": batch[0].to(device), "attention_mask": batch[1].to(device),
              "entity_pos": batch[3], "hts": batch[4]}
    if labels:
        inputs["labels"] = batch[2]
    return inputs


def build_optimizer(model, args):
    import torch
    groups = [{"params": [p for name, p in model.named_parameters()
                          if not any(key in name for key in ("extractor", "bilinear"))]},
              {"params": [p for name, p in model.named_parameters()
                          if any(key in name for key in ("extractor", "bilinear"))], "lr": 1e-5}]
    return torch.optim.AdamW(groups, lr=args.learning_rate, eps=1e-6, weight_decay=0.01)


def evaluate_dev(model, loader, device, documents, annotated, distant=None):
    import numpy as np
    import torch
    from evaluation import to_official
    predictions = []
    offset = 0
    model.eval()
    with torch.no_grad():
        for batch in loader:
            logits = model(**inputs_from_batch(batch, device, labels=False))
            if not torch.isfinite(logits).all().item():
                raise RuntimeError("Nonfinite development logits")
            pred = np.zeros(tuple(logits.shape), dtype=np.float32)
            values = logits.cpu().numpy()
            pred[:, 1:] = values[:, 1:] > values[:, :1]
            pred[:, 0] = pred[:, 1:].sum(axis=1) == 0
            batch_size = batch[0].shape[0]
            features = loader.dataset[offset:offset + batch_size]
            predictions.extend(to_official(pred, features))
            offset += batch_size
    if offset != len(documents):
        raise RuntimeError("Development document count mismatch")
    return score_predictions(predictions, documents, annotated, distant)


def cpu_snapshot(model) -> dict:
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def record_best(model, score, best_score, epoch, run_dir):
    import torch
    if not math.isfinite(score):
        raise RuntimeError("Nonfinite selection score")
    if score > best_score:
        torch.save(cpu_snapshot(model), run_dir / "best_model.pth")
        return score, epoch
    return best_score, None


def write_json(path: Path, values: dict) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(values, handle, indent=2, ensure_ascii=False, default=str, allow_nan=False)
        handle.write("\n")


def run_training(args, manifest):
    # This module is imported only after validation; train2.py is never imported.
    import numpy as np
    import torch
    import transformers
    from torch.utils.data import DataLoader
    from transformers import AutoConfig, AutoModel, AutoTokenizer
    from transformers.optimization import get_linear_schedule_with_warmup
    from model2 import DocREModel
    from prepro import read_docred

    if (torch.__version__.split("+")[0] != "2.0.1" or torch.version.cuda != "11.8"
            or transformers.__version__ != "4.34.0" or np.__version__ != "1.24.3"):
        raise RuntimeError("Training requires the verified cloud dependency versions")
    if not torch.cuda.is_available():
        raise RuntimeError("The official S-PU baseline requires CUDA")
    args.device = torch.device("cuda:0")
    args.model_type, args.m_tag, args.isrank = "ttmre", "S-PU", 1
    args.num_class, args.num_layers = 97, 4
    args.e, args.m, args.beta, args.gamma = 1.0, 1.0, 0.0, 1.0
    args.learning_rate = 3e-5
    args.max_seq_length, args.memory_size = 1024, 200
    seed_all(args.seed, torch, np)
    config = AutoConfig.from_pretrained(str(args.model_dir), num_labels=97, local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(str(args.model_dir), local_files_only=True)
    config.cls_token_id, config.sep_token_id = tokenizer.cls_token_id, tokenizer.sep_token_id
    config.transformer_type = "roberta"
    training, priors = read_docred(args, str(args.data_dir / "train_revised.json"), tokenizer, 1024)
    development, _ = read_docred(args, str(args.data_dir / "dev_revised.json"), tokenizer, 1024)
    gold = read_json(args.data_dir / "dev_revised.json")
    annotated = train_facts(read_json(args.data_dir / "train_revised.json"))
    distant = load_distant_facts(args.distant_facts)
    backbone = AutoModel.from_pretrained(str(args.model_dir), config=config, local_files_only=True)
    model = DocREModel(args, config, torch.tensor(priors + 1e-9, device=args.device),
                       backbone, tokenizer).to(args.device)
    optimizer = build_optimizer(model, args)
    # Upstream train() resets the seed before shuffling and dropout.
    seed_all(args.seed, torch, np)
    loader = DataLoader(training, batch_size=args.train_batch_size, shuffle=True,
                        collate_fn=collate_fn, drop_last=True)
    dev_loader = DataLoader(development, batch_size=args.dev_batch_size, shuffle=False,
                            collate_fn=collate_fn, drop_last=False)
    if not len(loader):
        raise ValueError("Training loader is empty")
    total = len(loader) * args.num_train_epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, int(total * 0.06), total)
    args.run_dir.mkdir(parents=True, exist_ok=False)
    manifest.update({"created_utc": datetime.now(timezone.utc).isoformat(),
                     "versions": {"torch": torch.__version__, "transformers": transformers.__version__,
                                  "numpy": np.__version__, "cuda": torch.version.cuda},
                     "gpu": torch.cuda.get_device_name(0), "total_steps": total,
                     "warmup_steps": int(total * 0.06), "selection": "dev_F1; earliest on ties",
                     "protocol": "Human Annotation Only; no task checkpoint; no final test",
                     "parameters": {"head_learning_rate": 1e-5, "weight_decay": 0.01,
                                    "adam_epsilon": 1e-6, "gradient_accumulation_steps": 1,
                                    "warmup_ratio": 0.06, "max_grad_norm": 1.0,
                                    "precision": "float32", "drop_last": True},
                     "deviations": ["epoch-end development evaluation", "seed before model creation",
                                    "test isolation", "empty predictions score zero"]})
    write_json(args.run_dir / "manifest.json", manifest)
    write_json(args.run_dir / "args.json", vars(args))
    best_score, best_epoch, steps = -math.inf, None, 0
    with (args.run_dir / "metrics.jsonl").open("x", encoding="utf-8") as metrics_log, \
            (args.run_dir / "train.log").open("x", encoding="utf-8") as train_log:
        for epoch in range(1, args.num_train_epochs + 1):
            total_loss = 0.0
            for batch in loader:
                model.train()
                optimizer.zero_grad(set_to_none=True)
                risks, _ = model(**inputs_from_batch(batch, args.device))
                loss = sum(risks)
                if not torch.isfinite(loss).item():
                    raise RuntimeError(f"Nonfinite training loss at step {steps}")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
                optimizer.step()
                scheduler.step()
                steps += 1
                total_loss += loss.detach().item()
                if steps == 1 or steps % 100 == 0:
                    progress = json.dumps({"epoch": epoch, "step": steps, "loss": loss.detach().item()})
                    print(progress, flush=True)
                    train_log.write(progress + "\n")
                    train_log.flush()
            optimizer.zero_grad(set_to_none=True)
            metrics = evaluate_dev(model, dev_loader, args.device, gold, annotated, distant)
            best_score, selected = record_best(model, metrics["dev_F1"], best_score, epoch, args.run_dir)
            if selected is not None:
                best_epoch = selected
                write_json(args.run_dir / "best_metrics.json", {"epoch": epoch, "step": steps, **metrics})
            record = {"epoch": epoch, "step": steps, "train_loss": total_loss / len(loader),
                      "selected": selected is not None, **metrics}
            line = json.dumps(record, allow_nan=False)
            print(line, flush=True)
            for handle in (metrics_log, train_log):
                handle.write(line + "\n")
                handle.flush()
    model.load_state_dict(torch.load(args.run_dir / "best_model.pth", map_location="cpu", weights_only=True))
    write_json(args.run_dir / "completion.json", {"best_epoch": best_epoch, "best_dev_F1": best_score,
               "optimizer_steps": steps, "test_data_read": False, "test_evaluated": False})


def main() -> None:
    args = parser().parse_args()
    manifest = validate(args)
    if args.mode == "validate":
        print(json.dumps({"validation": "passed", "manifest": manifest,
                          "optimizer_steps": 0, "model_download": False, "files_written": False}, indent=2))
        return
    os.chdir(ROOT)
    sys.path.insert(0, str(ROOT))
    run_training(args, manifest)


if __name__ == "__main__":
    main()
