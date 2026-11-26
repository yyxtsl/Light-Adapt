"""Offline real-training-data forward/backward check, with no optimizer or artifact writes."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace

from train_ttmre_human_only import (ROOT, collate_fn, inputs_from_batch, parser as baseline_parser,
                                   read_json, seed_all, validate)


def select_batches(features: list, batch_size: int) -> dict:
    if len(features) < batch_size:
        raise ValueError("Not enough training documents")
    pairs = lambda index: len(features[index]["hts"])
    length = lambda index: len(features[index]["input_ids"])
    # Probe two different pressure sources; no claim that these cover every batch.
    by_pairs = sorted(range(len(features)), key=lambda index: (pairs(index), length(index)), reverse=True)
    by_length = sorted(range(len(features)), key=lambda index: (length(index), pairs(index)), reverse=True)
    return {"first": list(range(batch_size)), "most_pairs": by_pairs[:batch_size],
            "longest": by_length[:batch_size]}


def finite_backward(model, batch, device, torch):
    model.zero_grad(set_to_none=True)
    model.train()
    risks, logits = model(**inputs_from_batch(batch, device))
    loss = sum(risks)
    if not torch.isfinite(loss).item() or not torch.isfinite(logits).all().item():
        raise RuntimeError("Nonfinite full-model forward values")
    loss.backward()
    gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    if not gradients or not all(torch.isfinite(grad).all().item() for grad in gradients):
        raise RuntimeError("Missing/nonfinite full-model gradients")
    for name, parameter in (("memory", model.mu_encoder.memory_tokens),
                            ("extractor", model.head_extractor.weight),
                            ("classifier", model.bilinear.weight),
                            ("backbone", model.model.embeddings.word_embeddings.weight)):
        if parameter.grad is None or not parameter.grad.abs().sum().item() > 0:
            raise RuntimeError(f"Missing/zero {name} gradient")
    return loss.detach().item(), len(gradients)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    if args.model_dir is None:
        source = read_json(ROOT / "models" / "roberta-large-source.json")
        args.model_dir = (ROOT / "models" / "hf-cache" / "models--FacebookAI--roberta-large"
                          / "snapshots" / source["revision"])
    settings = baseline_parser().parse_args(["--model-dir", str(args.model_dir)])
    manifest = validate(settings)
    print(json.dumps({"identity_validation": "passed", "model": manifest["model"]}, indent=2), flush=True)
    os.chdir(ROOT)
    sys.path.insert(0, str(ROOT))
    import numpy as np
    import torch
    import transformers
    from transformers import AutoConfig, AutoTokenizer, AutoModel
    from prepro import read_docred
    from model2 import DocREModel
    if (torch.__version__.split("+")[0] != "2.0.1" or torch.version.cuda != "11.8"
            or transformers.__version__ != "4.34.0" or np.__version__ != "1.24.3"):
        raise RuntimeError("Full-model check requires the verified cloud versions")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for upstream S-PU")
    device = torch.device("cuda:0")
    seed_all(74, torch, np)
    config = AutoConfig.from_pretrained(str(settings.model_dir), num_labels=97, local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(str(settings.model_dir), local_files_only=True)
    features, priors = read_docred(settings, str(settings.data_dir / "train_revised.json"), tokenizer, 1024)
    selected = select_batches(features, args.batch_size)
    backbone = AutoModel.from_pretrained(str(settings.model_dir), config=config, local_files_only=True)
    config.cls_token_id, config.sep_token_id = tokenizer.cls_token_id, tokenizer.sep_token_id
    config.transformer_type = "roberta"
    model_args = SimpleNamespace(device=device, model_type="ttmre", num_class=97, isrank=1,
                                 e=1.0, m=1.0, m_tag="S-PU", beta=0.0, gamma=1.0, num_layers=4)
    model = DocREModel(model_args, config, torch.tensor(priors + 1e-9, device=device),
                       backbone, tokenizer).to(device)
    parameter_bytes = sum(parameter.numel() * parameter.element_size() for parameter in model.parameters())
    trainable_bytes = sum(parameter.numel() * parameter.element_size() for parameter in model.parameters()
                          if parameter.requires_grad)
    print(json.dumps({"gpu": torch.cuda.get_device_name(0), "batch_size": args.batch_size,
                      "parameter_GiB": parameter_bytes / 1024**3,
                      "adam_moments_estimate_GiB": 2 * trainable_bytes / 1024**3,
                      "precision": "float32", "seed": 74}), flush=True)
    for name, indices in selected.items():
        model.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        free, total = torch.cuda.mem_get_info()
        start = time.perf_counter()
        batch = collate_fn([features[index] for index in indices])
        loss, gradients = finite_backward(model, batch, device, torch)
        torch.cuda.synchronize()
        peak = torch.cuda.max_memory_allocated() / 1024**3
        report = {"probe": name, "document_indices": indices,
                  "titles": [features[index]["title"] for index in indices],
                  "token_lengths": [len(features[index]["input_ids"]) for index in indices],
                  "entity_pairs": [len(features[index]["hts"]) for index in indices],
                  "loss": loss, "finite_gradient_tensors": gradients,
                  "seconds": time.perf_counter() - start, "peak_allocated_GiB": peak,
                  "peak_reserved_GiB": torch.cuda.max_memory_reserved() / 1024**3,
                  "free_before_probe_GiB": free / 1024**3, "total_GiB": total / 1024**3,
                  "optimizer_steps": 0}
        print(json.dumps(report, allow_nan=False), flush=True)
    model.zero_grad(set_to_none=True)
    print("TTMRE_FULL_MODEL_CHECK_OK; optimizer_steps=0; test_labels_read=false; model_download=false; artifacts_saved=false")


if __name__ == "__main__":
    main()
