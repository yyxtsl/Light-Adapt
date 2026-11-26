"""One authorized frozen-Control75 six-head screen; new six-hour total cap.

Default validate is read-only. Training is allowed only through the one-use
Linux launcher. All source checkpoints and historical reports remain read-only.
"""
from __future__ import annotations

import argparse
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
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/"scripts")]
RUN_ROOT = ROOT/"experiments/2026-10-19-nonempty-head-screen"
DONE = ROOT/("experiments/2026-10-18-n1-weighted-completion/"
             "weighted-completion-launch-20261018-121010-EkYgSZ/workspace")
IDENTITY_SHA = "a8c228970956e46db14055de8c87ba1038f70fe45b7f09b4d7db01e2cf019d66"
CHECKPOINT_SHA = "b92f36a6adab9f2779d00e1d210a1ab72c0653961ecebd61685821a708bebbd7"
CONTROL_REPORT_SHA = "bd69c8a79b2cfc7b1f0df4fb4ba7981d247bad498c67925960e926e336043c8e"
REPLAY_SHA = "f2e84849cfa85097641ae2e7d78821e3bac38d9f09d152f8f955e7a3abe6668e"
CHECKPOINT = DONE/"control/human-frozen/artifacts/best_model.pth"
FILES = tuple("scripts/"+n for n in ("ttmre_nonempty_head.py",
    "train_ttmre_nonempty_head.py", "check_ttmre_nonempty_head.py", "run_ttmre_nonempty_head.sh"))
KINDS = ("spu", "atgl", "amtl", "bce", "factor", "set")
HARD_SECONDS, SOFT_SECONDS, TERM_SECONDS = 21600, 120, 30
COUNTS = {"train": 1193092, "dev": 193232}
DOCS = {"train": 3053, "dev": 500}
EPOCHS, PAIR_BATCH, PRODUCT_CHUNK, TOP_K = 30, 2048, 256, 16315
EMBEDDING, BLOCK_SIZE = 768, 64
TRAIN_POSITIVE, TRAIN_MULTILABEL, TRAIN_FACTS, DEV_FACTS = 67808, 17364, 85932, 17284
METRICS = ("dev_F1", "dev_precision", "dev_recall", "dev_IgnF1_annotated", "dev_IgnF1_distant")


class BudgetStop(TimeoutError):
    pass


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def digest(path, budget=None):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b""):
            if budget is not None:
                check_budget(budget)
            h.update(block)
    return h.hexdigest()


def publish(path, writer, replace=False):
    path = Path(path)
    if path.exists() and not replace:
        raise FileExistsError("Preserve existing artifact: "+str(path))
    tmp = path.with_name(path.name+".partial-"+uuid.uuid4().hex)
    with tmp.open("xb") as stream:
        writer(stream)
        stream.flush()
        os.fsync(stream.fileno())
    if replace:
        os.replace(tmp, path)
    else:
        os.link(tmp, path)  # Keep incomplete/historical files; no cleanup.
    return digest(path)


def write_json(path, data, replace=False):
    raw = (json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False)+"\n").encode("utf-8")
    return publish(path, lambda f: f.write(raw), replace)


def save_tensor(path, value):
    import torch
    return publish(path, lambda f: torch.save(value, f))


def save_array(path, value):
    import numpy as np
    return publish(path, lambda f: np.save(f, value, allow_pickle=False))


def emit(event, **fields):
    print(json.dumps({"event": event, **fields}, ensure_ascii=False, allow_nan=False), flush=True)


def protocol():
    return {"version": "nonempty-head-screen-v1", "seed": 75, "heads": list(KINDS),
        "frozen_carrier": "original unweighted complete Control75 Human best",
        "checkpoint_sha256": CHECKPOINT_SHA, "entire_carrier_frozen_eval": True,
        "hard_seconds": HARD_SECONDS, "soft_reserve_seconds": SOFT_SECONDS,
        "term_reserve_seconds": TERM_SECONDS, "human_pair_passes": EPOCHS,
        "effective_pair_batch": PAIR_BATCH, "product_chunk": PRODUCT_CHUNK,
        "shuffle": "private numpy RandomState(75), identical natural pair stream per head",
        "train_pairs": COUNTS["train"], "dev_pairs": COUNTS["dev"],
        "precision": "FP32 parameters/products; factor/set scalar likelihood in FP64 for numerical stability",
        "lr": .0001, "eps": 1e-6, "weight_decay": .01, "bias_weight_decay": 0.,
        "clip": 1., "warmup_ratio": .06, "selection": "end of every pair pass, native raw Dev F1; earliest tie",
        "atgl_alpha": 16.5, "amtl_lambda": 3.5, "amtl_groups": [10, 20, 46, 20],
        "amtl_decoder": "official raw TH rows; training uses sum of TH rows/3.5",
        "presence_prior": "Human train positive pairs / all pairs",
        "set_normalization": "zero-truncated independent Bernoulli product, no probability floor",
        "set_decode": "a*q/Z > .5; no hard gate or forced top1",
        "factor_decode": "a*q > .5", "bce_decode": "sigmoid > .5",
        "fixed_K": TOP_K, "main_gate_pp": .5, "mechanism_gate_pp": .2,
        "train_full_backbone": False, "new_distant_training": False, "test_data_read": False,
        "retry": False, "resume": False, "extension": False, "automatic_followup": False}


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=("validate", "train", "worker"), default="validate")
    p.add_argument("--phase", choices=("prepare", *KINDS, "finish"))
    p.add_argument("--run-dir", type=Path)
    return p


def check_budget(budget, reserve=SOFT_SECONDS):
    if time.monotonic() >= budget["deadline_monotonic"]-reserve:
        raise BudgetStop("Six-hour reserve reached; no retry/resume/extension")
    if os.name == "posix":
        boot = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
        if boot != budget["boot_id"]:
            raise BudgetStop("Boot changed; no restart of this budget")


def verify_run_context(run_dir, budget):
    if (run_dir.name != "workspace" or run_dir.parent.parent != RUN_ROOT.resolve()
            or budget.get("hard_seconds") != HARD_SECONDS
            or budget.get("deadline_monotonic")-budget.get("start_monotonic") != HARD_SECONDS
            or budget.get("automatic_retry") is not False
            or budget.get("automatic_resume") is not False or budget.get("extension") is not False):
        raise ValueError("Invalid immutable six-hour workspace/budget")
    claim = read(RUN_ROOT/"nonempty-head-once-claim.json")
    if (claim.get("workspace") != str(run_dir) or claim.get("protocol") != protocol()
            or claim.get("budget_sha256") != digest(run_dir/"budget.json")):
        raise ValueError("One-use claim/budget differs")
    identity_path = run_dir/"identity.json"
    if identity_path.exists() and read(identity_path).get("budget_sha256") != digest(run_dir/"budget.json"):
        raise ValueError("Identity does not bind this budget")


def require_lock():
    if os.name != "posix" or os.environ.get("TTMRE_NONEMPTY_HEAD_LOCK_FD") != "9":
        raise RuntimeError("Use only run_ttmre_nonempty_head.sh")
    import fcntl
    expected = (ROOT/"experiments/2026-10-08-baseline-alignment/.training.lock").stat()
    actual = os.fstat(9)
    if (actual.st_dev, actual.st_ino) != (expected.st_dev, expected.st_ino):
        raise RuntimeError("Descriptor9 is not the shared training lock")
    fcntl.flock(9, fcntl.LOCK_EX | fcntl.LOCK_NB)
    return 9


def validate(budget=None):
    if digest(DONE/"identity.json", budget) != IDENTITY_SHA:
        raise ValueError("Pinned completed-run identity changed")
    parent = read(DONE/"identity.json")
    if digest(DONE/"control/completion.json", budget) != CONTROL_REPORT_SHA:
        raise ValueError("Pinned Control completion changed")
    if digest(DONE/"control/human-frozen/artifacts/checkpoint_replay.json", budget) != REPLAY_SHA:
        raise ValueError("Pinned Control replay changed")
    report = read(DONE/"control/completion.json")
    replay = read(DONE/"control/human-frozen/artifacts/checkpoint_replay.json")
    if (report.get("passed") is not True or report.get("seed") != 75
            or report.get("new_optimizer_updates") != 0
            or report["human"]["checkpoint_sha256"] != CHECKPOINT_SHA
            or replay.get("passed") is not True
            or replay["full_dev_grouping"]["changed_relation_decisions"] != 0):
        raise ValueError("Original Control75 is not the completed accepted carrier")
    if digest(CHECKPOINT, budget) != CHECKPOINT_SHA:
        raise ValueError("Control checkpoint changed")
    sources = dict(parent["source_files"])
    for path, expected in sources.items():
        if digest(ROOT/path, budget) != expected:
            raise ValueError("Historical source changed: "+path)
    sources.update({p: digest(ROOT/p, budget) for p in FILES})
    cache = Path(parent["baseline_workspace"])/"cache"
    inputs = {str(CHECKPOINT): CHECKPOINT_SHA, str(DONE/"identity.json"): IDENTITY_SHA,
              str(DONE/"control/completion.json"): CONTROL_REPORT_SHA,
              str(DONE/"control/human-frozen/artifacts/checkpoint_replay.json"): REPLAY_SHA}
    # Read only Human/Dev data and the PRECOMPUTED Distant fact metadata.
    wanted = [cache/split/f for split in ("human", "dev") for f in ("index.json", "features.bin")]
    wanted += [cache/"distant/index.json"]
    wanted += [Path(parent["data_dir"])/f for f in ("train_revised.json", "dev_revised.json")]
    wanted += [Path(parent["model_dir"])/f for f in parent["model"]["sha256"]]
    for path in wanted:
        expected = parent["input_files"][str(path)]
        if digest(path, budget) != expected:
            raise ValueError("Approved input changed: "+str(path))
        inputs[str(path)] = expected
    return {"protocol": protocol(), "parent_identity_sha256": IDENTITY_SHA,
        "source_files": sources, "input_files": inputs, "checkpoint": str(CHECKPOINT),
        "model_dir": parent["model_dir"], "data_dir": parent["data_dir"],
        "baseline_workspace": parent["baseline_workspace"], "environment": parent["environment"],
        "baseline_identity": parent["baseline_identity"], "control_best": report["human_best"],
        "historical_control_strict_numeric_passed": replay["original_strict_grouping"]["passed"],
        "old_numeric_failure_preserved": True, "test_data_read": False,
        "distant_documents_decoded": False}


def verify_sources(identity, budget):
    if identity["protocol"] != protocol():
        raise ValueError("Frozen protocol changed")
    for path, expected in identity["source_files"].items():
        if digest(ROOT/path, budget) != expected:
            raise ValueError("Source changed during run: "+path)


def tensor_digest(state):
    h = hashlib.sha256()
    for name, value in sorted(state.items()):
        t = value.detach().cpu().contiguous()
        h.update((name+str(t.dtype)+str(tuple(t.shape))).encode())
        h.update(t.reshape(-1).view(__import__("torch").uint8).numpy().tobytes())
    return h.hexdigest()


def load_carrier(identity, run_dir):
    import torch
    from types import SimpleNamespace
    import train_ttmre_n1_full_seed75 as carrier
    args = SimpleNamespace(run_dir=run_dir, baseline_workspace=Path(identity["baseline_workspace"]),
        model_dir=Path(identity["model_dir"]), data_dir=Path(identity["data_dir"]),
        device=torch.device("cuda:0"))
    model = carrier.load_generic_model_seed75(args, identity["baseline_identity"])
    model.load_state_dict(torch.load(identity["checkpoint"], map_location="cpu", weights_only=True), strict=True)
    model.requires_grad_(False)
    model.eval()
    if any(p.requires_grad for p in model.parameters()):
        raise RuntimeError("Carrier freezing failed")
    return model


def original_features(identity):
    import ttmre_human_distant_data as data
    cache = Path(identity["baseline_workspace"])/"cache"
    return {name: data.DiskFeatures(cache/split, read(cache/split/"index.json"), name == "train")
            for name, split in (("train", "human"), ("dev", "dev"))}


def build_dev_targets(dataset, gold, rel2id, annotated, distant):
    import numpy as np
    by_title = {d["title"]: d for d in gold}
    labels, ign_a, ign_d, documents, reference = [], [], [], [], []
    offset = 0
    id2rel = {v: k for k, v in rel2id.items()}
    for f in dataset:
        doc = by_title[f["title"]]
        if len(f["hts"]) != len(doc["vertexSet"])*(len(doc["vertexSet"])-1):
            raise ValueError("Development pair scope changed")
        y = np.zeros((len(f["hts"]), 96), dtype=np.uint8)
        a, d = np.zeros_like(y), np.zeros_like(y)
        pair_index = {tuple(p): i for i, p in enumerate(f["hts"])}
        for fact in doc["labels"]:
            row, col = pair_index[(fact["h"], fact["t"])], rel2id[fact["r"]]-1
            y[row, col] = 1
            matches = [(h["name"], t["name"], fact["r"])
                for h in doc["vertexSet"][fact["h"]] for t in doc["vertexSet"][fact["t"]]]
            a[row, col] = any(x in annotated for x in matches)
            d[row, col] = any(x in distant for x in matches)
        for row, col in np.argwhere(y):
            h, t = f["hts"][int(row)]
            reference.append({"title": doc["title"], "h_idx": h, "t_idx": t,
                "r": id2rel[int(col)+1], "ignore_annotated": bool(a[row, col]),
                "ignore_distant": bool(d[row, col])})
        documents.append({"title": doc["title"], "offset": offset, "pairs": f["hts"]})
        offset += len(y)
        labels.append(y); ign_a.append(a); ign_d.append(d)
    return np.concatenate(labels), np.concatenate(ign_a), np.concatenate(ign_d), documents, reference


def predict_head(head, vectors, budget, device, chunk=PRODUCT_CHUNK):
    import numpy as np
    import torch
    out = np.empty((len(vectors), head.relations), dtype=np.float32)
    head.eval()
    with torch.no_grad():
        for start in range(0, len(vectors), chunk):
            check_budget(budget)
            v = torch.from_numpy(np.array(vectors[start:start+chunk], copy=True)).to(device)
            value = head.scores(head(v))
            if not torch.isfinite(value).all():
                raise RuntimeError("Nonfinite native decision scores")
            out[start:start+len(v)] = value.cpu().float().numpy()
    return out


def raw_predictions(pred, documents, id2rel):
    import numpy as np
    result = []
    for doc in documents:
        start = doc["offset"]
        for row, col in np.argwhere(pred[start:start+len(doc["pairs"])]):
            h, t = doc["pairs"][int(row)]
            result.append({"title": doc["title"], "h_idx": h, "t_idx": t, "r": id2rel[int(col)+1]})
    return result


def prepare(run_dir, budget):
    import numpy as np
    import torch
    import train_ttmre_human_only as original
    import ttmre_nonempty_head as method
    started = time.monotonic()
    identity = validate(budget)
    identity["budget_sha256"] = digest(run_dir/"budget.json")
    identity["source_environment_report"] = identity["environment"]
    write_json(run_dir/"identity.json", identity)
    # Exact vector bytes plus masks, scores, head weights and report headroom.
    vector_bytes = sum(COUNTS.values())*2*EMBEDDING*4
    required = max(15*1024**3, vector_bytes+4*1024**3)
    if shutil.disk_usage(run_dir).free < required:
        raise RuntimeError("Require at least15GiB free; preserve all old artifacts")
    cache = run_dir/"cache"
    cache.mkdir()
    datasets = original_features(identity)
    train_meta = datasets["train"].metadata
    if (len(datasets["train"]) != DOCS["train"] or len(datasets["dev"]) != DOCS["dev"]
            or train_meta["ordered_pairs"] != COUNTS["train"]
            or datasets["dev"].metadata["ordered_pairs"] != COUNTS["dev"]):
        raise ValueError("Pinned cached data counts differ")
    gold = read(Path(identity["data_dir"])/"dev_revised.json")
    rel2id = read(ROOT/"meta/rel2id.json")
    annotated = {tuple(x) for x in train_meta["matching_dev_facts"]}
    distant_meta = read(Path(identity["baseline_workspace"])/"cache/distant/index.json")
    distant = {tuple(x) for x in distant_meta["matching_dev_facts"]}
    y_dev, ign_a, ign_d, documents, reference = build_dev_targets(
        datasets["dev"], gold, rel2id, annotated, distant)
    if len(y_dev) != COUNTS["dev"] or int(y_dev.sum()) != DEV_FACTS:
        raise ValueError("FullDev target count mismatch")
    for name, values in (("dev_labels", y_dev), ("ignore_annotated", ign_a), ("ignore_distant", ign_d)):
        save_array(cache/(name+".npy"), values)
    write_json(cache/"dev_documents.json", documents)
    write_json(cache/"gold_reference.json", reference)
    model = load_carrier(identity, run_dir)
    before = tensor_digest(model.state_dict())
    if model.emb_size != EMBEDDING or model.block_size != BLOCK_SIZE:
        raise ValueError("Unexpected carrier representation shape")
    reference_head = {"weight": model.bilinear.weight.detach().cpu().clone(),
        "bias": model.bilinear.bias.detach().cpu().clone(),
        "priors": np.array(train_meta["priors"])+1e-9, "block_size": model.block_size,
        "emb_size": model.emb_size, "positive_rate": TRAIN_POSITIVE/COUNTS["train"], "rel2id": rel2id}
    reference_head["priors"] = torch.tensor(reference_head["priors"], dtype=torch.float64)
    save_tensor(cache/"reference_head.pth", reference_head)
    direct_scores = np.empty_like(y_dev, dtype=np.float32)
    manifests = {}
    positive = multilabel = facts = 0
    for split in ("dev", "train"):
        folder = cache/split
        folder.mkdir()
        temp = folder/"vectors.partial.npy"
        vectors = np.lib.format.open_memmap(temp, mode="w+", dtype=np.float32,
            shape=(COUNTS[split], 2, model.emb_size))
        labels = (np.lib.format.open_memmap(folder/"labels.partial.npy", mode="w+",
            dtype=np.uint8, shape=(COUNTS[split], 96)) if split == "train" else None)
        offset = changed = numeric = 0
        max_delta = 0.
        seen = []
        for doc_start in range(0, len(datasets[split]), 4):
            check_budget(budget)
            rows = datasets[split][doc_start:doc_start+4]
            batch = original.inputs_from_batch(original.collate_fn(rows), model.args.device, labels=False)
            with torch.no_grad():
                v = method.extract_vectors(model, batch)
                if v.shape[1:] != (2, EMBEDDING) or not torch.isfinite(v).all():
                    raise RuntimeError("Invalid frozen pair vectors")
                values = v.cpu().numpy()
                vectors[offset:offset+len(v)] = values
                if split == "dev":
                    actual = model(**batch)
                    cached = torch.cat([model.bilinear(method.block_product(v[i:i+PRODUCT_CHUNK], BLOCK_SIZE))
                                        for i in range(0, len(v), PRODUCT_CHUNK)])
                    if not torch.isfinite(actual).all() or not torch.isfinite(cached).all():
                        raise RuntimeError("Nonfinite original/cache replay logits")
                    da = actual[:, 1:] > actual[:, :1]
                    db = cached[:, 1:] > cached[:, :1]
                    changed += int((da != db).sum())
                    delta = (actual-cached).abs()
                    numeric += int((delta > (1e-4+1e-4*actual.abs())).sum())
                    max_delta = max(max_delta, float(delta.max()))
                    direct_scores[offset:offset+len(v)] = (actual[:, 1:]-actual[:, :1]).cpu().numpy()
                else:
                    y = np.array([z[1:] for row in rows for z in row["labels"]], dtype=np.uint8)
                    if len(y) != len(v) or not np.isin(y, (0, 1)).all():
                        raise ValueError("Natural multilabel training targets changed")
                    labels[offset:offset+len(v)] = y
                    cardinality = y.sum(-1)
                    positive += int((cardinality > 0).sum())
                    multilabel += int((cardinality > 1).sum())
                    facts += int(cardinality.sum())
                offset += len(v)
                seen.extend(row["title"] for row in rows)
            if doc_start == 0 or (doc_start//4+1) % 25 == 0:
                emit("NONEMPTY_CACHE_PROGRESS", split=split, documents=len(seen), pairs=offset,
                     elapsed_seconds=time.monotonic()-started)
        vectors.flush()
        del vectors
        os.link(temp, folder/"vectors.npy")
        if labels is not None:
            labels.flush()
            del labels
            os.link(folder/"labels.partial.npy", folder/"labels.npy")
        if offset != COUNTS[split] or len(set(seen)) != DOCS[split]:
            raise ValueError("Incomplete/duplicated source document scope")
        manifests[split] = {"pairs": offset, "documents": len(seen),
            "vectors_sha256": digest(folder/"vectors.npy", budget),
            "shape": [offset, 2, EMBEDDING], "dtype": "float32", "source_frozen_eval": True}
        if split == "dev":
            report = {"passed": changed == 0, "changed_relation_decisions": changed,
                "numeric_violations_original_tolerance": numeric, "max_abs_delta": max_delta,
                "rtol": 1e-4, "atol": 1e-4, "documents": len(seen), "pairs": offset,
                "policy": "new-cache-exact-decisions-no-tolerance-growth",
                "historical_control_strict_numeric_passed": identity["historical_control_strict_numeric_passed"]}
            write_json(run_dir/"cache_replay.json", report)
            if changed:
                raise RuntimeError("New cache changes original Dev decisions; stop without threshold repair")
            metrics = method.score_array(direct_scores > 0, y_dev.astype(bool),
                                          ign_a.astype(bool), ign_d.astype(bool))
            if not all(math.isclose(metrics[k], identity["control_best"][k], rel_tol=0, abs_tol=1e-6)
                       for k in METRICS):
                raise RuntimeError("Direct Control75 metric replay differs")
            write_json(run_dir/"control_metrics.json", metrics)
            save_array(cache/"control_scores.npy", direct_scores)
        write_json(folder/"manifest.json", manifests[split])
    if (positive, multilabel, facts) != (TRAIN_POSITIVE, TRAIN_MULTILABEL, TRAIN_FACTS):
        raise ValueError("Training NA/multilabel/fact counts differ from reviewed inventory")
    after = tensor_digest(model.state_dict())
    frozen = {"passed": before == after and not any(p.grad is not None for p in model.parameters()),
              "state_before": before, "state_after": after, "optimizer_steps": 0,
              "all_parameters_requires_grad_false": not any(p.requires_grad for p in model.parameters())}
    write_json(run_dir/"carrier_freeze.json", frozen)
    if not frozen["passed"] or digest(CHECKPOINT, budget) != CHECKPOINT_SHA:
        raise RuntimeError("Read-only carrier changed")
    manifests["train"]["labels_sha256"] = digest(cache/"train/labels.npy", budget)
    ids = sorted(range(1, 97), key=lambda r: (float(reference_head["priors"][r-1]), r))
    buckets = {str(r): ("rare", "middle", "frequent")[i//32] for i, r in enumerate(ids)}
    manifest = {"passed": True, "splits": manifests, "positive_pairs": positive,
        "multilabel_positive_pairs": multilabel, "facts": facts, "frequency_buckets": buckets,
        "reference_head_sha256": digest(cache/"reference_head.pth", budget),
        "files": {}, "elapsed_seconds": time.monotonic()-started, "gpu_optimizer_steps": 0}
    for path in sorted(cache.rglob("*")):
        if path.is_file() and ".partial" not in path.name:
            manifest["files"][str(path.relative_to(cache))] = digest(path, budget)
    write_json(cache/"completion.json", manifest)
    emit("NONEMPTY_CACHE_COMPLETE", train_pairs=COUNTS["train"], dev_pairs=COUNTS["dev"],
         original_F1=metrics["dev_F1"], new_carrier_optimizer_steps=0)


def load_cache(run_dir, budget, verify=True):
    import numpy as np
    import torch
    cache = run_dir/"cache"
    manifest = read(cache/"completion.json")
    if manifest.get("passed") is not True:
        raise ValueError("Incomplete cache cannot start an optimizer")
    if verify:
        for name, expected in manifest["files"].items():
            if digest(cache/name, budget) != expected:
                raise ValueError("Cache artifact changed: "+name)
    reference = torch.load(cache/"reference_head.pth", map_location="cpu", weights_only=True)
    arrays = {name: np.load(cache/path, mmap_mode="r", allow_pickle=False) for name, path in
        (("train", "train/vectors.npy"), ("train_labels", "train/labels.npy"),
         ("dev", "dev/vectors.npy"), ("dev_labels", "dev_labels.npy"),
         ("ignore_annotated", "ignore_annotated.npy"), ("ignore_distant", "ignore_distant.npy"))}
    for name in ("train", "dev"):
        if arrays[name].shape != (COUNTS[name], 2, EMBEDDING) or arrays[name].dtype != np.float32:
            raise ValueError("Invalid compact representation")
    return arrays, reference, manifest


def make_head(kind, reference, device):
    import ttmre_nonempty_head as method
    return method.PairHead(kind, reference["weight"], reference["bias"], reference["priors"],
        reference["positive_rate"], reference["rel2id"], reference["block_size"]).to(device)


def train_head(kind, run_dir, budget):
    import numpy as np
    import torch
    import ttmre_nonempty_head as method
    identity = read(run_dir/"identity.json")
    verify_sources(identity, budget)
    import train_ttmre_b2_5090 as runtime
    device = runtime.runtime(identity["environment"])
    arrays, reference, manifest = load_cache(run_dir, budget)
    directory = run_dir/"heads"/kind
    directory.mkdir(parents=True, exist_ok=False)
    head = make_head(kind, reference, device)
    initial = tensor_digest(head.state_dict())
    optimizer = torch.optim.AdamW([
        {"params": [p for n, p in head.named_parameters() if not n.endswith("bias")], "weight_decay": .01},
        {"params": [p for n, p in head.named_parameters() if n.endswith("bias")], "weight_decay": 0.}],
        lr=1e-4, eps=1e-6)
    batches = math.ceil(len(arrays["train"])/PAIR_BATCH)
    total = EPOCHS*batches
    warmup = int(total*.06)
    def scale(step):
        if step < warmup:
            return step/max(1, warmup)
        return max(0., (total-step)/max(1, total-warmup))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, scale)
    rng = np.random.RandomState(75)
    order_hash = hashlib.sha256()
    best, best_epoch, step = -math.inf, None, 0
    witness = None
    started = time.monotonic()
    gold = np.asarray(arrays["dev_labels"], dtype=bool)
    ign_a = np.asarray(arrays["ignore_annotated"], dtype=bool)
    ign_d = np.asarray(arrays["ignore_distant"], dtype=bool)
    with (directory/"metrics.jsonl").open("x", encoding="utf-8") as log:
        for epoch in range(1, EPOCHS+1):
            epoch_started = time.monotonic()
            order = rng.permutation(len(arrays["train"])).astype(np.int64)
            order_hash.update(order.astype("<i8").tobytes())
            losses = 0.
            head.train()
            for start in range(0, len(order), PAIR_BATCH):
                check_budget(budget)
                take = order[start:start+PAIR_BATCH]
                vectors = torch.from_numpy(np.array(arrays["train"][take], copy=True)).to(device)
                labels = torch.from_numpy(np.array(arrays["train_labels"][take], copy=True)).to(device)
                optimizer.zero_grad(set_to_none=True)
                # All chunk logits share ONE complete effective-batch S-PU risk.
                values = head.batched_forward(vectors, PRODUCT_CHUNK)
                loss = head.loss(values, labels)
                if not torch.isfinite(loss):
                    raise RuntimeError("Nonfinite "+kind+" loss")
                loss.backward()
                grads = [p for p in head.parameters() if p.grad is not None]
                if not grads or any(not torch.isfinite(p.grad).all() for p in grads):
                    raise RuntimeError("Missing/nonfinite head gradients")
                norm = torch.nn.utils.clip_grad_norm_(head.parameters(), 1.)
                if not torch.isfinite(norm):
                    raise RuntimeError("Nonfinite global head gradient norm")
                learning = witness is None and optimizer.param_groups[0]["lr"] > 0
                before = {n: p.detach().clone() for n, p in head.named_parameters()} if learning else None
                optimizer.step()
                if learning:
                    changes = {n: float((p.detach()-before[n]).abs().max()) for n, p in head.named_parameters()}
                    moments = sum("exp_avg" in s and "exp_avg_sq" in s for s in optimizer.state.values())
                    if not any(v > 0 for v in changes.values()) or not moments:
                        raise RuntimeError("PositiveLR call did not learn")
                    if kind in ("factor", "set") and labels.any():
                        if not any(v > 0 for n, v in changes.items() if n.startswith("conditional.")):
                            raise RuntimeError("Positive examples did not update conditional head")
                    witness = {"passed": True, "step": step+1, "lr": optimizer.param_groups[0]["lr"],
                               "parameter_max_changes": changes, "adam_moment_tensors": moments,
                               "actual_backward": True}
                    write_json(directory/"learning_update.json", witness)
                    emit("NONEMPTY_LEARNING_UPDATE_VERIFIED", head=kind, **witness)
                scheduler.step()
                step += 1
                losses += float(loss.detach())*len(take)
                if step == 1 or step % 100 == 0:
                    emit("nonempty_optimizer_update", head=kind, epoch=epoch, step=step,
                         batch=start//PAIR_BATCH+1, loss=float(loss.detach()), actual_learning_verified=witness is not None)
            scores = predict_head(head, arrays["dev"], budget, device)
            metrics = method.score_array(scores > 0, gold, ign_a, ign_d)
            selected = metrics["dev_F1"] > best
            if selected:
                best, best_epoch = metrics["dev_F1"], epoch
                snapshot = {n: p.detach().cpu().clone() for n, p in head.state_dict().items()}
                name = "selected-epoch%02d.pth" % epoch
                save_tensor(directory/name, snapshot)
                best_record = {"head": kind, "epoch": epoch, "step": step, **metrics,
                    "checkpoint": name, "checkpoint_sha256": digest(directory/name, budget)}
                write_json(directory/"best.json", best_record, replace=True)
            row = {"epoch": epoch, "step": step, "mean_train_loss": losses/len(order),
                   "selected": selected, **metrics, "epoch_seconds": time.monotonic()-epoch_started}
            log.write(json.dumps(row, allow_nan=False)+"\n")
            log.flush()
            emit("nonempty_epoch_complete", head=kind, **row)
            # Measured projection belongs to this authorized run and budget.
            if epoch == 1:
                remaining_passes = EPOCHS-1+(len(KINDS)-KINDS.index(kind)-1)*EPOCHS
                estimate = row["epoch_seconds"]*remaining_passes+180
                remaining = budget["deadline_monotonic"]-time.monotonic()-SOFT_SECONDS
                write_json(directory/"throughput.json", {"first_pass_seconds": row["epoch_seconds"],
                    "conservative_remaining_estimate_seconds": estimate, "remaining_seconds": remaining,
                    "estimate_not_guarantee": True, "no_extra_probe_or_updates": True})
                if estimate > remaining:
                    raise BudgetStop("Measured remaining six-head projection exceeds original six-hour cap")
    if witness is None or step != total:
        raise RuntimeError("Incomplete head update schedule")
    best_record = read(directory/"best.json")
    head.load_state_dict(torch.load(directory/best_record["checkpoint"], map_location="cpu", weights_only=True))
    scores = predict_head(head, arrays["dev"], budget, device)
    metrics = method.score_array(scores > 0, gold, ign_a, ign_d)
    if any(abs(metrics[k]-best_record[k]) > 1e-6 for k in METRICS):
        raise RuntimeError("Best-head fullDev replay differs")
    save_array(directory/"dev_scores.npy", scores)
    documents = read(run_dir/"cache/dev_documents.json")
    write_json(directory/"predictions.json", raw_predictions(scores > 0, documents,
        {v: k for k, v in reference["rel2id"].items()}))
    buckets = {int(k): v for k, v in manifest["frequency_buckets"].items()}
    diag = method.diagnostics(scores, gold, buckets, TOP_K)
    write_json(directory/"diagnostics.json", diag)
    activity = mechanism_activity(head, arrays["dev"], gold, budget, device) if kind in ("factor", "set") else None
    if activity is not None:
        write_json(directory/"mechanism.json", activity)
    completion = {"passed": True, "head": kind, "optimizer_steps": step,
        "best": best_record, "metrics": metrics, "diagnostics": diag, "mechanism": activity,
        "initial_head_sha256": initial, "sample_order_sha256": order_hash.hexdigest(),
        "learning_update_verified": witness is not None, "carrier_optimizer_updates": 0,
        "elapsed_seconds": time.monotonic()-started, "test_data_read": False,
        "scores_sha256": digest(directory/"dev_scores.npy", budget),
        "predictions_sha256": digest(directory/"predictions.json", budget),
        "checkpoint_sha256": digest(directory/best_record["checkpoint"], budget)}
    completion["final_state_sha256"] = tensor_digest(head.state_dict())
    completion["initial_parameters_match_final"] = completion["final_state_sha256"] == initial
    write_json(directory/"completion.json", completion)
    emit("NONEMPTY_HEAD_COMPLETE", head=kind, F1=metrics["dev_F1"], epoch=best_epoch, optimizer_steps=step)


def mechanism_activity(head, vectors, gold, budget, device):
    import numpy as np
    import torch
    import ttmre_nonempty_head as method
    all_a, all_z = [], []
    head.eval()
    with torch.no_grad():
        for start in range(0, len(vectors), PRODUCT_CHUNK):
            check_budget(budget)
            v = torch.from_numpy(np.array(vectors[start:start+PRODUCT_CHUNK], copy=True)).to(device)
            out = head(v)
            all_a.append(torch.sigmoid(out["g"]).cpu().double().numpy())
            all_z.append(method.log_nonempty(out["b"]).exp().cpu().numpy())
    a, z = np.concatenate(all_a), np.concatenate(all_z)
    cardinality = gold.sum(-1)
    result = {}
    for name, mask in (("all", cardinality >= 0), ("positive", cardinality > 0),
                       ("multi_label", cardinality > 1)):
        zz, aa = z[mask], a[mask]
        result[name] = {"pairs": int(mask.sum()), "mean_Z": float(zz.mean()),
            "Z_quantiles": np.quantile(zz, [0, .25, .5, .75, 1]).tolist(),
            "mean_implicit_nonempty_gap": float((aa*(1-zz)).mean())}
    result["interpretation"] = "formal activity only; no causal performance conclusion"
    return result


def screen_gate(results, control, control_diag):
    existing = {name: results[name] for name in ("spu", "atgl", "amtl", "bce")}
    existing["original"] = {"metrics": control, "diagnostics": control_diag}
    strongest_name = max(existing, key=lambda name: existing[name]["metrics"]["dev_F1"])
    strongest = existing[strongest_name]
    candidate, factor = results["set"], results["factor"]
    m, s, f = candidate["metrics"], strongest["metrics"], factor["metrics"]
    md, sd, fd = candidate["diagnostics"], strongest["diagnostics"], factor["diagnostics"]
    checks = {
        "F1_vs_strongest_existing_at_least_0_5pp": m["dev_F1"]-s["dev_F1"] >= .5,
        "both_Ign_positive_vs_strongest": all(m[k] > s[k] for k in METRICS[-2:]),
        "both_Ign_positive_vs_original": all(m[k] > control[k] for k in METRICS[-2:]),
        "F1_vs_factor_at_least_0_2pp": m["dev_F1"]-f["dev_F1"] >= .2,
        "multilabel_F1_improves_vs_factor": md["subsets"]["multi_label"]["F1"] > fd["subsets"]["multi_label"]["F1"],
        "fixed_K_ranking_beats_strongest": md["fixed_K"]["F1"] > sd["fixed_K"]["F1"],
        "rare_middle_not_worse": md["frequency"]["rare_middle"]["F1"] >= sd["frequency"]["rare_middle"]["F1"],
        "NA_false_positives_not_increased": md["subsets"]["na_pairs"]["FP"] <= sd["subsets"]["na_pairs"]["FP"],
        "normalization_active_on_positive_pairs":
            candidate["mechanism"]["positive"]["mean_implicit_nonempty_gap"] > 0,
    }
    return {"screening_gate_passed": all(checks.values()), "checks": checks,
        "strongest_existing": strongest_name, "delta_F1_pp": m["dev_F1"]-s["dev_F1"],
        "delta_vs_factor_pp": m["dev_F1"]-f["dev_F1"],
        "NA_FP_acceptance": "conservative no-increase relative to strongest existing",
        "reliable_improvement_established": False, "novelty_established": False,
        "automatic_followup": False,
        "decision": "candidate_for_separately_reviewed_confirmation" if all(checks.values())
                    else "stop_this_fixed_representation_candidate"}


def finish(run_dir, budget):
    import numpy as np
    import ttmre_nonempty_head as method
    identity = read(run_dir/"identity.json")
    verify_sources(identity, budget)
    manifest = read(run_dir/"cache/completion.json")
    if not read(run_dir/"carrier_freeze.json")["passed"] or not read(run_dir/"cache_replay.json")["passed"]:
        raise ValueError("Missing source/cache acceptance")
    results = {kind: read(run_dir/"heads"/kind/"completion.json") for kind in KINDS}
    expected_steps = EPOCHS*math.ceil(COUNTS["train"]/PAIR_BATCH)
    order_hashes = set()
    for kind, result in results.items():
        if (result.get("passed") is not True or result["optimizer_steps"] != expected_steps
                or result["carrier_optimizer_updates"] != 0 or not result["learning_update_verified"]):
            raise ValueError("Missing full "+kind+" acceptance")
        path = run_dir/"heads"/kind
        for name, key in (("dev_scores.npy", "scores_sha256"), ("predictions.json", "predictions_sha256")):
            if digest(path/name, budget) != result[key]:
                raise ValueError("Head final artifact changed")
        if digest(path/result["best"]["checkpoint"], budget) != result["checkpoint_sha256"]:
            raise ValueError("Selected head changed")
        order_hashes.add(result["sample_order_sha256"])
    if len(order_hashes) != 1 or results["factor"]["initial_head_sha256"] != results["set"]["initial_head_sha256"]:
        raise ValueError("Paired natural order or factor/set initialization mismatch")
    if digest(CHECKPOINT, budget) != CHECKPOINT_SHA:
        raise ValueError("Read-only parent checkpoint changed")
    scores = np.load(run_dir/"cache/control_scores.npy", allow_pickle=False)
    gold = np.load(run_dir/"cache/dev_labels.npy", allow_pickle=False).astype(bool)
    buckets = {int(k): v for k, v in manifest["frequency_buckets"].items()}
    control_diag = method.diagnostics(scores, gold, buckets, TOP_K)
    control = read(run_dir/"control_metrics.json")
    directory = run_dir/"comparison"
    directory.mkdir()
    write_json(directory/"original_diagnostics.json", control_diag)
    gate = screen_gate(results, control, control_diag)
    completion = {"passed": True, "marker": "NONEMPTY_HEAD_SCREEN_COMPLETE",
        "protocol": protocol(), "original_control": control,
        "heads": results, **gate, "optimizer_steps_per_head": expected_steps,
        "new_head_optimizer_steps": expected_steps*len(KINDS), "carrier_optimizer_steps": 0,
        "elapsed_seconds": time.monotonic()-budget["start_monotonic"],
        "historical_control_strict_numeric_passed": identity["historical_control_strict_numeric_passed"],
        "test_data_read": False, "test_evaluated": False}
    # Native checkpoint/score artifacts have all finished before the result seal.
    check_budget(budget)
    write_json(directory/"completion.json", completion)
    # Report-only export excludes all task weights, vectors and raw test data.
    archive = directory/"reports.tar.gz"
    selected = [p for p in run_dir.rglob("*") if p.is_file()
        and ".partial" not in p.name and p.suffix in (".json", ".jsonl")
        and not p.is_relative_to(run_dir/"cache")]
    def export(stream):
        with tarfile.open(fileobj=stream, mode="w:gz") as tar:
            for path in selected:
                check_budget(budget)
                tar.add(path, arcname=str(path.relative_to(run_dir)), recursive=False)
    publish(archive, export)
    emit("NONEMPTY_HEAD_SCREEN_COMPLETE", screening_gate_passed=gate["screening_gate_passed"],
         strongest_existing=gate["strongest_existing"], delta_F1_pp=gate["delta_F1_pp"],
         reports=str(archive), automatic_followup=False)


def worker(args):
    require_lock()
    run_dir = args.run_dir.resolve()
    budget = read(run_dir/"budget.json")
    verify_run_context(run_dir, budget)
    check_budget(budget)
    def stop(*_):
        raise BudgetStop("Worker stopped; partial artifacts retained, no retry")
    signal.signal(signal.SIGTERM, stop)
    try:
        if args.phase == "prepare":
            prepare(run_dir, budget)
        elif args.phase in KINDS:
            train_head(args.phase, run_dir, budget)
        elif args.phase == "finish":
            finish(run_dir, budget)
        else:
            raise ValueError("Unknown phase")
    except BaseException as error:
        write_json(run_dir/("failure-"+str(args.phase)+".json"),
            {"error": str(error), "type": type(error).__name__, "complete": False,
             "elapsed_seconds": time.monotonic()-budget["start_monotonic"],
             "automatic_retry": False, "resumable": False})
        raise


def supervise(command, budget, lock_fd):
    check_budget(budget)
    child = subprocess.Popen(command, start_new_session=True, pass_fds=(lock_fd,))
    term_sent = False
    try:
        while True:
            status = child.poll()
            now = time.monotonic()
            if now >= budget["deadline_monotonic"]:
                if status is None:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait()
                raise BudgetStop("Six-hour hard cap; worker group stopped")
            if status is not None:
                if status != 0 or term_sent:
                    raise RuntimeError("Worker failed/stopped exit="+str(status)+"; no retry")
                return
            if now >= budget["deadline_monotonic"]-TERM_SECONDS and not term_sent:
                os.killpg(child.pid, signal.SIGTERM)
                term_sent = True
            time.sleep(.2)
    finally:
        if child.poll() is None:
            os.killpg(child.pid, signal.SIGKILL)
            child.wait()


def train(args):
    if args.run_dir is None:
        raise ValueError("Use the unique launcher workspace")
    lock_fd = require_lock()
    started = time.monotonic()
    run_dir = args.run_dir.resolve()
    if run_dir.name != "workspace" or run_dir.parent.parent != RUN_ROOT.resolve() or run_dir.exists():
        raise ValueError("Only a new dated launch/workspace is permitted")
    claim = RUN_ROOT/"nonempty-head-once-claim.json"
    if claim.exists():
        raise FileExistsError("Authorization already claimed; do not retry")
    run_dir.mkdir()
    budget = {"start_monotonic": started, "deadline_monotonic": started+HARD_SECONDS,
        "hard_seconds": HARD_SECONDS, "soft_reserve_seconds": SOFT_SECONDS,
        "term_reserve_seconds": TERM_SECONDS, "start_utc": datetime.now(timezone.utc).isoformat(),
        "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
        "authorization": "user-confirmed-DESIGN-20261019-once-six-hour-frozen-head-screen",
        "automatic_retry": False, "automatic_resume": False, "extension": False}
    write_json(run_dir/"budget.json", budget)
    write_json(claim, {"workspace": str(run_dir), "budget_sha256": digest(run_dir/"budget.json"),
                       "protocol": protocol()})
    write_json(run_dir.parent/"pid.json", {"supervisor_pid": os.getpid()})
    def stop(*_):
        raise BudgetStop("Supervisor interrupted; no retry/resume/extension")
    signal.signal(signal.SIGTERM, stop)
    command = [sys.executable, "-u", "-B", str(Path(__file__).resolve()),
               "--mode", "worker", "--run-dir", str(run_dir)]
    try:
        emit("NONEMPTY_HEAD_SCREEN_BUDGET_STARTED", **budget)
        supervise([sys.executable, "-B", str(ROOT/FILES[2])], budget, lock_fd)
        for phase in ("prepare", *KINDS, "finish"):
            check_budget(budget)
            supervise(command+["--phase", phase], budget, lock_fd)
    except BaseException as error:
        write_json(run_dir/"failure.json", {"error": str(error), "type": type(error).__name__,
            "complete": False, "automatic_retry": False, "resumable": False,
            "elapsed_seconds": time.monotonic()-started})
        raise


def main():
    args = parser().parse_args()
    os.chdir(ROOT)
    if args.mode == "train":
        train(args)
    elif args.mode == "worker":
        worker(args)
    else:
        print(json.dumps({"validation": "passed", "identity": validate(),
                          "files_written": False, "optimizer_steps": 0}, ensure_ascii=False))


if __name__ == "__main__":
    main()
