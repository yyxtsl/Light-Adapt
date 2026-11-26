"""One authorized zero-training Dev module/fixed-K diagnostic, total cap30min.

Default validate is read-only. Use the one-use Bash launcher; no Test or tuning.
"""
from __future__ import annotations

import argparse
from collections import OrderedDict
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import evaluate_ttmre_atgl_soup as soup

base = soup.base
read, write_json, digest, emit = soup.read, soup.write_json, soup.digest, soup.emit
RUN_ROOT = ROOT / "experiments/2026-10-21-atgl-soup-diagnostic"
CLAIM = "atgl-soup-diagnostic-once-claim.json"
FILES = tuple("scripts/" + n for n in (
    "evaluate_ttmre_atgl_soup_diagnostic.py", "check_ttmre_atgl_soup_diagnostic.py",
    "run_ttmre_atgl_soup_diagnostic.sh"))
HARD_SECONDS, SOFT_SECONDS, TERM_SECONDS = 1800, 120, 30
TRAIN_DOCS, DEV_DOCS, DEV_PAIRS, K = 3053, 500, 193232, 16210
GROUPS = ("ttm", "backbone", "head")
MEMORY = soup.MEMORY
SOUP_SHA = "d399ce68cd144855692628e31b9f0dc4427425107d423c0d20a7ada07177b22e"
OLD_PINS = {
    "identity.json": "80324d53eba2d4d99032cfcd0f6cff6bbc0bc1ea28c1fa51e1971415cbbfe5eb",
    "comparison/completion.json": "51172ff952936ed611f109a4fbb1c76da6d0a244c31b414e67a5723080a9f01d",
    "predictions.json": "73471e037ba427b4496cf3e0f355e9d680de0a92979df0f71c451c0aa1f4afb3",
    "dev_pair_order.json": "ea399322decf3903b557ffe034567d9f57e90c2b73350fabdcf65bbbf1ec96dd"}
PARENT_PINS = {
    "frequency_diagnostics.json": soup.REPORT_PINS["77"]["atgl/human-frozen/artifacts/frequency_diagnostics.json"],
    "predictions.json": "0ba38b73f61f506358b274fb46eb186a54841bf6217615c45901665e1ca8ede2",
    "dev_pair_order.json": OLD_PINS["dev_pair_order.json"],
    "dev_margins.npy": "2de9c554620eaef287c76749c08ece42f5a07d61d4eaa560ba7a60c2a7b8126a"}


def protocol():
    return {"version": "atgl-soup-diagnostic-v1", "hard_seconds": HARD_SECONDS,
            "soft_reserve_seconds": SOFT_SECONDS, "term_reserve_seconds": TERM_SECONDS,
            "configurations": ["soup-margin-capture", "restore-ttm", "restore-backbone", "restore-head"],
            "restore_anchor_seed": 77, "soup_sha256": SOUP_SHA,
            "anchor_sha256": soup.CHECKPOINTS[77], "fixed_K": K,
            "fixed_K_ties": "descending margin, ascending original pair index, ascending relation ID",
            "native_decoder": soup.protocol()["decoder"], "optimizer_updates": 0,
            "training": False, "test_data_read": False, "dev_documents": DEV_DOCS,
            "dev_pairs": DEV_PAIRS, "dev_batch_size": 4, "precision": "float32",
            "choose_best_hybrid": False, "TTM_investment_min_F1_drop_pp": .20,
            "automatic_retry": False, "automatic_resume": False, "extension": False,
            "automatic_followup": False, "novelty_established": False}


def check_budget(budget, reserve=SOFT_SECONDS):
    if budget is None:
        return
    if time.monotonic() >= budget["deadline_monotonic"] - reserve:
        raise TimeoutError("30-minute diagnostic reserve reached; no extension")
    if os.name == "posix" and Path("/proc/sys/kernel/random/boot_id").read_text().strip() != budget["boot_id"]:
        raise TimeoutError("Boot identity changed; cannot reset budget")


def verify_files(paths, budget):
    for name, expected in paths.items():
        check_budget(budget)
        if digest(Path(name), budget) != expected:
            raise ValueError("Read-only source changed/missing: " + name)


def validate(budget=None):
    """Inherit the pinned completed soup provenance, check only used assets."""
    old = Path(read(soup.RUN_ROOT / "atgl-soup-once-claim.json")["workspace"])
    if old != old.resolve() or old.name != "workspace" or old.parent.parent != soup.RUN_ROOT.resolve():
        raise ValueError("Unexpected completed soup workspace")
    pinned = {str(old / name): sha for name, sha in OLD_PINS.items()}
    verify_files(pinned, budget)
    origin, completion = read(old / "identity.json"), read(old / "comparison/completion.json")
    if (not completion["passed"] or completion["marker"] != "ATGL_UNIFORM_SOUP_COMPLETE"
            or completion["checkpoint_sha256"] != SOUP_SHA or completion["optimizer_updates"] != 0
            or completion["test_data_read"] is not False):
        raise ValueError("Expected completed zero-training soup")
    source = next(x for x in origin["sources"] if x["seed"] == 77)
    checkpoint = Path(source["checkpoint"])
    if not checkpoint.resolve().is_relative_to(ROOT.resolve()):
        raise ValueError("Anchor is outside project")
    art = checkpoint.parent
    pinned.update({str(art / name): sha for name, sha in PARENT_PINS.items()})
    pinned.update({str(checkpoint): soup.CHECKPOINTS[77], str(old / "soup_model.pth"): SOUP_SHA})
    parent = origin["parent"]
    cache = Path(parent["baseline_workspace"]) / "cache"
    wanted = [cache / split / n for split in ("human", "dev") for n in ("index.json", "features.bin")]
    wanted += [cache / "distant/index.json"]  # Precomputed matching facts only.
    wanted += [Path(parent["data_dir"]) / n for n in ("train_revised.json", "dev_revised.json")]
    model_dir = Path(parent["model_dir"])
    wanted += [Path(n) for n in parent["input_files"] if Path(n).is_relative_to(model_dir)]
    wanted += [ROOT / "meta/rel2id.json"]
    for path in wanted:
        if not path.resolve().is_relative_to(ROOT.resolve()):
            raise ValueError("Used asset is outside project")
        expected = origin["pinned_files"].get(str(path))
        if expected is None:
            expected = origin["source_files"].get(path.relative_to(ROOT).as_posix())
        if expected is None:
            raise ValueError("Used asset has no prior digest: " + str(path))
        pinned[str(path)] = expected
    verify_files(pinned, budget)
    sources = dict(origin["source_files"])
    verify_files({str(ROOT / n): sha for n, sha in sources.items()}, budget)
    sources.update({n: digest(ROOT / n, budget) for n in FILES})
    frequency = read(art / "frequency_diagnostics.json")
    if (frequency["margins_sha256"] != PARENT_PINS["dev_margins.npy"]
            or frequency["pair_order_sha256"] != OLD_PINS["dev_pair_order.json"]
            or not frequency["passed"]):
        raise ValueError("Anchor margin provenance changed")
    return {"protocol": protocol(), "parent": parent, "source_files": sources,
            "pinned_files": pinned, "soup_checkpoint": str(old / "soup_model.pth"),
            "anchor_checkpoint": str(checkpoint), "soup_predictions": str(old / "predictions.json"),
            "anchor_predictions": str(art / "predictions.json"), "anchor_margins": str(art / "dev_margins.npy"),
            "pair_order": str(old / "dev_pair_order.json"), "soup_metrics": completion["metrics"],
            "anchor_metrics": frequency["metrics"], "optimizer_updates": 0, "test_data_read": False}


def resources(workspace, identity):
    free = shutil.disk_usage(workspace).free
    required = 1024 ** 3  # One 74MB margin array plus predictions, reports, reserve; no checkpoints.
    if free < required:
        raise RuntimeError("Project disk free %.3fGiB < required 1GiB" % (free / 1024 ** 3))
    return {"free_bytes": free, "required_bytes": required, "recommended_available_RAM_GiB": 32,
            "new_full_checkpoints": 0}


def module_group(name):
    if name == MEMORY:
        return "memory"
    for prefix, group in (("model.", "backbone"), ("mu_encoder.", "ttm"),
                          ("head_extractor.", "head"), ("tail_extractor.", "head"), ("bilinear.", "head")):
        if name.startswith(prefix):
            return group
    raise ValueError("Unmapped state key: " + name)


def tensor_bytes_equal(left, right):
    import torch
    if left.shape != right.shape or left.dtype != right.dtype:
        return False
    return torch.equal(left.detach().cpu().contiguous().reshape(-1).view(torch.uint8),
                       right.detach().cpu().contiguous().reshape(-1).view(torch.uint8))


def state_report(full, anchor, budget):
    import torch
    if set(full) != set(anchor) or MEMORY not in full:
        raise ValueError("State keys differ or memory missing")
    rows = {g: {"keys": [], "numel": 0, "changed_numel": 0, "squared_l2": 0., "max_abs_delta": 0.}
            for g in (*GROUPS, "memory")}
    tensors = {}
    for name, value in full.items():
        check_budget(budget)
        group, other = module_group(name), anchor[name]
        if (not isinstance(value, torch.Tensor) or not isinstance(other, torch.Tensor)
                or value.shape != other.shape or value.dtype != other.dtype
                or value.layout != torch.strided or other.layout != torch.strided or value.is_complex()):
            raise ValueError("State tensor schema mismatch: " + name)
        if value.is_floating_point():
            if value.dtype != torch.float32 or not torch.isfinite(value).all() or not torch.isfinite(other).all():
                raise ValueError("Expected finite FP32 state: " + name)
        if (group == "memory" or not value.is_floating_point()) and not tensor_bytes_equal(value, other):
            raise ValueError("Frozen memory/nonfloat differs: " + name)
        delta = value.double() - other.double()
        row = rows[group]
        row["keys"].append(name)
        row["numel"] += value.numel()
        changed = int(torch.count_nonzero(delta))
        row["changed_numel"] += changed
        row["squared_l2"] += float(torch.linalg.vector_norm(delta)) ** 2
        maximum = float(delta.abs().max()) if value.numel() else 0.
        row["max_abs_delta"] = max(row["max_abs_delta"], maximum)
        tensors[name] = {"group": group, "shape": list(value.shape), "dtype": str(value.dtype),
                         "changed_numel": changed, "max_abs_delta": maximum}
    if any(not rows[g]["keys"] for g in rows):
        raise ValueError("A required module group is absent")
    return {"groups": rows, "tensors": tensors, "all_state_keys_mapped": True, "memory_bytes_identical": True,
            "anchor": "ATGL seed77", "no_op_notes": [
                "TTM branch uses head_extractor for both entities; tail_extractor is retained without changing upstream.",
                "Unused write bypass/pooler are retained; no separate no-op ablation."],
            "recipe": "replace exactly one mapped group from seed77; all other state from fixed full soup"}


def restored_state(full, anchor, group):
    if group not in GROUPS:
        raise ValueError("Unknown restoration group")
    return OrderedDict((n, anchor[n] if module_group(n) == group else v) for n, v in full.items())


def pairs(order):
    return [(doc["title"], h, t) for doc in order for h, t in doc["hts"]]


def margin_predictions(margins, order, rel2id, k=None):
    """No gold input. Stable global top-K; native path uses margin>0."""
    import numpy as np
    flat = margins.reshape(-1)
    ordered_pairs = pairs(order)
    relations = len(rel2id) - 1
    if margins.shape != (len(ordered_pairs), relations) or margins.dtype != np.float32 or not np.isfinite(margins).all():
        raise ValueError("Invalid margin shape/dtype/values")
    if k is None:
        selected = np.flatnonzero(flat > 0)
    else:
        if not isinstance(k, int) or not 0 < k <= flat.size:
            raise ValueError("K outside fixed margin population")
        boundary = np.partition(flat, flat.size - k)[flat.size - k]
        above = np.flatnonzero(flat > boundary)
        tied = np.flatnonzero(flat == boundary)[:k - len(above)]
        selected = np.concatenate((above, tied))
        selected = selected[np.lexsort((selected, -flat[selected]))]
    id2rel = {v: key for key, v in rel2id.items()}
    result = []
    for index in selected:
        row, rel = divmod(int(index), relations)
        title, h, t = ordered_pairs[row]
        result.append({"title": title, "h_idx": h, "t_idx": t, "r": id2rel[rel + 1]})
    return result


def capture(model, loader, device, budget):
    import numpy as np
    import torch
    from evaluation import to_official
    predictions, order, margins, offset = [], [], [], 0
    model.eval()
    with torch.no_grad():
        for batch in loader:
            check_budget(budget)
            values = model(**base.core.shared.inputs_from_batch(batch, device, labels=False)).cpu().numpy()
            if values.dtype != np.float32 or not np.isfinite(values).all():
                raise ValueError("Nonfinite/nonFP32 diagnostic logits")
            margin = values[:, 1:] - values[:, :1]
            if not np.isfinite(margin).all():
                raise ValueError("Nonfinite relation margins")
            margins.append(margin)
            flags = np.zeros_like(values)
            flags[:, 1:] = values[:, 1:] > values[:, :1]
            flags[:, 0] = flags[:, 1:].sum(1) == 0
            features = loader.dataset[offset:offset + len(batch[0])]
            predictions.extend(to_official(flags, features))
            order.extend({"title": f["title"], "hts": f["hts"]} for f in features)
            offset += len(features)
    return predictions, order, np.concatenate(margins)


def frequency_report(predictions, gold, buckets, rel2id):
    truth = {(d["title"], x["h"], x["t"], x["r"]) for d in gold for x in d["labels"]}
    predicted = soup.facts(predictions)
    result = {}
    for name in ("rare", "middle", "frequent"):
        p, g = ({x for x in seq if buckets[rel2id[x[3]]] == name} for seq in (predicted, truth))
        tp, fp, fn = len(g & p), len(p - g), len(g - p)
        result[name] = {"TP": tp, "FP": fp, "FN": fn, "F1": 200 * tp / (2 * tp + fp + fn) if tp + fp + fn else 0.}
    return result


def investment_gate(full, restored, fixed_full, fixed_anchor):
    return (full["dev_F1"] - restored["dev_F1"] >= .20
            and all(full[k] >= restored[k] for k in ("dev_IgnF1_annotated", "dev_IgnF1_distant"))
            and fixed_full["dev_F1"] > fixed_anchor["dev_F1"])


def execute(workspace, identity, budget):
    import numpy as np
    import torch
    from torch.utils.data import DataLoader
    parent = identity["parent"]
    args = SimpleNamespace(run_dir=workspace, baseline_workspace=Path(parent["baseline_workspace"]),
                           model_dir=Path(parent["model_dir"]), data_dir=Path(parent["data_dir"]), budget=budget)
    write_json(workspace / "resource_check.json", resources(workspace, identity))
    args.device = base.core.baseline.control.runtime(parent["environment"])
    datasets = base.frozen.original_features(parent)
    human, dev = datasets["train"], datasets["dev"]
    if len(human) != TRAIN_DOCS or len(dev) != DEV_DOCS:
        raise ValueError("Human/Dev scope differs")
    rel2id, expected_order = read(ROOT / "meta/rel2id.json"), read(Path(identity["pair_order"]))
    old_soup, old_anchor = read(Path(identity["soup_predictions"])), read(Path(identity["anchor_predictions"]))
    old_margins = np.load(identity["anchor_margins"], mmap_mode="r", allow_pickle=False)
    if old_margins.shape != (DEV_PAIRS, 96) or len(old_anchor) != K:
        raise ValueError("Anchor margins/population/K changed")
    anchor_native = margin_predictions(old_margins, expected_order, rel2id)
    anchor_fixed = margin_predictions(old_margins, expected_order, rel2id, K)
    if soup.facts(anchor_native) != soup.facts(old_anchor) or soup.facts(anchor_fixed) != soup.facts(old_anchor):
        raise ValueError("Reused ATGL77 margins do not reproduce native/fixed-K predictions")
    del old_margins
    full = torch.load(identity["soup_checkpoint"], map_location="cpu", weights_only=True)
    anchor = torch.load(identity["anchor_checkpoint"], map_location="cpu", weights_only=True)
    summary = state_report(full, anchor, budget)
    write_json(workspace / "module_mapping.json", summary)
    model = base.core.load_generic_model_seed75(args, parent["baseline_identity"])
    model.requires_grad_(False)
    model.eval()
    loader = DataLoader(dev, batch_size=4, shuffle=False, collate_fn=base.core.shared.collate_fn)
    results, generated = {}, {}
    # All predictions (including fixed-K) are produced without passing gold to any decoder.
    for kind in ("soup-margin-capture", *("restore-" + g for g in GROUPS)):
        check_budget(budget)
        directory = workspace / kind
        directory.mkdir()
        state = full if kind == "soup-margin-capture" else restored_state(full, anchor, kind[8:])
        model.load_state_dict(state, strict=True)
        before = base.core.preflight.state_digest(model)
        predictions, order, margins = capture(model, loader, args.device, budget)
        if order != expected_order or margins.shape != (DEV_PAIRS, 96):
            raise ValueError("Dev pair order/population differs")
        if kind == "soup-margin-capture":
            if soup.facts(predictions) != soup.facts(old_soup):
                raise ValueError("Captured soup predictions differ from prior native result")
            generated["soup_fixed"] = margin_predictions(margins, order, rel2id, K)
            np.save(directory / "dev_margins.npy", margins, allow_pickle=False)
            write_json(directory / "margin_evidence.json", {"sha256": digest(directory / "dev_margins.npy", budget),
                       "shape": list(margins.shape), "dtype": str(margins.dtype),
                       "reason_for_new_forward": "prior soup export did not contain margins", "native_matches_prior": True})
        del margins
        strict = base.core.aligned.grouping_invariance(model, dev, args.device)
        grouped, replay = base.accepted.full_dev_grouping(model, dev, args.device, budget)
        valid = (grouped["passed"] and strict["changed_relation_decisions"] == 0
                 and all(strict["checks"][k]["close"] for k in ("ttm_grouped_vs_single", "ttm_permutation"))
                 and soup.facts(predictions) == soup.facts(replay)
                 and before == base.core.preflight.state_digest(model)
                 and tensor_bytes_equal(model.mu_encoder.memory_tokens, full[MEMORY]))
        write_json(directory / "replay.json", {"passed": bool(valid), "original_strict_grouping": strict,
                   "full_dev_grouping": grouped, "state_unchanged": before == base.core.preflight.state_digest(model),
                   "optimizer_updates": 0})
        write_json(directory / "predictions.json", predictions)
        write_json(directory / "dev_pair_order.json", order)
        write_json(directory / "recipe.json", {"soup_sha256": SOUP_SHA, "anchor_sha256": soup.CHECKPOINTS[77],
                   "restore_group": None if kind == "soup-margin-capture" else kind[8:], "memory_unchanged": True,
                   "state_sha256": before, "full_checkpoint_saved": False})
        if not valid:
            raise RuntimeError("Diagnostic acceptance failed: " + kind)
        generated[kind] = predictions
        emit("ATGL_SOUP_DIAGNOSTIC_CONFIGURATION_EVALUATED", configuration=kind, predictions=len(predictions), optimizer_updates=0)
    # Gold enters only scoring after every configuration and fixed-K prediction is fixed.
    gold = read(args.data_dir / "dev_revised.json")
    annotated = {tuple(x) for x in human.metadata["matching_dev_facts"]}
    distant = {tuple(x) for x in read(args.baseline_workspace / "cache/distant/index.json")["matching_dev_facts"]}
    buckets = base.core.method.frequency_buckets(human.metadata["priors"])
    for kind, predictions in {**generated, "anchor_fixed": anchor_fixed}.items():
        score = base.core.shared.score_predictions(predictions, gold, annotated, distant)
        results[kind] = score
        directory = workspace / kind
        directory.mkdir(exist_ok=True)
        write_json(directory / "metrics.json", score)
        write_json(directory / "predictions.json", predictions)
        write_json(directory / "frequency_diagnostics.json", frequency_report(predictions, gold, buckets, rel2id))
    for name, reference in (("soup-margin-capture", identity["soup_metrics"]), ("anchor_fixed", identity["anchor_metrics"])):
        if any(not math.isclose(results[name][k], reference[k], rel_tol=0, abs_tol=1e-6) for k in soup.METRICS):
            raise ValueError("Prior native metrics do not replay")
    verify_files(identity["pinned_files"], budget)
    verify_files({str(ROOT / n): sha for n, sha in identity["source_files"].items()}, budget)
    comparison = workspace / "comparison"
    comparison.mkdir()
    completion = {"passed": True, "marker": "ATGL_SOUP_DIAGNOSTIC_COMPLETE", "protocol": protocol(),
                  "metrics": results, "native_references": {"soup": identity["soup_metrics"], "ATGL77": identity["anchor_metrics"]},
                  "full_minus_restored_pp": {g: {k: results["soup-margin-capture"][k] - results["restore-" + g][k]
                                                  for k in soup.METRICS} for g in GROUPS},
                  "fixedK_soup_minus_anchor_pp": {k: results["soup_fixed"][k] - results["anchor_fixed"][k] for k in soup.METRICS},
                  "TTM_investment_gate_passed": bool(investment_gate(results["soup-margin-capture"], results["restore-ttm"],
                                                                   results["soup_fixed"], results["anchor_fixed"])),
                  "optimizer_updates": 0, "anchor_model_forwards": 0, "frozen_memory_unchanged": True,
                  "test_data_read": False, "test_evaluated": False, "new_full_checkpoints": 0,
                  "elapsed_seconds": time.monotonic() - budget["start_monotonic"], "automatic_followup": False,
                  "hybrid_selection_performed": False, "novelty_established": False,
                  "interpretation": "conditional module replacement and quantity diagnostic; not additive causal attribution or novelty proof"}
    write_json(comparison / "completion.json", completion)
    soup.archive_reports(workspace, budget)
    emit("ATGL_SOUP_DIAGNOSTIC_COMPLETE", TTM_investment_gate_passed=completion["TTM_investment_gate_passed"],
         reports=str(comparison / "reports.tar.gz"), optimizer_updates=0, automatic_followup=False)


def require_lock():
    if os.name != "posix" or os.environ.get("TTMRE_ATGL_SOUP_DIAGNOSTIC_LOCK_FD") != "9":
        raise RuntimeError("Use the one-use diagnostic Bash launcher")
    import fcntl
    a, b = os.fstat(9), (ROOT / "experiments/2026-10-08-baseline-alignment/.training.lock").stat()
    if (a.st_dev, a.st_ino) != (b.st_dev, b.st_ino):
        raise RuntimeError("Wrong shared training lock")
    fcntl.flock(9, fcntl.LOCK_EX | fcntl.LOCK_NB)
    return 9


def verify_context(workspace, budget):
    claim = read(RUN_ROOT / CLAIM)
    if (workspace != workspace.resolve() or workspace.name != "workspace" or workspace.parent.parent != RUN_ROOT.resolve()
            or budget["hard_seconds"] != HARD_SECONDS
            or budget["deadline_monotonic"] - budget["start_monotonic"] != HARD_SECONDS
            or any(budget[k] is not False for k in ("automatic_retry", "automatic_resume", "extension"))
            or claim["workspace"] != str(workspace) or claim["protocol"] != protocol()
            or claim["budget_sha256"] != digest(workspace / "budget.json")
            or claim["identity_sha256"] != digest(workspace / "identity.json")):
        raise ValueError("Invalid immutable diagnostic workspace/claim/budget")
    check_budget(budget)


def supervise(command, budget, lock):
    check_budget(budget)
    child = subprocess.Popen(command, start_new_session=True, pass_fds=(lock,))
    term_sent = False
    try:
        while True:
            now, status = time.monotonic(), child.poll()
            if now >= budget["deadline_monotonic"]:
                raise TimeoutError("30-minute hard cap; stop worker group")
            if status is not None:
                if status or term_sent:
                    raise RuntimeError("Diagnostic worker/checker failed exit=" + str(status) + "; no retry")
                return
            check_budget(budget, reserve=0)
            if now >= budget["deadline_monotonic"] - TERM_SECONDS and not term_sent:
                os.killpg(child.pid, signal.SIGTERM)
                term_sent = True
            time.sleep(.2)
    finally:
        if child.poll() is None:
            os.killpg(child.pid, signal.SIGKILL)
            child.wait()


def worker(workspace):
    require_lock()
    budget = read(workspace / "budget.json")
    verify_context(workspace, budget)
    def stop(*_):
        raise TimeoutError("Diagnostic interrupted; preserve partial reports")
    signal.signal(signal.SIGTERM, stop)
    try:
        identity = read(workspace / "identity.json")
        if identity["protocol"] != protocol():
            raise ValueError("Diagnostic protocol changed")
        verify_files(identity["pinned_files"], budget)
        verify_files({str(ROOT / n): sha for n, sha in identity["source_files"].items()}, budget)
        execute(workspace, identity, budget)
    except BaseException as error:
        write_json(workspace / "failure-worker.json", {"error": str(error), "type": type(error).__name__,
                   "complete": False, "optimizer_updates": 0, "automatic_retry": False})
        if not (workspace / "comparison/reports.tar.gz").exists():
            try:
                soup.archive_reports(workspace, budget)
            except BaseException:
                pass
        raise


def start(workspace):
    lock = require_lock()
    if workspace.name != "workspace" or workspace.parent.parent != RUN_ROOT.resolve() or workspace.exists():
        raise ValueError("Only a new diagnostic launch workspace is permitted")
    claim_path = RUN_ROOT / CLAIM
    if claim_path.exists():
        raise FileExistsError("Diagnostic already claimed; no retry")
    now = time.monotonic()
    budget = {"start_monotonic": now, "deadline_monotonic": now + HARD_SECONDS,
              "hard_seconds": HARD_SECONDS, "soft_reserve_seconds": SOFT_SECONDS,
              "term_reserve_seconds": TERM_SECONDS, "start_utc": datetime.now(timezone.utc).isoformat(),
              "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
              "automatic_retry": False, "automatic_resume": False, "extension": False}
    def stop(*_):
        raise TimeoutError("Diagnostic supervisor interrupted")
    signal.signal(signal.SIGTERM, stop)
    try:
        emit("ATGL_SOUP_DIAGNOSTIC_BUDGET_STARTED", **budget)
        identity = validate(budget)
        emit("ATGL_SOUP_DIAGNOSTIC_SOURCES_VERIFIED", resources=resources(workspace.parent, identity))
        supervise([sys.executable, "-B", str(ROOT / FILES[1])], budget, lock)
        check_budget(budget)
        workspace.mkdir()
        write_json(workspace / "budget.json", budget)
        write_json(workspace / "identity.json", identity)
        with claim_path.open("x", encoding="utf-8") as handle:
            json.dump({"workspace": str(workspace), "protocol": protocol(),
                       "budget_sha256": digest(workspace / "budget.json"),
                       "identity_sha256": digest(workspace / "identity.json"),
                       "authorization": "user continued fixed NEXT_PLAN, one Dev-only diagnostic cap30min"}, handle, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
        supervise([sys.executable, "-u", "-B", str(Path(__file__).resolve()),
                   "--mode", "worker", "--run-dir", str(workspace)], budget, lock)
    except BaseException as error:
        write_json(workspace.parent / "failure-supervisor.json", {"error": str(error), "type": type(error).__name__,
                   "complete": False, "elapsed_seconds": time.monotonic() - now,
                   "optimizer_updates": 0, "automatic_retry": False})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("validate", "start", "worker"), default="validate")
    parser.add_argument("--run-dir", type=Path)
    args = parser.parse_args()
    if args.mode == "validate":
        print(json.dumps(validate(), indent=2))
    elif args.run_dir is None:
        raise ValueError("Use the single Bash launcher")
    elif args.mode == "start":
        start(args.run_dir.resolve())
    else:
        worker(args.run_dir.resolve())


if __name__ == "__main__":
    main()
