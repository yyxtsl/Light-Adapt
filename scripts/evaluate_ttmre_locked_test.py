"""One authorized locked eight-system Test evaluation, zero training, cap30min.

Default validate is read-only. Predictions are sealed in a separate process
before any Test scoring. No method selection, retuning or automatic follow-up.
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
import sys
import time
from types import SimpleNamespace
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import train_ttmre_spu_soup_control as control
import ttmre_human_distant_data as features_api

soup, base, core = control.soup, control.base, control.core
read, write_json, digest, emit = control.read, control.write_json, control.digest, control.emit
supervise = control.diagnostic.supervise
RUN_ROOT = ROOT / "experiments/2026-10-21-locked-test"
CLAIM = "locked-test-once-claim.json"
FILES = tuple("scripts/" + n for n in ("evaluate_ttmre_locked_test.py", "check_ttmre_locked_test.py", "run_ttmre_locked_test.sh"))
HARD_SECONDS, SOFT_SECONDS, TERM_SECONDS = 1800, 120, 30
TEST_MEMBER = "docred/test_revised.json"
TEST_SHA = "ea673b7ef91e16510d7eb88863d3903e47318dc5653ef361266f27ae3b84723c"
TEST_BYTES, TEST_DOCS, TEST_PAIRS = 3207228, 500, 198670
MEMORY = soup.MEMORY
SYSTEMS = ("spu75", "spu76", "spu77", "atgl75", "atgl76", "atgl77", "spu-soup", "atgl-soup")
WEIGHTS = {"spu75": base.frozen.CHECKPOINT_SHA,
           "spu76": "a521dfa650844e11476ac43edb352e920566e24ad2bef2abe1f742b4c2e0619f",
           "spu77": "03ee2537bbd7c53bc0983caf1e92708fcaf46cac7a4faddacb7da56f38c41c93",
           **{"atgl"+str(s): h for s,h in soup.CHECKPOINTS.items()}}
CONTROL_PINS = {
    "identity.json": "a45042c20c95e91cadd44c6980c676f1fd7e2937c2020487d5a9c32315eb9c35",
    "comparison/completion.json": "f83ca122a8103464d433389a343e0e4e8bf49d5da8df412c1f9e172970e014ab",
    "spu-soup/recipe.json": "a03c123fb2ef3fbab1bfdc2e13f262a2e7ebdd211cb795095e6cfd158546a760",
    "seed76/completion.json": "83dc26c43955333f06a6f06fc0591f88d3801a8ad2cb23265c9790d66264b790",
    "seed77/completion.json": "b79e13213e9653036356cfc7424d8255d3ecca95bcaea368f7452bb2fe2101d7"}
METRICS = ("test_F1", "test_precision", "test_recall", "test_IgnF1_annotated")


def protocol():
    return {"version": "locked-test-v1", "systems": list(SYSTEMS), "checkpoint_sha256": WEIGHTS,
            "soup_members": {loss+"-soup": [loss+str(s) for s in (75,76,77)] for loss in ("spu","atgl")},
            "soup_weights": [1/3]*3, "soup_accumulation": "CPU float64 then float32",
            "decoder": soup.protocol()["decoder"], "precision": "float32", "batch_size": 4,
            "maximum_sequence_length": 1024, "pair_order": "all directed non-self pairs in h,t order; label independent",
            "test_member": TEST_MEMBER, "test_sha256": TEST_SHA, "test_documents": TEST_DOCS, "test_pairs": TEST_PAIRS,
            "primary_comparison": "ATGLsoup minus SPUsoup Test F1 and Human-train Ign-A",
            "fixed_Dev_best_single_references": {"spu": "spu75", "atgl": "atgl77"},
            "freeze": "all parameters", "optimizer_updates": 0, "backward_calls": 0,
            "prediction_seal_before_any_scoring": True, "IgnD": "null; no Distant rescan or Dev-match cache substitution",
            "investment_min_soup_delta_F1_pp": .30, "require_IgnA_not_lower": True,
            "require_ATGLsoup_F1_above_fixed_ATGL77": True, "Test_driven_retuning": False,
            "hard_seconds": HARD_SECONDS, "soft_reserve_seconds": SOFT_SECONDS, "term_reserve_seconds": TERM_SECONDS,
            "automatic_retry": False, "automatic_resume": False, "extension": False, "automatic_followup": False}


def check_budget(budget, reserve=SOFT_SECONDS):
    if budget is None:
        return
    if time.monotonic() >= budget["deadline_monotonic"] - reserve:
        raise TimeoutError("30-minute locked Test reserve reached; no extension")
    if os.name == "posix" and Path("/proc/sys/kernel/random/boot_id").read_text().strip() != budget["boot_id"]:
        raise TimeoutError("Boot identity changed; do not reset budget")


def verify_files(files, budget):
    for name, expected in files.items():
        check_budget(budget)
        if digest(Path(name), budget) != expected:
            raise ValueError("Pinned file missing/changed: " + name)


def test_bytes(archive):
    """Read this exact pinned ZIP member; never extract arbitrary archive paths."""
    with zipfile.ZipFile(archive) as source:
        members = [m for m in source.infolist() if m.filename == TEST_MEMBER]
        if len(members) != 1 or members[0].file_size != TEST_BYTES or members[0].is_dir():
            raise ValueError("Exact Test member missing/duplicated or size changed")
        raw = source.read(TEST_MEMBER)
    if len(raw) != TEST_BYTES or hashlib.sha256(raw).hexdigest() != TEST_SHA:
        raise ValueError("Pinned Test member bytes changed")
    return raw


def validate(budget=None):
    old = Path(read(control.RUN_ROOT / control.CLAIM)["workspace"])
    if old != old.resolve() or old.name != "workspace" or old.parent.parent != control.RUN_ROOT.resolve():
        raise ValueError("Unexpected completed S-PU control workspace")
    pinned = {str(old / name): h for name,h in CONTROL_PINS.items()}
    verify_files(pinned, budget)
    prior, done = read(old / "identity.json"), read(old / "comparison/completion.json")
    if (not done["passed"] or done["marker"] != "SPU_SOUP_CONTROL_COMPLETE"
            or done["formal_optimizer_updates_new"] != 45780 or done["test_data_read"] is not False):
        raise ValueError("Expected completed original S-PU controls, without Test")
    parent = prior["parent"]
    atgl_root = Path(prior["ATGL_soup"]["predictions"]).parent
    for name in ("identity.json", "comparison/completion.json"):
        pinned[str(atgl_root / name)] = control.diagnostic.OLD_PINS[name]
    verify_files(pinned, budget)
    atgl_origin = read(atgl_root / "identity.json")
    if atgl_origin["parent"] != parent:
        raise ValueError("Human parent provenance differs")
    sources = {"spu75": prior["control75_checkpoint"], **{
        "spu"+str(seed): str(old / ("seed"+str(seed)) / "best_model.pth") for seed in (76,77)}}
    sources.update({"atgl"+str(row["seed"]): row["checkpoint"] for row in atgl_origin["sources"]})
    if set(sources) != set(SYSTEMS[:6]):
        raise ValueError("Expected exactly six locked source checkpoints")
    for name, path in sources.items():
        if not Path(path).resolve().is_relative_to(ROOT.resolve()):
            raise ValueError("Source checkpoint outside project")
        pinned[path] = WEIGHTS[name]
    for seed in (76,77):
        stage = read(old / ("seed"+str(seed)) / "completion.json")
        if stage["checkpoint_sha256"] != WEIGHTS["spu"+str(seed)] or not stage["passed"]:
            raise ValueError("Locked S-PU member changed")
    cache = Path(parent["baseline_workspace"]) / "cache"
    # Only the Human index/priors and generic constructor assets are needed.
    wanted = [cache / "human/index.json", Path(parent["data_dir"]) / "train_revised.json"]
    model_dir = Path(parent["model_dir"])
    wanted += [Path(n) for n in parent["input_files"] if Path(n).is_relative_to(model_dir)]
    for path in wanted:
        if not path.resolve().is_relative_to(ROOT.resolve()):
            raise ValueError("Asset outside project")
        pinned[str(path)] = prior["pinned_files"][str(path)]
    verify_files(pinned, budget)
    source_files = dict(prior["source_files"])
    verify_files({str(ROOT / n): h for n,h in source_files.items()}, budget)
    source_files.update({n: digest(ROOT / n, budget) for n in FILES})
    archive = ROOT / "dataset/docred.zip"
    raw = test_bytes(archive)
    # Schema validation only. No Test metric or label-frequency construction.
    documents = json.loads(raw)
    validate_documents(documents)
    return {"protocol": protocol(), "parent": parent, "source_files": source_files,
            "pinned_files": pinned, "sources": sources, "test_archive": str(archive),
            "test_member_sha256": TEST_SHA, "old_Test_data_read": False,
            "test_scope_authorized": True, "optimizer_updates": 0}


def validate_documents(documents):
    rel2id = read(ROOT / "meta/rel2id.json")
    if (not isinstance(documents, list) or len(documents) != TEST_DOCS
            or len({d["title"] for d in documents}) != TEST_DOCS
            or sum(len(d["vertexSet"])*(len(d["vertexSet"])-1) for d in documents) != TEST_PAIRS):
        raise ValueError("Declared complete Test population changed")
    for doc in documents:
        features_api.check_document(doc, rel2id)


def model_document(doc):
    """Copy only model inputs; even malformed gold is not consulted here."""
    return {"title": doc["title"], "sents": doc["sents"], "vertexSet": doc["vertexSet"], "labels": []}


def build_features(documents, tokenizer, rel2id, budget):
    features = []
    for doc in documents:
        check_budget(budget)
        sparse = features_api.sparse_feature(model_document(doc), tokenizer, rel2id, with_labels=False)
        feature = features_api.expand_feature(sparse, with_labels=False)
        if feature["labels"] or sparse["positive"]:
            raise ValueError("Test gold leaked into model features")
        features.append(feature)
    return features


def resources(path, identity):
    free = shutil.disk_usage(path).free
    if free < 1024**3:
        raise RuntimeError("Require 1GiB actual project disk for locked Test reports/features; free=%.3fGiB" % (free/1024**3))
    return {"free_bytes": free, "required_bytes": 1024**3, "recommended_available_RAM_GiB": 32,
            "new_model_checkpoints": 0}


def tokenizer_for(identity):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(identity["parent"]["model_dir"], local_files_only=True)


def order_from(features):
    return [{"title": f["title"], "hts": f["hts"]} for f in features]


def predict_grouping(model, dataset, args, budget):
    """Keep native batch4 predictions and record complete single-document replay."""
    import numpy as np
    import torch
    from evaluation import to_official
    native, single = [], []
    changed = numeric = pair_count = 0
    maximum = 0.
    model.eval()
    with torch.no_grad():
        for offset in range(0, len(dataset), 4):
            check_budget(budget)
            batch_features = dataset[offset:offset+4]
            a = model(**core.shared.inputs_from_batch(core.shared.collate_fn(batch_features), args.device, labels=False))
            b = torch.cat([model(**core.shared.inputs_from_batch(core.shared.collate_fn([f]), args.device, labels=False)) for f in batch_features])
            if a.shape != b.shape or a.dtype != torch.float32 or not torch.isfinite(a).all() or not torch.isfinite(b).all():
                raise ValueError("Invalid Test inference logits")
            delta = (a-b).abs()
            numeric += int((delta > (1e-4 + 1e-4*b.abs())).sum())
            maximum = max(maximum, float(delta.max()))
            masks = []
            for logits in (a,b):
                values = logits.cpu().numpy()
                flags = np.zeros_like(values)
                flags[:,1:] = values[:,1:] > values[:,:1]
                flags[:,0] = flags[:,1:].sum(1) == 0
                masks.append(flags)
            changed += int(np.count_nonzero(masks[0][:,1:] != masks[1][:,1:]))
            pair_count += len(a)
            native.extend(to_official(masks[0], batch_features))
            single.extend(to_official(masks[1], batch_features))
    same = soup.facts(native) == soup.facts(single)
    return native, {"passed": changed == 0 and same, "documents": len(dataset), "pairs": pair_count,
                    "changed_relation_decisions": changed, "prediction_sets_equal": same,
                    "numeric_violations_original_tolerance": numeric, "max_abs_delta": maximum,
                    "original_atol": 1e-4, "original_rtol": 1e-4,
                    "policy": "fixed native batch4; exact relation decisions; retain original numeric outcomes"}


def predict(workspace, identity, budget):
    import torch
    parent = identity["parent"]
    args = SimpleNamespace(run_dir=workspace, baseline_workspace=Path(parent["baseline_workspace"]),
                           model_dir=Path(parent["model_dir"]), device=core.baseline.control.runtime(parent["environment"]), budget=budget)
    write_json(workspace / "resource_check.json", resources(workspace, identity))
    raw = test_bytes(Path(identity["test_archive"]))
    documents = json.loads(raw)
    validate_documents(documents)
    with (workspace / "test_source.json").open("xb") as handle:
        handle.write(raw); handle.flush(); os.fsync(handle.fileno())
    features = build_features(documents, tokenizer_for(identity), read(ROOT / "meta/rel2id.json"), budget)
    del documents, raw  # Scoring process alone opens gold after predictions are sealed.
    order = order_from(features)
    if len(features) != TEST_DOCS or sum(len(f["hts"]) for f in features) != TEST_PAIRS:
        raise ValueError("Complete Test feature count mismatch")
    write_json(workspace / "test_pair_order.json", order)
    encoded = json.dumps(features, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    write_json(workspace / "feature_provenance.json", {"features_sha256": hashlib.sha256(encoded).hexdigest(),
               "label_rows": 0, "documents": len(features), "pairs": TEST_PAIRS,
               "tokenizer_source": parent["model_dir"], "gold_stripped_before_tokenization": True,
               "maximum_input_ids": max(len(f["input_ids"]) for f in features)})
    del encoded
    model = core.load_generic_model_seed75(args, parent["baseline_identity"])
    model.requires_grad_(False); model.eval()
    memory, sealed, accepted = None, {}, True
    hook = model.register_forward_pre_hook(lambda *_: check_budget(budget))
    try:
        for system in SYSTEMS:
            check_budget(budget)
            directory = workspace / system
            directory.mkdir()
            if system in identity["sources"]:
                state = torch.load(identity["sources"][system], map_location="cpu", weights_only=True)
                recipe = {"checkpoint_sha256": WEIGHTS[system], "checkpoint": identity["sources"][system]}
            else:
                loss = system.split("-",1)[0]
                members = [loss+str(s) for s in (75,76,77)]
                old_emit = soup.emit
                try:
                    soup.emit = lambda _event, **fields: emit("LOCKED_TEST_SOUP_SOURCE_LOADED", system=system, **fields)
                    state, average = soup.uniform_state([identity["sources"][m] for m in members], budget)
                finally:
                    soup.emit = old_emit
                recipe = {"members": members, "checkpoint_sha256": {m: WEIGHTS[m] for m in members},
                          "averaging": average, "new_checkpoint_saved": False}
            if memory is None:
                memory = state[MEMORY].clone()
            if not control.diagnostic.tensor_bytes_equal(memory, state[MEMORY]):
                raise ValueError("Frozen memory differs across locked models")
            model.load_state_dict(state, strict=True)
            del state
            before = core.preflight.state_digest(model)
            predictions, grouping = predict_grouping(model, features, args, budget)
            strict = core.aligned.grouping_invariance(model, features, args.device)
            ttm_ok = all(strict["checks"][n]["close"] for n in ("ttm_grouped_vs_single", "ttm_permutation"))
            passed = (grouping["passed"] and ttm_ok and strict["changed_relation_decisions"] == 0)
            state_ok = (before == core.preflight.state_digest(model)
                        and control.diagnostic.tensor_bytes_equal(memory, model.mu_encoder.memory_tokens)
                        and all(not p.requires_grad and p.grad is None for p in model.parameters()))
            if not state_ok:
                raise ValueError("Frozen Test model state changed")
            # Decision disagreement is disclosed without selecting a different decoder/group.
            write_json(directory / "predictions.json", predictions)
            write_json(directory / "replay.json", {"passed": bool(passed), "full_test_grouping": grouping,
                       "original_strict_grouping": strict, "parameters_unchanged": True, "memory_unchanged": True,
                       "optimizer_updates": 0, "no_result_based_group_choice": True})
            write_json(directory / "recipe.json", {"system": system, "state_sha256": before, **recipe})
            sealed[system] = {"predictions_sha256": digest(directory / "predictions.json", budget),
                              "replay_sha256": digest(directory / "replay.json", budget),
                              "recipe_sha256": digest(directory / "recipe.json", budget),
                              "acceptance_passed": bool(passed)}
            accepted = accepted and passed
            emit("LOCKED_TEST_SYSTEM_PREDICTED", system=system, predictions=len(predictions),
                 decision_acceptance_passed=bool(passed), optimizer_updates=0, metrics_computed=False)
    finally:
        hook.remove()
    verify_files(identity["pinned_files"], budget)
    verify_files({str(ROOT / n): h for n,h in identity["source_files"].items()}, budget)
    write_json(workspace / "predictions_sealed.json", {"protocol": protocol(), "systems": sealed,
               "identity_sha256": digest(workspace / "identity.json"), "test_source_sha256": digest(workspace / "test_source.json"),
               "pair_order_sha256": digest(workspace / "test_pair_order.json"), "all_predictions_saved": True,
               "all_acceptance_passed": bool(accepted), "scoring_started": False, "optimizer_updates": 0})
    emit("LOCKED_TEST_ALL_PREDICTIONS_SEALED", systems=len(sealed), metrics_computed=False)


def score_predictions(predictions, gold, annotated):
    native = core.shared.score_predictions(predictions, gold, annotated, None)
    result = {name.replace("dev_", "test_", 1): value for name,value in native.items()}
    truth = {(d["title"], x["h"], x["t"], x["r"]) for d in gold for x in d["labels"]}
    facts = soup.facts(predictions)
    result.update({"TP": len(facts & truth), "FP": len(facts-truth), "FN": len(truth-facts),
                   "predicted": len(facts), "gold": len(truth)})
    return result


def comparisons(metrics):
    result = {"paired_ATGL_minus_SPU_pp": {str(s): {k: metrics["atgl"+str(s)][k]-metrics["spu"+str(s)][k] for k in METRICS} for s in (75,76,77)},
              "ATGL_minus_SPU_soup_pp": {k: metrics["atgl-soup"][k]-metrics["spu-soup"][k] for k in METRICS}}
    result["soup_minus_single_mean_pp"] = {loss: {k: metrics[loss+"-soup"][k]-sum(metrics[loss+str(s)][k] for s in (75,76,77))/3
                                                  for k in METRICS} for loss in ("spu","atgl")}
    result["soup_minus_fixed_Dev_best_single_pp"] = {loss: {k: metrics[loss+"-soup"][k]-metrics[ref][k] for k in METRICS}
                                                       for loss,ref in (("spu","spu75"),("atgl","atgl77"))}
    delta = result["ATGL_minus_SPU_soup_pp"]
    result["investment_gate_passed"] = bool(delta["test_F1"] >= .30 and delta["test_IgnF1_annotated"] >= 0
        and result["soup_minus_fixed_Dev_best_single_pp"]["atgl"]["test_F1"] > 0)
    return result


def score(workspace, identity, budget):
    """Only called in a fresh CPU process after all eight predictions exist."""
    seal = read(workspace / "predictions_sealed.json")
    if (seal["protocol"] != protocol() or set(seal["systems"]) != set(SYSTEMS)
            or seal["identity_sha256"] != digest(workspace / "identity.json")
            or not seal["all_predictions_saved"] or seal["scoring_started"] is not False
            or seal["optimizer_updates"] != 0
            or seal["test_source_sha256"] != TEST_SHA or digest(workspace / "test_source.json", budget) != TEST_SHA
            or seal["pair_order_sha256"] != digest(workspace / "test_pair_order.json", budget)):
        raise ValueError("All-system prediction seal missing/changed")
    all_predictions = {}
    for system in SYSTEMS:
        row = seal["systems"][system]
        for name in ("predictions", "replay", "recipe"):
            if digest(workspace / system / (name+".json"), budget) != row[name+"_sha256"]:
                raise ValueError("Sealed Test output changed: " + system)
        all_predictions[system] = read(workspace / system / "predictions.json")
        replay = read(workspace / system / "replay.json")
        if row["acceptance_passed"] != replay["passed"] or not replay["parameters_unchanged"] or not replay["memory_unchanged"]:
            raise ValueError("Sealed acceptance witness changed: " + system)
    with (workspace / "scoring_started.json").open("x", encoding="utf-8") as handle:
        json.dump({"prediction_seal_sha256": digest(workspace / "predictions_sealed.json"),
                   "systems": list(SYSTEMS), "all_predictions_verified_before_scoring": True,
                   "start_utc": datetime.now(timezone.utc).isoformat()}, handle, indent=2)
        handle.flush(); os.fsync(handle.fileno())
    # No gold-dependent score is computed above this barrier.
    gold = read(workspace / "test_source.json")
    validate_documents(gold)
    train = read(Path(identity["parent"]["data_dir"]) / "train_revised.json")
    annotated = core.shared.train_facts(train)
    counts = {r: 0 for r in range(1,97)}
    rel2id = read(ROOT / "meta/rel2id.json")
    for doc in train:
        for label in doc["labels"]:
            counts[rel2id[label["r"]]] += 1
    ranked = sorted(counts, key=lambda r: (counts[r],r))
    buckets = {r: ("rare","middle","frequent")[i//32] for i,r in enumerate(ranked)}
    metric_rows = {}
    titles = {d["title"]: len(d["vertexSet"]) for d in gold}
    for system, predictions in all_predictions.items():
        check_budget(budget)
        if (len(predictions) != len(soup.facts(predictions))
                or any(p["title"] not in titles or not 0 <= p["h_idx"] < titles[p["title"]]
                       or not 0 <= p["t_idx"] < titles[p["title"]] or p["h_idx"] == p["t_idx"]
                       or rel2id.get(p["r"],0) <= 0 for p in predictions)):
            raise ValueError("Invalid sealed prediction coordinates")
        metric_rows[system] = score_predictions(predictions, gold, annotated)
        write_json(workspace / system / "metrics.json", metric_rows[system])
        write_json(workspace / system / "frequency_diagnostics.json", control.diagnostic.frequency_report(predictions,gold,buckets,rel2id))
    accepted = all(seal["systems"][s]["acceptance_passed"] for s in SYSTEMS)
    summary = comparisons(metric_rows)
    comparison = workspace / "comparison"
    comparison.mkdir()
    write_json(comparison / "metrics.json", metric_rows)
    write_json(comparison / "summary.json", summary)
    verify_files(identity["pinned_files"], budget)
    verify_files({str(ROOT / n): h for n,h in identity["source_files"].items()}, budget)
    completion = {"passed": bool(accepted), "marker": "LOCKED_TEST_COMPLETE" if accepted else "LOCKED_TEST_EVALUATED_ACCEPTANCE_FAILED",
                  "protocol": protocol(), "metrics": metric_rows, **summary, "all_predictions_sealed_before_scoring": True,
                  "prediction_seal_sha256": digest(workspace / "predictions_sealed.json"),
                  "test_source_sha256": TEST_SHA, "optimizer_updates": 0, "backward_calls": 0,
                  "test_data_read": True, "test_evaluated": True, "Test_driven_retuning": False,
                  "IgnD_available": False, "IgnD_reason": "only Dev-matching Distant facts cached; no Distant rescan authorized",
                  "elapsed_seconds": time.monotonic()-budget["start_monotonic"], "automatic_followup": False,
                  "novelty_established": False, "independent_training_groups": 1,
                  "interpretation": "frozen models on held-out Test; no claim of new mechanism or independent Distant reproducibility"}
    write_json(comparison / "completion.json", completion)
    soup.archive_reports(workspace, budget)
    emit(completion["marker"], ATGL_minus_SPU_soup_F1_pp=summary["ATGL_minus_SPU_soup_pp"]["test_F1"],
         investment_gate_passed=summary["investment_gate_passed"], reports=str(comparison / "reports.tar.gz"), automatic_followup=False)


def require_lock():
    if os.name != "posix" or os.environ.get("TTMRE_LOCKED_TEST_LOCK_FD") != "9":
        raise RuntimeError("Use the one-use locked Test Bash launcher")
    import fcntl
    a,b = os.fstat(9), (ROOT / "experiments/2026-10-08-baseline-alignment/.training.lock").stat()
    if (a.st_dev,a.st_ino) != (b.st_dev,b.st_ino):
        raise RuntimeError("Wrong shared training lock")
    fcntl.flock(9,fcntl.LOCK_EX|fcntl.LOCK_NB)
    return 9


def verify_context(workspace,budget):
    claim = read(RUN_ROOT / CLAIM)
    if (workspace != workspace.resolve() or workspace.name != "workspace" or workspace.parent.parent != RUN_ROOT.resolve()
            or budget["hard_seconds"] != HARD_SECONDS or budget["deadline_monotonic"]-budget["start_monotonic"] != HARD_SECONDS
            or any(budget[k] is not False for k in ("automatic_retry","automatic_resume","extension"))
            or claim["workspace"] != str(workspace) or claim["protocol"] != protocol()
            or claim["budget_sha256"] != digest(workspace / "budget.json") or claim["identity_sha256"] != digest(workspace / "identity.json")):
        raise ValueError("Invalid locked Test immutable claim/budget")
    check_budget(budget)


def worker(workspace,phase):
    require_lock()
    budget = read(workspace / "budget.json")
    verify_context(workspace,budget)
    identity = read(workspace / "identity.json")
    def stop(*_):
        raise TimeoutError("Locked Test interrupted; preserve partials")
    signal.signal(signal.SIGTERM,stop)
    try:
        verify_files(identity["pinned_files"],budget)
        verify_files({str(ROOT/n):h for n,h in identity["source_files"].items()},budget)
        if identity["protocol"] != protocol():
            raise ValueError("Frozen Test protocol changed")
        (predict if phase == "predict" else score)(workspace,identity,budget)
    except BaseException as error:
        write_json(workspace / ("failure-"+phase+".json"),{"error":str(error),"type":type(error).__name__,
                   "complete":False,"optimizer_updates":0,"automatic_retry":False})
        try:
            if not (workspace / "comparison/reports.tar.gz").exists():
                soup.archive_reports(workspace,budget)
        except BaseException:
            pass
        raise


def start(workspace):
    lock = require_lock()
    if workspace.name != "workspace" or workspace.parent.parent != RUN_ROOT.resolve() or workspace.exists():
        raise ValueError("Use a new locked Test workspace")
    claim_path = RUN_ROOT / CLAIM
    if claim_path.exists():
        raise FileExistsError("Test already claimed; no retry")
    now=time.monotonic()
    budget={"start_monotonic":now,"deadline_monotonic":now+HARD_SECONDS,"hard_seconds":HARD_SECONDS,
            "soft_reserve_seconds":SOFT_SECONDS,"term_reserve_seconds":TERM_SECONDS,
            "start_utc":datetime.now(timezone.utc).isoformat(),"boot_id":Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
            "automatic_retry":False,"automatic_resume":False,"extension":False}
    def stop(*_):
        raise TimeoutError("Locked Test supervisor interrupted")
    signal.signal(signal.SIGTERM,stop)
    try:
        emit("LOCKED_TEST_BUDGET_STARTED",**budget)
        identity=validate(budget)
        emit("LOCKED_TEST_SOURCES_VERIFIED",resources=resources(workspace.parent,identity))
        # Reuse the verified 30-minute process-group supervisor and deadline.
        supervise([sys.executable,"-B",str(ROOT/FILES[1])],budget,lock)
        check_budget(budget)
        workspace.mkdir()
        write_json(workspace/"budget.json",budget)
        write_json(workspace/"identity.json",identity)
        with claim_path.open("x",encoding="utf-8") as handle:
            json.dump({"workspace":str(workspace),"protocol":protocol(),"budget_sha256":digest(workspace/"budget.json"),
                       "identity_sha256":digest(workspace/"identity.json"),
                       "authorization":"user confirmed locked8system Test once, zero training, cap30min; no retuning"},handle,indent=2)
            handle.flush();os.fsync(handle.fileno())
        command=[sys.executable,"-u","-B",str(Path(__file__).resolve()),"--mode","worker","--run-dir",str(workspace)]
        for phase in ("predict","score"):
            supervise(command+["--phase",phase],budget,lock)
    except BaseException as error:
        write_json(workspace.parent/"failure-supervisor.json",{"error":str(error),"type":type(error).__name__,
                   "complete":False,"elapsed_seconds":time.monotonic()-now,"automatic_retry":False})
        raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode",choices=("validate","start","worker"),default="validate")
    parser.add_argument("--run-dir",type=Path)
    parser.add_argument("--phase",choices=("predict","score"))
    args=parser.parse_args()
    if args.mode == "validate":
        print(json.dumps(validate(),indent=2))
    elif args.run_dir is None or (args.mode == "worker" and args.phase is None):
        raise ValueError("Use the single Bash launcher")
    elif args.mode == "start":
        start(args.run_dir.resolve())
    else:
        worker(args.run_dir.resolve(),args.phase)


if __name__ == "__main__":main()
