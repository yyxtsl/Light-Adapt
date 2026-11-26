"""Approved CPU-only DWIE v1 package; stages share one immutable deadline.

No model/optimizer/GPU imports. Download and build stages are separately
claimed, never reset a budget, and preserve every old cache file.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
import hashlib
import json
from pathlib import Path
import socket
import threading
import time
import urllib.error
import urllib.request
import zipfile

import prepare_dwie_cpu_admission as original

ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT / "dataset/dwie-admission-20261023"
NEW = ROOT / "dataset/dwie-adaptation-20261023"
ARCHIVE_SHA = "3fe52d9c18c8cdb5e85486afa5d5dd2417b75a27bfb32fe01b83d2312ae681f8"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write(path, value):
    original.write_new(path, value)


def transient(error):
    if isinstance(error, urllib.error.HTTPError):
        return error.code in (429, 502, 503, 504)
    if isinstance(error, (TimeoutError, socket.timeout)):
        return True
    return isinstance(error, urllib.error.URLError) and isinstance(error.reason, (TimeoutError, socket.timeout))


class Budget:
    def __init__(self, root=NEW):
        self.root = root
        self.path = root / "manifests/budget.json"
        self.state = read(self.path)
        self.lock, self.stop = threading.RLock(), threading.Event()
        if self.state["hard_seconds"] != 1800 or self.state["gpu"] is not False:
            raise ValueError("Wrong approved CPU budget")
        if self.state.get("closed", False):
            raise RuntimeError("CPU package is closed; no automatic restart")
        self.check()

    def check(self, reserve=30):
        if self.stop.is_set() or time.monotonic() >= self.state["deadline_monotonic"] - reserve:
            raise TimeoutError("CPU package deadline/stop; no reset or extension")
        if self.state["network_bytes"] > self.state["network_cap_bytes"]:
            raise ValueError("Network cap exceeded")

    def written_bytes(self):
        return sum(p.stat().st_size for p in self.root.rglob("*") if p.is_file())

    def persist(self):
        self.path.write_text(json.dumps(self.state, indent=2) + "\n", encoding="utf-8")

    def check_writes(self):
        if self.written_bytes() > self.state["write_cap_bytes"]:
            raise ValueError("New dataset write cap exceeded")


def load_source():
    archive = next((OLD / "raw").glob("official-*.zip"))
    if digest(archive) != ARCHIVE_SHA:
        raise ValueError("Pinned official archive changed")
    partitions = read(OLD / "manifests/partitions.json")
    rows = [r for r in partitions["documents"] if "train" in r["tags"] and "test" not in r["tags"]]
    with zipfile.ZipFile(archive) as source:
        prefix = source.namelist()[0].split("/")[0] + "/"
        metadata = {r["id"]: r for r in json.loads(source.read(prefix + "data/article_id_to_url.json"))}
        annotations = {r["id"]: json.loads(source.read(r["archive_member"])) for r in rows}
    if len(annotations) != 702 or any(a["id"] != k or "train" not in a["tags"] or "test" in a["tags"]
                                      for k, a in annotations.items()):
        raise ValueError("Train partition changed")
    return metadata, annotations


def fetch_article(url, path, budget, opener=urllib.request.urlopen, pause=time.sleep):
    if not original.re.fullmatch(r"https://api\.dw\.com/api/detail/article/[0-9]+", url):
        raise ValueError("Nonofficial article URL")
    if path.exists():
        raise FileExistsError(str(path))
    attempts = []
    for attempt in (1, 2):
        budget.check(reserve=90)
        row = {"url": url, "attempt": attempt}
        started = time.monotonic()
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "TTMRE-DWIE-CPU-adaptation/1.0"})
            chunks, size = [], 0
            with opener(request, timeout=12) as response:
                row["http_status"] = response.status
                # read1 returns after one buffered/socket chunk; a slowly
                # trickling server cannot hide the absolute deadline inside
                # a request to fill an entire64KiB read buffer.
                reader = getattr(response, "read1", response.read)
                while True:
                    budget.check(reserve=60)
                    data = reader(65536)
                    if not data:
                        break
                    size += len(data)
                    with budget.lock:
                        budget.state["network_bytes"] += len(data)
                        budget.check(reserve=60)
                    if size > 2 * 1024**2:
                        raise ValueError("Article response too large")
                    chunks.append(data)
            raw = b"".join(chunks)
            with budget.lock:
                budget.check(reserve=60)
                with path.open("xb") as handle:
                    handle.write(raw)
                row.update(ok=True, bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
                budget.check_writes()
                budget.persist()
            attempts.append({**row, "seconds": time.monotonic() - started})
            return raw, attempts
        except Exception as error:
            attempts.append({**row, "ok": False, "error": str(error), "seconds": time.monotonic() - started})
            with budget.lock:
                budget.persist()
            if attempt == 2 or not transient(error):
                setattr(error, "dwie_attempts", attempts)
                raise
            budget.check(reserve=90)
            pause(1)
    raise AssertionError("Unreachable")


def verify_old_cache():
    pins = read(NEW / "manifests/old_cache_pins.json")
    current = {str(p.relative_to(OLD)).replace("\\", "/") for p in OLD.rglob("*") if p.is_file()}
    changes = [name for name, expected in pins.items() if not (OLD / name).is_file() or digest(OLD / name) != expected]
    return {"checked": len(pins), "changed": changes, "new_old_cache_files": sorted(current - set(pins)),
            "passed": not changes and current == set(pins)}


def download():
    budget = Budget()
    write(NEW / "manifests/download_once_claim.json", {"deadline_monotonic": budget.state["deadline_monotonic"]})
    metadata, annotations = load_source()
    previous = read(OLD / "manifests/recovery_preparation_result.json")
    reuse = {r["id"]: r for r in previous["records"]}
    if len(reuse) != 462:
        raise ValueError("Expected462 verified cache records")
    result = {"records": [], "failures": [], "passed": False, "reused": 462, "new_expected": 240,
              "test_gold_read": False, "gpu": False, "optimizer_updates": 0}

    def one(ident):
        budget.check(reserve=90)
        meta = metadata[ident]
        attempts = []
        if ident in reuse:
            path = OLD / "raw" / (ident + "-article-response.json")
            raw = path.read_bytes()
            if hashlib.sha256(raw).hexdigest() != reuse[ident]["response_sha256"]:
                raise ValueError("Original response digest changed")
            if digest(OLD / "converted" / (ident + ".json")) != reuse[ident]["converted_sha256"]:
                raise ValueError("Original converted digest changed")
        else:
            path = NEW / "raw" / (ident + "-article-response.json")
            url = meta["url"].replace("18525950", "19026607") if ident == "DW_18525950" else meta["url"]
            raw, attempts = fetch_article(url, path, budget)
        content = original.reconstruct(json.loads(raw), ident)
        converted, stats = original.validate_and_convert(annotations[ident], content, meta["hash"])
        return {"id": ident, "raw_path": str(path), "response_sha256": hashlib.sha256(raw).hexdigest(),
                "content_sha1": meta["hash"], "reused": ident in reuse, "attempts": attempts,
                "entities": len(converted["entities"]), "gold_relations": len(converted["relations"]), **stats}

    # Verify old caches immediately, then submit only the missing240 responses.
    for ident in sorted(reuse):
        result["records"].append(one(ident))
    pending_ids = iter(sorted(set(annotations) - set(reuse)))
    with ThreadPoolExecutor(max_workers=4) as pool:
        pending = {pool.submit(one, ident): ident for ident in [next(pending_ids) for _ in range(4)]}
        while pending:
            finished, _ = wait(pending, timeout=1, return_when=FIRST_COMPLETED)
            for future in finished:
                ident = pending.pop(future)
                try:
                    result["records"].append(future.result())
                except Exception as error:
                    budget.stop.set()
                    result["failures"].append({"id": ident, "error": str(error),
                                               "attempts": getattr(error, "dwie_attempts", [])})
                if not budget.stop.is_set():
                    ident = next(pending_ids, None)
                    if ident is not None:
                        pending[pool.submit(one, ident)] = ident
            if finished and not result["failures"] and len(result["records"]) % 20 == 2:
                print(json.dumps({"verified": len(result["records"]), "expected": 702}), flush=True)
    result["records"].sort(key=lambda r: r["id"])
    result.update(passed=len(result["records"]) == 702 and not result["failures"],
                  network_bytes=budget.state["network_bytes"], elapsed_seconds=time.monotonic()-budget.state["start_monotonic"])
    write(NEW / "manifests/download_result.json", result)
    print(json.dumps({k: v for k, v in result.items() if k != "records"}), flush=True)
    return 0 if result["passed"] else 2


def build(partial_diagnostic=False):
    # Lazy CPU-only import; no network is performed in the build stage.
    import ttmre_dwie_input as adapter
    from tokenizers import Tokenizer, AddedToken
    budget = Budget()
    stage = "diagnostic_build" if partial_diagnostic else "build"
    write(NEW / "manifests" / (stage+"_once_claim.json"), {"deadline_monotonic": budget.state["deadline_monotonic"]})
    result = read(NEW / "manifests/download_result.json")
    if not result["passed"] and not partial_diagnostic:
        raise ValueError("Corpus not fully recovered; cannot certify full adaptation")
    metadata, annotations = load_source()
    split = read(OLD / "manifests/target_split.json")
    split = {key: split[key] for key in ("train", "dev")}
    vocab = sorted({r["p"] for ident in split["train"] for r in annotations[ident]["relations"]})
    if len(vocab) != 62 or len(split["train"]) != 602 or len(split["dev"]) != 100:
        raise ValueError("Predeclared split/vocabulary changed")
    rel2id = {name: index+1 for index, name in enumerate(vocab)}
    tokenizer_path = OLD / "raw/roberta-large-pinned-tokenizer.json"
    if digest(tokenizer_path) != "847bbeab6174d66a88898f729d52fa8d355fafe1bea101cf960dd404581df70e":
        raise ValueError("Pinned tokenizer changed")
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    tokenizer.no_truncation(); tokenizer.no_padding()
    before = tokenizer.get_vocab_size()
    tokenizer.add_tokens([AddedToken("*", normalized=False)])
    if tokenizer.get_vocab_size() != before or tokenizer.token_to_id("*") != 3226:
        raise ValueError("Marker registration changed vocabulary")
    by_id = {row["id"]: row for row in result["records"]}
    rows, feature_files = [], {}
    for scope in ("train", "dev"):
        output = NEW / "features-v1" / (("diagnostic_" if partial_diagnostic else "") + scope + ".jsonl")
        with output.open("x", encoding="utf-8") as handle:
            for ident in split[scope]:
                if ident not in by_id:
                    continue
                budget.check()
                row = by_id[ident]
                path = Path(row["raw_path"])
                if digest(path) != row["response_sha256"]:
                    raise ValueError("Article response changed")
                content = original.reconstruct(read(path), ident)
                record, stats = original.validate_and_convert(annotations[ident], content, metadata[ident]["hash"])
                feature, audit = adapter.make_feature(record, tokenizer, rel2id)
                handle.write(json.dumps(feature, ensure_ascii=False, separators=(",", ":")) + "\n")
                rows.append({"id": ident, "split": scope, **stats, **audit})
        feature_files[scope] = {"path": str(output), "sha256": digest(output), "bytes": output.stat().st_size}
        budget.check_writes()
    prefix = "diagnostic_" if partial_diagnostic else ""
    write(NEW / "manifests" / (prefix+"target_split.json"), split)
    write(NEW / "manifests" / (prefix+"rel2id.json"), {"TH": 0, **rel2id})
    old_check = verify_old_cache()
    max_tokens = max(r["content_tokens"] + 2 for r in rows)
    summary = {"passed": old_check["passed"] and result["passed"] and not partial_diagnostic,
               "diagnostic_only": partial_diagnostic, "full_corpus_prepared": result["passed"],
               "documents": len(rows), "train": sum(r["split"] == "train" for r in rows),
               "dev": sum(r["split"] == "dev" for r in rows), "fixed_full_target_split": {"train":602,"dev":100},
               "positive_classes": len(rel2id), "output_classes": len(rel2id)+1,
               "max_tokens_with_special": max_tokens, "max_windows": max(r["windows"] for r in rows),
               "documents_over1024": sum(r["content_tokens"]+2 > 1024 for r in rows),
               "total_gold": sum(r["gold_relations"] for r in rows),
               "unknown_gold": sum(r["unknown_gold"] for r in rows),
               "self_gold": sum(r["self_gold"] for r in rows),
               "unanchored_gold": sum(r["unanchored_gold"] for r in rows),
               "all_token_positions_covered": all(r["min_coverage"] >= 1 for r in rows),
               "all_marker_spans_mapped": True, "features": feature_files,
               "fp32_dense_attention_bytes_max_single_doc_16heads": 16*max_tokens*max_tokens*4,
               "attention_estimate_excludes_activations_gradients_batch_and_parameters": True,
               "old_cache": old_check, "rows": rows, "test_gold_read": False,
               "gpu": False, "real_weights_loaded": False, "optimizer_updates": 0,
               "network_bytes": budget.state["network_bytes"],
               "elapsed_seconds": time.monotonic()-budget.state["start_monotonic"]}
    if not summary["all_token_positions_covered"]:
        raise ValueError("Token coverage incomplete")
    budget.check_writes()
    write(NEW / "manifests" / (stage+"_result.json"), summary)
    print(json.dumps({k:v for k,v in summary.items() if k not in ("rows", "old_cache")}), flush=True)
    return 0 if summary["passed"] or partial_diagnostic else 2


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("status", "download", "build", "diagnose"), default="status")
    args = parser.parse_args()
    if args.phase == "status":
        state = read(NEW / "manifests/budget.json")
        print(json.dumps({"remaining_seconds": state["deadline_monotonic"]-time.monotonic(), "network_bytes": state["network_bytes"]}))
    else:
        try:
            raise SystemExit(download() if args.phase == "download" else build(args.phase == "diagnose"))
        except Exception as error:
            failure = NEW / "manifests" / (args.phase+"_failure.json")
            if not failure.exists():
                write(failure, {"error": str(error), "type": type(error).__name__, "automatic_restart": False})
            raise
