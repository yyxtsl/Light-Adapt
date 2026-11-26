"""Bounded DWIE train/dev preparation; no Test gold parsing or model imports.

Uses the already-created authorization/deadline and the pinned official ZIP.
Never creates or resets a budget, installs dependencies, or runs training.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
import hashlib
import json
from pathlib import Path
import re
import threading
import time
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "dataset/dwie-admission-20261023"


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def split_header(prefix):
    """Read only the official leading id/tags; never decode following gold.

    Accept an incomplete JSON prefix deliberately: Test annotation bodies may
    not even be supplied. Fail closed if the official field order changes.
    """
    text = prefix.decode("utf-8") if isinstance(prefix, bytes) else prefix
    decoder, pos, result = json.JSONDecoder(), 1, {}
    if not text.startswith("{"):
        raise ValueError("Annotation must start with an object")
    for expected in ("id", "tags"):
        while pos < len(text) and (text[pos].isspace() or text[pos] == ","):
            pos += 1
        key, pos = decoder.raw_decode(text, pos)
        if key != expected:
            raise ValueError("Refuse annotation body: expected leading " + expected)
        while pos < len(text) and text[pos].isspace():
            pos += 1
        if text[pos] != ":":
            raise ValueError("Missing header separator")
        pos += 1
        while pos < len(text) and text[pos].isspace():
            pos += 1
        result[key], pos = decoder.raw_decode(text, pos)
    if not isinstance(result["id"], str) or not isinstance(result["tags"], list):
        raise ValueError("Invalid partition header")
    return result


def reconstruct(article, document_id):
    content = "{}\n{}\n{}".format(article.get("name", ""),
                                    article.get("teaser", ""), article.get("text", ""))
    # Exact historical corrections published in the pinned official downloader.
    if document_id == "DW_40663341":
        for old, new in [("starting with Sunday's", "starting Sunday's"),
                         ("$1 million (€840,000)", "one million dollars\u00a0(840,000 euros)"),
                         ("who kneel in protest during", "to kneel in protest during")]:
            content = content.replace(old, new)
    return content


def validate_and_convert(annotation, content, expected_sha1):
    if "train" not in annotation["tags"] or "test" in annotation["tags"]:
        raise ValueError("Train-only conversion refuses Test/ambiguous tags")
    if hashlib.sha1(content.encode("utf-8")).hexdigest() != expected_sha1:
        raise ValueError("Official article SHA1 mismatch: " + annotation["id"])
    concepts = {c["concept"]: c for c in annotation["concepts"]}
    if len(concepts) != len(annotation["concepts"]):
        raise ValueError("Duplicate concept IDs")
    if not all(isinstance(k, int) and not isinstance(k, bool) for k in concepts):
        raise ValueError("Unexpected official concept ID type")
    mentions = {k: [] for k in concepts}
    normalized = 0
    for mention in annotation["mentions"]:
        key, begin, end = mention["concept"], mention["begin"], mention["end"]
        if key not in concepts or not 0 <= begin < end <= len(content):
            raise ValueError("Invalid mention concept/span")
        text = mention["text"].replace("\u00a0", " ").replace("\u200b", "")
        if content[begin:end] != text:
            raise ValueError("Official mention span mismatch: " + annotation["id"])
        normalized += text != mention["text"]
        mentions[key].append({"begin": begin, "end": end, "text": content[begin:end]})
    entities = [{"id": k, "mentions": sorted(mentions[k], key=lambda m: (m["begin"], m["end"]))}
                for k in sorted(concepts)]
    triples = set()
    self_relations = 0
    for relation in annotation["relations"]:
        head, tail, label = relation["s"], relation["o"], relation["p"]
        if head not in concepts or tail not in concepts or not isinstance(label, str):
            raise ValueError("Invalid relation endpoints/label")
        self_relations += head == tail
        triples.add((head, label, tail))
    result = {"id": annotation["id"], "content": content, "entities": entities,
              "relations": [{"h": h, "r": label, "t": t} for h, label, t in sorted(triples)],
              "hts": [[h, t] for h in sorted(concepts) for t in sorted(concepts) if h != t]}
    stats = {"normalized_mention_texts": normalized,
             "unanchored_entities": sum(not x["mentions"] for x in entities),
             "self_relations": self_relations,
             "duplicate_relation_annotations": len(annotation["relations"]) - len(triples),
             "unanchored_gold_relations": sum(not mentions[h] or not mentions[t] for h, _, t in triples)}
    # Keep these cases visible, including all gold, for later admission judgment.
    return result, stats


def fixed_split(ids):
    if len(set(ids)) != len(ids) or len(ids) < 200:
        raise ValueError("Unexpected official train IDs")
    ordered = sorted(ids, key=lambda k: hashlib.sha256(("ttmre-dwie-dev-v1:" + k).encode()).hexdigest())
    return sorted(ordered[100:]), sorted(ordered[:100])


def score_triples(predictions, gold):
    pred, true = set(predictions), set(gold)
    tp = len(pred & true)
    precision, recall = tp / len(pred) if pred else 0., tp / len(true) if true else 0.
    return {"tp": tp, "fp": len(pred - true), "fn": len(true - pred),
            "precision": precision, "recall": recall,
            "f1": 2 * tp / (len(pred) + len(true)) if pred or true else 0.}


class Budget:
    def __init__(self, root):
        self.root, self.path = root, root / "manifests/budget.json"
        self.values = json.loads(self.path.read_text(encoding="utf-8"))
        self.lock, self.stop = threading.RLock(), threading.Event()
        self.check()

    def check(self):
        if self.stop.is_set() or time.monotonic() >= self.values["deadline_monotonic"] - 15:
            raise TimeoutError("Original CPU admission deadline/stop reached")
        written = sum(p.stat().st_size for p in self.root.rglob("*") if p.is_file())
        if written > self.values["new_dataset_write_bytes_cap"]:
            raise ValueError("Dataset write cap exceeded")

    def persist(self):
        self.path.write_text(json.dumps(self.values, indent=2) + "\n", encoding="utf-8")

    def fetch(self, url, path):
        self.check()
        if path.exists():
            return path.read_bytes()
        if not re.fullmatch(r"https://api\.dw\.com/api/detail/article/[0-9]+", url):
            raise ValueError("Unexpected article URL")
        chunks, size = [], 0
        request = urllib.request.Request(url, headers={"User-Agent": "TTMRE-DWIE-CPU-admission/1.0"})
        started = time.monotonic()
        try:
            with urllib.request.urlopen(request, timeout=12) as response:
                status = response.status
                while True:
                    self.check()
                    data = response.read(65536)
                    if not data:
                        break
                    with self.lock:
                        self.values["network_bytes"] += len(data)
                        if self.values["network_bytes"] > self.values["network_bytes_cap"]:
                            raise ValueError("Network cap exceeded")
                    size += len(data)
                    if size > 2 * 1024**2:
                        raise ValueError("Article response cap exceeded")
                    chunks.append(data)
            raw = b"".join(chunks)
            with self.lock:
                self.check()
                with path.open("xb") as output:
                    output.write(raw)
                self.values["downloads"].append({"url": url, "path": str(path),
                    "bytes": len(raw), "sha256": sha256(raw), "http_status": status,
                    "seconds": time.monotonic() - started})
                self.persist()
            return raw
        except Exception as error:
            with self.lock:
                self.values["downloads"].append({"url": url, "ok": False,
                    "error": str(error), "seconds": time.monotonic() - started})
                self.persist()
            raise


def write_new(path, value):
    with path.open("x", encoding="utf-8") as output:
        json.dump(value, output, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
        output.write("\n")


def prepare(root, resume_once=False):
    budget = Budget(root)
    original_result = root / "manifests/preparation_result.json"
    previous = None
    if resume_once:
        receipt = root / "manifests/recovery_authorization.json"
        authorization = json.loads(receipt.read_text(encoding="utf-8"))
        previous = json.loads(original_result.read_text(encoding="utf-8"))
        if not authorization.get("user_confirmed") or authorization["deadline_monotonic"] != budget.values["deadline_monotonic"]:
            raise ValueError("Recovery must preserve the original authorized deadline")
        if previous["passed"] or (root / "manifests/recovery_once_claim.json").exists():
            raise ValueError("Recovery already claimed or original preparation complete")
        errors = [r["error"] for r in previous["failures"]]
        if not errors or any("timed out" not in e and "deadline/stop reached" not in e for e in errors):
            raise ValueError("Only the recorded network timeout/stop may be recovered")
        write_new(root / "manifests/recovery_once_claim.json", {
            "deadline_monotonic": budget.values["deadline_monotonic"],
            "original_failure_report_sha256": sha256(original_result.read_bytes()),
            "new_budget_created": False})
    elif original_result.exists():
        raise ValueError("Preparation already has a final result; no automatic retry")
    archive = next((root / "raw").glob("official-*.zip"))
    initial = json.loads((root / "manifests/initial_probe.json").read_text())
    if len(initial["attempts"]) != 3 or not all(x.get("ok") and x.get("hash_match")
            and x.get("span_mismatches") == 0 for x in initial["attempts"]):
        raise ValueError("The three initial train probes must pass first")
    partition = json.loads((root / "manifests/partitions.json").read_text())
    train_rows = [r for r in partition["documents"] if "train" in r["tags"] and "test" not in r["tags"]]
    ids = sorted(x["id"] for x in train_rows)
    train_ids, dev_ids = fixed_split(ids)
    with zipfile.ZipFile(archive) as source:
        prefix = source.namelist()[0].split("/")[0] + "/"
        metadata = {r["id"]: r for r in json.loads(source.read(prefix + "data/article_id_to_url.json"))}
        # Only train annotation bodies may be deserialized.
        annotations = {r["id"]: json.loads(source.read(r["archive_member"])) for r in train_rows}
    result = {"official_commit": budget.values["official_commit"], "passed": False,
              "official_train_documents": len(ids), "official_test_partition_ids": partition["counts"]["test"],
              "target_train_documents": len(train_ids), "target_dev_documents": len(dev_ids),
              "test_gold_parsed": False, "test_content_downloaded": False, "gpu_used": False,
              "optimizer_updates": 0, "records": previous["records"][:] if previous else [], "failures": [],
              "explicit_user_authorized_recovery": resume_once}
    split_path = root / "manifests/target_split.json"
    if previous:
        existing = json.loads(split_path.read_text())
        if existing["train"] != train_ids or existing["dev"] != dev_ids:
            raise ValueError("Recovery cannot change the target split")
    else:
        write_new(split_path, {"train": train_ids, "dev": dev_ids,
                  "algorithm": "sha256('ttmre-dwie-dev-v1:' + id); first 100 official train IDs form Dev"})

    def one(ident):
        budget.check()
        meta = metadata[ident]
        url = meta["url"]
        if ident == "DW_18525950":
            url = url.replace("18525950", "19026607")
        raw = budget.fetch(url, root / "raw" / (ident + "-article-response.json"))
        content = reconstruct(json.loads(raw), ident)
        converted, statistics = validate_and_convert(annotations[ident], content, meta["hash"])
        path = root / "converted" / (ident + ".json")
        write_new(path, converted)
        return {"id": ident, "content_sha1": meta["hash"], "response_sha256": sha256(raw),
                "converted_sha256": sha256(path.read_bytes()), "characters": len(content),
                "entities": len(converted["entities"]), "relations": len(converted["relations"]),
                "pairs": len(converted["hts"]), **statistics}

    # A rolling pool limits outstanding work to four requests. Stop submitting
    # on the first error; retain already-running receipts rather than erase them.
    completed_ids = {r["id"] for r in result["records"]}
    pending_ids = [ident for ident in ids if ident not in completed_ids]
    remaining = iter(pending_ids)
    with ThreadPoolExecutor(max_workers=4) as pool:
        pending = {pool.submit(one, ident): ident for ident in [next(remaining) for _ in range(min(4, len(pending_ids)))]}
        while pending:
            done, _ = wait(pending, return_when=FIRST_COMPLETED, timeout=1)
            if not done:
                try:
                    budget.check()
                except Exception as error:
                    budget.stop.set()
                    result["failures"].append({"phase": "budget", "error": str(error)})
                continue
            for future in done:
                ident = pending.pop(future)
                try:
                    row = future.result()
                    result["records"].append(row)
                except Exception as error:
                    budget.stop.set()
                    result["failures"].append({"id": ident, "error": str(error)})
                if not budget.stop.is_set():
                    ident = next(remaining, None)
                    if ident is not None:
                        pending[pool.submit(one, ident)] = ident
            if len(result["records"]) % 25 == 0:
                print(json.dumps({"prepared": len(result["records"]),
                                  "failed": len(result["failures"])}), flush=True)
    result["records"].sort(key=lambda r: r["id"])
    result["passed"] = len(result["records"]) == len(ids) and not result["failures"]
    result["network_bytes"] = budget.values["network_bytes"]
    result["elapsed_seconds"] = time.monotonic() - budget.values["start_monotonic"]
    output = "recovery_preparation_result.json" if resume_once else "preparation_result.json"
    write_new(root / "manifests" / output, result)
    print(json.dumps({k: v for k, v in result.items() if k not in ("records",)}), flush=True)
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DATA_ROOT)
    parser.add_argument("--resume-once", action="store_true", help="Requires the explicit user recovery receipt; original deadline unchanged")
    args = parser.parse_args()
    raise SystemExit(prepare(args.root.resolve(), args.resume_once))
