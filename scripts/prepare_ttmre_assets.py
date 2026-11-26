"""Prepare exact train/dev bytes and optionally download pinned generic RoBERTa-large."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import urllib.request
import zipfile

from train_ttmre_human_only import ROOT, SPLITS, read_json, sha256

ARCHIVE_SHA256 = "24f304c2612a625183b9cf337ddebc1ceb46ce02bbdc548391978c3cde444f47"
REPOSITORY = "FacebookAI/roberta-large"
ENDPOINT = "https://hf-mirror.com"


def prepare_data(root: Path, dry_run=False) -> dict:
    archive_path = root / "dataset" / "docred.zip"
    if sha256(archive_path) != ARCHIVE_SHA256:
        raise ValueError("Archive identity mismatch; no extraction performed")
    pending = {}
    result = {}
    with zipfile.ZipFile(archive_path) as archive:
        for name, expected in SPLITS.items():
            raw = archive.read("docred/" + name)
            if hashlib.sha256(raw).hexdigest() != expected:
                raise ValueError(f"Archive member identity mismatch: {name}")
            target = root / "dataset" / "docred" / name
            if target.exists():
                if not target.is_file() or sha256(target) != expected:
                    raise FileExistsError(f"Conflicting existing data; preserved: {target}")
                result[name] = "reused_identical"
            else:
                pending[target] = raw
                result[name] = "would_extract" if dry_run else "extracted"
    # Validate both members and both targets before creating anything.
    if pending and not dry_run:
        (root / "dataset" / "docred").mkdir(exist_ok=True)
        for target, raw in pending.items():
            with target.open("xb") as handle:
                handle.write(raw)
            if sha256(target) != SPLITS[target.name]:
                raise ValueError(f"Extracted identity mismatch; file preserved: {target}")
    return {"splits": result, "test_extracted": False, "distant_extracted": False}


def resolve_source(models: Path) -> dict:
    record = models / "roberta-large-source.json"
    if record.exists():
        source = read_json(record)
        if (source.get("repository") != REPOSITORY or source.get("endpoint") != ENDPOINT
                or not re.fullmatch(r"[0-9a-f]{40}", source.get("revision", ""))):
            raise ValueError("Existing source record is incompatible; preserved")
        return source
    request = urllib.request.Request(ENDPOINT + "/api/models/" + REPOSITORY + "?blobs=true",
                                     headers={"User-Agent": "ttmre-reproduction-assets/1"})
    with urllib.request.urlopen(request, timeout=30) as response:
        info = json.load(response)
    revision = info["sha"]
    if info["id"] != REPOSITORY or not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("Invalid official model metadata")
    siblings = {item["rfilename"]: item for item in info["siblings"]}
    required = {"config.json", "vocab.json", "merges.txt"}
    weight = "model.safetensors" if "model.safetensors" in siblings else "pytorch_model.bin"
    required.add(weight)
    required.update(name for name in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json")
                    if name in siblings)
    files = {}
    for name in sorted(required):
        item = siblings[name]
        lfs = item.get("lfs")
        files[name] = {"bytes": item["size"], "sha256": lfs["sha256"] if lfs else None,
                       "git_blob_sha1": item["blobId"] if not lfs else None}
    source = {"repository": REPOSITORY, "endpoint": ENDPOINT, "revision": revision,
              "files": files, "kind": "generic_pretrained_encoder; not TTM-RE task checkpoint"}
    models.mkdir(exist_ok=True)
    with record.open("x", encoding="utf-8") as handle:
        json.dump(source, handle, indent=2)
        handle.write("\n")
    return source


def verify_model(snapshot: Path, source: dict) -> dict:
    results = {}
    for name, expected in source["files"].items():
        path = snapshot / name
        size = path.stat().st_size
        if size != expected["bytes"]:
            raise ValueError(f"Official size mismatch: {name}")
        digest = hashlib.sha256()
        git_digest = hashlib.sha1(f"blob {size}\0".encode())
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
                git_digest.update(block)
        if expected["sha256"] and digest.hexdigest() != expected["sha256"]:
            raise ValueError(f"Official LFS SHA-256 mismatch: {name}")
        if expected["git_blob_sha1"] and git_digest.hexdigest() != expected["git_blob_sha1"]:
            raise ValueError(f"Official Git blob identity mismatch: {name}")
        results[name] = {"bytes": size, "sha256": digest.hexdigest()}
    return results


def prepare_model(root: Path) -> dict:
    from huggingface_hub import snapshot_download
    models = root / "models"
    source = resolve_source(models)
    print(f"Official source: {source['repository']}@{source['revision']}", flush=True)
    snapshot = Path(snapshot_download(repo_id=REPOSITORY, revision=source["revision"],
                                     cache_dir=str(models / "hf-cache"),
                                     allow_patterns=list(source["files"]), token=False,
                                     max_workers=2, resume_download=True))
    files = verify_model(snapshot, source)
    return {"repository": REPOSITORY, "revision": source["revision"],
            "model_dir": str(snapshot.resolve()), "verified_files": files}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="Validate data only; no writes or network")
    parser.add_argument("--download-model", action="store_true", help="Download about 1.4 GB of official model files")
    args = parser.parse_args()
    if args.dry_run and args.download_model:
        parser.error("--dry-run cannot be combined with --download-model")
    result = {"data": prepare_data(ROOT, args.dry_run), "model": None,
              "test_labels_read": False, "distant_data_read": False, "optimizer_steps": 0}
    if args.download_model:
        result["model"] = prepare_model(ROOT)
    print(json.dumps(result, indent=2))
    print("TTMRE_ASSETS_DRY_RUN_OK" if args.dry_run else "TTMRE_ASSETS_OK; optimizer_steps=0; test_labels_read=false")


if __name__ == "__main__":
    main()
