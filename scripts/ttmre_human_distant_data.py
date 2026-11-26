"""Streaming DocRED input and sparse disk features for the approved two-stage run."""
from __future__ import annotations

import codecs
import hashlib
import json
from pathlib import Path
import zlib

CACHE_VERSION = 1
MAX_RECORD_BYTES = 16 * 1024 * 1024


def iter_documents(stream, digest=None, chunk_size=65536):
    """Read a UTF-8 JSON array one document at a time, hashing exact input bytes."""
    decoder, text_decoder = json.JSONDecoder(), codecs.getincrementaldecoder("utf-8")()
    buffer, ended = "", False

    def more():
        nonlocal buffer, ended
        block = stream.read(chunk_size)
        if digest is not None:
            digest.update(block)
        ended = not block
        buffer += text_decoder.decode(block, final=ended)
        if len(buffer.encode("utf-8")) > MAX_RECORD_BYTES:
            raise ValueError("Oversized or malformed JSON document")

    def ready():
        nonlocal buffer
        buffer = buffer.lstrip()
        while not buffer and not ended:
            more()
            buffer = buffer.lstrip()

    ready()
    if not buffer.startswith("["):
        raise ValueError("Expected a JSON document array")
    buffer = buffer[1:]
    first = True
    while True:
        ready()
        if first and buffer.startswith("]"):
            buffer = buffer[1:]
            break
        if not buffer or buffer.startswith("]"):
            raise ValueError("Truncated array or trailing comma")
        while True:
            try:
                doc, end = decoder.raw_decode(buffer)
                break
            except json.JSONDecodeError as error:
                if ended:
                    raise ValueError("Malformed document JSON") from error
                more()
        if not isinstance(doc, dict):
            raise ValueError("Each document must be an object")
        buffer = buffer[end:]
        yield doc
        first = False
        ready()
        if buffer.startswith(","):
            buffer = buffer[1:]
        elif buffer.startswith("]"):
            buffer = buffer[1:]
            break
        else:
            raise ValueError("Missing array delimiter")
    ready()
    if buffer:
        raise ValueError("Trailing content after document array")


def check_document(doc, rel2id):
    sents, entities, labels = doc.get("sents"), doc.get("vertexSet"), doc.get("labels")
    if (not isinstance(doc.get("title"), str) or not isinstance(sents, list) or not sents
            or any(not isinstance(sent, list) or not sent or
                   any(not isinstance(token, str) for token in sent) for sent in sents)
            or not isinstance(entities, list) or len(entities) < 2 or not isinstance(labels, list)):
        raise ValueError("Invalid document schema")
    for entity in entities:
        if not isinstance(entity, list) or not entity:
            raise ValueError("Empty/invalid entity")
        for mention in entity:
            sentence, pos = mention.get("sent_id"), mention.get("pos")
            if (type(sentence) is not int or not 0 <= sentence < len(sents)
                    or not isinstance(pos, list) or len(pos) != 2
                    or any(type(x) is not int for x in pos)
                    or not 0 <= pos[0] < pos[1] <= len(sents[sentence])
                    or not isinstance(mention.get("name"), str)):
                raise ValueError("Invalid mention position/name")
    for label in labels:
        h, t, rel = label.get("h"), label.get("t"), label.get("r")
        if (type(h) is not int or type(t) is not int or
                not 0 <= h < len(entities) or not 0 <= t < len(entities) or h == t
                or rel not in rel2id or rel2id[rel] == 0):
            raise ValueError("Invalid directed relation")


def sparse_feature(doc, tokenizer, rel2id, with_labels=True):
    """Match pinned prepro.py token/mention processing, using canonical pairs."""
    check_document(doc, rel2id)
    entities = doc["vertexSet"]
    starts = {(m["sent_id"], m["pos"][0]) for e in entities for m in e}
    ends = {(m["sent_id"], m["pos"][1] - 1) for e in entities for m in e}
    tokens, maps = [], []
    for sentence_index, sentence in enumerate(doc["sents"]):
        positions = {}
        for token_index, token in enumerate(sentence):
            wordpieces = tokenizer.tokenize(token)
            if (sentence_index, token_index) in starts:
                wordpieces = ["*"] + wordpieces
            if (sentence_index, token_index) in ends:
                wordpieces += ["*"]
            positions[token_index] = len(tokens)
            tokens.extend(wordpieces)
        positions[len(sentence)] = len(tokens)
        maps.append(positions)
    entity_pos = [[(maps[m["sent_id"]][m["pos"][0]], maps[m["sent_id"]][m["pos"][1]])
                   for m in e] for e in entities]
    input_ids = tokenizer.build_inputs_with_special_tokens(tokenizer.convert_tokens_to_ids(tokens[:1022]))
    positive = {}
    if with_labels:
        count = len(entities)
        for label in doc["labels"]:
            h, t = label["h"], label["t"]
            index = h * (count - 1) + t - int(t > h)
            positive.setdefault(str(index), set()).add(rel2id[label["r"]])
    return {"input_ids": input_ids, "entity_pos": entity_pos, "title": doc["title"],
            "positive": {key: sorted(value) for key, value in positive.items()}}


def expand_feature(item, with_labels=True):
    count = len(item["entity_pos"])
    pairs = [[h, t] for h in range(count) for t in range(count) if h != t]
    labels = []
    if with_labels:
        for index in range(len(pairs)):
            row = [0] * 97
            relations = item["positive"].get(str(index), [])
            for rel in relations:
                row[rel] = 1
            row[0] = int(not relations)
            labels.append(row)
    return {"input_ids": list(item["input_ids"]),
            "entity_pos": [[tuple(position) for position in entity] for entity in item["entity_pos"]],
            "hts": pairs, "labels": labels, "title": item["title"]}


def relevant_gold_facts(documents):
    return {(h["name"], t["name"], label["r"])
            for doc in documents for label in doc["labels"]
            for h in doc["vertexSet"][label["h"]] for t in doc["vertexSet"][label["t"]]}


def matching_facts(doc, candidates):
    # Only correct Dev predictions can enter the official ignored-precision count.
    return {(h["name"], t["name"], label["r"])
            for label in doc["labels"]
            for h in doc["vertexSet"][label["h"]] for t in doc["vertexSet"][label["t"]]
            if (h["name"], t["name"], label["r"]) in candidates}


def encode_record(item):
    return zlib.compress(json.dumps(item, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))


def build_cache(stream, directory, tokenizer, rel2id, expected_documents, candidates,
                with_labels=True, progress=None, expected_sha256=None):
    """Exclusively create one cache, bounded by one sparse document in memory."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    digest = hashlib.sha256()
    index, frequencies, facts, total_pairs, label_count = [], [0] * 96, set(), 0, 0
    titles = set()
    with (directory / "features.bin").open("xb") as out:
        for number, doc in enumerate(iter_documents(stream, digest), 1):
            check_document(doc, rel2id)
            if expected_documents in (3053, 500) and doc["title"] in titles:
                raise ValueError("Duplicate Human/Dev title")
            if expected_documents in (3053, 500):
                titles.add(doc["title"])
            item = sparse_feature(doc, tokenizer, rel2id, with_labels)
            payload = encode_record(item)
            pairs = len(item["entity_pos"]) * (len(item["entity_pos"]) - 1)
            index.append([out.tell(), len(payload), len(item["input_ids"]), pairs])
            out.write(payload)
            total_pairs += pairs
            for label in doc["labels"]:
                frequencies[rel2id[label["r"]] - 1] += 1
            label_count += len(doc["labels"])
            facts.update(matching_facts(doc, candidates))
            if progress and (number == 1 or number % 1000 == 0):
                progress(documents=number, compressed_bytes=out.tell(), ordered_pairs=total_pairs)
    if len(index) != expected_documents or total_pairs == 0:
        raise ValueError(f"Expected {expected_documents} documents, found {len(index)}")
    checksum = digest.hexdigest()
    if expected_sha256 is not None and checksum != expected_sha256:
        raise ValueError("Source split SHA256 mismatch")
    def file_hash(path):
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1048576), b""):
                digest.update(block)
        return digest.hexdigest()
    metadata = {"cache_version": CACHE_VERSION, "documents": len(index), "ordered_pairs": total_pairs,
                "label_entries": label_count, "source_sha256": checksum, "with_labels": with_labels,
                "priors": [value / total_pairs for value in frequencies],
                "features_sha256": file_hash(directory / "features.bin"), "index": index,
                "matching_dev_facts": sorted(facts)}
    with (directory / "index.json").open("x", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, allow_nan=False)
    return metadata


class DiskFeatures:
    """Map-style dataset; no persistent file handles and no full dense-label cache."""
    def __init__(self, directory, metadata, with_labels=True):
        self.directory, self.metadata, self.with_labels = Path(directory), metadata, with_labels

    def __len__(self):
        return len(self.metadata["index"])

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self[i] for i in range(*index.indices(len(self)))]
        offset, size, _, _ = self.metadata["index"][index]
        with (self.directory / "features.bin").open("rb") as handle:
            handle.seek(offset)
            payload = handle.read(size)
        if len(payload) != size:
            raise ValueError("Truncated feature cache")
        item = json.loads(zlib.decompress(payload))
        return expand_feature(item, self.with_labels)

    def probe_indices(self, batch_size=4):
        index = self.metadata["index"]
        if len(index) < batch_size:
            raise ValueError("Insufficient documents for probe")
        pairs = sorted(range(len(index)), key=lambda i: (index[i][3], index[i][2]), reverse=True)
        lengths = sorted(range(len(index)), key=lambda i: (index[i][2], index[i][3]), reverse=True)
        return {"first": list(range(batch_size)), "most_pairs": pairs[:batch_size], "longest": lengths[:batch_size]}


def validate_cache(directory, sha256, expected_documents, expected_source=None, with_labels=True):
    directory = Path(directory)
    with (directory / "index.json").open(encoding="utf-8") as handle:
        metadata = json.load(handle)
    if (metadata.get("cache_version") != CACHE_VERSION or metadata.get("documents") != expected_documents
            or metadata.get("with_labels") is not with_labels or len(metadata.get("index", [])) != expected_documents
            or (expected_source is not None and metadata.get("source_sha256") != expected_source)
            or metadata.get("features_sha256") != sha256(directory / "features.bin")):
        raise ValueError("Prepared cache identity mismatch")
    offset, total_pairs = 0, 0
    for row in metadata["index"]:
        if (len(row) != 4 or any(type(x) is not int for x in row) or row[0] != offset
                or row[1] <= 0 or not 2 <= row[2] <= 1024 or row[3] <= 0):
            raise ValueError("Invalid feature index")
        offset += row[1]
        total_pairs += row[3]
    if offset != (directory / "features.bin").stat().st_size or total_pairs != metadata["ordered_pairs"]:
        raise ValueError("Feature index size/count mismatch")
    return metadata
