"""Isolated full-window/full-gold DWIE input and scoring functions.

Pure data functions import no tensor runtime. Tensor functions import torch
lazily, never load weights or create an optimizer, and preserve the caller's
device. This module does not modify model2.py or any historical runner.
"""
from __future__ import annotations

import collections
import math

WINDOW = 512
PAYLOAD = 510
STRIDE = 256
STAR = 3226


def window_starts(length, payload=PAYLOAD, stride=STRIDE):
    if length < 0 or payload < 1 or not 0 < stride <= payload:
        raise ValueError("Invalid full-window geometry")
    if length <= payload:
        return [0]
    starts = list(range(0, length-payload+1, stride))
    if starts[-1] != length-payload:
        starts.append(length-payload)
    return starts


def window_layout(length, payload=PAYLOAD, stride=STRIDE):
    maps = [[0] + list(range(start+1, min(length, start+payload)+1)) + [length+1]
            for start in window_starts(length, payload, stride)]
    coverage = [0] * (length+2)
    for indices in maps:
        for index in indices:
            coverage[index] += 1
    if min(coverage) < 1:
        raise ValueError("Uncovered global token position")
    return maps, coverage


def window_inputs(content_ids, cls_id=0, sep_id=2):
    length = len(content_ids)
    maps, coverage = window_layout(length)
    windows = []
    for start, mapping in zip(window_starts(length), maps):
        ids = [cls_id] + content_ids[start:start+PAYLOAD] + [sep_id]
        if len(ids) != len(mapping) or len(ids) > WINDOW:
            raise ValueError("Window input/map mismatch")
        windows.append({"input_ids": ids, "attention_mask": [1]*len(ids), "global_indices": mapping})
    return windows, coverage


def merge_window_outputs(sequences, attentions, mappings, global_length):
    import torch
    if not sequences or len(sequences) != len(attentions) or len(sequences) != len(mappings):
        raise ValueError("Window output counts disagree")
    seq, att = sequences[0], attentions[0]
    if seq.ndim != 2 or att.ndim != 3:
        raise ValueError("Expected [tokens,hidden] and [heads,tokens,tokens]")
    if len(sequences) == 1 and mappings[0] == list(range(global_length)):
        if seq.shape[0] < global_length or att.shape[1] < global_length or att.shape[2] < global_length:
            raise ValueError("Single window output is shorter than its mapping")
        return seq[:global_length], att[:, :global_length, :global_length]
    width, heads = seq.shape[1], att.shape[0]
    total = seq.new_zeros(global_length, width)
    attention_total = att.new_zeros(heads, global_length*global_length)
    coverage = seq.new_zeros(global_length)
    for output, attention, mapping in zip(sequences, attentions, mappings):
        size = len(mapping)
        if (output.ndim != 2 or output.shape[0] < size or output.shape[1] != width
                or attention.shape[0] != heads or attention.shape[1] < size or attention.shape[2] < size
                or not mapping or min(mapping) < 0 or max(mapping) >= global_length
                or len(set(mapping)) != size):
            raise ValueError("Window tensor/map shape mismatch")
        index = torch.tensor(mapping, dtype=torch.long, device=output.device)
        total = total.index_add(0, index, output[:size])
        coverage = coverage.index_add(0, index, output.new_ones(size))
        flat_index = (index[:, None]*global_length + index[None, :]).reshape(-1)
        attention_total = attention_total.index_add(1, flat_index, attention[:, :size, :size].reshape(heads, -1))
    if (coverage == 0).any().item():
        raise ValueError("Window merger left a hole")
    attention_total = attention_total.reshape(heads, global_length, global_length)
    denominator = attention_total.sum(-1, keepdim=True)
    merged_attention = attention_total / denominator.clamp_min(1e-10)
    return total / coverage[:, None], merged_attention


def pool_entities(sequence, attention, entity_pos):
    import torch
    n, width, heads = sequence.shape[0], sequence.shape[1], attention.shape[0]
    embeddings, rows = [], []
    for mentions in entity_pos:
        if not mentions:
            embeddings.append(sequence.new_zeros(width))
            rows.append(attention.new_zeros(heads, n))
            continue
        indices = []
        for start, end in mentions:
            if not 0 <= start < end <= n-2:
                raise ValueError("Mention span outside full content tokens")
            indices.append(start+1)  # global CLS offset
        emb = sequence[indices]
        embeddings.append(emb[0] if len(indices) == 1 else torch.logsumexp(emb, dim=0))
        rows.append(attention[:, indices, :].mean(1))
    if not embeddings:
        return sequence.new_zeros(0, width), attention.new_zeros(0, heads, n)
    return torch.stack(embeddings), torch.stack(rows)


def candidate_pairs(entity_ids):
    if entity_ids != sorted(set(entity_ids)):
        raise ValueError("Entity IDs must be unique and sorted independently of labels")
    return [(h, t) for h in range(len(entity_ids)) for t in range(len(entity_ids))]


def pair_representations(sequence, attention, entity_pos):
    import torch
    embeddings, rows = pool_entities(sequence, attention, entity_pos)
    count = len(entity_pos)
    pairs = candidate_pairs(list(range(count)))
    if not pairs:
        empty = sequence.new_zeros(0, sequence.shape[1])
        return empty, empty.clone(), empty.clone(), 0
    index = torch.tensor(pairs, dtype=torch.long, device=sequence.device)
    hs, ts = embeddings[index[:, 0]], embeddings[index[:, 1]]
    weights = (rows[index[:, 0]] * rows[index[:, 1]]).mean(1)
    mass = weights.sum(1, keepdim=True)
    context = (weights / (mass+1e-5)) @ sequence
    return hs, context, ts, int((mass == 0).sum().item())


def mark_document(record):
    starts, ends = set(), set()
    for entity in record["entities"]:
        for mention in entity["mentions"]:
            start, end = mention["begin"], mention["end"]
            if not 0 <= start < end <= len(record["content"]) or record["content"][start:end] != mention["text"]:
                raise ValueError("Original mention character offsets changed")
            starts.add(start); ends.add(end)
    chunks, start_offsets, end_offsets, cursor, size = [], {}, {}, 0, 0
    for boundary in sorted(starts | ends):
        chunk = record["content"][cursor:boundary]
        chunks.append(chunk); size += len(chunk)
        if boundary in ends:
            end_offsets[boundary] = size
            chunks.append("*"); size += 1
        if boundary in starts:
            start_offsets[boundary] = size
            chunks.append("*"); size += 1
        cursor = boundary
    chunks.append(record["content"][cursor:])
    return "".join(chunks), start_offsets, end_offsets


def make_feature(record, tokenizer, rel2id):
    marked, start_offsets, end_offsets = mark_document(record)
    encoded = tokenizer.encode(marked, add_special_tokens=False)
    lookup = {a: i for i, ((a, b), ident) in enumerate(zip(encoded.offsets, encoded.ids))
              if ident == STAR and b == a+1}
    if any(offset not in lookup for offset in list(start_offsets.values()) + list(end_offsets.values())):
        raise ValueError("Marker character-to-token mapping failed")
    entities = sorted(record["entities"], key=lambda entity: entity["id"])
    ids = [e["id"] for e in entities]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate entity IDs")
    positions = [[[lookup[start_offsets[m["begin"]]], lookup[end_offsets[m["end"]]]+1]
                  for m in e["mentions"]] for e in entities]
    mappings, coverage = window_layout(len(encoded.ids))
    gold = sorted({(r["h"], r["r"], r["t"]) for r in record["relations"]})
    if any(h not in ids or t not in ids for h, _, t in gold):
        raise ValueError("Gold endpoint missing from concept IDs")
    anchored = {e["id"] for e in entities if e["mentions"]}
    # Gold is kept separate; changing labels must not change model inputs.
    feature = {"id": record["id"], "inputs": {"content_ids": encoded.ids, "entity_ids": ids,
               "entity_pos": positions, "candidate_rule": "cartesian_including_self"},
               "gold": [{"h": h, "r": r, "t": t} for h, r, t in gold]}
    audit = {"content_tokens": len(encoded.ids), "windows": len(mappings),
             "min_coverage": min(coverage), "max_coverage": max(coverage),
             "entities": len(ids), "mentions_mapped": sum(map(len, positions)),
             "pairs": len(ids)**2, "gold_relations": len(gold),
             "unknown_gold": sum(r not in rel2id for _, r, _ in gold),
             "self_gold": sum(h == t for h, _, t in gold),
             "unanchored_gold": sum(h not in anchored or t not in anchored for h, _, t in gold)}
    return feature, audit


def full_id_score(predictions, gold):
    predicted, truth = set(predictions), set(gold)
    tp = len(predicted & truth)
    return {"tp": tp, "fp": len(predicted-truth), "fn": len(truth-predicted),
            "precision": tp/len(predicted) if predicted else 0.,
            "recall": tp/len(truth) if truth else 0.,
            "f1": 2*tp/(len(predicted)+len(truth)) if predicted or truth else 0.}


def known_training_targets(feature, rel2id):
    """Sparse positives only; unknown gold cannot silently become NA/TH."""
    result = collections.defaultdict(list)
    by_id = {ident: i for i, ident in enumerate(feature["inputs"]["entity_ids"])}
    for relation in feature["gold"]:
        if relation["r"] not in rel2id:
            raise ValueError("Unknown gold cannot be converted to training TH")
        result[(by_id[relation["h"]], by_id[relation["t"]])].append(rel2id[relation["r"]])
    return dict(result)


def official_metric_types(source_path):
    """Extract only inspected pure metric definitions from pinned source."""
    import ast
    import hashlib
    from pathlib import Path
    raw = Path(source_path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != "0a939d851405a619f4af008b2c224899960c3bc16d54ae098ba2fe0181ed2006":
        raise ValueError("Official metric source digest changed")
    wanted = {"to_pairwise", "to_pairs", "MetricRelationF1", "MetricRelationF1Hard"}
    source = ast.parse(raw.decode("utf-8"))
    definitions = [n for n in source.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in wanted]
    if {n.name for n in definitions} != wanted:
        raise ValueError("Pinned metric definitions missing")
    module = ast.Module(body=definitions, type_ignores=[])
    for node in ast.walk(module):
        if isinstance(node, (ast.Import, ast.ImportFrom)) or (
            isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
            and node.func.id in {"open", "eval", "exec", "compile", "__import__"}
        ):
            raise ValueError("Unexpected effect in metric definitions")
    namespace = {}
    exec(compile(module, str(source_path), "exec"), namespace)
    return namespace["MetricRelationF1"], namespace["MetricRelationF1Hard"]


def score_documents(documents, predictions, known_relations, official_source=None):
    """Full-ID primary plus fixed slices and distinctly labeled official metrics.

    documents use canonical full gold entity IDs/character mentions. Predictions
    are (document_id, head_ID, relation_name, tail_ID), never TH. This function
    consumes gold solely for scoring, not input construction or vocabulary.
    """
    by_id = {d["id"]: d for d in documents}
    if len(by_id) != len(documents):
        raise ValueError("Duplicate document IDs")
    anchored, clusters, entity_sets, gold = {}, {}, {}, set()
    for ident, document in by_id.items():
        entity_sets[ident] = {e["id"] for e in document["entities"]}
        anchored[ident] = {e["id"] for e in document["entities"] if e["mentions"]}
        clusters[ident] = {e["id"]: tuple(sorted((m["begin"], m["end"]) for m in e["mentions"]))
                           for e in document["entities"]}
        for relation in document["relations"]:
            if relation["h"] not in entity_sets[ident] or relation["t"] not in entity_sets[ident]:
                raise ValueError("Gold endpoint missing")
            gold.add((ident, relation["h"], relation["r"], relation["t"]))
    pred, known = set(predictions), set(known_relations)
    for ident, head, relation, tail in pred:
        if ident not in by_id or head not in entity_sets[ident] or tail not in entity_sets[ident] or relation not in known:
            raise ValueError("Prediction violates fixed input IDs/train vocabulary")
    selectors = {
        "anchored_endpoints": lambda x: x[1] in anchored[x[0]] and x[3] in anchored[x[0]],
        "unanchored_endpoint": lambda x: x[1] not in anchored[x[0]] or x[3] not in anchored[x[0]],
        "self": lambda x: x[1] == x[3], "nonself": lambda x: x[1] != x[3],
        "train_known": lambda x: x[2] in known, "train_unknown": lambda x: x[2] not in known,
    }
    report = {"primary_full_id": full_id_score(pred, gold),
              "slices": {name: full_id_score([x for x in pred if choose(x)],
                                              [x for x in gold if choose(x)])
                         for name, choose in selectors.items()},
              "official_metrics_are_supplemental_not_id_equivalent": True}
    if official_source is not None:
        soft_type, hard_type = official_metric_types(official_source)
        soft, hard = soft_type(), hard_type()
        for ident in sorted(by_id):
            def expand(values):
                return [(clusters[ident][h], clusters[ident][t], relation)
                        for doc, h, relation, t in sorted(values) if doc == ident]
            expanded_pred, expanded_gold = expand(pred), expand(gold)
            soft.add(expanded_pred, expanded_gold)
            hard.add(expanded_pred, expanded_gold)
        report["official_expanded_clusters"] = {
            "soft": {"precision":soft.get_pr(), "recall":soft.get_re(), "f1":soft.get_f1()},
            "hard": {"precision":hard.get_pr(), "recall":hard.get_re(), "f1":hard.get_f1()},
            "gold_entity_clusters_provided": True,
            "empty_clusters_can_collapse_distinct_concept_ids": True}
    return report
