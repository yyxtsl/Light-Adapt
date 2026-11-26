"""Pure NumPy symmetric decision-offset diagnostics; no Torch/model/Test I/O."""
from __future__ import annotations
import numpy as np


def fixed_grid():
    return np.arange(-2000, 2001, dtype=np.float64) / 20.0


def margins(logits):
    x = np.asarray(logits)
    if x.ndim != 2 or x.shape[1] < 2 or not np.isfinite(x).all():
        raise ValueError('Need finite pair-by-(TH+relations) logits')
    return x[:, 1:].astype(np.float64) - x[:, :1].astype(np.float64)


def counts_to_metrics(tp, predicted, gold):
    tp, predicted, gold = int(tp), int(predicted), int(gold)
    if not 0 <= tp <= min(predicted, gold):
        raise ValueError('Invalid confusion counts')
    return {'TP': tp, 'FP': predicted-tp, 'FN': gold-tp,
            'precision': 100*tp/predicted if predicted else 0.0,
            'recall': 100*tp/gold if gold else 0.0,
            'F1': 200*tp/(predicted+gold) if predicted+gold else 0.0}


def scan(logits, gold_mask, grid=None):
    gap = margins(logits)
    y = np.asarray(gold_mask)
    if y.dtype != np.bool_ or y.shape != gap.shape:
        raise ValueError('Boolean gold mask must match positive-relation margins')
    grid = fixed_grid() if grid is None else np.asarray(grid, dtype=np.float64)
    if grid.ndim != 1 or not len(grid) or not np.isfinite(grid).all() or not (np.diff(grid)>0).all():
        raise ValueError('Need finite strictly increasing threshold grid')
    positive = np.sort(gap[y])
    all_scores = np.sort(gap, axis=None)
    predicted = len(all_scores)-np.searchsorted(all_scores, grid, side='right')
    tp = len(positive)-np.searchsorted(positive, grid, side='right')
    gold = int(y.sum())
    rows = [{'delta':float(d), **counts_to_metrics(t, p, gold)} for d,t,p in zip(grid,tp,predicted)]
    # Compare exact integer-ratio F1, avoiding roundoff-based tie decisions.
    best = 0
    for i in range(1,len(rows)):
        a,b=rows[i],rows[best]
        da=a['TP']+a['FP']+gold; db=b['TP']+b['FP']+gold
        cmp=a['TP']*db-b['TP']*da
        if cmp>0 or (cmp==0 and (abs(a['delta']),-a['delta']) < (abs(b['delta']),-b['delta'])):
            best=i
    return {'rows':rows,'selected':rows[best],
            'boundary_limited':best in (0,len(rows)-1),
            'gold_triples':gold,'pairs':len(y),'positive_relations':y.shape[1],
            'margin_min':float(all_scores[0]),'margin_max':float(all_scores[-1])}


def make_gold_mask(pair_order, documents, rel2id):
    mapping={d['title']:d for d in documents}
    if len(mapping)!=len(documents):raise ValueError('Duplicate document titles')
    relation_count=len(rel2id)-1
    if sorted(rel2id.values())!=list(range(relation_count+1)):
        raise ValueError('Noncontiguous relation mapping')
    mask=np.zeros((sum(len(d['hts']) for d in pair_order),relation_count),dtype=np.bool_)
    offset=0;seen=set();gold_total=0
    for item in pair_order:
        title=item['title']
        if title in seen or title not in mapping:raise ValueError('Changed document identity/order')
        seen.add(title);doc=mapping[title];pairs=[tuple(p) for p in item['hts']]
        n=len(doc['vertexSet'])
        if len(pairs)!=n*(n-1) or len(set(pairs))!=len(pairs) or set(pairs)!={(i,j) for i in range(n) for j in range(n) if i!=j}:
            raise ValueError('Candidate pair loss/duplication')
        lookup={p:i for i,p in enumerate(pairs)}
        gold={(v['h'],v['t'],v['r']) for v in doc['labels']}
        for h,t,r in gold:
            if (h,t) not in lookup or r not in rel2id or rel2id[r]==0:
                raise ValueError('Gold not representable; filtering forbidden')
            mask[offset+lookup[h,t],rel2id[r]-1]=True
        offset+=len(pairs);gold_total+=len(gold)
    if seen!=set(mapping) or int(mask.sum())!=gold_total:raise ValueError('Lost gold/document')
    return mask


def prediction_records(logits, delta, pair_order, id2rel):
    flags=margins(logits)>float(delta)
    order=[(doc['title'],h,t) for doc in pair_order for h,t in doc['hts']]
    if len(order)!=len(flags):raise ValueError('Pair order size mismatch')
    rows,columns=np.nonzero(flags)
    return [{'title':order[i][0],'h_idx':order[i][1],'t_idx':order[i][2],
             'r':id2rel[int(j)+1]} for i,j in zip(rows,columns)]
