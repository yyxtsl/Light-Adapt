"""Dev-only latency measurements and symmetric calibration from saved logits."""

import argparse
import gc
import json
from pathlib import Path
import statistics
import sys
import time

import reproduce_redocred as common


ROOT = Path(__file__).resolve().parent


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    commands = result.add_subparsers(dest='command', required=True)
    calibration = commands.add_parser('calibrate')
    calibration.add_argument('--logits', type=Path, required=True, help='FP32 logits .npy, columns TH then 96 relations')
    calibration.add_argument('--pair-order', type=Path, required=True, help='List of title/hts records in logit order')
    calibration.add_argument('--dev-gold', type=Path, required=True)
    calibration.add_argument('--output-dir', type=Path, required=True)
    latency = commands.add_parser('benchmark-redocred')
    latency.add_argument('--model-dir', type=Path, required=True)
    latency.add_argument('--data-dir', type=Path, required=True)
    latency.add_argument('--checkpoint-c', type=Path, required=True)
    latency.add_argument('--checkpoint-b', type=Path, required=True)
    latency.add_argument('--output-dir', type=Path, required=True)
    latency.add_argument('--device', default='cuda:0')
    latency.add_argument('--batch-size', type=int, choices=[1, 4], default=4)
    return result


def calibrate(arguments):
    import numpy as np
    calibration_source = ROOT / 'refine-logs/DEV_THRESHOLD_CALIBRATION_20261103'
    sys.path.insert(0, str(calibration_source))
    try:
        import calibration_core as calibration
        if common.digest(arguments.dev_gold) != common.DEV_SHA256:
            raise ValueError('Expected the fixed Re-DocRED Dev gold file')
        documents = common.read_json(arguments.dev_gold)
        pair_order = common.read_json(arguments.pair_order)
        mapping = common.read_json(ROOT / 'meta/rel2id.json')
        logits = np.load(arguments.logits, mmap_mode='r', allow_pickle=False)
        if logits.dtype != np.float32 or logits.ndim != 2 or logits.shape[1] != 97:
            raise ValueError('Expected a complete FP32 [pairs,97] logit matrix')
        if len(documents) != 500 or len(pair_order) != 500:
            raise ValueError('Use the complete 500-document Dev split')
        gold = calibration.make_gold_mask(pair_order, documents, mapping)
        result = calibration.scan(logits, gold)
        original = next(row for row in result['rows'] if row['delta'] == 0)
        output = arguments.output_dir
        output.mkdir(parents=True, exist_ok=False)
        common.write_json(output / 'calibration.json', {
            **result, 'original': original, 'selection_scope': 'Dev-only diagnostic',
            'logits_sha256': common.digest(arguments.logits),
            'pair_order_sha256': common.digest(arguments.pair_order),
            'dev_gold_sha256': common.digest(arguments.dev_gold),
            'model_execution': False, 'optimizer_updates': 0, 'test_read': False})
        print(json.dumps({'original': original, 'selected': result['selected']}, indent=2))
    finally:
        sys.path.remove(str(calibration_source))


def bin_name(count):
    if count <= 15:
        return 'N<=15'
    if count <= 20:
        return '16<=N<=20'
    if count <= 25:
        return '21<=N<=25'
    return 'N>=26'


def benchmark(arguments):
    import numpy as np
    import torch
    import train_ttmre_human_only as shared
    import ttmre_human_distant_data as data
    output = arguments.output_dir
    output.mkdir(parents=True, exist_ok=False)
    args = common.parser().parse_args([
        '--mode', 'evaluate', '--variant', 'C', '--model-dir', str(arguments.model_dir),
        '--data-dir', str(arguments.data_dir), '--checkpoint', str(arguments.checkpoint_c),
        '--output-dir', str(output / 'unused'), '--device', arguments.device])
    args = common.normalized_args(args)
    _, documents, mapping, priors, data_identity = common.read_data(args, 'dev')
    result = {}
    features = None
    device = torch.device(arguments.device)
    def sync():
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
    for variant, checkpoint in [('C', arguments.checkpoint_c), ('B', arguments.checkpoint_b)]:
        args.variant, args.checkpoint = variant, checkpoint
        model, tokenizer, _ = common.load_model(args, priors, target_checkpoint=True)
        model.requires_grad_(False)
        model.eval()
        if features is None:
            features = []
            for document in documents:
                public = {'title': document['title'], 'sents': document['sents'],
                          'vertexSet': document['vertexSet'], 'labels': []}
                feature = data.expand_feature(data.sparse_feature(public, tokenizer, mapping, False), False)
                features.append(feature)
        # Keep prepared batches on CPU; transfer one batch outside each timed call.
        batches = [shared.collate_fn(features[i:i+arguments.batch_size])
                   for i in range(0, len(features), arguments.batch_size)]
        seconds = []
        per_document = []
        with torch.inference_mode():
            warmup = shared.inputs_from_batch(batches[0], device, labels=False)
            model(**warmup)
            del warmup
            for round_index in range(3):
                total = 0.0
                for index, batch in enumerate(batches):
                    inputs = shared.inputs_from_batch(batch, device, labels=False)
                    sync()
                    start = time.perf_counter()
                    logits = model(**inputs)
                    sync()
                    elapsed = time.perf_counter() - start
                    total += elapsed
                    if arguments.batch_size == 1:
                        per_document.append({'title': features[index]['title'], 'round': round_index,
                                             'entities': len(features[index]['entity_pos']),
                                             'seconds': elapsed})
                    del logits, inputs
                seconds.append(total)
        result[variant] = {'round_seconds': seconds, 'median_seconds': statistics.median(seconds),
                           'checkpoint_sha256': common.digest(checkpoint)}
        if per_document:
            result[variant]['entity_bins'] = {
                name: {'mean_seconds': float(np.mean([r['seconds'] for r in per_document
                                                     if bin_name(r['entities']) == name])),
                       'document_count': sum(bin_name(len(f['entity_pos'])) == name for f in features)}
                for name in ['N<=15', '16<=N<=20', '21<=N<=25', 'N>=26']}
            common.write_json(output / (variant + '_per_document.json'), per_document)
        del model, batches
        gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    common.write_json(output / 'timing.json', {
        'models': result, 'documents': 500, 'batch_size': arguments.batch_size,
        'scope': 'sum of synchronized batch forward calls, excluding preparation/loading',
        'protocol': 'new portable measurement, C followed by B, three passes per model',
        'historical_alternating_timing_replay': False,
        'data': data_identity, 'runtime': common.runtime(),
        'test_read': False, 'model_quality_metrics_recomputed': False})


def main(argv=None):
    command = parser()
    values = sys.argv[1:] if argv is None else argv
    if not values:
        command.print_help()
        return 0
    args = command.parse_args(values)
    for name, value in vars(args).items():
        if isinstance(value, Path):
            setattr(args, name, value.expanduser().resolve())
    if args.output_dir.exists():
        raise FileExistsError('Use a new output directory')
    with common.project_imports():
        if args.command == 'calibrate':
            calibrate(args)
        else:
            benchmark(args)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
