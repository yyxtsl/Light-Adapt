"""Portable preparation, target adaptation and evaluation for the adapted DWIE study."""

import argparse
import gc
import hashlib
import json
from pathlib import Path
import random
import statistics
import sys
import time

import reproduce_redocred as common


ROOT = Path(__file__).resolve().parent
COUNTS = {'train': 602, 'dev': 100, 'test': 100}


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument('--mode', choices=['validate', 'prepare', 'train', 'evaluate', 'benchmark'],
                        required=True)
    result.add_argument('--validate-for', choices=['prepare', 'train', 'evaluate', 'benchmark'],
                        default='evaluate', help='Operation whose paths are checked without model execution')
    result.add_argument('--variant', choices=['C', 'B', 'D', 'G'], default='B')
    result.add_argument('--model-dir', type=Path)
    result.add_argument('--data-dir', type=Path, help='Output of --mode prepare')
    result.add_argument('--annotations-dir', type=Path, help='Official annos_with_content directory')
    result.add_argument('--article-index', type=Path, help='Official data/article_id_to_url.json')
    result.add_argument('--checkpoint', type=Path, help='Source state for train, target state for evaluate')
    result.add_argument('--checkpoint-c', type=Path, help='Full-TTM target state for benchmark')
    result.add_argument('--checkpoint-b', type=Path, help='Light-Adapt target state for benchmark')
    result.add_argument('--expected-checkpoint-sha256')
    result.add_argument('--split', choices=['train', 'dev', 'test'], default='dev')
    result.add_argument('--include-test', action='store_true', help='Explicitly prepare the official test split')
    result.add_argument('--output-dir', type=Path)
    result.add_argument('--device', default='cuda:0')
    return result


def load_mapping():
    mapping = common.read_json(ROOT / 'metadata/dwie/rel2id.json')
    if len(mapping) != 63 or mapping.get('TH') != 0 or sorted(mapping.values()) != list(range(63)):
        raise ValueError('Expected the fixed 62 positive relations plus TH')
    return mapping


def validation(arguments):
    operation = arguments.validate_for if arguments.mode == 'validate' else arguments.mode
    if arguments.output_dir is not None and arguments.output_dir.exists():
        raise FileExistsError('Use a new output directory')
    required = []
    if operation == 'prepare':
        if arguments.annotations_dir is None or arguments.article_index is None:
            raise ValueError('Prepare requires --annotations-dir and --article-index')
        if not arguments.annotations_dir.is_dir():
            raise FileNotFoundError(arguments.annotations_dir)
        required.append(arguments.article_index)
    elif operation == 'benchmark':
        if arguments.split != 'dev':
            raise ValueError('Benchmark uses the fixed Dev split')
        if arguments.checkpoint_c is None or arguments.checkpoint_b is None:
            raise ValueError('Supply --checkpoint-c and --checkpoint-b')
        required.extend([arguments.checkpoint_c, arguments.checkpoint_b])
    elif operation in {'train', 'evaluate'}:
        if operation == 'evaluate' and arguments.split == 'train':
            raise ValueError('Evaluation supports Dev or Test')
        if operation == 'train' and arguments.split != 'dev':
            raise ValueError('Training selects checkpoints on Dev')
        if operation == 'train' and arguments.variant == 'G':
            if arguments.checkpoint is not None or arguments.expected_checkpoint_sha256:
                raise ValueError('Generic training uses --model-dir only')
        elif arguments.checkpoint is None:
            raise ValueError('Supply --checkpoint')
        if arguments.checkpoint is not None:
            required.append(arguments.checkpoint)
    if arguments.model_dir is None:
        raise ValueError('Supply --model-dir')
    required.extend([arguments.model_dir / 'config.json', arguments.model_dir / 'tokenizer.json'])
    if arguments.mode != 'validate' and arguments.output_dir is None:
        raise ValueError('Supply a new --output-dir')
    if operation in {'train', 'evaluate', 'benchmark'}:
        if arguments.data_dir is None:
            raise ValueError('Supply prepared --data-dir')
        splits = ['train', 'dev'] if operation == 'train' else [arguments.split]
        for split in splits:
            required.extend([arguments.data_dir / 'features' / (split + '.jsonl'),
                             arguments.data_dir / 'gold' / (split + '.json')])
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)
    config = common.read_json(arguments.model_dir / 'config.json')
    if config.get('model_type') != 'roberta' or config.get('hidden_size') != 1024:
        raise ValueError('Expected RoBERTa-large configuration')
    if arguments.expected_checkpoint_sha256:
        checksum = arguments.expected_checkpoint_sha256.lower()
        if len(checksum) != 64 or any(c not in '0123456789abcdef' for c in checksum):
            raise ValueError('Expected checkpoint digest must contain 64 hexadecimal characters')
    return {'operation': operation, 'files': [{'path': str(p), 'bytes': p.stat().st_size} for p in required],
            'mapping_classes': len(load_mapping()), 'model_execution': False}


def convert_annotation(annotation, expected_hash, is_train):
    import prepare_dwie_cpu_admission as conversion
    import evaluate_ttmre_dwie_fixed_test as evaluation
    content = annotation.get('content')
    if not isinstance(content, str):
        raise ValueError('Run the official article downloader first')
    if is_train:
        converted, _ = conversion.validate_and_convert(annotation, content, expected_hash)
        return converted
    _, converted = evaluation.test_convert(annotation, content, expected_hash)
    converted['content'] = content
    return converted


def prepare(arguments):
    from tokenizers import AddedToken, Tokenizer
    import ttmre_dwie_input as dwie
    mapping = load_mapping()
    split = common.read_json(ROOT / 'metadata/dwie/target_split.json')
    article_rows = common.read_json(arguments.article_index)
    article_index = {row['id']: row for row in article_rows}
    if len(article_index) != len(article_rows):
        raise ValueError('Duplicate official article index IDs')
    tokenizer_path = arguments.model_dir / 'tokenizer.json'
    if common.digest(tokenizer_path) != '847bbeab6174d66a88898f729d52fa8d355fafe1bea101cf960dd404581df70e':
        raise ValueError('Expected the pinned RoBERTa tokenizer')
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    tokenizer.no_truncation()
    tokenizer.no_padding()
    vocabulary_size = tokenizer.get_vocab_size()
    tokenizer.add_tokens([AddedToken('*', normalized=False)])
    if tokenizer.get_vocab_size() != vocabulary_size or tokenizer.token_to_id('*') != 3226:
        raise ValueError('Entity marker changed the vocabulary')
    if arguments.include_test:
        remaining = sorted(set(article_index) - set(split['train']) - set(split['dev']))
        if len(remaining) != 100:
            raise ValueError('Expected exactly 100 official test IDs')
        split['test'] = remaining
    for name, ids in split.items():
        if len(ids) != COUNTS[name] or len(set(ids)) != len(ids):
            raise ValueError('Unexpected complete split size')
    output = arguments.output_dir
    output.mkdir(parents=True, exist_ok=False)
    (output / 'features').mkdir()
    (output / 'gold').mkdir()
    summary = {}
    train_relations = set()
    for name, ids in split.items():
        documents = []
        audits = []
        with (output / 'features' / (name + '.jsonl')).open('x', encoding='utf-8') as stream:
            for ident in ids:
                annotation = common.read_json(arguments.annotations_dir / (ident + '.json'))
                if annotation['id'] != ident:
                    raise ValueError('Document ID differs from its filename')
                record = convert_annotation(annotation, article_index[ident]['hash'], name != 'test')
                if name == 'train':
                    train_relations.update(row['r'] for row in record['relations'])
                public = record if name == 'train' else {**record, 'relations': []}
                feature, audit = dwie.make_feature(public, tokenizer, mapping)
                stream.write(json.dumps(feature, ensure_ascii=False, separators=(',', ':')) + '\n')
                documents.append({key: record[key] for key in ['id', 'entities', 'relations']})
                audits.append({'id': ident, **audit})
        common.write_json(output / 'gold' / (name + '.json'), documents)
        summary[name] = {'documents': len(documents), 'audits': audits,
                         'feature_sha256': common.digest(output / 'features' / (name + '.jsonl')),
                         'gold_sha256': common.digest(output / 'gold' / (name + '.json'))}
    if train_relations != set(mapping) - {'TH'}:
        raise ValueError('Prepared train vocabulary differs from the fixed mapping')
    common.write_json(output / 'preparation.json', {
        'splits': summary, 'tokenizer_sha256': common.digest(tokenizer_path),
        'article_index_sha256': common.digest(arguments.article_index),
        'test_prepared': arguments.include_test, 'model_execution': False})
    common.emit('DWIE_preparation_complete', splits={k: v['documents'] for k, v in summary.items()})


def read_features(directory, split, training=False):
    features = []
    with (directory / 'features' / (split + '.jsonl')).open(encoding='utf-8') as stream:
        for line in stream:
            if line.strip():
                features.append(json.loads(line))
    documents = common.read_json(directory / 'gold' / (split + '.json'))
    if len(features) != COUNTS[split] or len(documents) != COUNTS[split]:
        raise ValueError('A complete study split is required')
    if [row['id'] for row in features] != [row['id'] for row in documents]:
        raise ValueError('Input and scoring document order differs')
    if len({row['id'] for row in features}) != len(features):
        raise ValueError('Duplicate prepared document IDs')
    fixed = common.read_json(ROOT / 'metadata/dwie/target_split.json')
    if split in fixed and [row['id'] for row in features] != fixed[split]:
        raise ValueError('Prepared split does not match the fixed train/Dev IDs')
    if split == 'test' and set(fixed['train'] + fixed['dev']) & {row['id'] for row in features}:
        raise ValueError('Test overlaps the fixed train/Dev IDs')
    if not training:
        features = [{'id': row['id'], 'inputs': row['inputs'], 'gold': []} for row in features]
    return features, documents


def configure_math():
    import torch
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = False


def build_model(arguments, evaluation):
    import torch
    import train_ttmre_dwie_adaptation as training
    import ttmre_dwie_gpu_probe_v4 as probe
    configure_math()
    device = torch.device(arguments.device)
    generic = arguments.variant == 'G' and not evaluation
    if generic:
        import train_ttmre_dwie_generic_budget_replacement as control
        model, report = control.build_generic({'model_config_dir': str(arguments.model_dir)}, device)
        return model, report
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(75)
        model, report = probe.build_model(arguments.model_dir)
    model.requires_grad_(True)
    if arguments.variant == 'D' or (evaluation and arguments.variant != 'C'):
        probe.replacement.install(model, 'bypass')
    if evaluation:
        probe.set_classifier(model, 63)
    state = torch.load(arguments.checkpoint, map_location='cpu', weights_only=True)
    probe.strict_load(model, state, 'explicit DWIE ' + arguments.variant + ' checkpoint')
    del state
    if not evaluation:
        if arguments.variant == 'B':
            probe.replacement.install(model, 'bypass')
        head, head_sha = training._target_head(model.bilinear.in_features)
        probe.set_classifier(model, 63, head)
        report['target_head_sha256'] = head_sha
    if arguments.variant == 'C':
        model.mu_encoder.memory_tokens.requires_grad_(False)
    model.args.device = device
    model.train_mode = 'finetune'
    model.to(device)
    gc.collect()
    expected = 379504199 if arguments.variant == 'C' else 361603647
    if sum(p.numel() for p in model.parameters()) != expected:
        raise ValueError('Unexpected DWIE parameter geometry')
    return model, report


def checkpoint_record(arguments):
    if arguments.checkpoint is None:
        return {'initialization': 'generic RoBERTa-large only'}
    checksum = common.digest(arguments.checkpoint)
    if arguments.expected_checkpoint_sha256 and checksum != arguments.expected_checkpoint_sha256.lower():
        raise ValueError('Checkpoint hash differs from the requested identity')
    return {'path': str(arguments.checkpoint), 'sha256': checksum}


def run_model(arguments):
    import torch
    import train_ttmre_dwie_adaptation as training
    mapping = load_mapping()
    checkpoint = checkpoint_record(arguments)
    model, report = build_model(arguments, evaluation=arguments.mode == 'evaluate')
    output = arguments.output_dir
    official = ROOT / 'refine-logs/SECOND_DATASET_ADMISSION_20261023/DWIE_OFFICIAL_METRICS_20261024.py'
    if arguments.mode == 'train':
        features, _ = read_features(arguments.data_dir, 'train', training=True)
        development, documents = read_features(arguments.data_dir, 'dev')
        result = training.train_arm(model, features, development, documents, mapping,
                                    torch.device(arguments.device), output,
                                    official_source=official)
    else:
        output.mkdir(parents=True, exist_ok=False)
        features, documents = read_features(arguments.data_dir, arguments.split)
        result, predictions = training.evaluate(model, features, documents, mapping,
                                                torch.device(arguments.device), official_source=official)
        common.write_json(output / 'predictions.json', [list(row) for row in predictions])
        common.write_json(output / 'metrics.json', result)
    common.write_json(output / 'execution.json', {
        'mode': arguments.mode, 'variant': arguments.variant, 'split': arguments.split,
        'checkpoint': checkpoint, 'implementation': report, 'runtime': common.runtime(),
        'test_read': arguments.mode == 'evaluate' and arguments.split == 'test'})
    common.emit('DWIE_execution_complete', mode=arguments.mode, variant=arguments.variant)


def benchmark(arguments):
    import torch
    import ttmre_dwie_gpu_probe_v4 as probe
    features, _ = read_features(arguments.data_dir, 'dev')
    mapping = load_mapping()
    output = arguments.output_dir
    output.mkdir(parents=True, exist_ok=False)
    result = {}
    for variant, checkpoint in [('C', arguments.checkpoint_c), ('B', arguments.checkpoint_b)]:
        arguments.variant, arguments.checkpoint = variant, checkpoint
        model, _ = build_model(arguments, evaluation=True)
        model.eval()
        device = torch.device(arguments.device)
        def sync():
            if device.type == 'cuda':
                torch.cuda.synchronize(device)
        with torch.inference_mode():
            probe.feature_logits(model, features[0], mapping, device)
            rounds = []
            for _ in range(3):
                sync()
                start = time.perf_counter()
                for feature in features:
                    logits, _, _ = probe.feature_logits(model, feature, mapping, device)
                    del logits
                sync()
                rounds.append(time.perf_counter() - start)
        result[variant] = {'seconds': rounds, 'median_seconds': statistics.median(rounds),
                           'checkpoint_sha256': common.digest(checkpoint)}
        del model
        gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    common.write_json(output / 'timing.json', {
        'models': result, 'documents': 100, 'scope': 'sequential one-document calls; window tensors and input transfer included',
        'protocol': 'new portable measurement, C followed by B, three passes per model',
        'historical_paired_rotating_measurement': False, 'runtime': common.runtime(),
        'F1_recomputed': False, 'test_read': False})


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
    result = validation(args)
    if args.mode == 'validate':
        print(json.dumps(result, indent=2))
        return 0
    with common.project_imports():
        if args.mode == 'prepare':
            prepare(args)
        elif args.mode == 'benchmark':
            benchmark(args)
        else:
            run_model(args)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
