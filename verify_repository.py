"""Check an extracted Light-Adapt release using Python's standard library."""

import ast
import hashlib
import json
import math
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parent


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def verify(root):
    manifest_path = root / 'RELEASE_MANIFEST.json'
    if not manifest_path.is_file():
        raise FileNotFoundError('Missing RELEASE_MANIFEST.json. Upload the complete repository contents.')
    manifest = read_json(manifest_path)
    records = manifest['files']
    missing = [name for name in records if not (root / name).is_file()]
    if missing:
        raise FileNotFoundError('Missing release files: ' + ', '.join(missing))
    changed = [name for name, record in records.items()
               if sha256(root / name) != record['sha256']]
    if changed:
        raise ValueError('Files differ from this release manifest: ' + ', '.join(changed))

    source = read_json(root / 'SOURCE_PROVENANCE.json')['files']
    for name, record in source.items():
        if sha256(root / name) != record['sha256']:
            raise ValueError('Recorded research source differs: ' + name)

    for name in records:
        path = root / name
        if path.suffix == '.py':
            ast.parse(path.read_text(encoding='utf-8-sig'), filename=name)

    checked_results = 0
    for name in ['redocred_single_models.json', 'redocred_B_average.json',
                 'redocred_C_average.json']:
        metrics = read_json(root / 'results' / name)
        rows = [metrics] if 'TP' in metrics else metrics.values()
        for row in rows:
            denominator = 2 * row['TP'] + row['FP'] + row['FN']
            expected = 100 * 2 * row['TP'] / denominator
            if not math.isclose(expected, row['test_F1'], rel_tol=0, abs_tol=1e-10):
                raise ValueError('Stored F1 differs from its counts in ' + name)
            checked_results += 1

    split = read_json(root / 'metadata/dwie/target_split.json')
    mapping = read_json(root / 'metadata/dwie/rel2id.json')
    if len(split['train']) != 602 or len(split['dev']) != 100:
        raise ValueError('Unexpected DWIE train/development split size')
    if len(set(split['train'])) != 602 or len(set(split['dev'])) != 100:
        raise ValueError('Duplicate DWIE split identifiers')
    if set(split['train']) & set(split['dev']):
        raise ValueError('Overlapping DWIE train/development identifiers')
    if len(mapping) != 63 or sorted(mapping.values()) != list(range(63)):
        raise ValueError('Unexpected DWIE relation vocabulary')
    return {
        'release_files_verified': len(records),
        'recorded_source_files_verified': len(source),
        'stored_F1_values_verified': checked_results,
        'DWIE_relations_including_TH': len(mapping),
        'model_execution': False,
        'release_manifest_self_hash': 'The manifest is verified separately by the distribution checksum.',
    }


def main():
    try:
        result = verify(ROOT)
    except (FileNotFoundError, ValueError, KeyError, SyntaxError) as error:
        print('Repository check failed: ' + str(error), file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
