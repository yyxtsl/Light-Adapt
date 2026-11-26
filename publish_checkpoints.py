"""Verify selected retained weights and optionally upload to a private model repository."""

import argparse
import getpass
import hashlib
import json
from pathlib import Path
import sys


def digest(path):
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            result.update(block)
    return result.hexdigest()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True,
                        help='Author-reviewed mapping with source_root and files path/hash entries')
    parser.add_argument('--source-root', type=Path)
    parser.add_argument('--ids', nargs='+', help='Checkpoint IDs; defaults to reviewed default_upload_subset')
    parser.add_argument('--repo-id', help='Hugging Face owner/model-repository')
    parser.add_argument('--model-card', type=Path)
    parser.add_argument('--upload', action='store_true', help='Explicitly create/use a private HF model repository and upload')
    args = parser.parse_args(argv)
    plan = json.loads(args.plan.read_text(encoding='utf-8'))
    root = (args.source_root or Path(plan['source_root'])).expanduser().resolve()
    selected = args.ids or plan['default_upload_subset']
    if not selected or len(set(selected)) != len(selected):
        raise ValueError('Choose distinct checkpoint IDs')
    by_id = {row['id']: row for row in plan['files']}
    if len(by_id) != len(plan['files']):
        raise ValueError('Duplicate checkpoint IDs in release plan')
    rows = []
    filenames = set()
    for ident in selected:
        if ident not in by_id:
            raise ValueError('Unknown checkpoint ID: ' + ident)
        row = by_id[ident]
        path = (root / row['path']).resolve()
        if not path.is_relative_to(root):
            raise ValueError('Checkpoint path leaves the specified source directory')
        if path.stat().st_size != row['bytes'] or digest(path) != row['sha256']:
            raise ValueError('Checkpoint size/hash differs: ' + ident)
        if Path(row['filename']).name != row['filename']:
            raise ValueError('Repository filename must be a plain basename')
        if row['filename'] in filenames or row['filename'] in {'README.md', 'CHECKPOINTS.json'}:
            raise ValueError('Duplicate or reserved repository filename')
        filenames.add(row['filename'])
        rows.append((row, path))
        print(json.dumps({'verified_checkpoint': ident, 'bytes': row['bytes']}), flush=True)
    if not args.upload:
        print(json.dumps({'verified': len(rows), 'upload': False, 'models_loaded': False}))
        return 0
    if not args.repo_id or args.repo_id.count('/') != 1 or not args.model_card:
        raise ValueError('Upload requires --repo-id owner/name and --model-card')
    if not args.model_card.is_file():
        raise FileNotFoundError(args.model_card)
    from huggingface_hub import HfApi
    token = getpass.getpass('Hugging Face write token (input hidden, kept in memory): ')
    if not token:
        raise ValueError('An authenticated write token is required')
    api = HfApi(token=token)
    api.create_repo(repo_id=args.repo_id, repo_type='model', private=True, exist_ok=True)
    info = api.repo_info(repo_id=args.repo_id, repo_type='model')
    if not info.private:
        raise ValueError('This preparation uploader requires a private repository')
    api.upload_file(path_or_fileobj=str(args.model_card), path_in_repo='README.md',
                    repo_id=args.repo_id, repo_type='model', commit_message='Add model provenance and usage')
    for row, path in rows:
        api.upload_file(path_or_fileobj=str(path), path_in_repo=row['filename'],
                        repo_id=args.repo_id, repo_type='model',
                        commit_message='Upload verified ' + row['id'] + ' checkpoint')
    metadata = {'checkpoints': [{k: row[k] for k in ['id', 'filename', 'sha256', 'bytes']}
                               for row, _ in rows]}
    api.upload_file(path_or_fileobj=(json.dumps(metadata, indent=2) + '\n').encode(),
                    path_in_repo='CHECKPOINTS.json', repo_id=args.repo_id, repo_type='model',
                    commit_message='Record uploaded checkpoint hashes')
    print(json.dumps({'repo_id': args.repo_id, 'uploaded_checkpoints': len(rows),
                      'private': True, 'visibility_changed': False, 'models_loaded': False}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
