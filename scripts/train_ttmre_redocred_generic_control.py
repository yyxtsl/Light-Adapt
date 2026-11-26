"""One pending Re-DocRED generic-only G, same Human/Dev protocol as B75.

Default validation is read-only and needs no cloud assets. No task checkpoint,
Distant training records, Test, download, installation, deletion or automatic
retry/resume/extension. Training requires a separately pinned approval.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tarfile
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts')]
DOC = ROOT / 'refine-logs/REDOCRED_GENERIC_CONTROL_20261025'
REFERENCE = DOC / 'REFERENCE.json'
REFERENCE_SHA = 'f9818e8bc8f6975c53cc9de65fb59590d76c57c570724c201f97b2789bba04c7'
AUTHORIZATION = DOC / 'AUTHORIZATION_APPROVED_20261025.json'
AUTHORIZATION_SHA = '8cf47647884084c1dad9c3057ecc18ac5d24dc21988762e9fed9e8bf852d888c'
SOURCE_MANIFEST = DOC / 'SOURCE_MANIFEST.json'
SOURCE_MANIFEST_SHA = 'f5e76219cf89669f2d94758e64709db022f5bfe3144833fc34024c8f8dfb8cbc'
RUN_ROOT = ROOT / 'experiments/2026-10-25-redocred-generic-control'
CLAIM = RUN_ROOT / 'generic75-once-claim.json'
HARD_SECONDS, SOFT_SECONDS, TERM_SECONDS = 7200, 120, 30
HUMAN_DOCS, DEV_DOCS, EPOCHS, SEED = 3053, 500, 30, 75
UPDATES = HUMAN_DOCS // 4 * EPOCHS
CHECKPOINT_BOUND, REPORT_MARGIN = 1453845935, 512 * 1024**2
METRICS = ('dev_F1', 'dev_precision', 'dev_recall', 'dev_IgnF1_annotated', 'dev_IgnF1_distant')
PROGRESS = {'phase': 'not-started', 'formal_optimizer_updates': 0}


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def emit(event, **fields):
    print(json.dumps({'event': event, **fields}, ensure_ascii=False, allow_nan=False), flush=True)


def interrupted(*_):
    raise TimeoutError('G process interrupted; no automatic retry/resume')


def check_budget(budget, reserve=SOFT_SECONDS):
    if budget is None:
        return
    if time.monotonic() >= budget['deadline_monotonic'] - reserve:
        raise TimeoutError('Original Re-DocRED G cap/reserve reached; no automatic extension')
    if os.name == 'posix' and Path('/proc/sys/kernel/random/boot_id').read_text().strip() != budget['boot_id']:
        raise RuntimeError('Boot changed; deadline cannot reset')


def digest(path, budget=None):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b''):
            check_budget(budget)
            value.update(block)
    return value.hexdigest()


def protocol():
    return {'version': 'redocred-generic-control-v1', 'arms': ['G'], 'seed': SEED,
        'epochs': EPOCHS, 'optimizer_updates': UPDATES, 'Human_documents': HUMAN_DOCS,
        'Dev_documents': DEV_DOCS, 'num_classes': 97, 'batch_size': 4, 'drop_last': True,
        'precision': 'float32', 'loss': 'original ATGL alpha16.5', 'backbone_lr': 3e-5,
        'head_lr': 1e-5, 'eps': 1e-6, 'weight_decay': .01, 'clip': 1., 'warmup_steps': int(UPDATES * .06),
        'initialization': 'pinned generic RoBERTa only; fresh original seed75 task constructor',
        'architecture': 'same noTTM shared-projector B/D; no new module', 'native_ATLOP': False,
        'selection': 'penultimate batch raw Dev F1; strict increase, earliest tie',
        'decoder': 'all relation logits > TH; no top-k cap',
        'checkpoint_policy': 'CPU best snapshots; one final best write and strict replay',
        'Distant_training_records_read': 0, 'task_source_tensor_loads': 0,
        'Distant_index_use': 'matching Dev facts for unchanged IgnF1_Distant only',
        'Test': False, 'new_Distant_updates': 0, 'reference_optimizer_updates': 0,
        'hard_seconds': HARD_SECONDS, 'soft_reserve_seconds': SOFT_SECONDS,
        'term_reserve_seconds': TERM_SECONDS, 'retry': False, 'resume': False,
        'extension': False, 'automatic_followup': False}


def authorization(require_approved=False):
    if digest(AUTHORIZATION) != AUTHORIZATION_SHA:
        raise ValueError('Authorization bytes changed')
    value = read(AUTHORIZATION)
    if (value['version'] != 'redocred-generic-authorization-v1'
            or (value['seed'], value['epochs'], value['updates'], value['arms']) != (75, 30, 22890, ['G'])
            or value['proposed_hard_seconds'] != HARD_SECONDS
            or any(value[k] is not False for k in ('Test', 'new_seeds', 'retry', 'resume', 'extension', 'deletion', 'download', 'install'))):
        raise ValueError('Authorization scope mismatch')
    if require_approved and (value['approved'] is not True or value['hard_seconds'] != HARD_SECONDS):
        raise PermissionError('New Re-DocRED G GPU budget is pending; no launch')
    if value['approved'] is False and value['hard_seconds'] is not None:
        raise ValueError('Pending authorization cannot allocate a clock')
    return value


def validate_local():
    auth = authorization()
    if digest(REFERENCE) != REFERENCE_SHA or digest(SOURCE_MANIFEST) != SOURCE_MANIFEST_SHA:
        raise ValueError('Reference/source manifest changed')
    pins = read(SOURCE_MANIFEST)
    for name, expected in pins.items():
        if digest(ROOT / name) != expected:
            raise ValueError('Pinned source changed: ' + name)
    return {'protocol': protocol(), 'authorization': auth, 'source_files': pins,
        'reference_sha256': REFERENCE_SHA, 'entry_sha256': digest(__file__),
        'CUDA_initialized': False, 'tensor_loads': 0, 'training_started': False}


def resources():
    from inventory_ttmre_redocred_generic_control import memory_metadata
    free = shutil.disk_usage(ROOT).free
    required = CHECKPOINT_BOUND + REPORT_MARGIN
    memory = memory_metadata()
    if free < required:
        raise RuntimeError('Insufficient disk for one new best plus margin; no deletion')
    if memory.get('effective_available_bytes') is None or memory['effective_available_bytes'] < 32 * 1024**3:
        raise RuntimeError('Effective available host/container RAM must be at least 32GiB')
    return {'free_bytes': free, 'required_bytes': required, 'memory': memory, 'files_deleted': 0}


def admit(budget):
    authorization(True)
    identity = validate_local()
    check_budget(budget)
    if CLAIM.exists():
        raise FileExistsError('Re-DocRED G already claimed; no repeat')
    identity['resources'] = resources()
    ref = read(REFERENCE)
    files = dict(ref['input_files'])
    files.update({ref['model_dir'] + '/' + name: sha for name, sha in ref['generic_files'].items()})
    for name, expected in files.items():
        if digest(ROOT / name, budget) != expected:
            raise ValueError('Cloud asset/cache changed: ' + name)
    from ttmre_nvidia_smi_preflight import inspect_gpu
    external = inspect_gpu()
    if not external['passed']:
        raise RuntimeError('External GPU occupancy exceeds 500MiB before claim/CUDA')
    identity.update(input_files=files, reference=ref, external_gpu_preflight=external,
                    authorization_sha256=AUTHORIZATION_SHA, budget_sha256=None)
    return identity


def modules():
    import train_ttmre_pair_replacement as prior
    return prior, prior.core, prior.base, prior.old


def data_context(workspace, identity, budget):
    _, core, _, _ = modules()
    ref = identity['reference']
    args = SimpleNamespace(run_dir=workspace, baseline_workspace=ROOT / ref['baseline_workspace'],
        model_dir=ROOT / ref['model_dir'], data_dir=ROOT / ref['data_dir'], budget=budget,
        device=core.baseline.control.runtime(ref['environment']))
    datasets = []
    for split, count, labels in (('human', HUMAN_DOCS, True), ('dev', DEV_DOCS, False)):
        directory = args.baseline_workspace / 'cache' / split
        metadata = read(directory / 'index.json')
        dataset = core.baseline.data.DiskFeatures(directory, metadata, labels)
        if metadata['documents'] != count or len(dataset) != count or metadata['with_labels'] is not labels:
            raise ValueError('Human/Dev cache schema/count/labels changed')
        datasets.append(dataset)
    human, dev = datasets
    gold = read(args.data_dir / 'dev_revised.json')
    if len(gold) != DEV_DOCS:
        raise ValueError('Dev gold count changed')
    # This legacy index is only an ignored-metric lookup; never open its features.bin.
    ignored = read(args.baseline_workspace / 'cache/distant/index.json')
    annotated = {tuple(v) for v in human.metadata['matching_dev_facts']}
    distant = {tuple(v) for v in ignored['matching_dev_facts']}
    return args, human, dev, gold, annotated, distant


def backbone_loader(model_dir, config):
    from transformers import AutoModel
    backbone, info = AutoModel.from_pretrained(str(model_dir), config=config, local_files_only=True,
        use_safetensors=True, output_loading_info=True)
    allowed = {p + n for p in ('', 'roberta.') for n in ('pooler.dense.weight', 'pooler.dense.bias')}
    if (set(info.get('missing_keys', [])) - allowed or info.get('mismatched_keys')
            or info.get('error_msgs') or any(not n.startswith('lm_head.') for n in info.get('unexpected_keys', []))):
        raise ValueError('Generic load differs beyond unused MLM/pooler: ' + repr(info))
    return backbone, info


def construct(args, human, model_factory=None):
    import numpy as np
    import torch
    import ttmre_pair_replacement as replacement
    prior, core, _, _ = modules()
    if model_factory is None:
        from transformers import AutoConfig, AutoTokenizer
        from model2 import DocREModel
        from ttmre_memory_fit import install_unused_write_bypass, install_ttm_checkpoint
        for name, value in core.baseline.aligned.protocol('pair-independent').items():
            setattr(args, name, value)
        args.seed = SEED
        config = AutoConfig.from_pretrained(str(args.model_dir), num_labels=97, local_files_only=True)
        tokenizer = AutoTokenizer.from_pretrained(str(args.model_dir), local_files_only=True)
        config.cls_token_id, config.sep_token_id, config.transformer_type = tokenizer.cls_token_id, tokenizer.sep_token_id, 'roberta'
        backbone, load = backbone_loader(args.model_dir, config)
        core.shared.seed_all(SEED, torch, np)
        model = DocREModel(args, config, torch.tensor(np.array(human.metadata['priors']) + 1e-9,
                            device=args.device), backbone.to(args.device), tokenizer).to(args.device)
        core.baseline.aligned.apply_variant(model, 'pair-independent')
        core.baseline.control.assert_original_read(model)
        if sum(p.numel() for p in model.parameters()) != 381175401:
            raise ValueError('Original generic task geometry changed')
        install_unused_write_bypass(model)
        install_ttm_checkpoint(model)
    else:
        model, load = model_factory(), {'synthetic_factory': True}
    before, rng = prior.common_digest(model), core.preflight.rng_state()
    model.requires_grad_(True)
    installation = replacement.install(model, 'bypass')
    gc.collect()
    core.preflight.assert_rng_equal(rng, core.preflight.rng_state())
    if prior.common_digest(model) != before or any(n.startswith('mu_encoder.') for n, _ in model.named_parameters()):
        raise ValueError('Bypass changed common fresh state or retains TTM parameters')
    if model_factory is None and sum(p.numel() for p in model.parameters()) != 363274849:
        raise ValueError('G structure differs from B/D')
    task = {n: v for n, v in model.state_dict().items() if not n.startswith(('model.', 'mu_encoder.'))}
    return model, {'generic_loading_info': load, 'replacement': installation,
        'fresh_task_state_sha256': prior.base.frozen.tensor_digest(task),
        'common_initial_state_sha256': before, 'task_constructor_seed': SEED,
        'task_source_tensor_loads': 0, 'all_task_parameters_fresh': True,
        'parameters': sum(p.numel() for p in model.parameters())}


@contextmanager
def training_scope(budget):
    _, core, base, _ = modules()
    saved = (core.training_forward, core.emit, core.aligned.emit, core.atomic_torch_save)
    snapshot = {}
    def save(value, path):
        if Path(path).name != 'best_model.pth':
            raise ValueError('Unexpected checkpoint target')
        snapshot.clear()
        snapshot.update(path=Path(path), state=value)
    def event(name, **fields):
        if name == 'N1_FULL_SEED75_LEARNING_UPDATE_VERIFIED':
            name = 'REDOCRED_G_LEARNING_UPDATE_VERIFIED'
        if 'step' in fields:
            PROGRESS['formal_optimizer_updates'] = fields['step']
        emit(name, **fields)
        if name == 'epoch_complete':
            remaining = budget['deadline_monotonic'] - time.monotonic() - SOFT_SECONDS
            estimate = 1.25 * fields['seconds'] * (EPOCHS - fields['epoch']) + 600
            emit('REDOCRED_G_REMAINING_PROJECTION', estimate_seconds=estimate, available_seconds=remaining)
            if estimate > remaining:
                raise TimeoutError('Measured remaining projection exceeds original two-hour cap')
    core.training_forward, core.emit, core.aligned.emit, core.atomic_torch_save = base.training_forward, event, event, save
    try:
        yield snapshot
    finally:
        core.training_forward, core.emit, core.aligned.emit, core.atomic_torch_save = saved


def check_selection(directory):
    rows = [json.loads(s) for s in (directory / 'metrics.jsonl').read_text().splitlines()]
    batches = HUMAN_DOCS // 4
    if len(rows) != EPOCHS or any(r['epoch'] != k or r['step'] != k * batches - 1
            or r['batch'] != batches - 1 for k, r in enumerate(rows, 1)):
        raise ValueError('Epoch count or penultimate selection timing changed')
    selected, best = max(rows, key=lambda r: r['dev_F1']), read(directory / 'best_metrics.json')
    if any(selected[k] != best[k] for k in ('epoch', 'step', *METRICS)):
        raise ValueError('Best must be earliest raw-F1 maximum')
    return best


def execute(workspace, identity, budget):
    import numpy as np
    import torch
    from torch.utils.data import DataLoader
    from transformers.optimization import get_linear_schedule_with_warmup
    prior, core, base, old = modules()
    args, human, dev, gold, annotated, distant = data_context(workspace, identity, budget)
    directory = workspace / 'G'
    directory.mkdir()
    model, initial = construct(args, human)
    write(directory / 'initialization.json', initial)
    memory = model.mu_encoder.memory_tokens.detach().cpu().clone()
    hook = model.register_forward_pre_hook(lambda *_: check_budget(budget))
    snapshot = None
    try:
        write(directory / 'full_model_check.json', base.probe(model, human, args, budget))
        expected = identity['reference']['pairing']
        optimizer = core.baseline.optimizer_for_stage(model, 'human-frozen')
        groups = core.baseline.optimizer_report(model, optimizer)
        if optimizer.state or groups != expected['optimizer_groups']:
            raise ValueError('Fresh optimizer/groups differ from B75')
        core.shared.seed_all(SEED, torch, np)
        loader = DataLoader(human, batch_size=4, shuffle=True, drop_last=True, collate_fn=core.shared.collate_fn)
        dl = DataLoader(dev, batch_size=4, shuffle=False, collate_fn=core.shared.collate_fn)
        scheduler = get_linear_schedule_with_warmup(optimizer, int(UPDATES * .06), UPDATES)
        initial_rng = core.rng_digest()
        if len(loader) * EPOCHS != UPDATES or initial_rng != expected['initial_rng_sha256']:
            raise ValueError('Initial training RNG/update count differs from B75')
        write(directory / 'manifest.json', {'protocol': protocol(), 'optimizer_groups': groups,
            'initial_rng_sha256': initial_rng, 'total_steps': UPDATES})
        with training_scope(budget) as snapshot:
            result = core.train_loop(model, loader, dl, args, 'generic', 'human-frozen', directory,
                gold, annotated, distant, core.method.frequency_buckets(human.metadata['priors']), budget, optimizer, scheduler)
        del optimizer, scheduler
        best = check_selection(directory)
        pairing = {'sample_order': result['sample_order_sha256'] == expected['sample_order_sha256'],
                   'final_rng': result['final_rng_sha256'] == expected['final_rng_sha256']}
        if result['optimizer_steps'] != UPDATES or not all(pairing.values()):
            raise ValueError('Actual order/RNG/update count differs from B75')
        write(directory / 'training_order.json', {**result, 'same_as_B75': pairing})
        if not snapshot or snapshot['path'] != directory / 'best_model.pth' or snapshot['path'].exists():
            raise ValueError('Missing/conflicting CPU best')
        check_budget(budget)
        core.atomic_torch_save(snapshot['state'], directory / 'best_model.pth')
        snapshot = None
        model.zero_grad(set_to_none=True)
        model.load_state_dict(torch.load(directory / 'best_model.pth', map_location='cpu', weights_only=True), strict=True)
        replay, pred = old.grouped_replay(model, dev, args, budget, best, memory)
        score = core.shared.score_predictions(pred, gold, annotated, distant)
        replay['metrics'] = score
        replay['metric_replay_passed'] = all(math.isclose(score[k], best[k], abs_tol=1e-6, rel_tol=0) for k in METRICS)
        replay['passed'] = replay['passed'] and replay['metric_replay_passed']
        write(directory / 'checkpoint_replay.json', replay)
        if not replay['passed'] or not torch.equal(memory, model.mu_encoder.memory_tokens.detach().cpu()):
            raise ValueError('Checkpoint metric/grouping/state replay failed')
        old.save_prediction_report(directory, pred, [{'title': f['title'], 'hts': f['hts']} for f in dev], score, gold, human)
        result.update(passed=True, protocol=protocol(), best=best, best_checkpoint_writes=1,
            checkpoint_sha256=digest(directory / 'best_model.pth', budget), checkpoint_replay_passed=True,
            memory_buffer_unchanged=True, same_as_B75=pairing, task_source_tensor_loads=0)
        write(directory / 'completion.json', result)
        return result
    except BaseException:
        # Preserve an available incomplete best when time/space allow; it is not resumable.
        if snapshot and not snapshot['path'].exists() and time.monotonic() < budget['deadline_monotonic'] - 60:
            needed = sum(v.numel() * v.element_size() for v in snapshot['state'].values()) + 64 * 1024**2
            if shutil.disk_usage(directory).free >= needed:
                core.atomic_torch_save(snapshot['state'], snapshot['path'])
                write(directory / 'incomplete_best_preserved.json', {'complete': False, 'resumable': False})
        raise
    finally:
        PROGRESS['formal_optimizer_updates'] = int(getattr(model, '_n1_completed_optimizer_steps', 0))
        hook.remove()


def require_lock():
    if os.name != 'posix' or os.environ.get('TTMRE_REDOCRED_G_LOCK_FD') != '9':
        raise RuntimeError('Use the sole Re-DocRED G launcher')
    import fcntl
    actual, expected = os.fstat(9), (ROOT / 'experiments/2026-10-08-baseline-alignment/.training.lock').stat()
    if (actual.st_dev, actual.st_ino) != (expected.st_dev, expected.st_ino):
        raise RuntimeError('Descriptor9 is not the shared training lock')
    fcntl.flock(9, fcntl.LOCK_EX | fcntl.LOCK_NB)
    return 9


def verify_context(workspace, budget):
    authorization(True)
    claim = read(CLAIM)
    if (workspace.name != 'workspace' or workspace != workspace.resolve() or workspace.parent.parent != RUN_ROOT.resolve()
            or claim['workspace'] != str(workspace) or claim['protocol'] != protocol()
            or claim['budget_sha256'] != digest(workspace / 'budget.json')
            or claim['identity_sha256'] != digest(workspace / 'identity.json')
            or claim['authorization_sha256'] != AUTHORIZATION_SHA
            or budget['hard_seconds'] != HARD_SECONDS or any(budget[k] is not False for k in ('retry', 'resume', 'extension'))
            or not math.isclose(budget['deadline_monotonic'] - budget['start_monotonic'], HARD_SECONDS, abs_tol=1e-6, rel_tol=0)):
        raise ValueError('Changed immutable G context')
    check_budget(budget)


def archive(workspace):
    target = workspace / 'reports.tar.gz'
    with tarfile.open(target, 'x:gz') as stream:
        for path in sorted(workspace.rglob('*')):
            if path.is_file() and path.suffix in ('.json', '.jsonl', '.log'):
                stream.add(path, arcname=path.relative_to(workspace).as_posix())


def verify_completed_arm(workspace, identity, budget):
    directory = workspace / 'G'
    result = read(directory / 'completion.json')
    learning = read(directory / 'learning_update_check.json')
    replay = read(directory / 'checkpoint_replay.json')
    order = read(directory / 'training_order.json')
    best = check_selection(directory)
    expected = identity['reference']['pairing']
    if (result['passed'] is not True or result['protocol'] != protocol() or result['optimizer_steps'] != UPDATES
            or result['best_checkpoint_writes'] != 1 or result['checkpoint_replay_passed'] is not True
            or result['memory_buffer_unchanged'] is not True or result['task_source_tensor_loads'] != 0
            or result['checkpoint_sha256'] != digest(directory / 'best_model.pth', budget)
            or learning['passed'] is not True or learning['parameter_changed'] is not True or learning['lr'] <= 0
            or learning['step'] != learning['optimizer_state_step'] or learning['extra_optimizer_updates'] != 0
            or replay['passed'] is not True or order['optimizer_steps'] != UPDATES
            or any(order[k] != expected[k] for k in ('sample_order_sha256', 'final_rng_sha256'))
            or any(not math.isclose(result['best'][k], best[k], abs_tol=1e-6, rel_tol=0) for k in METRICS)):
        raise ValueError('Incomplete or changed G arm')
    return result


def worker(workspace):
    authorization(True)
    require_lock()
    budget = read(workspace / 'budget.json')
    verify_context(workspace, budget)
    signal.signal(signal.SIGTERM, interrupted)
    identity = read(workspace / 'identity.json')
    PROGRESS['phase'] = 'train'
    try:
        current = validate_local()
        if current['source_files'] != identity['source_files'] or current['entry_sha256'] != identity['entry_sha256']:
            raise ValueError('Worker source changed since admission')
        for name, expected in identity['input_files'].items():
            if digest(ROOT / name, budget) != expected:
                raise ValueError('Worker input changed')
        resources()
        from ttmre_nvidia_smi_preflight import inspect_gpu
        if not inspect_gpu()['passed']:
            raise RuntimeError('GPU occupied before worker CUDA initialization')
        from ttmre_dwie_runtime_fix import runtime_identity
        runtime = runtime_identity()
        write(workspace / 'runtime.json', runtime)
        execute(workspace, identity, budget)
        result = verify_completed_arm(workspace, identity, budget)
        references = identity['reference']['reference_metrics']
        comparison = {name: {k: result['best'][k] - row[k] for k in METRICS} for name, row in references.items()}
        done = {'passed': True, 'marker': 'REDOCRED_GENERIC_CONTROL_COMPLETE', 'protocol': protocol(),
            'optimizer_updates': result['optimizer_steps'], 'best': result['best'],
            'G_minus_reference_pp': comparison, 'references': references,
            'elapsed_seconds': time.monotonic() - budget['start_monotonic'], 'Test': False,
            'claim_scope': 'single seed Dev control, no stable superiority or source specificity conclusion'}
        check_budget(budget)
        write(workspace / 'completion.json', done)
        archive(workspace)
        check_budget(budget)
        emit('REDOCRED_GENERIC_CONTROL_COMPLETE', **done, reports=str(workspace / 'reports.tar.gz'))
    except BaseException as error:
        write(workspace / 'failure.json', {'error': str(error), 'type': type(error).__name__,
            'progress': PROGRESS, 'complete': False, 'retry': False, 'resumable': False})
        raise


def supervise(command, budget, fd):
    child = subprocess.Popen(command, pass_fds=(fd,), start_new_session=True)
    terminated = False
    try:
        while True:
            check_budget(budget, reserve=0)
            code = child.poll()
            if code is not None:
                if code or terminated:
                    raise RuntimeError('CPU checker/worker failed; no automatic retry')
                return
            if time.monotonic() >= budget['deadline_monotonic'] - TERM_SECONDS and not terminated:
                os.killpg(child.pid, signal.SIGTERM)
                terminated = True
            time.sleep(.2)
    finally:
        if child.poll() is None:
            os.killpg(child.pid, signal.SIGKILL)
            child.wait()


def budget_from_environment():
    start = float(os.environ['TTMRE_REDOCRED_G_START'])
    budget = {'start_monotonic': start, 'deadline_monotonic': start + HARD_SECONDS,
        'hard_seconds': HARD_SECONDS, 'boot_id': os.environ['TTMRE_REDOCRED_G_BOOT'],
        'start_utc': os.environ['TTMRE_REDOCRED_G_UTC'], 'retry': False, 'resume': False, 'extension': False}
    if not math.isfinite(start) or start > time.monotonic():
        raise ValueError('Invalid launcher clock')
    check_budget(budget)
    return budget


def start(workspace):
    authorization(True)
    fd = require_lock()
    budget = budget_from_environment()
    if workspace.name != 'workspace' or workspace.parent.parent != RUN_ROOT.resolve() or workspace.exists():
        raise ValueError('Use one new dated workspace')
    signal.signal(signal.SIGTERM, interrupted)
    try:
        identity = admit(budget)
        supervise([sys.executable, '-B', str(ROOT / 'scripts/check_ttmre_redocred_generic_control.py')], budget, fd)
        check_budget(budget)
        workspace.mkdir()
        write(workspace / 'budget.json', budget)
        write(workspace / 'identity.json', identity)
        write(CLAIM, {'workspace': str(workspace), 'protocol': protocol(), 'budget_sha256': digest(workspace / 'budget.json'),
            'identity_sha256': digest(workspace / 'identity.json'), 'authorization_sha256': AUTHORIZATION_SHA})
        command = [sys.executable, '-u', '-B', str(Path(__file__).resolve()), '--mode', 'worker', '--run-dir', str(workspace)]
        supervise(command, budget, fd)
    except BaseException as error:
        write(workspace.parent / 'supervisor-failure.json', {'error': str(error), 'type': type(error).__name__,
            'complete': False, 'retry': False, 'resumable': False, 'budget': budget,
            'elapsed_seconds': time.monotonic() - budget['start_monotonic']})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('validate', 'authorize', 'start', 'worker'), default='validate')
    parser.add_argument('--run-dir', type=Path)
    args = parser.parse_args()
    if args.mode == 'validate':
        print(json.dumps(validate_local(), indent=2))
    elif args.mode == 'authorize':
        authorization(True)
        print(json.dumps({'approved': True, 'hard_seconds': HARD_SECONDS}))
    elif args.run_dir is None:
        raise ValueError('Use sole launcher; workspace required')
    elif args.mode == 'start':
        start(args.run_dir.resolve())
    else:
        worker(args.run_dir.resolve())


if __name__ == '__main__':
    main()
