"""One authorized C/C-off/B Dev inference control, zero optimizer, total900s.

Default validate is local/read-only. Old models/data/sources/reports are never
modified. No Test, training, new seed, model saving, download or automatic retry.
"""
from __future__ import annotations
import argparse
from contextlib import contextmanager
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import signal
import statistics
import subprocess
import sys
import tarfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts')]
DOC = ROOT / 'refine-logs/C_OFF_CONTROL_PLAN_20261025'
AUTH = DOC / 'AUTHORIZATION_APPROVED.json'
AUTH_SHA = '8397b244304bcecf73b505502f4a500b338649e8a9d2459b67e34c6e6c7a0bd8'
REFERENCE = DOC / 'REFERENCE.json'
REFERENCE_SHA = '88c2c1587decf7db6547b2805bd627ae7ed3120f333078345ab382c7b5f0df7a'
PINS = DOC / 'SOURCE_MANIFEST.json'
PINS_SHA = '96634a65911b926fb39cf8ea0402c048ebc32f9c6fff6a808e3cf57f3355ebf2'
RUN_ROOT = ROOT / 'experiments/2026-10-25-c-off-control'
CLAIM = RUN_ROOT / 'c-off-once-claim.json'
HARD_SECONDS, RESERVE, TERM = 900, 120, 30
DEV_DOCS = 500
ORDER = (('C', 'C-off', 'B'), ('C-off', 'B', 'C'), ('B', 'C', 'C-off'))
METRICS = ('dev_F1', 'dev_precision', 'dev_recall', 'dev_IgnF1_annotated', 'dev_IgnF1_distant')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    with Path(path).open('x', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')


def emit(event, **fields):
    print(json.dumps({'event': event, **fields}, allow_nan=False), flush=True)


def check(budget, reserve=RESERVE):
    if budget is None:
        return
    if time.monotonic() >= budget['deadline_monotonic'] - reserve:
        raise TimeoutError('Original900s inference budget reached; no extension')
    if os.name == 'posix' and Path('/proc/sys/kernel/random/boot_id').read_text().strip() != budget['boot_id']:
        raise RuntimeError('Boot changed; clock cannot reset')


def sha(path, budget=None):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024**2), b''):
            check(budget)
            h.update(block)
    return h.hexdigest()


def authorize():
    if sha(AUTH) != AUTH_SHA:
        raise ValueError('Authorization bytes changed')
    a = read(AUTH)
    if a['approved'] is not True or a['hard_seconds'] != 900:
        raise PermissionError('C-off900s permission missing')
    if ((a['configs'], a['rounds'], a['Dev_documents'], a['optimizer_updates'], a['seed']) != (['C', 'C-off', 'B'], 3, 500, 0, 75)
            or any(a[k] is not False for k in ('Test', 'retry', 'resume', 'extension', 'checkpoint_writes', 'download', 'install', 'delete'))):
        raise ValueError('Authorization scope differs')
    return a


def protocol():
    return {'version': 'C-off-v1', 'configs': ['C', 'C-off', 'B'], 'seed': 75, 'Dev_documents': DEV_DOCS,
        'round_order': [list(v) for v in ORDER], 'passes': 9, 'batch_size': 4, 'precision': 'float32',
        'warmup_batches': 3, 'optimizer_updates': 0, 'checkpoint_writes': 0, 'Test': False,
        'operation': 'C-off loads completed C strictly, then existing identity pair replacement; all other weights unchanged',
        'decoder': 'all relation logits>TH; no top-k', 'metrics': list(METRICS),
        'primary_timing': 'median3 synchronized full-Dev model-forward passes; load/preparation/host transfer excluded',
        'secondary_timing': 'full pass including collate/transfer/decode, load separately recorded',
        'hard_seconds': HARD_SECONDS, 'retry': False, 'resume': False, 'extension': False}


def validate():
    a = authorize()
    if sha(REFERENCE) != REFERENCE_SHA or sha(PINS) != PINS_SHA:
        raise ValueError('Reference/source manifest changed')
    pins = read(PINS)
    for name, expected in pins.items():
        if sha(ROOT / name) != expected:
            raise ValueError('Pinned source changed: ' + name)
    ref = read(REFERENCE)
    for kind in ('C', 'B'):
        if set(METRICS) - set(ref['reference_metrics'][kind]):
            raise ValueError('Incomplete actual reference metrics')
    return {'protocol': protocol(), 'authorization': a, 'authorization_sha256': AUTH_SHA,
        'source_files': pins, 'reference': ref, 'reference_sha256': REFERENCE_SHA,
        'entry_sha256': sha(__file__), 'CUDA_initialized': False, 'new_optimizer_updates': 0}


def resources():
    from inventory_ttmre_redocred_generic_control import memory_metadata
    mem = memory_metadata()
    if not mem.get('effective_available_bytes') or mem['effective_available_bytes'] < 16 * 1024**3:
        raise RuntimeError('Need16GiB effective host memory; no environment change')
    free = shutil.disk_usage(ROOT).free
    if free < 256 * 1024**2:
        raise RuntimeError('Need256MiB report margin; no cleanup')
    return {'memory': mem, 'free_bytes': free, 'required_bytes': 256 * 1024**2}


def admit(budget):
    identity = validate()
    check(budget)
    if CLAIM.exists():
        raise FileExistsError('C-off already claimed; no repeat')
    identity['resources'] = resources()
    ref = identity['reference']
    inputs = dict(ref['input_files'])
    inputs.update({ref['model_dir'] + '/' + n: h for n, h in ref['generic_files'].items()})
    for group in ('checkpoints', 'reference_predictions'):
        inputs.update({v['path']: v['sha256'] for v in ref[group].values()})
    for n, expected in inputs.items():
        if sha(ROOT / n, budget) != expected:
            raise ValueError('Input/checkpoint changed: ' + n)
    from ttmre_nvidia_smi_preflight import inspect_gpu
    external = inspect_gpu()
    if not external['passed']:
        raise RuntimeError('GPU occupancy exceeds500MiB before CUDA/claim')
    identity.update(input_files=inputs, external_GPU=external)
    return identity


def modules():
    import train_ttmre_pair_replacement as prior
    import train_ttmre_redocred_generic_control as generic
    return prior, generic, prior.core


def facts(predictions):
    return {(v['title'], v['h_idx'], v['t_idx'], v['r']) for v in predictions}


def build(args, ref, kind, factory=None, loader=None):
    import torch
    import ttmre_pair_replacement as method
    prior, _, core = modules()
    if kind not in ('C', 'C-off', 'B'):
        raise ValueError('Unknown configuration')
    model = factory() if factory else core.load_generic_model_seed75(args, ref['baseline_identity'])
    source = 'B' if kind == 'B' else 'C'
    get = loader or (lambda path: torch.load(path, map_location='cpu', weights_only=True))
    if kind == 'B':
        model.requires_grad_(True)
        method.install(model, 'bypass')
    model.load_state_dict(get(ROOT / ref['checkpoints'][source]['path']), strict=True)
    before = prior.common_digest(model)
    memory = model.mu_encoder.memory_tokens.detach().clone()
    rng = core.preflight.rng_state()
    replacement = None
    if kind == 'C-off':
        model.requires_grad_(True)
        replacement = method.install(model, 'bypass')
        gc.collect()
    core.preflight.assert_rng_equal(rng, core.preflight.rng_state())
    if prior.common_digest(model) != before or not torch.equal(memory, model.mu_encoder.memory_tokens):
        raise ValueError('Branch removal changed common state or provenance memory')
    count = sum(p.numel() for p in model.parameters())
    if count != ref['parameters'][kind]:
        raise ValueError('Unexpected model geometry')
    model.requires_grad_(False)
    model.eval()
    return model, {'kind': kind, 'loaded_checkpoint': source, 'common_state_sha256': before,
        'common_state_preserved': True, 'RNG_preserved': True, 'memory_preserved': True,
        'replacement': replacement, 'parameters': count, 'optimizer_updates': 0}


@contextmanager
def no_training():
    from unittest.mock import patch
    import torch
    def reject(*_, **__):
        raise RuntimeError('Inference control forbids optimizer/backward/model saving')
    with patch.object(torch.optim.Optimizer, '__init__', reject), patch.object(torch.Tensor, 'backward', reject), \
         patch.object(torch.autograd, 'backward', reject), patch.object(torch, 'save', reject):
        yield


def execute(workspace, identity, budget, factory=None, loader=None):
    import torch
    prior, generic, core = modules()
    args, _, dev, gold, annotated, distant = generic.data_context(workspace, identity, budget)
    ref = identity['reference']
    if len(dev) != DEV_DOCS:
        raise ValueError('Use complete fixed Dev')
    outputs = []
    fixed_predictions = {kind: read(ROOT / v['path']) for kind, v in ref['reference_predictions'].items()}
    seen = {}
    for round_number, order in enumerate(ORDER, 1):
        for kind in order:
            check(budget)
            started = time.perf_counter()
            with no_training():
                model, construction = build(args, ref, kind, factory, loader)
                loaded = time.perf_counter()
                all_before = core.preflight.state_digest(model)
                rng = core.preflight.rng_state()
                with torch.no_grad():
                    benchmark, predictions = prior.benchmark_pass(model, dev, args, budget)
                core.preflight.assert_rng_equal(rng, core.preflight.rng_state())
                if all_before != core.preflight.state_digest(model) or any(p.grad is not None for p in model.parameters()):
                    raise ValueError('Inference modified model state/gradients')
                metrics = core.shared.score_predictions(predictions, gold, annotated, distant)
                decisions = facts(predictions)
                if kind in fixed_predictions:
                    if decisions != facts(fixed_predictions[kind]) or any(not math.isclose(metrics[k], ref['reference_metrics'][kind][k], abs_tol=1e-6, rel_tol=0) for k in METRICS):
                        raise ValueError('C/B replay disagrees with original archived predictions/metrics')
                if kind in seen and decisions != seen[kind]:
                    raise ValueError('Predictions changed across timing rounds')
                seen[kind] = decisions
                row = {'round': round_number, 'kind': kind, 'construction': construction,
                    'benchmark': benchmark, 'metrics': metrics, 'state_unchanged': True,
                    'RNG_unchanged': True, 'load_seconds': loaded - started, 'optimizer_updates': 0}
                stem = 'round%d-%s' % (round_number, kind)
                write(workspace / (stem + '.json'), row)
                write(workspace / (stem + '-predictions.json'), predictions)
                outputs.append(row)
                emit('C_OFF_PASS_COMPLETE', round=round_number, kind=kind, F1=metrics['dev_F1'],
                     forward_seconds=benchmark['model_forward_seconds'], passes_done=len(outputs), optimizer_updates=0)
                del model
                gc.collect()
                if args.device.type == 'cuda':
                    torch.cuda.empty_cache()
            elapsed = time.perf_counter() - started
            available = budget['deadline_monotonic'] - time.monotonic() - RESERVE
            estimate = 1.5 * elapsed * (9 - len(outputs)) + 30
            emit('C_OFF_PROJECTION', estimate_seconds=estimate, available_seconds=available)
            if estimate > available:
                raise TimeoutError('Measured remaining work exceeds original900s clock')
    if len(outputs) != 9:
        raise ValueError('Incomplete balanced3rounds')
    summary = {}
    for kind in ('C', 'C-off', 'B'):
        rows = [v for v in outputs if v['kind'] == kind]
        summary[kind] = {'metrics': rows[0]['metrics'], 'model_forward_seconds_median3': statistics.median(v['benchmark']['model_forward_seconds'] for v in rows),
            'end_to_end_seconds_median3': statistics.median(v['benchmark']['end_to_end_seconds'] for v in rows),
            'peak_allocated_bytes_max3': max(v['benchmark']['peak_allocated_bytes'] for v in rows),
            'peak_reserved_bytes_max3': max(v['benchmark']['peak_reserved_bytes'] for v in rows),
            'parameters': rows[0]['construction']['parameters'], 'rounds': 3}
    write(workspace / 'dev_pair_order.json', [{'title': f['title'], 'hts': f['hts']} for f in dev])
    return {'passed': True, 'protocol': protocol(), 'passes_completed': 9, 'summary': summary,
        'B_minus_Coff_F1_pp': summary['B']['metrics']['dev_F1'] - summary['C-off']['metrics']['dev_F1'],
        'Coff_minus_C_F1_pp': summary['C-off']['metrics']['dev_F1'] - summary['C']['metrics']['dev_F1'],
        'optimizer_updates': 0, 'Test': False, 'checkpoint_writes': 0,
        'scope': 'single fixed seed Dev, direct removal versus target readaptation; no indispensability claim'}


def require_lock():
    if os.name != 'posix' or os.environ.get('TTMRE_C_OFF_LOCK_FD') != '9':
        raise RuntimeError('Use the sole C-off launcher')
    import fcntl
    a, b = os.fstat(9), (ROOT / 'experiments/2026-10-08-baseline-alignment/.training.lock').stat()
    if (a.st_dev, a.st_ino) != (b.st_dev, b.st_ino):
        raise ValueError('Wrong shared lock descriptor')
    fcntl.flock(9, fcntl.LOCK_EX | fcntl.LOCK_NB)
    return 9


def stopped(*_):
    raise TimeoutError('Inference interrupted; no retry/extension')


def budget_from_env():
    start = float(os.environ['TTMRE_C_OFF_START'])
    b = {'start_monotonic': start, 'deadline_monotonic': start + HARD_SECONDS, 'hard_seconds': HARD_SECONDS,
         'boot_id': os.environ['TTMRE_C_OFF_BOOT'], 'start_utc': os.environ['TTMRE_C_OFF_UTC'],
         'retry': False, 'resume': False, 'extension': False}
    if not math.isfinite(start) or start > time.monotonic():
        raise ValueError('Invalid original launcher clock')
    check(b)
    return b


def supervise(command, budget, fd):
    child = subprocess.Popen(command, start_new_session=True, pass_fds=(fd,))
    terminated = False
    try:
        while True:
            check(budget, reserve=0)
            code = child.poll()
            if code is not None:
                if code or terminated:
                    raise RuntimeError('CPU checker/worker failed; no automatic retry')
                return
            if time.monotonic() >= budget['deadline_monotonic'] - TERM and not terminated:
                os.killpg(child.pid, signal.SIGTERM)
                terminated = True
            time.sleep(.2)
    finally:
        if child.poll() is None:
            os.killpg(child.pid, signal.SIGKILL)
            child.wait()


def verify_context(workspace, budget):
    authorize()
    c = read(CLAIM)
    if (workspace.name != 'workspace' or workspace != workspace.resolve() or workspace.parent.parent != RUN_ROOT.resolve()
            or c['workspace'] != str(workspace) or c['protocol'] != protocol()
            or c['budget_sha256'] != sha(workspace / 'budget.json') or c['identity_sha256'] != sha(workspace / 'identity.json')
            or c['authorization_sha256'] != AUTH_SHA or budget['hard_seconds'] != 900
            or not math.isclose(budget['deadline_monotonic'] - budget['start_monotonic'], 900, abs_tol=1e-6, rel_tol=0)
            or any(budget[k] is not False for k in ('retry', 'resume', 'extension'))):
        raise ValueError('Changed original inference context')
    check(budget)


def worker(workspace):
    authorize()
    require_lock()
    budget = read(workspace / 'budget.json')
    verify_context(workspace, budget)
    signal.signal(signal.SIGTERM, stopped)
    identity = read(workspace / 'identity.json')
    try:
        current = validate()
        if current['source_files'] != identity['source_files'] or current['entry_sha256'] != identity['entry_sha256']:
            raise ValueError('Worker source changed')
        resources()
        from ttmre_nvidia_smi_preflight import inspect_gpu
        if not inspect_gpu()['passed']:
            raise RuntimeError('GPU occupied before worker CUDA')
        for name, expected in identity['input_files'].items():
            if sha(ROOT / name, budget) != expected:
                raise ValueError('Worker input changed')
        from ttmre_dwie_runtime_fix import runtime_identity
        write(workspace / 'runtime.json', runtime_identity())
        protected = {str(ROOT / v['path']): (ROOT / v['path']).stat() for v in identity['reference']['checkpoints'].values()}
        result = execute(workspace, identity, budget)
        for name, before in protected.items():
            now = Path(name).stat()
            if (now.st_ino, now.st_size, now.st_mtime_ns) != (before.st_ino, before.st_size, before.st_mtime_ns):
                raise ValueError('Original checkpoint changed')
        check(budget)
        result['elapsed_seconds'] = time.monotonic() - budget['start_monotonic']
        write(workspace / 'completion.json', result)
        with tarfile.open(workspace / 'reports.tar.gz', 'x:gz') as archive:
            for p in sorted(workspace.glob('*.json')):
                check(budget)
                archive.add(p, arcname=p.name)
        check(budget)
        emit('C_OFF_COMPLETE', reports=str(workspace / 'reports.tar.gz'), **result)
    except BaseException as error:
        write(workspace / 'failure.json', {'error': str(error), 'type': type(error).__name__, 'complete': False,
              'optimizer_updates': 0, 'retry': False, 'resumable': False, 'elapsed_seconds': time.monotonic() - budget['start_monotonic']})
        raise


def start(workspace):
    authorize()
    fd = require_lock()
    budget = budget_from_env()
    if workspace.name != 'workspace' or workspace.parent.parent != RUN_ROOT.resolve() or workspace.exists():
        raise ValueError('Use new dated C-off workspace')
    signal.signal(signal.SIGTERM, stopped)
    try:
        identity = admit(budget)
        supervise([sys.executable, '-B', str(ROOT / 'scripts/check_ttmre_c_off_control.py')], budget, fd)
        check(budget)
        workspace.mkdir()
        write(workspace / 'budget.json', budget)
        write(workspace / 'identity.json', identity)
        write(CLAIM, {'workspace': str(workspace), 'protocol': protocol(), 'authorization_sha256': AUTH_SHA,
                      'budget_sha256': sha(workspace / 'budget.json'), 'identity_sha256': sha(workspace / 'identity.json')})
        supervise([sys.executable, '-u', '-B', str(Path(__file__).resolve()), '--mode', 'worker', '--run-dir', str(workspace)], budget, fd)
    except BaseException as error:
        write(workspace.parent / 'supervisor-failure.json', {'error': str(error), 'type': type(error).__name__,
              'budget': budget, 'complete': False, 'optimizer_updates': 0, 'retry': False,
              'elapsed_seconds': time.monotonic() - budget['start_monotonic']})
        raise


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode', choices=('validate', 'authorize', 'start', 'worker'), default='validate')
    p.add_argument('--run-dir', type=Path)
    a = p.parse_args()
    if a.mode == 'validate':
        print(json.dumps(validate(), indent=2))
    elif a.mode == 'authorize':
        print(json.dumps(authorize()))
    elif a.run_dir is None:
        raise ValueError('Use sole launcher')
    elif a.mode == 'start':
        start(a.run_dir.resolve())
    else:
        worker(a.run_dir.resolve())


if __name__ == '__main__':
    main()
