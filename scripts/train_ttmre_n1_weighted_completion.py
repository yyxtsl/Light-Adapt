"""Complete the authorized N1 pair: preserve Control and train Weighted only.

New 12-hour remaining-work budget. No optimizer-state restore, Control retrain,
threshold inflation, extra seed, Test, retry or automatic follow-up.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import shutil
import signal
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts')]
import train_ttmre_n1_full_seed75 as core
import train_ttmre_n1_full_seed75_restart20h as prior

RUN_ROOT = ROOT / 'experiments/2026-10-18-n1-weighted-completion'
PARENT = ROOT / ('experiments/2026-10-17-n1-independent-seed75-restart20h/'
    'n1-full75-restart20h-launch-20261017-200600-o3v6Ke/workspace')
DIAGNOSIS = ROOT / ('experiments/2026-10-18-n1-full75-grouping-diagnosis/'
    'run-20261018T035636Z-827d0f2d/diagnostic.json')
PARENT_IDENTITY_SHA = '7e1eadc02c437185e2352806c2fe770de63a70b9f1dd0f330ceb39f398418e6b'
DIAGNOSIS_SHA = '269140fb19e849d905192c97db18f8de089bf58ff51cc61f3a6520dcd8247777'
CHECKPOINTS = {'distant': '4fd5aca22c20584dde442c2d80dba8a72db8fe20a4b9fa920fa7c43a7fe13ce9',
    'human-frozen': 'b92f36a6adab9f2779d00e1d210a1ab72c0653961ecebd61685821a708bebbd7'}
FILES = ('scripts/train_ttmre_n1_weighted_completion.py',
         'scripts/check_ttmre_n1_weighted_completion.py',
         'scripts/run_ttmre_n1_weighted_completion.sh')
POLICY = 'full-Dev-exact-decisions-original-TTM-checks-numeric-results-retained-v1'
HARD_SECONDS = 12 * 3600
DEV_DOCUMENTS = 500
_base_protocol = core.protocol
_base_verify_stage = core.verify_stage
_base_lock = core.require_lock
_base_loader = core.load_generic_model_seed75


def protocol():
    p = _base_protocol()
    p.update({'version': 'n1-weighted-completion-v1', 'hard_seconds': HARD_SECONDS,
        'variant': 'n1-weighted-completion', 'new_formal_optimizer_updates': core.UPDATES_PER_BRANCH,
        'reused_control_optimizer_updates': core.UPDATES_PER_BRANCH,
        'acceptance_policy': POLICY, 'control_retraining': False})
    return p


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode', choices=('validate', 'train', 'worker'), default='validate')
    p.add_argument('--phase', choices=('prepare', 'control', 'weighted', 'finish'))
    p.add_argument('--run-dir', type=Path)
    return p


def parent_context():
    if core.digest(PARENT / 'identity.json') != PARENT_IDENTITY_SHA or core.digest(DIAGNOSIS) != DIAGNOSIS_SHA:
        raise ValueError('Pinned parent/diagnosis identity mismatch')
    identity, diagnosis = core.read(PARENT / 'identity.json'), core.read(DIAGNOSIS)
    if (identity['budget_sha256'] != core.digest(PARENT / 'budget.json')
            or diagnosis['checkpoint_sha256'] != CHECKPOINTS['human-frozen']
            or not all(diagnosis[k] is True for k in ('dev_metric_replay_passed', 'parameters_unchanged',
                                                    'checkpoint_file_unchanged', 'diagnosis_completed'))
            or diagnosis['optimizer_updates'] != 0):
        raise ValueError('Parent diagnostic contract failed')
    for name, expected in identity['source_files'].items():
        if core.digest(ROOT / name) != expected:
            raise ValueError('Frozen source changed: ' + name)
    for stage, expected in CHECKPOINTS.items():
        if core.digest(PARENT / 'control' / stage / 'artifacts/best_model.pth') != expected:
            raise ValueError('Preserved Control checkpoint changed: ' + stage)
    return identity, diagnosis


def validate(args, budget=None):
    parent, diagnosis = parent_context()
    if args.run_dir is not None:
        path = args.run_dir.resolve()
        if path.name != 'workspace' or path.parent.parent != RUN_ROOT.resolve():
            raise ValueError('Use the new dated launch/workspace')
        if path.exists():
            raise FileExistsError('Preserve existing workspace')
    if budget is not None:
        core.check_budget(budget)
    # All train/model input bytes still have to match the approved parent.
    for path, expected in parent['input_files'].items():
        if budget is not None:
            core.check_budget(budget)
        if core.digest(path) != expected:
            raise ValueError('Original input changed: ' + path)
    required = max(8 * 1024**3, (PARENT / 'control/human-frozen/artifacts/best_model.pth').stat().st_size * 3 + 1024**3)
    if shutil.disk_usage(RUN_ROOT.parent).free < required:
        raise RuntimeError('Insufficient free disk; no artifacts deleted')
    identity = {**parent, 'protocol': protocol(), 'variant': 'n1-weighted-completion',
        'source_files': {**parent['source_files'], **{f: core.digest(ROOT / f) for f in FILES}},
        'entrypoint_sha256': core.digest(Path(__file__)), 'historical_parent': str(PARENT),
        'historical_parent_identity_sha256': PARENT_IDENTITY_SHA,
        'diagnosis_sha256': DIAGNOSIS_SHA, 'required_free_bytes': required,
        'historical_control_grouping': diagnosis['grouping'],
        'historical_budget': core.read(PARENT / 'budget.json')}
    return identity


def configure():
    core.protocol = protocol
    core.verify_stage = verify_stage
    core.baseline.replay = replay
    core.load_generic_model_seed75 = load_and_verify_generic


def load_and_verify_generic(args, identity):
    """Prove the effective initial state equals Control while retaining raw bytes.

    RoBERTa's unused pooler may initialize randomly before the task seed. The
    original loss never consumes it. Substitute its original bytes only while
    calculating a comparison digest, then restore them before any forward.
    """
    import torch
    model = _base_loader(args, identity)
    raw = core.preflight.state_digest(model)
    expected = core.read(PARENT / 'control/initial_model.json')['state_sha256']
    names = [n for n in model.state_dict() if n.startswith('model.pooler.')]
    allowed = {'model.pooler.dense.weight', 'model.pooler.dense.bias'}
    if set(names) not in (set(), allowed):
        raise ValueError('Unexpected unused pooler structure')
    restored = {n: model.state_dict()[n].detach().clone() for n in names}
    try:
        if names:
            control = torch.load(PARENT / 'control/distant/artifacts/best_model.pth', map_location='cpu', weights_only=True)
            with torch.no_grad():
                for n in names:
                    model.state_dict()[n].copy_(control[n].to(model.state_dict()[n].device))
            del control
        canonical = core.preflight.state_digest(model)
    finally:
        with torch.no_grad():
            for n, value in restored.items(): model.state_dict()[n].copy_(value)
    if canonical != expected or core.preflight.state_digest(model) != raw:
        raise ValueError('Effective generic initialization differs from original Control')
    core.shared.write_json(args.run_dir / ('initialization_validation-' + args.phase + '.json'), {
        'passed': True, 'raw_initial_sha256': raw, 'canonical_initial_sha256': canonical,
        'original_control_initial_sha256': expected, 'hash_only_substitution': names,
        'model_restored_before_forward': True, 'trained_parameter_initialization_reused': False})
    return model


def require_lock():
    if os.environ.get('TTMRE_N1_WEIGHTED_COMPLETION_LOCK_FD') != '9':
        raise RuntimeError('Use only the weighted-completion launcher')
    return _base_lock()


def read_rows(path):
    return [json.loads(x) for x in Path(path).read_text(encoding='utf-8').splitlines() if x.strip()]


def check_trajectory(directory, stage, branch='control'):
    best = core.read(directory / 'best_metrics.json')
    manifest = core.read(directory / 'manifest.json')
    rows = read_rows(directory / 'metrics.jsonl')
    batches, epochs = core.baseline.STAGES[stage][0] // 4, core.baseline.STAGES[stage][1]
    if (len(rows) != epochs or manifest['total_steps'] != batches * epochs
            or manifest['branch'] != branch or manifest['stage'] != stage
            or any(r.get('branch') != branch or r.get('stage') != stage or r['epoch'] != i
                   or r['batch'] != batches - 1 or r['step'] != i * batches - 1
                   or not math.isfinite(r['dev_F1']) for i, r in enumerate(rows, 1))):
        raise ValueError('Saved trajectory contract failed')
    selected = max(rows, key=lambda r: r['dev_F1'])
    if selected['epoch'] != best['epoch'] or any(selected[k] != best[k] for k in core.baseline.m1.METRIC_KEYS):
        raise ValueError('Saved earliest-best selection differs')
    order = core.read(directory / 'training_order.json')
    if order['formal_optimizer_updates'] != batches * epochs or order['seed'] != 75:
        raise ValueError('Saved training order incomplete')
    return best, manifest, order


def full_dev_grouping(model, dataset, device, budget):
    import numpy as np
    import torch
    from evaluation import to_official
    if len(dataset) != DEV_DOCUMENTS:
        raise ValueError('Full-Dev comparison must include the complete declared Dev set')
    grouped_predictions, single_predictions = [], []
    changed, pair_count, numeric_violations, max_delta = 0, 0, 0, 0.0
    model.eval()
    with torch.no_grad():
        for offset in range(0, len(dataset), 4):
            core.check_budget(budget)
            features = dataset[offset:offset + 4]
            a = model(**core.shared.inputs_from_batch(core.shared.collate_fn(features), device, labels=False))
            b = torch.cat([model(**core.shared.inputs_from_batch(core.shared.collate_fn([f]), device, labels=False))
                           for f in features])
            if a.shape != b.shape or not torch.isfinite(a).all() or not torch.isfinite(b).all():
                raise RuntimeError('Invalid full-Dev grouping logits')
            delta = (a - b).abs()
            numeric_violations += int((delta > (1e-4 + 1e-4 * b.abs())).sum())
            max_delta = max(max_delta, float(delta.max()))
            preds = []
            for logits in (a, b):
                pred = np.zeros(tuple(logits.shape), dtype=np.float32)
                values = logits.cpu().numpy()
                pred[:, 1:] = values[:, 1:] > values[:, :1]
                pred[:, 0] = pred[:, 1:].sum(axis=1) == 0
                preds.append(pred)
            changed += int(np.count_nonzero(preds[0][:, 1:] != preds[1][:, 1:]))
            pair_count += len(a)
            grouped_predictions.extend(to_official(preds[0], features))
            single_predictions.extend(to_official(preds[1], features))
    def facts(predictions):
        return {(p['title'], p['h_idx'], p['t_idx'], p['r']) for p in predictions}
    same = facts(grouped_predictions) == facts(single_predictions)
    return ({'policy': POLICY, 'passed': changed == 0 and same, 'documents': len(dataset),
             'pairs': pair_count, 'changed_relation_decisions': changed, 'prediction_sets_equal': same,
             'numeric_violations_original_tolerance': numeric_violations, 'max_abs_delta': max_delta,
             'original_rtol': 1e-4, 'original_atol': 1e-4}, grouped_predictions)


def replay(model, directory, dev_loader, args, stage, gold, annotated, distant):
    import torch
    model.zero_grad(set_to_none=True)
    model.load_state_dict(torch.load(directory / 'best_model.pth', map_location='cpu', weights_only=True), strict=True)
    best = core.read(directory / 'best_metrics.json')
    strict = core.aligned.grouping_invariance(model, dev_loader.dataset, args.device)
    report = {'passed': False, 'stage': stage, 'epoch': best['epoch'], 'step': best['step'],
        'checkpoint_sha256': core.digest(directory / 'best_model.pth'), 'optimizer_steps': 0,
        'test_data_read': False, 'test_evaluated': False, 'original_strict_grouping': strict,
        'acceptance_policy': POLICY if stage == 'human-frozen' else 'original-strict'}
    core.shared.write_json(directory / 'replay_evidence.json', report)
    if stage == 'human-frozen':
        full, predictions = full_dev_grouping(model, dev_loader.dataset, args.device, args.budget)
        report['full_dev_grouping'] = full
        metrics = core.shared.score_predictions(predictions, gold, annotated, distant)
        ttm_ok = all(strict['checks'][key]['close'] for key in ('ttm_grouped_vs_single', 'ttm_permutation'))
        accepted = full['passed'] and ttm_ok and strict['changed_relation_decisions'] == 0
    else:
        metrics = core.evaluate(model, dev_loader, args, gold, annotated, distant)
        accepted = strict['passed']
    report['metrics'] = metrics
    report['metric_replay_passed'] = all(math.isclose(metrics[k], best[k], rel_tol=0, abs_tol=1e-6)
                                         for k in core.baseline.m1.METRIC_KEYS)
    report['passed'] = bool(accepted and report['metric_replay_passed'])
    # "passed" below is the approved operational policy, NOT the original
    # numeric result. The original check is retained verbatim alongside it.
    report['invariance'] = {'passed': bool(accepted), 'policy': report['acceptance_policy'],
                             'original_strict_passed': strict['passed']}
    core.shared.write_json(directory / 'replay_evidence.json', report)
    if not report['passed']:
        raise RuntimeError('Checkpoint acceptance failed; see replay_evidence.json')
    return report


def verify_stage(directory, identity, branch, stage):
    result = _base_verify_stage(directory, identity, branch, stage)
    if stage == 'human-frozen':
        report = core.read(directory / 'checkpoint_replay.json')
        full = report.get('full_dev_grouping', {})
        strict = report.get('original_strict_grouping', {})
        if (report.get('acceptance_policy') != POLICY or not full.get('passed')
                or full.get('documents') != DEV_DOCUMENTS
                or full.get('changed_relation_decisions') != 0 or not full.get('prediction_sets_equal')
                or report.get('metric_replay_passed') is not True
                or strict.get('changed_relation_decisions') != 0
                or not all(strict.get('checks', {}).get(k, {}).get('close') for k in ('ttm_grouped_vs_single', 'ttm_permutation'))):
            raise ValueError('Missing full-Dev decision acceptance evidence')
    return result


def metadata_copy(source, target, identity):
    target.mkdir(parents=True, exist_ok=False)
    for path in source.iterdir():
        if path.is_file() and path.suffix in ('.json', '.jsonl', '.log') and path.name not in (
                'completion.json', 'failure.json', 'checkpoint_replay.json', 'frequency_diagnostics.json', 'timing.json'):
            shutil.copyfile(path, target / path.name)
    # Both directories live on the experiment volume. This is a read-only
    # reference, never a target for training or checkpoint replacement.
    os.link(source / 'best_model.pth', target / 'best_model.pth')
    manifest = core.read(target / 'manifest.json')
    manifest.update({'protocol': protocol(), 'identity_sha256': core.digest(identity),
                     'reused_from': str(source), 'no_new_training': True})
    core.shared.write_json(target / 'manifest.json', manifest)


def complete_control(args, identity):
    import torch
    from torch.utils.data import DataLoader
    started = time.monotonic()
    source = PARENT / 'control'
    # Temporarily use the exact parent protocol to verify its completed Distant.
    current = core.protocol
    try:
        core.protocol = lambda: core.read(PARENT / 'identity.json')['protocol']
        _base_verify_stage(source / 'distant/artifacts', core.read(PARENT / 'identity.json'), 'control', 'distant')
    finally:
        core.protocol = current
    for stage in ('distant', 'human-frozen'):
        check_trajectory(source / stage / 'artifacts', stage)
    failure = core.read(source / 'human-frozen/artifacts/failure.json')
    if failure.get('optimizer_step_calls_completed') != core.HUMAN_UPDATES or failure.get('error') != 'Grouping invariance failed':
        raise ValueError('Human did not complete the approved updates before grouping failure')
    target = args.run_dir / 'control'
    target.mkdir()
    shutil.copyfile(source / 'initial_model.json', target / 'initial_model.json')
    for stage in ('distant', 'human-frozen'):
        metadata_copy(source / stage / 'artifacts', target / stage / 'artifacts', args.run_dir / 'identity.json')
    d, h = target / 'distant/artifacts', target / 'human-frozen/artifacts'
    for name in ('completion.json', 'checkpoint_replay.json', 'frequency_diagnostics.json', 'timing.json'):
        shutil.copyfile(source / 'distant/artifacts' / name, d / name)
    _, _, dev, gold, annotated, distant, buckets = core.datasets(args)
    model = core.load_generic_model_seed75(args, identity['baseline_identity'])
    # The loader validated all effective initial bytes against Control and kept
    # the raw unused-pooler bytes separately; no task parameter was initialized from Control.
    model.load_state_dict(torch.load(d / 'best_model.pth', map_location='cpu', weights_only=True), strict=True)
    if core.preflight.state_digest(model) != core.read(h / 'manifest.json')['initial_state_sha256']:
        raise ValueError('Human did not initialize from selected Control Distant')
    memory = model.mu_encoder.memory_tokens.detach().cpu().clone()
    core.baseline.set_stage(model, 'human-frozen')
    loader = DataLoader(dev, batch_size=4, shuffle=False, collate_fn=core.shared.collate_fn)
    report = replay(model, h, loader, args, 'human-frozen', gold, annotated, distant)
    if not torch.equal(memory, model.mu_encoder.memory_tokens.detach().cpu()):
        raise ValueError('Control Human frozen memory differs from Distant parent')
    core.shared.write_json(h / 'checkpoint_replay.json', report)
    best, manifest, order = check_trajectory(h, 'human-frozen')
    frequency = core.n1.frequency_report(model, loader, args, gold, annotated, distant, buckets, best)
    if frequency.get('passed') is not True:
        raise RuntimeError('Control frequency diagnostics failed')
    core.shared.write_json(h / 'frequency_diagnostics.json', frequency)
    manifest.update({'best_checkpoint_sha256': report['checkpoint_sha256'], 'frequency_diagnostics_passed': True})
    core.shared.write_json(h / 'manifest.json', manifest)
    core.shared.write_json(h / 'completion.json', {'stage': 'human-frozen', 'branch': 'control',
        'best_epoch': best['epoch'], 'best_step': best['step'], 'best_dev_F1': best['dev_F1'],
        'optimizer_steps': core.HUMAN_UPDATES, 'memory_unchanged': True,
        'learning_update_check_passed': True, 'checkpoint_replay_passed': True,
        'frequency_diagnostics_passed': True, 'test_data_read': False, 'test_evaluated': False,
        'reused_optimizer_updates': core.HUMAN_UPDATES, 'new_optimizer_updates': 0})
    core.shared.write_json(h / 'timing.json', {'new_completion_seconds': time.monotonic() - started,
        'historical_training_reused': True, 'new_optimizer_updates': 0})
    refs = {stage: verify_stage(target / stage / 'artifacts', identity, 'control', stage)
            for stage in ('distant', 'human-frozen')}
    result = {'passed': True, 'branch': 'control', 'seed': 75,
        'formal_optimizer_updates': core.UPDATES_PER_BRANCH, 'new_optimizer_updates': 0,
        'reused_optimizer_updates': core.UPDATES_PER_BRANCH,
        'initial_model_sha256': core.read(target / 'initial_model.json')['state_sha256'],
        'distant': refs['distant'], 'human': refs['human-frozen'],
        'distant_best': refs['distant']['best_metrics'], 'human_best': best,
        'diagnostic_optimizer_updates': 0, 'test_data_read': False, 'automatic_retry': False,
        'elapsed_seconds': time.monotonic() - started}
    core.shared.write_json(target / 'completion.json', result)
    core.emit('N1_PRESERVED_CONTROL_ACCEPTED', new_optimizer_updates=0, human_F1=best['dev_F1'])


def finish(args, identity, budget):
    core.verify_workspace(identity, budget)
    core.verify_selection_files(args, identity, budget)
    control = core.read(args.run_dir / 'control/completion.json')
    weighted = core.read(args.run_dir / 'weighted/completion.json')
    certificate = core.read(args.run_dir / 'initialization_validation-weighted.json')
    if (certificate.get('passed') is not True or not certificate.get('model_restored_before_forward')
            or certificate.get('trained_parameter_initialization_reused') is not False
            or certificate.get('canonical_initial_sha256') != control['initial_model_sha256']
            or certificate.get('raw_initial_sha256') != weighted['initial_model_sha256']
            or set(certificate.get('hash_only_substitution', [])) - {
                'model.pooler.dense.weight', 'model.pooler.dense.bias'}):
        raise ValueError('Paired effective initialization certificate failed')
    for stage in ('distant', 'human-frozen'):
        dirs = [args.run_dir / b / stage / 'artifacts' for b in ('control','weighted')]
        c,w = [core.read(d/'manifest.json') for d in dirs]
        co,wo = [core.read(d/'training_order.json') for d in dirs]
        if any(c[k] != w[k] for k in ('optimizer_groups','batches_per_epoch','initial_rng_sha256')) or any(
                co[k] != wo[k] for k in ('sample_order_sha256','final_rng_sha256')):
            raise ValueError('Paired optimizer/RNG/sample order differs')
    deltas = {stage:core.paired_delta(control[stage+'_best'],weighted[stage+'_best']) for stage in ('distant','human')}
    h = deltas['human']
    result = {'passed': True, 'marker': 'N1_WEIGHTED_COMPLETION_COMPLETE', 'protocol':protocol(),
        'formal_optimizer_updates':core.FORMAL_UPDATES,'new_formal_optimizer_updates':core.UPDATES_PER_BRANCH,
        'reused_control_optimizer_updates':core.UPDATES_PER_BRANCH,'diagnostic_optimizer_updates':0,
        'delta_pp':deltas,'metrics':{s:{'control':control[s+'_best'],'weighted':weighted[s+'_best']} for s in ('distant','human')},
        'screening_gate_passed':h['dev_F1']>=.5 and h['dev_IgnF1_annotated']>0 and h['dev_IgnF1_distant']>0,
        'positive_human_signal':h['dev_F1']>0,'reliable_improvement_established':False,
        'selected_after_weighting':weighted['distant']['selected_after_weighting'],
        'acceptance_policy':POLICY,'effective_initialization':certificate,
        'strict_grouping_passed':{b:core.read(args.run_dir/b/'human-frozen/artifacts/checkpoint_replay.json')['original_strict_grouping']['passed'] for b in ('control','weighted')},
        'historical_control_grouping':identity.get('historical_control_grouping'),
        'historical_budget':identity['historical_budget'],'elapsed_seconds':time.monotonic()-budget['start_monotonic'],
        'test_evaluated':False,'automatic_followup':False,'independent_full_two_stage_seed':True}
    core.shared.write_json(args.run_dir/'comparison/completion.json',result)
    core.emit('N1_WEIGHTED_COMPLETION_COMPLETE',new_optimizer_updates=core.UPDATES_PER_BRANCH,
        reused_optimizer_updates=core.UPDATES_PER_BRANCH,human_delta_pp=h['dev_F1'])


def worker(args):
    require_lock()
    budget = core.read(args.run_dir / 'budget.json')
    args.budget = budget
    core.check_budget(budget)
    try:
        if args.phase == 'prepare':
            empty = args.run_dir
            args.run_dir = None
            try: identity = validate(args, budget)
            finally: args.run_dir = empty
            identity['budget_sha256'] = core.digest(args.run_dir / 'budget.json')
            core.shared.write_json(args.run_dir / 'identity.json', identity)
            return
        identity = core.read(args.run_dir / 'identity.json')
        if identity['protocol'] != protocol() or identity['budget_sha256'] != core.digest(args.run_dir / 'budget.json'):
            raise ValueError('New workspace identity/budget changed')
        core.verify_workspace(identity, budget)
        args.baseline_workspace = Path(identity['baseline_workspace'])
        args.model_dir, args.data_dir = Path(identity['model_dir']), Path(identity['data_dir'])
        args.device = core.baseline.control.runtime(identity['environment'])
        if args.phase == 'control':
            complete_control(args, identity)
        elif args.phase == 'weighted':
            if core.read(args.run_dir / 'control/completion.json')['new_optimizer_updates'] != 0:
                raise ValueError('Control must not be retrained')
            core.run_branch(args, identity, budget, 'weighted')
        elif args.phase == 'finish':
            (args.run_dir / 'comparison').mkdir()
            finish(args, identity, budget)
        else:
            raise ValueError('Unknown phase')
    except BaseException as error:
        core.shared.write_json(args.run_dir / ('failure-' + str(args.phase) + '.json'),
            {'error': str(error), 'type': type(error).__name__, 'complete': False, 'resumable': False})
        raise


def train(args):
    if os.name != 'posix' or args.run_dir is None:
        raise RuntimeError('Use Linux one-use launcher')
    fd = require_lock()
    started = time.monotonic()
    path = args.run_dir.resolve()
    if path.name != 'workspace' or path.parent.parent != RUN_ROOT.resolve() or path.exists():
        raise ValueError('Only a new dated workspace is permitted')
    claim = RUN_ROOT / 'weighted-completion-once-claim.json'
    if claim.exists():
        raise FileExistsError('Remaining-work authorization already claimed')
    path.mkdir(parents=True)
    identity, _ = parent_context()
    budget = {'start_monotonic': started, 'deadline_monotonic': started + HARD_SECONDS,
        'start_utc': datetime.now(timezone.utc).isoformat(), 'hard_seconds': HARD_SECONDS,
        'boot_id': Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
        'soft_reserve_seconds':120, 'term_reserve_seconds':30,
        'authorization': 'user-approved-remaining-Weighted-and-Control-reports-20261018',
        'historical_budget': core.read(PARENT / 'budget.json'),
        'automatic_retry': False, 'automatic_resume': False, 'extension': False}
    core.shared.write_json(path / 'budget.json', budget)
    prior.exclusive_json(claim, {'workspace': str(path), 'budget_sha256': core.digest(path / 'budget.json')})
    prior.exclusive_json(path.parent / 'pid', os.getpid())
    def stop(*_): raise core.BudgetStop('Supervisor stopped; no automatic retry')
    signal.signal(signal.SIGTERM, stop)
    command = [sys.executable, '-u', '-B', str(Path(__file__).resolve()), '--mode', 'worker', '--run-dir', str(path)]
    try:
        core.emit('N1_WEIGHTED_COMPLETION_BUDGET_STARTED', **budget)
        core.supervise([sys.executable, '-B', str(ROOT / FILES[1])], budget, fd)
        for phase in ('prepare', 'control', 'weighted', 'finish'):
            core.supervise(command + ['--phase', phase], budget, fd)
    except BaseException as error:
        core.shared.write_json(path / 'failure.json', {'error': str(error), 'complete': False,
            'automatic_retry': False, 'resumable': False, 'elapsed_seconds': time.monotonic() - started})
        raise


def main():
    os.chdir(ROOT)
    configure()
    args = parser().parse_args()
    if args.mode == 'train': train(args)
    elif args.mode == 'worker': worker(args)
    else: print(json.dumps({'validation': 'passed', 'identity': validate(args), 'files_written': False}, ensure_ascii=False))


if __name__ == '__main__':
    main()
