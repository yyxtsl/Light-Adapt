"""Approved12-condition Re-DocRED Test, zero training, one1800s clock.

Default validation never opens Test. Preparation strips labels; prediction and
scoring use separate processes; every native prediction is sealed before scoring.
No new model/seed/optimizer/tuning/download/deletion or automatic retry/extension.
"""
from __future__ import annotations
import argparse
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import statistics
import subprocess
import sys
import tarfile
import time
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts')]
DOC = ROOT / 'refine-logs/FINAL_EVALUATION_PREPARATION_20261026'
AUTH = DOC / 'AUTHORIZATION_APPROVED.json'
AUTH_SHA = '229e935db84a3e766bfdca863eac1398a4ee76991003dd8136a060db919e86d0'
REF = DOC / 'REFERENCE.json'
REF_SHA = '8fea99f478c885c73e07c6578a5f44962806daa8d36217a836938ee2d14ed0cb'
PINS = DOC / 'SOURCE_MANIFEST.json'
PINS_SHA = '3e2ef3fc2e500297a179ca4adbf0fbd0d450fb8587e3236a343f4dccc1e59b68'
RUN_ROOT = ROOT / 'experiments/2026-10-26-final-fixed-test'
CLAIM = RUN_ROOT / 'final-fixed-test-once-claim.json'
HARD_SECONDS, RESERVE, TERM = 1800, 120, 30
SYSTEMS = ('C75','B75','D75','C76','B76','D76','C77','B77','D77','G75','U75','C75-off')
METRICS = ('test_F1','test_precision','test_recall','test_IgnF1_annotated')
DEV_METRICS = tuple(k.replace('test_', 'dev_', 1) for k in METRICS) + ('dev_IgnF1_distant',)
TEST_DOCS, TEST_PAIRS, TEST_GOLD, HUMAN_DOCS = 500, 198670, 17448, 3053


def read(path): return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    with Path(path).open('x', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')


def emit(event, **fields):
    print(json.dumps({'event':event, **fields}, allow_nan=False), flush=True)


def check(budget, reserve=RESERVE):
    if budget is None: return
    if time.monotonic() >= budget['deadline_monotonic'] - reserve:
        raise TimeoutError('Original1800s final-Test reserve reached; no extension')
    if os.name == 'posix' and Path('/proc/sys/kernel/random/boot_id').read_text().strip() != budget['boot_id']:
        raise RuntimeError('Original boot changed')


def sha(path, budget=None):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(4*1024**2), b''):
            check(budget); h.update(block)
    return h.hexdigest()


def authorize():
    if sha(AUTH) != AUTH_SHA: raise ValueError('Authorization changed')
    a = read(AUTH)
    if a['approved'] is not True or a['hard_seconds'] != HARD_SECONDS:
        raise PermissionError('Explicit new Test/1800s permission missing')
    if (a['systems'] != list(SYSTEMS) or a['optimizer_updates'] != 0 or a['new_seeds'] != 0
        or any(a[k] is not False for k in ('checkpoint_writes','downloads','installs','deletions','retry','resume','extension','Test_based_retuning','DWIE_Test'))):
        raise ValueError('Approved Test scope changed')
    return a


def protocol():
    return {'version':'final-fixed-Test-v1','systems':list(SYSTEMS),'Test_documents':TEST_DOCS,
        'Test_pairs':TEST_PAIRS,'primary_seeds':[76,77],'primary':'mean paired B-C Test F1 at76/77',
        'metrics':list(METRICS),'IgnD':None,'IgnD_reason':'Dev-matching lookup is not a Test lookup',
        'precision':'float32','batch_size':4,'max_sequence_length':1024,
        'decoder':'all relation logits>TH;no top-k','prediction_seal_before_scoring':True,
        'prior_project_Test_exposure':True,'optimizer_updates':0,'checkpoint_writes':0,
        'new_seeds':0,'hard_seconds':HARD_SECONDS,'retry':False,'resume':False,'extension':False}


def validate():
    a = authorize()
    if sha(REF) != REF_SHA or sha(PINS) != PINS_SHA: raise ValueError('Reference/source manifest changed')
    pins = read(PINS)
    for name, expected in pins.items():
        if sha(ROOT/name) != expected: raise ValueError('Pinned source changed: '+name)
    ref = read(REF)
    if [v['id'] for v in ref['models']] != list(SYSTEMS): raise ValueError('Frozen system list changed')
    if len({v['checkpoint'] for v in ref['models']}) != 11: raise ValueError('Not the fixed11 weight files')
    for m in ref['models']:
        if any(k not in m['Dev_reference_metrics'] or not math.isfinite(m['Dev_reference_metrics'][k]) for k in DEV_METRICS):
            raise ValueError('Incomplete actual Dev reference schema')
    if a['candidate_freeze_SHA256'] != ref['candidate_freeze_SHA256']: raise ValueError('Wrong candidate freeze')
    return {'protocol':protocol(),'authorization':a,'authorization_sha256':AUTH_SHA,
        'source_files':pins,'reference':ref,'reference_SHA256':REF_SHA,'entry_SHA256':sha(__file__),
        'Test_content_opened_in_validate':False,'CUDA_initialized':False,'optimizer_updates':0}


def resources():
    import shutil
    from inventory_ttmre_redocred_generic_control import memory_metadata
    mem = memory_metadata(); free = shutil.disk_usage(ROOT).free
    if free < 1024**3 or mem.get('effective_available_bytes', 0) < 32*1024**3:
        raise RuntimeError('Require1GiB report margin and32GiB effective RAM; no cleanup/install')
    return {'free_bytes':free,'memory':mem,'new_model_checkpoints':0}


def admit(budget):
    identity = validate(); check(budget)
    if CLAIM.exists(): raise FileExistsError('Already claimed; no repeat')
    identity['resources'] = resources(); ref = identity['reference']
    inputs = dict(ref['input_files'])
    inputs.update({ref['model_dir']+'/'+n:h for n,h in ref['generic_files'].items()})
    inputs.update({m['checkpoint']:m['checkpoint_sha256'] for m in ref['models']})
    inputs.update({m['Dev_predictions']['path']:m['Dev_predictions']['sha256'] for m in ref['models']})
    for name, expected in inputs.items():
        if sha(ROOT/name, budget) != expected: raise ValueError('Input/weight changed: '+name)
    if not (ROOT/'dataset/docred.zip').is_file(): raise FileNotFoundError('Existing Test archive missing; no download')
    from ttmre_nvidia_smi_preflight import inspect_gpu
    external = inspect_gpu()
    if not external['passed']: raise RuntimeError('GPU occupied before claim/CUDA')
    identity.update(input_files=inputs, external_GPU=external)
    return identity


def modules():
    import evaluate_ttmre_c_off_control as coff
    prior, generic, core = coff.modules()
    return coff, generic, core


def prepare(w, identity, budget):
    # Authorized preparation process only; no model forward or scoring.
    ref = identity['reference']; check(budget)
    with zipfile.ZipFile(ROOT/'dataset/docred.zip') as archive: raw = archive.read(ref['Test_member'])
    if len(raw) != ref['Test_bytes'] or hashlib.sha256(raw).hexdigest() != ref['Test_SHA256']:
        raise ValueError('Exact frozen Test bytes changed')
    documents = json.loads(raw)
    import evaluate_ttmre_locked_test as legacy
    legacy.validate_documents(documents)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(ROOT/ref['model_dir']), local_files_only=True)
    features = legacy.build_features(documents, tokenizer, read(ROOT/'meta/rel2id.json'), budget)
    if len(features) != TEST_DOCS or sum(len(v['hts']) for v in features) != TEST_PAIRS or any(v['labels'] for v in features):
        raise ValueError('Wrong/unmasked Test model inputs')
    with (w/'test_source.json').open('xb') as f: f.write(raw)
    write(w/'model_inputs.json', features)
    write(w/'Test_pair_order.json', [{'title':v['title'],'hts':v['hts']} for v in features])
    write(w/'prepared.json', {'Test_source_SHA256':hashlib.sha256(raw).hexdigest(),
        'features_SHA256':sha(w/'model_inputs.json',budget),'pair_order_SHA256':sha(w/'Test_pair_order.json',budget),
        'documents':TEST_DOCS,'pairs':TEST_PAIRS,'label_rows':0,'metrics_computed':False})
    emit('FINAL_TEST_LABEL_FREE_INPUTS_READY', documents=TEST_DOCS,pairs=TEST_PAIRS,metrics_computed=False)


def build(args, ref, model, factory=None, loader=None):
    coff, _, _ = modules()
    mode = model['loader_mode']
    if mode not in ('full_TTM','load_full_C_then_identity','independent_tail_ATLOP','identity_shared_projector'):
        raise ValueError('Unknown frozen architecture mode')
    kind = 'C' if mode == 'full_TTM' else 'C-off' if mode == 'load_full_C_then_identity' else 'B'
    one = {**ref,'checkpoints':{'C':{'path':model['checkpoint']},'B':{'path':model['checkpoint']}},
           'parameters':{'C':381175401,'C-off':363274849,'B':363274849}}
    if factory is not None: one['parameters'][kind] = model['expected_parameters']
    net, recipe = coff.build(args, one, kind, factory, loader)
    if mode == 'independent_tail_ATLOP': net.args.model_type = 'ATLOP'
    recipe.update(system=model['id'],loader_mode=mode,checkpoint_sha256=model['checkpoint_sha256'],
                  checkpoint=model['checkpoint'],prior_checkpoint_Test_exposure=model['same_checkpoint_previously_Test_evaluated'])
    return net, recipe


def inference(model, dataset, device, budget, single_replay=False):
    import numpy as np
    import torch
    from evaluation import to_official
    _, _, core = modules()
    native=[]; single=[]; changed=numeric=pairs=0; maximum=0.
    model.eval()
    with torch.no_grad():
        for offset in range(0,len(dataset),4):
            check(budget); features=dataset[offset:offset+4]
            a=model(**core.shared.inputs_from_batch(core.shared.collate_fn(features),device,labels=False))
            if a.dtype != torch.float32 or not torch.isfinite(a).all(): raise ValueError('Nonfinite/wrong dtype native logits')
            logits=[a]
            if single_replay:
                b=torch.cat([model(**core.shared.inputs_from_batch(core.shared.collate_fn([f]),device,labels=False)) for f in features])
                if a.shape != b.shape or not torch.isfinite(b).all(): raise ValueError('Invalid single replay logits')
                delta=(a-b).abs(); maximum=max(maximum,float(delta.max()))
                numeric+=int((delta > (1e-4+1e-4*b.abs())).sum()); logits.append(b)
            flags=[]
            for output in logits:
                values=output.detach().cpu().numpy(); mask=np.zeros_like(values)
                mask[:,1:]=values[:,1:]>values[:,:1]; mask[:,0]=mask[:,1:].sum(1)==0; flags.append(mask)
            native.extend(to_official(flags[0],features)); pairs+=len(a)
            if single_replay:
                single.extend(to_official(flags[1],features)); changed+=int(np.count_nonzero(flags[0][:,1:]!=flags[1][:,1:]))
    def facts(p):return {(v['title'],v['h_idx'],v['t_idx'],v['r']) for v in p}
    same = not single_replay or facts(native)==facts(single)
    return native,single,{'passed':changed==0 and same,'documents':len(dataset),'pairs':pairs,
        'changed_relation_decisions':changed,'prediction_sets_equal':same,
        'numeric_violations_original_tolerance':numeric,'max_abs_delta':maximum,'original_atol':1e-4,'original_rtol':1e-4}


def predict(w, identity, budget, factory=None, loader=None):
    import torch
    coff, generic, core = modules(); ref=identity['reference']; prepared=read(w/'prepared.json')
    if prepared['Test_source_SHA256']!=ref['Test_SHA256'] or sha(w/'model_inputs.json',budget)!=prepared['features_SHA256'] or sha(w/'Test_pair_order.json',budget)!=prepared['pair_order_SHA256']:
        raise ValueError('Prepared feature/order bytes changed')
    features=read(w/'model_inputs.json')
    if len(features)!=TEST_DOCS or sum(len(v['hts']) for v in features)!=TEST_PAIRS or any(v['labels'] for v in features):
        raise ValueError('Incomplete/unmasked model features')
    # This call reads original Human/Dev and Dev ignored-metric metadata only.
    args,_,dev,gold,annotated,distant=generic.data_context(w,identity,budget)
    sealed={}
    for index,m in enumerate(ref['models']):
        check(budget); started=time.perf_counter(); system=m['id']; directory=w/system; directory.mkdir()
        with coff.no_training():
            model,recipe=build(args,ref,m,factory,loader); before=core.preflight.state_digest(model); rng=core.preflight.rng_state()
            dev_pred,_,_=inference(model,dev,args.device,budget,False)
            dev_metrics=core.shared.score_predictions(dev_pred,gold,annotated,distant)
            if any(not math.isclose(dev_metrics[k],m['Dev_reference_metrics'][k],abs_tol=1e-6,rel_tol=0) for k in DEV_METRICS):
                raise ValueError('Frozen Dev replay disagrees: '+system)
            reference_path=ROOT/m['Dev_predictions']['path']
            if sha(reference_path,budget)!=m['Dev_predictions']['sha256']:raise ValueError('Dev prediction reference changed')
            as_facts=lambda rows:{(v['title'],v['h_idx'],v['t_idx'],v['r']) for v in rows}
            if as_facts(dev_pred)!=as_facts(read(reference_path)):raise ValueError('Dev prediction-set replay disagrees: '+system)
            write(directory/'Dev_replay.json',{'passed':True,'metrics':dev_metrics,'optimizer_updates':0})
            write(directory/'Dev_predictions.json',dev_pred)
            pred,single,replay=inference(model,features,args.device,budget,True)
            write(directory/'native_predictions.json',pred); write(directory/'single_predictions.json',single)
            write(directory/'Test_grouping_check.json',replay)
            if not replay['passed'] or replay['pairs']!=TEST_PAIRS: raise ValueError('Strict Test grouping decision check failed: '+system)
            core.preflight.assert_rng_equal(rng,core.preflight.rng_state())
            if before!=core.preflight.state_digest(model) or any(p.grad is not None for p in model.parameters()):
                raise ValueError('Inference mutated state/gradient')
            recipe.update(model_state_sha256=before,state_unchanged=True,RNG_unchanged=True,optimizer_updates=0)
            write(directory/'recipe.json',recipe)
            sealed[system]={n:sha(directory/n,budget) for n in ('native_predictions.json','single_predictions.json','Test_grouping_check.json','recipe.json','Dev_replay.json')}
            del model;gc.collect()
            if args.device.type=='cuda':torch.cuda.empty_cache()
        elapsed=time.perf_counter()-started; remaining=budget['deadline_monotonic']-time.monotonic()-RESERVE
        estimate=1.5*elapsed*(len(ref['models'])-index-1)+60
        emit('FINAL_TEST_SYSTEM_SEALED',system=system,systems_done=index+1,systems_total=len(ref['models']),metrics_computed=False,
             estimate_remaining_seconds=estimate,available_seconds=remaining)
        if estimate>remaining:raise TimeoutError('Measured remaining work exceeds original1800s deadline')
    if set(sealed)!=set(SYSTEMS):raise ValueError('Not all frozen systems predicted')
    write(w/'predictions_sealed.json',{'protocol':protocol(),'systems':sealed,'prepared_SHA256':sha(w/'prepared.json',budget),
        'identity_SHA256':sha(w/'identity.json',budget),'Test_source_SHA256':prepared['Test_source_SHA256'],
        'metrics_computed':False,'sealed_monotonic':time.monotonic()})
    emit('FINAL_TEST_ALL12_PREDICTIONS_SEALED',systems=len(sealed),metrics_computed=False)


def require_seal(w, identity, budget):
    seal=read(w/'predictions_sealed.json')
    if (seal['protocol']!=protocol() or set(seal['systems'])!=set(SYSTEMS) or seal['metrics_computed'] is not False
        or seal['identity_SHA256']!=sha(w/'identity.json',budget) or seal['prepared_SHA256']!=sha(w/'prepared.json',budget)
        or seal['Test_source_SHA256']!=identity['reference']['Test_SHA256']): raise ValueError('Incomplete/changed global seal')
    for system,files in seal['systems'].items():
        for name,expected in files.items():
            if sha(w/system/name,budget)!=expected: raise ValueError('Sealed system bytes changed')
    return seal


def score_one(pred,gold,human):
    truth={(d['title'],y['h'],y['t'],y['r']) for d in gold for y in d['labels']}
    predicted={(v['title'],v['h_idx'],v['t_idx'],v['r']) for v in pred};correct=truth&predicted
    entities={d['title']:d['vertexSet'] for d in gold};tp=len(correct);recall=tp/len(truth) if truth else 0.
    precision=tp/len(predicted) if predicted else 0.
    overlap=sum(any((h['name'],t['name'],rel) in human for h in entities[title][a] for t in entities[title][b]) for title,a,b,rel in correct)
    ignp=(tp-overlap)/(len(predicted)-overlap+1e-5)
    f1=lambda p:200*p*recall/(p+recall) if p+recall else 0.
    return {'test_F1':f1(precision),'test_precision':100*precision,'test_recall':100*recall,
        'test_IgnF1_annotated':f1(ignp),'test_IgnF1_distant':None,'TP':tp,'FP':len(predicted-truth),
        'FN':len(truth-predicted),'gold':len(truth),'predicted':len(predicted),'overlap_Human':overlap}


def comparisons(metrics):
    if set(metrics)!=set(SYSTEMS) or any(k not in v or not math.isfinite(v[k]) for v in metrics.values() for k in METRICS):
        raise ValueError('Incomplete finite Test metric schema')
    pairs={str(s):{k:metrics['B'+str(s)][k]-metrics['C'+str(s)][k] for k in METRICS} for s in (75,76,77)}
    source_pairs={str(s):{k:metrics['B'+str(s)][k]-metrics['D'+str(s)][k] for k in METRICS} for s in (75,76,77)}
    return {'paired_B_minus_C_pp':pairs,'paired_B_minus_D_pp':source_pairs,
        'secondary_B_minus_D_mean76_77_pp':{k:statistics.mean(source_pairs[str(s)][k] for s in (76,77)) for k in METRICS},
        'primary_mean76_77_pp':{k:statistics.mean(pairs[str(s)][k] for s in (76,77)) for k in METRICS},
        'selected_inclusive_mean75_76_77_pp':{k:statistics.mean(pairs[str(s)][k] for s in (75,76,77)) for k in METRICS},
        'secondary_at75_pp':{label:{k:metrics[a][k]-metrics[b][k] for k in METRICS}
            for label,a,b in (('B-G','B75','G75'),('D-G','D75','G75'),('U-B','U75','B75'),('off-C','C75-off','C75'),('B-off','B75','C75-off'))}}


def score(w, identity, budget):
    seal=require_seal(w,identity,budget) # No Test source read before complete seal.
    write(w/'scoring_started.json',{'all_predictions_verified_before_scoring':True,'seal_SHA256':sha(w/'predictions_sealed.json',budget),'systems':list(SYSTEMS)})
    ref=identity['reference']
    if sha(w/'test_source.json',budget)!=ref['Test_SHA256']:raise ValueError('Test gold source changed')
    gold=read(w/'test_source.json')
    train_path=ROOT/ref['data_dir']/'train_revised.json'
    if sha(train_path,budget)!=ref['input_files'][(Path(ref['data_dir'])/'train_revised.json').as_posix()]:raise ValueError('Human reference changed')
    train=read(train_path)
    if len(gold)!=TEST_DOCS or len(train)!=HUMAN_DOCS:raise ValueError('Incomplete Test/Human')
    human={(h['name'],t['name'],y['r']) for d in train for y in d['labels'] for h in d['vertexSet'][y['h']] for t in d['vertexSet'][y['t']]}
    metrics={s:score_one(read(w/s/'native_predictions.json'),gold,human) for s in SYSTEMS}
    if any(v['gold']!=TEST_GOLD for v in metrics.values()):raise ValueError('Gold count changed/dropped')
    write(w/'Test_metrics.json',metrics);summary=comparisons(metrics)
    write(w/'comparisons.json',summary)
    result={'passed':True,'marker':'FINAL_FIXED_TEST_COMPLETE','protocol':protocol(),'metrics':metrics,'comparisons':summary,
        'all_predictions_sealed_before_scoring':True,'prior_project_Test_exposure':True,'optimizer_updates':0,
        'checkpoint_writes':0,'new_seeds':0,'elapsed_seconds':time.monotonic()-budget['start_monotonic'],
        'positive_gain_required_for_completion':False,'not_untouched_confirmation':True}
    check(budget);write(w/'completion.json',result)
    with tarfile.open(w/'reports.tar.gz','x:gz') as archive:
        for p in sorted(w.rglob('*.json')):
            check(budget);archive.add(p,arcname=p.relative_to(w).as_posix())
    check(budget);emit('FINAL_FIXED_TEST_COMPLETE',reports=str(w/'reports.tar.gz'),primary_mean_F1_pp=summary['primary_mean76_77_pp']['test_F1'])


def require_lock():
    if os.name!='posix' or os.environ.get('TTMRE_FINAL_FIXED_LOCK_FD')!='9':raise RuntimeError('Use sole final-fixed-Test launcher')
    import fcntl
    a,b=os.fstat(9),(ROOT/'experiments/2026-10-08-baseline-alignment/.training.lock').stat()
    if (a.st_dev,a.st_ino)!=(b.st_dev,b.st_ino):raise ValueError('Wrong shared GPU lock')
    fcntl.flock(9,fcntl.LOCK_EX|fcntl.LOCK_NB);return 9


def stopped(*_):raise TimeoutError('Final Test interrupted; no reset/retry')


def original_budget():
    start=float(os.environ['TTMRE_FINAL_FIXED_START'])
    if not math.isfinite(start) or start>time.monotonic():raise ValueError('Invalid launcher clock')
    b={'start_monotonic':start,'deadline_monotonic':start+HARD_SECONDS,'hard_seconds':HARD_SECONDS,
        'boot_id':os.environ['TTMRE_FINAL_FIXED_BOOT'],'start_utc':os.environ['TTMRE_FINAL_FIXED_UTC'],'retry':False,'resume':False,'extension':False}
    check(b);return b


def verify_context(w,b):
    authorize();c=read(CLAIM)
    if (w!=w.resolve() or w.name!='workspace' or w.parent.parent!=RUN_ROOT.resolve() or c['workspace']!=str(w)
        or c['protocol']!=protocol() or c['authorization_SHA256']!=AUTH_SHA or c['budget_SHA256']!=sha(w/'budget.json')
        or c['identity_SHA256']!=sha(w/'identity.json') or b['hard_seconds']!=HARD_SECONDS
        or not math.isclose(b['deadline_monotonic']-b['start_monotonic'],HARD_SECONDS,abs_tol=1e-6,rel_tol=0)
        or any(b[k] is not False for k in ('retry','resume','extension'))):raise ValueError('Original context changed')
    check(b)


def worker(w,phase):
    authorize();require_lock();b=read(w/'budget.json');verify_context(w,b);signal.signal(signal.SIGTERM,stopped);i=read(w/'identity.json')
    try:
        current=validate()
        if current['source_files']!=i['source_files'] or current['entry_SHA256']!=i['entry_SHA256']:raise ValueError('Worker source changed')
        if phase=='prepare':
            from ttmre_dwie_runtime_fix import runtime_identity
            write(w/'runtime_prepare.json',runtime_identity());prepare(w,i,b)
        elif phase=='predict':
            resources()
            from ttmre_nvidia_smi_preflight import inspect_gpu
            if not inspect_gpu()['passed']:raise RuntimeError('GPU occupied before prediction CUDA')
            from ttmre_dwie_runtime_fix import runtime_identity
            write(w/'runtime_predict.json',runtime_identity())
            for name,expected in i['input_files'].items():
                if sha(ROOT/name,b)!=expected:raise ValueError('Prediction input/weight changed')
            protected={m['checkpoint']:((ROOT/m['checkpoint']).stat().st_ino,(ROOT/m['checkpoint']).stat().st_size,(ROOT/m['checkpoint']).stat().st_mtime_ns) for m in i['reference']['models']}
            predict(w,i,b)
            for name,before in protected.items():
                now=(ROOT/name).stat()
                if (now.st_ino,now.st_size,now.st_mtime_ns)!=before:raise ValueError('Frozen checkpoint changed during prediction')
        elif phase=='score':score(w,i,b)
        else:raise ValueError('Unknown phase')
    except BaseException as error:
        write(w/('failure-'+phase+'.json'),{'error':str(error),'type':type(error).__name__,'phase':phase,'complete':False,
              'optimizer_updates':0,'retry':False,'resumable':False,'elapsed_seconds':time.monotonic()-b['start_monotonic']});raise


def supervise(command,b,fd):
    child=subprocess.Popen(command,start_new_session=True,pass_fds=(fd,));terminated=False
    try:
        while True:
            check(b,reserve=0);code=child.poll()
            if code is not None:
                if code or terminated:raise RuntimeError('CPU/phase failed; no automatic retry')
                return
            if time.monotonic()>=b['deadline_monotonic']-TERM and not terminated:
                os.killpg(child.pid,signal.SIGTERM);terminated=True
            time.sleep(.2)
    finally:
        if child.poll() is None:os.killpg(child.pid,signal.SIGKILL);child.wait()


def start(w):
    authorize();fd=require_lock();b=original_budget()
    if w.name!='workspace' or w.parent.parent!=RUN_ROOT.resolve() or w.exists():raise ValueError('Use a new final-Test workspace')
    signal.signal(signal.SIGTERM,stopped)
    try:
        i=admit(b);supervise([sys.executable,'-B',str(ROOT/'scripts/check_ttmre_final_fixed_test.py')],b,fd);check(b)
        w.mkdir();write(w/'budget.json',b);write(w/'identity.json',i)
        write(CLAIM,{'workspace':str(w),'protocol':protocol(),'authorization_SHA256':AUTH_SHA,
                     'budget_SHA256':sha(w/'budget.json'),'identity_SHA256':sha(w/'identity.json')})
        command=[sys.executable,'-u','-B',str(Path(__file__).resolve()),'--mode','worker','--run-dir',str(w)]
        for phase in ('prepare','predict','score'):supervise(command+['--phase',phase],b,fd)
    except BaseException as error:
        write(w.parent/'supervisor-failure.json',{'error':str(error),'type':type(error).__name__,'budget':b,'complete':False,
              'optimizer_updates':0,'retry':False,'elapsed_seconds':time.monotonic()-b['start_monotonic']});raise


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--mode',choices=('validate','authorize','start','worker'),default='validate')
    p.add_argument('--run-dir',type=Path);p.add_argument('--phase',choices=('prepare','predict','score'));a=p.parse_args()
    if a.mode=='validate':print(json.dumps(validate(),indent=2))
    elif a.mode=='authorize':print(json.dumps(authorize()))
    elif a.run_dir is None or (a.mode=='worker' and a.phase is None):raise ValueError('Use sole launcher and fixed phase')
    elif a.mode=='start':start(a.run_dir.resolve())
    else:worker(a.run_dir.resolve(),a.phase)


if __name__=='__main__':main()
