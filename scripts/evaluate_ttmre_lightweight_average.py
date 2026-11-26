"""One approved equal-weight B75/76/77 supplementary evaluation, 1200s total.

Original checkpoints read-only; new fixed derived state only. No optimizer,
coefficient/subset selection or replacement of the original locked12 comparison.
"""
from pathlib import Path
import argparse,contextlib,gc,hashlib,json,math,os,signal,statistics,subprocess,sys,tarfile,time
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
DOC=ROOT/'refine-logs/CASE_STUDY_20261027'
RUN_ROOT=ROOT/'experiments/2026-10-27-lightweight-average'
HARD=1200;RESERVE=90;TERM=20
MEMORY='mu_encoder.memory_tokens'

def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def write(p,v):
    with Path(p).open('x',encoding='utf-8') as f:json.dump(v,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def emit(event,**kw):print(json.dumps({'event':event,**kw},allow_nan=False),flush=True)
def check(b,reserve=RESERVE):
    if time.monotonic()>=b['deadline_monotonic']-reserve:raise TimeoutError('Approved1200s original clock reached')
    if Path('/proc/sys/kernel/random/boot_id').read_text().strip()!=b['boot_id']:raise RuntimeError('Boot changed; no clock reset')
def sha(p,b=None):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda:f.read(4*1024**2),b''):
            if b:check(b)
            h.update(block)
    return h.hexdigest()
def facts(rows):return {(r['title'],r['h_idx'],r['t_idx'],r['r']) for r in rows}

def uniform_state(paths,b,torch_module=None):
    """Three compatible raw FP32 states; preserve equal provenance/nonfloat buffers."""
    if torch_module is None:import torch as torch_module
    torch=torch_module
    if len(paths)!=3:raise ValueError('Exactly three fixed B checkpoints required')
    result={};schema={};preserved=set();raw_bytes=0
    for i,path in enumerate(paths):
        check(b);state=torch.load(path,map_location='cpu',weights_only=True)
        if not isinstance(state,dict) or MEMORY not in state:raise ValueError('Expected raw B state dict including memory')
        if i and set(state)!=set(schema):raise ValueError('State keys differ')
        for name,value in state.items():
            check(b)
            if not isinstance(value,torch.Tensor) or value.is_complex() or value.layout!=torch.strided:raise ValueError('Unsupported tensor '+name)
            if not i:schema[name]=(tuple(value.shape),str(value.dtype));raw_bytes+=value.numel()*value.element_size()
            elif schema[name]!=(tuple(value.shape),str(value.dtype)):raise ValueError('Shape/dtype differs '+name)
            if value.is_floating_point() and (value.dtype!=torch.float32 or not torch.isfinite(value).all()):raise ValueError('Nonfinite/nonFP32 state '+name)
            keep=name==MEMORY or not value.is_floating_point()
            if keep:
                preserved.add(name)
                if not i:result[name]=value.clone()
                elif not torch.equal(result[name],value):raise ValueError('Frozen buffer differs '+name)
            elif not i:result[name]=value.to(torch.float64)
            else:result[name].add_(value)
        del state;emit('B_AVERAGE_SOURCE_ACCUMULATED',source_index=i+1,optimizer_updates=0)
    for name in result:
        check(b)
        if name not in preserved:result[name]=result[name].div_(3).to(torch.float32)
        if result[name].is_floating_point() and not torch.isfinite(result[name]).all():raise ValueError('Nonfinite average')
    return result,{'models':3,'weights':[1/3]*3,'accumulation':'CPU FP64 then original FP32','preserved_equal_tensors':sorted(preserved),'state_entries':len(schema),'tensor_bytes':raw_bytes,'schema':schema,'optimizer_updates':0}

def reference():
    ref=read(DOC/'AVERAGE_REFERENCE.json');auth=read(DOC/'AVERAGE_AUTHORIZATION.json')
    if not auth['approved'] or auth['hard_seconds']!=HARD or auth['coefficients']!=[1/3]*3:raise PermissionError('Wrong averaging authorization')
    if ref['authorization_SHA256']!=sha(DOC/'AVERAGE_AUTHORIZATION.json'):raise ValueError('Authorization changed')
    if [m['id'] for m in ref['models']]!=['C75','B75','B76','B77']:raise ValueError('Fixed model list changed')
    return ref

def admit(b):
    import shutil
    ref=reference()
    for n,h in ref['source_pins'].items():
        if sha(ROOT/n,b)!=h:raise ValueError('Source changed '+n)
    inputs=dict(ref['input_files']);inputs.update({ref['model_dir']+'/'+n:h for n,h in ref['generic_files'].items()})
    inputs.update({m['checkpoint']:m['checkpoint_sha256'] for m in ref['models']})
    for m in ref['models']:
        if m['id'] in ['C75','B75']:inputs[m['Dev_predictions']['path']]=m['Dev_predictions']['sha256']
    for n,h in inputs.items():
        if sha(ROOT/n,b)!=h:raise ValueError('Frozen input changed '+n)
    from inventory_ttmre_redocred_generic_control import memory_metadata
    mem=memory_metadata();free=shutil.disk_usage(ROOT).free
    if free<3*1024**3 or mem.get('effective_available_bytes',0)<32*1024**3:raise RuntimeError('Insufficient disk/RAM; no cleanup')
    # This stdlib subprocess gate must precede every Torch CUDA API.
    from ttmre_nvidia_smi_preflight import inspect_gpu
    external=inspect_gpu()
    if not external['passed']:raise RuntimeError('GPU occupied; no launch')
    return {'reference':ref,'input_files':inputs,'external_GPU_before_Torch':external,'memory':mem,'disk_free':free,
            'entry_SHA256':sha(__file__),'reference_SHA256':sha(DOC/'AVERAGE_REFERENCE.json'),'budget_hard_seconds':HARD}

def prepare(w,i,b):
    import evaluate_ttmre_final_fixed_test as final
    final.prepare(w,i,b)

def build(args,ref,m):
    import evaluate_ttmre_final_fixed_test as final
    return final.build(args,ref,m)

def predict(w,i,b):
    import torch
    import evaluate_ttmre_final_fixed_test as final
    coff,generic,core=final.modules();prior,_,_=coff.modules();ref=i['reference']
    from ttmre_dwie_runtime_fix import runtime_identity
    write(w/'runtime_predict.json',runtime_identity())
    B=[m for m in ref['models'] if m['id'] in ['B75','B76','B77']]
    paths=[ROOT/m['checkpoint'] for m in B]
    state,details=uniform_state(paths,b)
    derived=w/'B_uniform_average.pth'
    with derived.open('xb') as f:
        torch.save(state,f);f.flush();os.fsync(f.fileno())
    del state;gc.collect()
    average_hash=sha(derived,b)
    write(w/'averaging.json',{**details,'checkpoint':str(derived.relative_to(ROOT)),'checkpoint_SHA256':average_hash,
          'sources':[{'id':m['id'],'path':m['checkpoint'],'SHA256':m['checkpoint_sha256']} for m in B]})
    emit('B_FIXED_AVERAGE_SAVED',checkpoint_SHA256=average_hash,tensor_bytes=details['tensor_bytes'])
    args,_,dev,gold,annotated,distant=generic.data_context(w,i,b)
    avg={**B[0],'id':'B-average','checkpoint':str(derived.relative_to(ROOT)),'checkpoint_sha256':average_hash,
         'same_checkpoint_previously_Test_evaluated':False}
    systems={m['id']:m for m in ref['models'] if m['id'] in ['C75','B75']};systems['B-average']=avg
    order=[['C75','B75','B-average'],['B75','B-average','C75'],['B-average','C75','B75']]
    seen={};metrics={};observations=[]
    for rnd,names in enumerate(order,1):
        for name in names:
            check(b);m=systems[name]
            with coff.no_training():
                model,recipe=build(args,ref,m);before=core.preflight.state_digest(model);rng=core.preflight.rng_state()
                benchmark,pred=prior.benchmark_pass(model,dev,args,b)
                values=core.shared.score_predictions(pred,gold,annotated,distant)
                decision=facts(pred)
                if name in ['C75','B75']:
                    if decision!=facts(read(ROOT/m['Dev_predictions']['path'])):raise ValueError('Original Dev decisions differ '+name)
                    if any(not math.isclose(values[k],m['Dev_reference_metrics'][k],rel_tol=0,abs_tol=1e-6) for k in final.DEV_METRICS):raise ValueError('Original Dev metric differs '+name)
                if name in seen and seen[name]!=decision:raise ValueError('Predictions differ across timing rounds '+name)
                seen[name]=decision;metrics[name]=values
                core.preflight.assert_rng_equal(rng,core.preflight.rng_state())
                if before!=core.preflight.state_digest(model) or any(p.grad is not None for p in model.parameters()):raise ValueError('State/gradient mutation')
                stem=f'round{rnd}-{name}';write(w/(stem+'-predictions.json'),pred)
                row={'round':rnd,'system':name,'benchmark':benchmark,'metrics':values,'recipe':recipe,'state_SHA256':before,
                     'state_unchanged':True,'RNG_unchanged':True,'optimizer_updates':0}
                write(w/(stem+'.json'),row);observations.append(row)
                emit('B_AVERAGE_DEV_TIMING_COMPLETE',round=rnd,system=name,dev_F1=values['dev_F1'],forward_seconds=benchmark['model_forward_seconds'])
                del model;gc.collect();torch.cuda.empty_cache()
    write(w/'Dev_metrics.json',metrics)
    times={n:statistics.median(r['benchmark']['model_forward_seconds'] for r in observations if r['system']==n) for n in systems}
    write(w/'timing_summary.json',{'boundary':'FP32 batch4 full500Dev synchronized warm model-forward; median3 rotating rounds',
       'seconds':times,'B_average_reduction_vs_C75_pct':100*(1-times['B-average']/times['C75']),
       'B_average_ratio_to_B75':times['B-average']/times['B75'],'historical_C_average_timing_measured':False,'observations':observations})
    with coff.no_training():
        check(b);model,recipe=build(args,ref,avg);before=core.preflight.state_digest(model);rng=core.preflight.rng_state()
        features=read(w/'model_inputs.json');prepared=read(w/'prepared.json')
        if sha(w/'model_inputs.json',b)!=prepared['features_SHA256'] or len(features)!=500 or sum(len(x['hts']) for x in features)!=198670 or any(x['labels'] for x in features):raise ValueError('Unmasked/changed Test inputs')
        pred,single,group=final.inference(model,features,args.device,b,True)
        if not group['passed'] or group['pairs']!=198670:raise ValueError('Test native/single decision consistency failed')
        core.preflight.assert_rng_equal(rng,core.preflight.rng_state())
        if before!=core.preflight.state_digest(model) or any(p.grad is not None for p in model.parameters()):raise ValueError('Test state changed')
        recipe.update(state_SHA256=before,state_unchanged=True,RNG_unchanged=True,exploratory_after_prior_Test=True)
        write(w/'B_average_Test_predictions.json',pred);write(w/'B_average_Test_single_predictions.json',single)
        write(w/'B_average_Test_grouping.json',group);write(w/'B_average_recipe.json',recipe)
        del model;gc.collect();torch.cuda.empty_cache()
    for n,h in i['input_files'].items():
        if sha(ROOT/n,b)!=h:raise ValueError('Original input/Checkpoint changed '+n)
    if sha(derived,b)!=average_hash:raise ValueError('Derived checkpoint changed')
    sealed={n:sha(w/n,b) for n in ['B_average_Test_predictions.json','B_average_Test_single_predictions.json','B_average_Test_grouping.json','B_average_recipe.json','averaging.json','Dev_metrics.json','timing_summary.json']}
    write(w/'predictions_sealed.json',{'files':sealed,'metrics_computed':False,'identity_SHA256':sha(w/'identity.json',b),
       'prepared_SHA256':sha(w/'prepared.json',b),'Test_SHA256':ref['Test_SHA256'],'sealed_monotonic':time.monotonic()})
    emit('B_AVERAGE_TEST_PREDICTIONS_SEALED',metrics_computed=False)

def score(w,i,b):
    import evaluate_ttmre_final_fixed_test as final
    seal=read(w/'predictions_sealed.json')
    if seal['metrics_computed'] is not False or seal['identity_SHA256']!=sha(w/'identity.json',b) or seal['prepared_SHA256']!=sha(w/'prepared.json',b):raise ValueError('Bad global seal')
    for n,h in seal['files'].items():
        if sha(w/n,b)!=h:raise ValueError('Prediction seal changed '+n)
    write(w/'scoring_started.json',{'all_predictions_verified_before_scoring':True,'seal_SHA256':sha(w/'predictions_sealed.json',b)})
    ref=i['reference']
    if sha(w/'test_source.json',b)!=ref['Test_SHA256']:raise ValueError('Test gold changed')
    gold=read(w/'test_source.json');train=read(ROOT/ref['data_dir']/'train_revised.json')
    if len(gold)!=500 or len(train)!=3053:raise ValueError('Incomplete corpora')
    human={(a['name'],c['name'],r['r']) for d in train for r in d['labels'] for a in d['vertexSet'][r['h']] for c in d['vertexSet'][r['t']]}
    metric=final.score_one(read(w/'B_average_Test_predictions.json'),gold,human)
    if metric['gold']!=17448:raise ValueError('Gold dropped')
    write(w/'Test_metrics.json',{'B-average':metric})
    prior=ref['historical_comparators'];comparison={name:metric['test_F1']-v['test_F1'] for name,v in prior.items()}
    result={'passed':True,'marker':'LIGHTWEIGHT_UNIFORM_AVERAGE_COMPLETE','test_metrics':metric,'Dev_metrics':read(w/'Dev_metrics.json'),
       'test_F1_difference_pp':comparison,'historical_comparators':prior,'timing':read(w/'timing_summary.json'),
       'optimizer_updates':0,'original_checkpoint_writes':0,'new_derived_checkpoint':1,'new_target_seeds':0,
       'Test_predictions_sealed_before_scoring':True,'exploratory_after_prior_Test_exposure':True,'original_locked12_unchanged':True,
       'positive_gain_required_for_completion':False,'elapsed_seconds':time.monotonic()-b['start_monotonic'],'hard_seconds':HARD}
    check(b);write(w/'completion.json',result)
    with tarfile.open(w/'reports.tar.gz','x:gz') as t:
        for p in sorted(w.glob('*.json')):check(b);t.add(p,arcname=p.name)
    check(b);emit('LIGHTWEIGHT_UNIFORM_AVERAGE_COMPLETE',test_F1=metric['test_F1'],seconds=time.monotonic()-b['start_monotonic'])

def lock():
    import fcntl
    fd=9;a=os.fstat(fd);p=(ROOT/'experiments/2026-10-08-baseline-alignment/.training.lock').stat()
    if (a.st_dev,a.st_ino)!=(p.st_dev,p.st_ino):raise RuntimeError('Wrong shared lock')
    fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB);return fd
def supervise(cmd,b,fd):
    child=subprocess.Popen(cmd,start_new_session=True,pass_fds=(fd,));terminated=False
    try:
        while True:
            check(b,0);code=child.poll()
            if code is not None:
                if code or terminated:raise RuntimeError('Phase failed; preserve original budget and evidence')
                return
            if time.monotonic()>=b['deadline_monotonic']-TERM and not terminated:os.killpg(child.pid,signal.SIGTERM);terminated=True
            time.sleep(.2)
    finally:
        if child.poll() is None:os.killpg(child.pid,signal.SIGKILL);child.wait()
def worker(w,phase):
    lock();i=read(w/'identity.json');b=read(w/'budget.json');check(b)
    if b['hard_seconds']!=HARD or b['deadline_monotonic']-b['start_monotonic']!=HARD:raise ValueError('Budget changed')
    if sha(__file__)!=i['entry_SHA256'] or sha(DOC/'AVERAGE_REFERENCE.json')!=i['reference_SHA256']:raise ValueError('Entry/reference changed')
    try:
        if phase=='prepare':prepare(w,i,b)
        elif phase=='predict':predict(w,i,b)
        elif phase=='score':score(w,i,b)
        else:raise ValueError('Unknown phase')
    except BaseException as exc:
        write(w/('failure-'+phase+'.json'),{'phase':phase,'error':str(exc),'type':type(exc).__name__,'elapsed_seconds':time.monotonic()-b['start_monotonic']});raise
def start(w):
    fd=lock();start=float(os.environ['TTMRE_B_AVERAGE_START']);b={'start_monotonic':start,'deadline_monotonic':start+HARD,'hard_seconds':HARD,
       'start_utc':os.environ['TTMRE_B_AVERAGE_UTC'],'boot_id':os.environ['TTMRE_B_AVERAGE_BOOT'],'extension':False}
    check(b);claim=RUN_ROOT/'uniform-average-once-claim.json'
    if claim.exists() or w.exists() or w.parent.parent!=RUN_ROOT.resolve():raise ValueError('Not a new authorized run')
    w.mkdir();write(w/'budget.json',b);write(claim,{'workspace':str(w),'budget':b,'authorization_SHA256':sha(DOC/'AVERAGE_AUTHORIZATION.json')})
    try:
        i=admit(b);write(w/'identity.json',i);emit('B_AVERAGE_ADMISSION_COMPLETE',hard_seconds=HARD,remaining_seconds=b['deadline_monotonic']-time.monotonic())
        for phase in ['prepare','predict','score']:
            supervise([sys.executable,'-u','-B',__file__,'--mode',phase,'--run-dir',str(w)],b,fd)
        write(w/'supervisor_complete.json',{'passed':True,'elapsed_seconds':time.monotonic()-start,'report_SHA256':sha(w/'reports.tar.gz',b)})
    except BaseException as exc:
        write(w/'supervisor-failure.json',{'error':str(exc),'type':type(exc).__name__,'elapsed_seconds':time.monotonic()-start,'budget':b});raise
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--mode',choices=['start','prepare','predict','score'],required=True);ap.add_argument('--run-dir',type=Path,required=True);a=ap.parse_args()
    def interrupted(*_):raise TimeoutError('External deadline/termination; preserve evidence and original clock')
    signal.signal(signal.SIGTERM,interrupted)
    if a.mode=='start':start(a.run_dir.resolve())
    else:worker(a.run_dir.resolve(),a.mode)
