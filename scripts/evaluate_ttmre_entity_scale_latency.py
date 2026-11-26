"""Requested C75/B75 Re-DocRED Dev entity-scale latency; no Test forward/training.

500 documents, four pre-fixed bins, FP32 batch1, three alternating rounds.
One conservative900s implementation cap for the explicitly requested task.
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
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
DOC=ROOT/'refine-logs/REDOCRED_BUCKET_LATENCY_20261026'
REF_SHA='ec228f1b4f1517165a6e748ce9878f1bcb1205c002ce5651bd49df3f23d46725'
RUN_ROOT=ROOT/'experiments/2026-10-26-redocred-bucket-latency'
CLAIM=RUN_ROOT/'entity-scale-once-claim.json'
HARD,RESERVE,TERM=900,60,15
ORDER=(('C75','B75'),('B75','C75'),('C75','B75'))
LABELS=('N<=15','16<=N<=20','21<=N<=25','N>=26')

def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def write(p,x):
    with Path(p).open('x',encoding='utf-8') as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def emit(event,**x):print(json.dumps({'event':event,**x},allow_nan=False),flush=True)
def check(b,reserve=RESERVE):
    if b is None:return
    if time.monotonic()>=b['deadline_monotonic']-reserve:raise TimeoutError('Original900s latency deadline; no extension')
    boot=Path('/proc/sys/kernel/random/boot_id')
    if boot.exists() and boot.read_text().strip()!=b['boot_id']:raise RuntimeError('Original boot changed')
def sha(p,b=None):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for part in iter(lambda:f.read(4*1024**2),b''):check(b);h.update(part)
    return h.hexdigest()
def stamp(p):
    s=Path(p).stat();return [s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns]
def bucket(n):
    if type(n) is not int or n<2:raise ValueError('At least two supplied entities required')
    return 0 if n<=15 else 1 if n<=20 else 2 if n<=25 else 3
def validate():
    if sha(DOC/'REFERENCE.json')!=REF_SHA:raise ValueError('Frozen reference changed')
    r=read(DOC/'REFERENCE.json');a=r['authorization']
    if (not a['user_explicitly_requested_zero_training_inference'] or a['conservative_implementation_cap_seconds']!=HARD
        or a['optimizer_updates']!=0 or a['new_seeds']!=0 or any(a[k] for k in ['checkpoint_writes','new_Test_forward','downloads','installs','deletions','extension'])):
        raise PermissionError('Not the requested bounded inference task')
    if r['bin_edges']!=[15,20,25] or r['round_order']!=[list(v) for v in ORDER] or r['batch_size']!=1 or r['precision']!='float32':raise ValueError('Frozen timing protocol changed')
    if [m['id'] for m in r['models']]!=['C75','B75'] or len(r['Dev_metadata'])!=500:raise ValueError('Fixed models/Dev required')
    counts=[sum(bucket(d['entities'])==k for d in r['Dev_metadata']) for k in range(4)]
    if counts!=[128,166,131,75] or counts!=r['expected_bucket_counts'] or sum(d['pairs'] for d in r['Dev_metadata'])!=193232:raise ValueError('Full pre-fixed bucket membership changed')
    for name,h in r['original_source_pins'].items():
        if sha(ROOT/name)!=h:raise ValueError('Original source changed: '+name)
    return {'reference':r,'reference_SHA256':REF_SHA,'entry_SHA256':sha(__file__),
        'checker_SHA256':sha(ROOT/'scripts/check_ttmre_entity_scale_latency.py'),
        'bins':list(LABELS),'bucket_counts':counts,'Test_forward':False,'optimizer_updates':0,'new_seeds':0,
        'checkpoint_writes':0,'CUDA_initialized':False}
def resources():
    import shutil
    from inventory_ttmre_redocred_generic_control import memory_metadata
    mem=memory_metadata();free=shutil.disk_usage(ROOT).free
    if mem.get('effective_available_bytes',0)<32*1024**3 or free<512*1024**2:raise RuntimeError('Need32GiB effective RAM/512MiB report space; no cleanup/install')
    return {'memory':mem,'free_bytes':free}
def admit(b):
    i=validate();r=i['reference'];i['resources']=resources()
    files=dict(r['input_files']);files.update({r['model_dir']+'/'+n:h for n,h in r['generic_files'].items()})
    for m in r['models']:
        files[m['checkpoint']]=m['checkpoint_sha256'];files[m['Dev_predictions']['path']]=m['Dev_predictions']['sha256']
    for name,h in files.items():
        if sha(ROOT/name,b)!=h:raise ValueError('Asset/weight changed: '+name)
    from ttmre_nvidia_smi_preflight import inspect_gpu
    external=inspect_gpu()
    if not external['passed']:raise RuntimeError('GPU occupied before CUDA')
    i.update(input_files=files,external_GPU=external)
    return i
def modules():
    import evaluate_ttmre_final_fixed_test as fixed
    coff,generic,core=fixed.modules()
    return fixed,coff,generic,core
def require_lock():
    import fcntl
    if os.environ.get('TTMRE_SCALE_LOCK_FD')!='9':raise RuntimeError('Use entity-scale shared-lock context')
    current=os.fstat(9);expected=(ROOT/'experiments/2026-10-08-baseline-alignment/.training.lock').stat()
    if (current.st_dev,current.st_ino)!=(expected.st_dev,expected.st_ino):raise ValueError('Wrong shared GPU lock descriptor')
    fcntl.flock(9,fcntl.LOCK_EX|fcntl.LOCK_NB)
    return 9
def metadata(dataset,expected):
    rows=[]
    if len(dataset)!=len(expected):raise ValueError('Require complete fixed Dev')
    for index,(f,d) in enumerate(zip(dataset,expected)):
        n=len(f['entity_pos']);pairs=n*(n-1)
        if (f['title']!=d['title'] or n!=d['entities'] or len(f['hts'])!=pairs or pairs!=d['pairs']
            or f['hts']!=[[h,t] for h in range(n) for t in range(n) if h!=t] or f['labels']):raise ValueError('Label-dependent/incomplete candidates')
        rows.append({'index':index,'title':f['title'],'entities':n,'pairs':pairs,'model_tokens':len(f['input_ids']),'bucket':bucket(n)})
    return rows
def synchronize(device):
    if device.type=='cuda':__import__('torch').cuda.synchronize(device)
def timed_pass(model,dev,args,rows,b,forward=None,clock=None):
    import numpy as np
    import torch
    from evaluation import to_official
    _,_,_,core=modules();forward=forward or (lambda inputs:model(**inputs));clock=clock or time.perf_counter
    warm=[]
    for k in range(4):
        candidate=next((r['index'] for r in rows if r['bucket']==k),None)
        if candidate is not None:warm.append(candidate)
    warm.append(max(rows,key=lambda r:(r['model_tokens'],-r['index']))['index']);warm=list(dict.fromkeys(warm))
    model.eval();records=[];predictions=[]
    with torch.no_grad():
        for index in warm:
            check(b);inputs=core.shared.inputs_from_batch(core.shared.collate_fn([dev[index]]),args.device,labels=False)
            output=forward(inputs);del output,inputs
        synchronize(args.device)
        for f,row in zip(dev,rows):
            check(b)
            # Collation and transfer complete before starting the synchronized timer.
            inputs=core.shared.inputs_from_batch(core.shared.collate_fn([f]),args.device,labels=False)
            synchronize(args.device);t0=clock();output=forward(inputs);synchronize(args.device);ms=1000*(clock()-t0)
            if output.dtype!=torch.float32 or output.shape!=(row['pairs'],97) or not bool(torch.isfinite(output).all()):raise ValueError('Nonfinite/wrong geometry logits')
            if not math.isfinite(ms) or ms<=0:raise ValueError('Invalid timing')
            values=output.detach().cpu().numpy();flags=np.zeros_like(values);flags[:,1:]=values[:,1:]>values[:,:1];flags[:,0]=flags[:,1:].sum(1)==0
            predictions.extend(to_official(flags,[f]));records.append({**row,'forward_ms':ms})
            del output,inputs
    return records,predictions,{'warmup_document_indices':warm,'batch_size':1,'synchronized':args.device.type=='cuda',
       'timer_includes':'model forward plus host invocation and final synchronization',
       'timer_excludes':['loading','collation','host_to_device_transfer','decoding','finite/state checks']}
def summarize(passes):
    if [(p['round'],p['system']) for p in passes]!=[(j+1,n) for j,v in enumerate(ORDER) for n in v]:raise ValueError('Not all six fixed passes')
    reference=None
    for p in passes:
        rows=p['observations'];ids=[r['title'] for r in rows]
        if reference is None:reference=ids
        if ids!=reference or len(set(ids))!=len(ids):raise ValueError('Doc order/duplicates differ across passes')
    table=[]
    for k,label in enumerate(LABELS):
        selected={s:[r for p in passes if p['system']==s for r in p['observations'] if r['bucket']==k] for s in ['C75','B75']}
        c,b=selected['C75'],selected['B75']
        if not c or len(c)!=len(b) or len(c)%3:raise ValueError('Empty/unpaired bucket')
        n=len(c)//3;cm=statistics.mean(r['forward_ms'] for r in c);bm=statistics.mean(r['forward_ms'] for r in b)
        base=c[:n]
        table.append({'bucket':k,'entity_range':label,'documents':n,'observations_per_model':len(c),
            'mean_entities':statistics.mean(r['entities'] for r in base),'mean_pairs':statistics.mean(r['pairs'] for r in base),
            'mean_model_tokens':statistics.mean(r['model_tokens'] for r in base),
            'C_mean_ms_doc':cm,'B_mean_ms_doc':bm,'C_median_ms_doc':statistics.median(r['forward_ms'] for r in c),
            'B_median_ms_doc':statistics.median(r['forward_ms'] for r in b),'speedup_C_over_B':cm/bm,'B_time_reduction_percent':100*(1-bm/cm),
            'round_means':{s:[statistics.mean(r['forward_ms'] for r in p['observations'] if r['bucket']==k) for p in passes if p['system']==s] for s in ['C75','B75']}})
    allc=[r['forward_ms'] for p in passes if p['system']=='C75' for r in p['observations']]
    allb=[r['forward_ms'] for p in passes if p['system']=='B75' for r in p['observations']]
    return {'bins':table,'overall_C_ms_doc':statistics.mean(allc),'overall_B_ms_doc':statistics.mean(allb),
            'overall_speedup':statistics.mean(allc)/statistics.mean(allb),'passes':6,'rounds':3,'batch_size':1,
            'independent_training_seeds':False,'entity_bins_do_not_control_document_length':True}
def execute(w,i,b,factory=None,loader=None,context=None,pass_fn=None):
    import torch
    fixed,coff,generic,core=modules();r=i['reference']
    args,_,dev,gold,annotated,distant=context or generic.data_context(w,i,b)
    rows=metadata(dev,r['Dev_metadata']);write(w/'document_buckets.json',rows)
    models={m['id']:m for m in r['models']};passes=[];seen={};native={}
    protection={m['checkpoint']:stamp(ROOT/m['checkpoint']) for m in r['models']}
    for round_number,order in enumerate(ORDER,1):
        for system in order:
            check(b);started=time.monotonic();m=models[system]
            with coff.no_training():
                net,recipe=fixed.build(args,r,m,factory,loader);before=core.preflight.state_digest(net);rng=core.preflight.rng_state()
                if round_number==1:
                    p,_,_=fixed.inference(net,dev,args.device,b,False);metrics=core.shared.score_predictions(p,gold,annotated,distant)
                    reference=read(ROOT/m['Dev_predictions']['path'])
                    if coff.facts(p)!=coff.facts(reference) or any(not math.isclose(metrics[k],m['Dev_reference_metrics'][k],abs_tol=1e-6,rel_tol=0) for k in fixed.DEV_METRICS):raise ValueError('Native Dev replay failed: '+system)
                    native[system]=coff.facts(p);write(w/(system+'-native-Dev-replay.json'),{'passed':True,'metrics':metrics,'predictions':p})
                observations,pred,timing=(pass_fn or timed_pass)(net,dev,args,rows,b)
                facts=coff.facts(pred)
                if system in seen and facts!=seen[system]:raise ValueError('Fixed batch1 decisions differ across rounds')
                seen[system]=facts;difference=len(facts.symmetric_difference(native[system]))
                core.preflight.assert_rng_equal(rng,core.preflight.rng_state())
                if before!=core.preflight.state_digest(net) or any(p.grad is not None for p in net.parameters()):raise ValueError('Forward mutated state/gradients')
                out={'round':round_number,'system':system,'recipe':recipe,'timing_protocol':timing,'observations':observations,
                     'model_state_SHA256':before,'state_unchanged':True,'RNG_unchanged':True,'native_batch4_vs_single_fact_symmetric_difference':difference}
                stem='round%d-%s'%(round_number,system);write(w/(stem+'.json'),out);write(w/(stem+'-predictions.json'),pred);passes.append(out)
                del net;gc.collect()
                if args.device.type=='cuda':torch.cuda.empty_cache()
            emit('ENTITY_SCALE_PASS_COMPLETE',round=round_number,system=system,documents=len(observations),native_single_fact_difference=difference,passes_complete=len(passes))
            if b and 1.5*(time.monotonic()-started)*(6-len(passes))+RESERVE>b['deadline_monotonic']-time.monotonic():raise TimeoutError('Projected remaining work exceeds original900s')
    summary=summarize(passes);write(w/'summary.json',summary)
    for m in r['models']:
        p=ROOT/m['checkpoint']
        if stamp(p)!=protection[m['checkpoint']] or (factory is None and sha(p,b)!=m['checkpoint_sha256']):raise ValueError('Saved checkpoint modified')
    write(w/'completion.json',{'passed':True,'passes':6,'Dev_documents':len(rows),'observations':6*len(rows),
        'optimizer_updates':0,'new_seeds':0,'checkpoint_writes':0,'Test_forward_passes':0,'checkpoint_files_unchanged':True,
        'elapsed_seconds':time.monotonic()-b['start_monotonic'] if b else None,'batch_size':1})
def verify(w,b):
    c=read(CLAIM)
    if w.name!='workspace' or w.parent.parent!=RUN_ROOT.resolve() or c['workspace']!=str(w) or sha(w/'budget.json')!=c['budget_SHA256'] or sha(w/'identity.json')!=c['identity_SHA256']:
        raise ValueError('Run context changed')
    current=validate();old=read(w/'identity.json')
    if current['entry_SHA256']!=old['entry_SHA256'] or current['checker_SHA256']!=old['checker_SHA256']:raise ValueError('New source changed after claim')
    if b['hard_seconds']!=HARD or b['deadline_monotonic']!=b['start_monotonic']+HARD:raise ValueError('Original clock changed')
    check(b)
def worker(w):
    require_lock();b=read(w/'budget.json');verify(w,b);i=read(w/'identity.json')
    try:
        resources()
        from ttmre_nvidia_smi_preflight import inspect_gpu
        if not inspect_gpu()['passed']:raise RuntimeError('GPU occupied before CUDA')
        execute(w,i,b)
    except BaseException as e:
        write(w/'failure.json',{'error':str(e),'type':type(e).__name__,'optimizer_updates':0,'elapsed_seconds':time.monotonic()-b['start_monotonic']});raise
def supervise(cmd,b,fd):
    child=subprocess.Popen(cmd,start_new_session=True,pass_fds=(fd,));term=False
    try:
        while True:
            check(b,0);code=child.poll()
            if code is not None:
                if code or term:raise RuntimeError('Worker/checker failed; preserve original clock')
                return
            if time.monotonic()>=b['deadline_monotonic']-TERM and not term:os.killpg(child.pid,signal.SIGTERM);term=True
            time.sleep(.2)
    finally:
        if child.poll() is None:os.killpg(child.pid,signal.SIGKILL);child.wait()
def start(w):
    fd=require_lock();s=float(os.environ['TTMRE_SCALE_START'])
    if not math.isfinite(s) or s>time.monotonic():raise ValueError('Invalid original clock')
    b={'start_monotonic':s,'deadline_monotonic':s+HARD,'hard_seconds':HARD,'boot_id':os.environ['TTMRE_SCALE_BOOT'],'start_UTC':os.environ['TTMRE_SCALE_UTC']}
    if w.exists() or w.name!='workspace' or w.parent.parent!=RUN_ROOT.resolve() or CLAIM.exists():raise ValueError('Use one new scope-specific task')
    w.mkdir();write(w/'budget.json',b)
    try:
        i=admit(b);write(w/'identity.json',i);write(CLAIM,{'workspace':str(w),'budget_SHA256':sha(w/'budget.json'),'identity_SHA256':sha(w/'identity.json')})
        supervise([sys.executable,'-B',str(ROOT/'scripts/check_ttmre_entity_scale_latency.py')],b,fd)
        supervise([sys.executable,'-u','-B',str(Path(__file__).resolve()),'--mode','worker','--run-dir',str(w)],b,fd)
    except BaseException as e:
        write(w/'supervisor-failure.json',{'error':str(e),'type':type(e).__name__,'budget':b,'optimizer_updates':0});raise
    finally:
        if time.monotonic()<b['deadline_monotonic']-10:
            with tarfile.open(w/'reports.tar.gz','x:gz') as t:
                for p in sorted(w.glob('*.json')):t.add(p,arcname=p.name)
            emit('ENTITY_SCALE_REPORT_READY',bytes=(w/'reports.tar.gz').stat().st_size,SHA256=sha(w/'reports.tar.gz',b))
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--mode',choices=('validate','start','worker'),default='validate');p.add_argument('--run-dir',type=Path);a=p.parse_args()
    if a.mode=='validate':v=validate();print(json.dumps({'passed':True,'buckets':v['bucket_counts'],'Test_forward':False,'new_training':False}))
    elif a.run_dir is None:raise ValueError('Use the sole900s launcher')
    elif a.mode=='start':start(a.run_dir.resolve())
    else:worker(a.run_dir.resolve())
if __name__=='__main__':main()
