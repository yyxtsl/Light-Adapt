"""Approved one U75 Human control, functional B start then untied projections, cap2h.

Default validate/storage read-only. No Test/Distant/B/C retraining, tuning,
extra seed, retry/resume/extension, automatic follow-up or deletion.
"""
from __future__ import annotations
import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import gc
import json
import math
import os
from pathlib import Path
import shutil
import signal
import stat
import statistics
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import train_ttmre_pair_replacement as prior

core,base,old,soup=prior.core,prior.base,prior.old,prior.soup
read,write_json,digest,emit=prior.read,prior.write_json,prior.digest,prior.emit
RUN_ROOT=ROOT/'experiments/2026-10-23-untied-pair-control'
CLAIM='untied-pair-control-once-claim.json'
FILES=tuple('scripts/'+n for n in ('train_ttmre_untied_pair_control.py','check_ttmre_untied_pair_control.py','run_ttmre_untied_pair_control.sh'))
SEED,HUMAN_DOCS,DEV_DOCS,EPOCHS=75,3053,500,30
HARD_SECONDS,SOFT_SECONDS,TERM_SECONDS=7200,120,30
GIB=1024**3
METRICS=prior.METRICS
B_SHA='cfe8c2685c7ab65975ed7ab82ae3cdf565a50a2e1f7a9cd3042969365d9d26d8'
PINS={'identity.json':'b08ca33dcb53325f5865718403b267426ffcd6fe881f82c611e7bdd2913ffd96',
 'comparison/completion.json':'52ea8e62de575ce0622a94ab7123069b2cadb49deec5c478538a40fd43e0c51b',
 'bypass/completion.json':'c322d5a2a6248d1c491826aac8b3c8a7f46b6900fe8699b59d45a38c0400520f',
 'bypass/manifest.json':'a72e7662a7f05d67cf3ffa6be8a8f619ea343e561b09473bdcb4ee05d1362808',
 'bypass/training_order.json':'4e402941275b236aad417c53cadafaca4d6b12edff81b47dacf80c6fcd9af8ec',
 'bypass/predictions.json':'778288ca19911726f1128c3c96b4ec13a70533cbdd7cfc65e3a17c442d20862d'}
PROGRESS={'phase':'not-started','formal_optimizer_updates':0}


def updates():return HUMAN_DOCS//4*EPOCHS


def protocol():
    return {'version':'untied-pair-control-v1','seed':SEED,'epochs':EPOCHS,'new_optimizer_updates':updates(),
      'Distant_parent_sha256':base.PARENT_SHA,'variant':'U: ATLOP-style independent head/tail, head copied into tail before Human',
      'official_ATLOP_full_reproduction':False,'loss':'original ATGL alpha16.5','precision':'float32','batch_size':4,
      'backbone_lr':3e-5,'head_and_tail_lr':1e-5,'eps':1e-6,'weight_decay':.01,'warmup_steps':int(updates()*.06),'clip':1.,
      'selection':'penultimate batch raw Dev F1; earliest tie','decoder':'all relation logits > TH; no top-k cap',
      'benchmark_order':[['bypass','untied'],['untied','bypass'],['bypass','untied']],
      'B_incremental_gate':{'B_minus_U_F1_min_pp':.20,'B_minus_U_IgnA_min_pp':0.,'B_over_U_latency_max':1.05,'B_over_U_peak_max':1.05},
      'U_competitive_gate':{'U_minus_B_F1_min_pp':-.10,'U_minus_B_IgnA_min_pp':-.10,'U_over_B_latency_max':1.05},
      'new_Distant_updates':0,'new_reference_updates':0,'test_data_read':False,'hard_seconds':HARD_SECONDS,
      'soft_reserve_seconds':SOFT_SECONDS,'term_reserve_seconds':TERM_SECONDS,
      'automatic_retry':False,'automatic_resume':False,'extension':False,'automatic_followup':False}


def check_budget(b,reserve=SOFT_SECONDS):
    if b is None:return
    if time.monotonic()>=b['deadline_monotonic']-reserve:raise TimeoutError('U75 two-hour reserve reached')
    if os.name=='posix' and Path('/proc/sys/kernel/random/boot_id').read_text().strip()!=b['boot_id']:raise TimeoutError('Boot changed; no reset')


def verify_files(files,b=None):
    for p,h in files.items():
        check_budget(b)
        if digest(Path(p),b)!=h:raise ValueError('Pinned file changed: '+p)


def validate(b=None):
    w=Path(read(prior.RUN_ROOT/prior.CLAIM)['workspace'])
    if w!=w.resolve() or w.name!='workspace' or w.parent.parent!=prior.RUN_ROOT.resolve():raise ValueError('Unexpected B75 source workspace')
    pins={str(w/n):h for n,h in PINS.items()};verify_files(pins,b)
    origin,done,bc,bm,bo=(read(w/n) for n in ('identity.json','comparison/completion.json','bypass/completion.json','bypass/manifest.json','bypass/training_order.json'))
    if (not done['passed'] or done['marker']!='PAIR_REPLACEMENT_COMPLETE' or not bc['passed'] or bc['optimizer_steps']!=updates()
        or bc['checkpoint_sha256']!=B_SHA or bo['seed']!=75 or origin['pinned_files'][origin['distant_checkpoint']]!=base.PARENT_SHA):raise ValueError('Original B75/C/Distant source differs')
    pins.update(origin['pinned_files']);pins[str(w/'bypass/best_model.pth')]=B_SHA
    verify_files(pins,b)
    sources=dict(origin['source_files']);verify_files({str(ROOT/n):h for n,h in sources.items()},b)
    sources.update({n:digest(ROOT/n,b) for n in FILES})
    return {'protocol':protocol(),'parent':origin['parent'],'pair_reference':origin,
      'distant_checkpoint':origin['distant_checkpoint'],'B_checkpoint':str(w/'bypass/best_model.pth'),
      'B_predictions':str(w/'bypass/predictions.json'),'B_metrics':{k:bc['best'][k] for k in METRICS},
      'C_predictions':origin['reference_predictions'],'C_metrics':origin['reference_metrics'],
      'pairing':{'optimizer_groups':bm['optimizer_groups'],'initial_rng_sha256':bm['initial_rng_sha256'],
                 'sample_order_sha256':bo['sample_order_sha256'],'final_rng_sha256':bo['final_rng_sha256']},
      'pinned_files':pins,'source_files':sources,'test_data_read':False,'Distant_features_read':False,
      'stronger_existing_ATGLsoup_Dev_metrics':origin['stronger_existing_ATGLsoup_Dev_metrics']}


def storage_inventory():
    art=ROOT/'experiments/2026-10-19-atgl-rdrop-human/atgl-rdrop-human-launch-20261019-165957-xYJ0t4/workspace/atgl-rdrop/human-frozen/artifacts'
    expected='adc97644eba4703561092c5c722af429e5ee33159c85e39fc830a709980c1c4f'
    verify_files({str(art/'completion.json'):'775087e5f88edb6b1796c0b5ee8fd6543ff409a89b2ca8c61d08db11561ef788'})
    c=read(art/'completion.json')
    if not c['passed'] or c['checkpoint_sha256']!=expected:raise ValueError('Old RDrop75 metadata differs')
    path=art/'best_model.pth';row={'path':str(path),'already_absent':not path.exists()}
    if path.exists():
        st=path.lstat()
        if path.resolve()!=path or path.is_symlink() or not stat.S_ISREG(st.st_mode) or st.st_nlink!=1 or st.st_dev!=ROOT.stat().st_dev:raise ValueError('Candidate is not a canonical single-link project file')
        verify_files({str(path):expected});row.update(size=st.st_size,allocated_bytes=getattr(st,'st_blocks',0)*512,sha256=expected,nlink=st.st_nlink)
    return {'mode':'read_only','free_bytes':shutil.disk_usage(ROOT).free,'candidates':[row],
            'allocated_bytes_potentially_releasable':row.get('allocated_bytes',0),'files_deleted':0,'deletion_authorized':False,'training_started':False,
            'impact_if_deleted':'Old RDrop75 weight cannot be directly replayed; reports/predictions/claims/C/B/D/Distant/cache preserved'}


def resources(path,i,phase='start'):
    required=max(2*GIB if phase=='start' else 512*1024**2,(Path(i['distant_checkpoint']).stat().st_size+512*1024**2) if phase in ('start','train') else 0)
    free=shutil.disk_usage(path).free
    if free<required:raise RuntimeError('Project disk free %.3fGiB < required %.3fGiB; no automatic deletion'%(free/GIB,required/GIB))
    return {'phase':phase,'free_bytes':free,'required_bytes':required,'recommended_available_RAM_GiB':32,'new_best_files':1,'old_files_deleted':False}


def data_context(w,i,b):return prior.data_context(w,i,b)


def construct(args,i,kind,checkpoint=None):
    import torch
    model,installation=prior.construct(args,i['pair_reference'],'bypass',None)
    if kind=='untied':
        rng=core.preflight.rng_state();before={n:p.detach().clone() for n,p in model.named_parameters() if n.startswith('tail_extractor.')}
        model.tail_extractor.load_state_dict(model.head_extractor.state_dict(),strict=True)
        if model.head_extractor.weight is model.tail_extractor.weight or model.head_extractor.weight.data_ptr()==model.tail_extractor.weight.data_ptr():raise ValueError('U projectors must be independent parameter objects')
        model.args.model_type='ATLOP';core.preflight.assert_rng_equal(rng,core.preflight.rng_state())
        installation={**installation,'variant':'untied','explicit_transform':'copy head_extractor weight+bias into independent tail_extractor',
                      'head_tail_equal_at_start':all(torch.equal(a,b) for a,b in zip(model.head_extractor.state_dict().values(),model.tail_extractor.state_dict().values())),
                      'head_tail_independent_objects':True,'copy_preserved_RNG':True,
                      'original_tail_changed_by_explicit_copy':{n:not torch.equal(v,dict(model.named_parameters())[n]) for n,v in before.items()}}
    elif kind!='bypass':raise ValueError('Only B/U allowed')
    if checkpoint is not None:model.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=True),strict=True)
    return model,installation


def initial_equivalence(model,human,args,b):
    import torch
    before=core.preflight.state_digest(model);rng=core.preflight.rng_state();flags=[(m,m.training) for m in model.modules()];rows=[]
    try:
        model.train()
        for name,indices in human.probe_indices().items():
            check_budget(b);batch=core.shared.collate_fn([human[j] for j in indices]);pair_rng=core.preflight.rng_state()
            with torch.no_grad():
                model.args.model_type='ttmre';bl,bp=base.training_forward(model,batch,args.device)
                core.preflight.restore_rng(pair_rng)
                model.args.model_type='ATLOP';ul,up=base.training_forward(model,batch,args.device)
            row={'probe':name,'indices':indices,'logits_equal':torch.equal(bp,up),'loss_equal':torch.equal(bl,ul),
                 'max_abs_logit_delta':float((bp-up).abs().max()),'loss_delta':float((bl-ul).abs()),'optimizer_updates':0}
            rows.append(row)
            if not row['logits_equal'] or not row['loss_equal']:raise ValueError('Initial U and B functions differ with matched dropout RNG')
            del bl,bp,ul,up,batch
            model.sims=None
    finally:
        model.args.model_type='ATLOP';core.preflight.restore_rng(rng)
        for m,f in flags:m.training=f
    if before!=core.preflight.state_digest(model):raise ValueError('Initial functional probe changed tensors')
    return {'passed':True,'probes':rows,'optimizer_updates':0,'parameters_unchanged':True,'RNG_restored':True,'matched_training_dropout':True}


def probe(model,human,args,b):
    import torch
    tail=[]
    def observe(grad):
        tail.append({'finite':bool(torch.isfinite(grad).all()),'nonzero':bool(torch.count_nonzero(grad))});return grad
    hook=model.tail_extractor.weight.register_hook(observe)
    try:report=base.probe(model,human,args,b)
    finally:hook.remove()
    if len(tail)!=len(report['probes']) or not all(r['finite'] and r['nonzero'] for r in tail):raise ValueError('Independent tail probe gradient missing')
    return {**report,'tail_gradients':tail,'tail_gradient_verified':True}


@contextmanager
def tail_witness(model,optimizer,d):
    import torch
    pending={};result={}
    def pre(opt,_args,_kwargs):
        if result:return
        p=model.tail_extractor.weight;lr=next(g['lr'] for g in opt.param_groups if any(v is p for v in g['params']))
        if lr>0 and p.grad is not None and torch.count_nonzero(p.grad):pending.update(before=p.detach().clone(),lr=lr)
    def post(opt,_args,_kwargs):
        if not pending:return
        p=model.tail_extractor.weight;step=int(opt.state[p]['step'].item());changed=not torch.equal(p.detach(),pending['before'])
        if not changed:raise RuntimeError('Positive-LR tail optimizer update did not change tail')
        result.update(passed=True,parameter='tail_extractor.weight',step=step,optimizer_state_step=step,lr=pending['lr'],parameter_changed=True,extra_optimizer_updates=0)
        write_json(d/'tail_learning_update_check.json',result);emit('UNTIED_PAIR_TAIL_LEARNING_VERIFIED',**result);pending.clear()
    a=optimizer.register_step_pre_hook(pre);z=optimizer.register_step_post_hook(post)
    try:yield result
    finally:a.remove();z.remove()


@contextmanager
def training_scope(b):
    saved=(core.training_forward,core.emit,core.aligned.emit,core.atomic_torch_save);snapshot={}
    def save(value,path):
        if Path(path).name!='best_model.pth':raise ValueError('Unexpected checkpoint target')
        snapshot.clear();snapshot.update(path=Path(path),state=value)
    def event(name,**fields):
        if name=='N1_FULL_SEED75_LEARNING_UPDATE_VERIFIED':name='UNTIED_PAIR_LEARNING_UPDATE_VERIFIED'
        if 'step' in fields:PROGRESS['formal_optimizer_updates']=fields['step']
        emit(name,**fields)
        if name=='epoch_complete' and fields['epoch']==1:
            estimate=fields['seconds']*(EPOCHS-1)+900;remaining=b['deadline_monotonic']-time.monotonic()-SOFT_SECONDS
            emit('UNTIED_PAIR_PROJECTION',estimate_seconds=estimate,remaining_seconds=remaining)
            if estimate>remaining:raise TimeoutError('Measured projection exceeds original two-hour cap')
    core.training_forward=base.training_forward;core.emit=core.aligned.emit=event;core.atomic_torch_save=save
    try:yield snapshot
    finally:core.training_forward,core.emit,core.aligned.emit,core.atomic_torch_save=saved


def execute(w,i,b):
    import torch
    import numpy as np
    from torch.utils.data import DataLoader
    from transformers.optimization import get_linear_schedule_with_warmup
    d=w/'untied';d.mkdir();write_json(d/'resource_check.json',resources(w,i,'train'))
    args,human,dev,gold,annotated,distant=data_context(w,i,b)
    model,install=construct(args,i,'untied');write_json(d/'initialization.json',install)
    memory=model.mu_encoder.memory_tokens.detach().cpu().clone();tail_initial=model.tail_extractor.weight.detach().cpu().clone()
    hook=model.register_forward_pre_hook(lambda *_:check_budget(b));snapshot=None
    try:
        write_json(d/'initial_function_equivalence.json',initial_equivalence(model,human,args,b))
        write_json(d/'full_model_check.json',probe(model,human,args,b))
        opt=core.baseline.optimizer_for_stage(model,'human-frozen');groups=core.baseline.optimizer_report(model,opt)
        if opt.state:raise ValueError('Optimizer must be fresh')
        for actual,ref in zip(groups,i['pairing']['optimizer_groups']):
            if actual['parameter_names']!=ref['parameter_names'] or actual['lr']!=ref['lr'] or actual['weight_decay']!=ref['weight_decay']:raise ValueError('Original optimizer groups changed')
        core.shared.seed_all(SEED,torch,np)
        loader=DataLoader(human,batch_size=4,shuffle=True,drop_last=True,collate_fn=core.shared.collate_fn)
        dl=DataLoader(dev,batch_size=4,shuffle=False,collate_fn=core.shared.collate_fn)
        scheduler=get_linear_schedule_with_warmup(opt,int(updates()*.06),updates());rng=core.rng_digest()
        if len(loader)*EPOCHS!=updates() or rng!=i['pairing']['initial_rng_sha256']:raise ValueError('Initial RNG/update budget differs from B')
        write_json(d/'manifest.json',{'protocol':protocol(),'optimizer_groups':groups,'initial_rng_sha256':rng,'total_steps':updates()})
        with tail_witness(model,opt,d) as tail,training_scope(b) as snapshot:
            result=core.train_loop(model,loader,dl,args,'untied','human-frozen',d,gold,annotated,distant,core.method.frequency_buckets(human.metadata['priors']),b,opt,scheduler)
        if not tail:raise ValueError('No actual independent tail learning witness')
        del opt,scheduler
        rows=[json.loads(l) for l in (d/'metrics.jsonl').read_text().splitlines()];best=max(rows,key=lambda r:r['dev_F1'])
        if len(rows)!=EPOCHS or best['epoch']!=result['best_epoch']:raise ValueError('Selection trajectory incomplete')
        pair=i['pairing'];order={'seed':SEED,'formal_optimizer_updates':result['optimizer_steps'],'sample_order_sha256':result['sample_order_sha256'],
          'final_rng_sha256':result['final_rng_sha256'],'same_as_reference_order':result['sample_order_sha256']==pair['sample_order_sha256'],
          'same_as_reference_final_rng':result['final_rng_sha256']==pair['final_rng_sha256']}
        write_json(d/'training_order.json',order)
        if result['optimizer_steps']!=updates() or not order['same_as_reference_order'] or not order['same_as_reference_final_rng']:raise ValueError('Actual order/RNG/count differs from B')
        if not snapshot or snapshot['path']!=d/'best_model.pth' or snapshot['path'].exists():raise ValueError('Missing/conflicting CPU best')
        check_budget(b);core.atomic_torch_save(snapshot['state'],d/'best_model.pth');snapshot=None
        model.zero_grad(set_to_none=True);model.load_state_dict(torch.load(d/'best_model.pth',map_location='cpu',weights_only=True),strict=True)
        changed=not torch.equal(tail_initial,model.tail_extractor.weight.detach().cpu())
        if not changed:raise ValueError('Selected best tail never changed')
        best=read(d/'best_metrics.json');replay,pred=old.grouped_replay(model,dev,args,b,best,memory)
        score=core.shared.score_predictions(pred,gold,annotated,distant)
        replay.update(metrics=score,metric_replay_passed=all(math.isclose(score[k],best[k],abs_tol=1e-6,rel_tol=0) for k in METRICS))
        replay['passed']=replay['passed'] and replay['metric_replay_passed'];write_json(d/'checkpoint_replay.json',replay)
        if not replay['passed']:raise ValueError('Selected checkpoint replay failed')
        old.save_prediction_report(d,pred,[{'title':f['title'],'hts':f['hts']} for f in dev],score,gold,human)
        result.update(passed=True,protocol=protocol(),best=best,checkpoint_sha256=digest(d/'best_model.pth',b),best_checkpoint_writes=1,
          source_checkpoint_sha256=base.PARENT_SHA,tail_learning_verified=True,selected_best_tail_changed=True,memory_buffer_unchanged=True,checkpoint_replay_passed=True)
        write_json(d/'completion.json',result);emit('UNTIED_PAIR_ARM_COMPLETE',optimizer_updates=updates(),F1=score['dev_F1'])
    except BaseException:
        if snapshot and not (d/'best_model.pth').exists() and time.monotonic()<b['deadline_monotonic']-60:
            needed=sum(v.numel()*v.element_size() for v in snapshot['state'].values())+64*1024**2
            if shutil.disk_usage(d).free>=needed:
                core.atomic_torch_save(snapshot['state'],d/'best_model.pth');write_json(d/'incomplete_best_preserved.json',{'complete':False,'resumable':False})
        raise
    finally:hook.remove();PROGRESS['formal_optimizer_updates']=int(getattr(model,'_n1_completed_optimizer_steps',0))


def verify_arm(w,i,b):
    d=w/'untied';c=read(d/'completion.json');o=read(d/'training_order.json');l=read(d/'tail_learning_update_check.json');h=read(d/'learning_update_check.json')
    if (not c['passed'] or c['protocol']!=protocol() or c['optimizer_steps']!=updates() or c['source_checkpoint_sha256']!=base.PARENT_SHA
      or c['checkpoint_sha256']!=digest(d/'best_model.pth',b) or c['best_checkpoint_writes']!=1 or not c['selected_best_tail_changed']
      or not c['memory_buffer_unchanged'] or not read(d/'checkpoint_replay.json')['passed'] or not read(d/'initial_function_equivalence.json')['passed']
      or not o['same_as_reference_order'] or not o['same_as_reference_final_rng']
      or any(o[k]!=i['pairing'][k] for k in ('sample_order_sha256','final_rng_sha256'))
      or any(not r['passed'] or not r['parameter_changed'] or r['lr']<=0 or r['step']!=r['optimizer_state_step'] or r['extra_optimizer_updates']!=0 for r in (l,h))):raise ValueError('Incomplete or changed U75 arm')
    rows=[json.loads(line) for line in (d/'metrics.jsonl').read_text().splitlines()];best=read(d/'best_metrics.json')
    if len(rows)!=EPOCHS or any(r['epoch']!=j or r['step']!=j*(HUMAN_DOCS//4)-1 or r['batch']!=HUMAN_DOCS//4-1 for j,r in enumerate(rows,1)) or max(rows,key=lambda r:r['dev_F1'])['epoch']!=best['epoch']:raise ValueError('Selection timing mismatch')
    return c


def investment(metrics,bench):
    b,u=metrics['B'],metrics['U'];bc,uc=bench['bypass'],bench['untied']
    specific=(b['dev_F1']-u['dev_F1']>=.20 and b['dev_IgnF1_annotated']>=u['dev_IgnF1_annotated']
       and bc['median_model_forward_seconds']<=1.05*uc['median_model_forward_seconds'] and bc['max_peak_allocated_bytes']<=1.05*uc['max_peak_allocated_bytes'])
    competitive=(u['dev_F1']>=b['dev_F1']-.10 and u['dev_IgnF1_annotated']>=b['dev_IgnF1_annotated']-.10 and uc['median_model_forward_seconds']<=1.05*bc['median_model_forward_seconds'])
    return {'B_shared_incremental_investment_gate':bool(specific),'U_simple_competitive_gate':bool(competitive),
            'interpretation':'Shared projection has single-seed incremental evidence' if specific else 'Simple untied projection is competitive' if competitive else 'Mixed evidence; no automatic tuning or follow-up'}


def finish(w,i,b):
    import torch
    c=verify_arm(w,i,b);args,human,dev,gold,annotated,distant=data_context(w,i,b)
    metrics={'U':{k:c['best'][k] for k in METRICS}};d=w/'comparison';d.mkdir()
    for label in ('B','C'):
        pred=read(i[label+'_predictions']);actual=core.shared.score_predictions(pred,gold,annotated,distant)
        if any(not math.isclose(actual[k],i[label+'_metrics'][k],abs_tol=1e-6,rel_tol=0) for k in METRICS):raise ValueError('Old reference predictions changed')
        metrics[label]=actual;write_json(d/(label+'-predictions.json'),pred)
    expected={'bypass':soup.facts(read(i['B_predictions'])),'untied':soup.facts(read(w/'untied/predictions.json'))};raw=[]
    for j,order in enumerate(protocol()['benchmark_order']):
        for kind in order:
            check_budget(b);start_load=time.perf_counter()
            model,_=construct(args,i,kind,i['B_checkpoint'] if kind=='bypass' else w/'untied/best_model.pth')
            model.requires_grad_(False);model.eval();prior.synchronize(args.device);load=time.perf_counter()-start_load
            before=core.preflight.state_digest(model);timing,pred=prior.benchmark_pass(model,dev,args,b)
            if before!=core.preflight.state_digest(model) or soup.facts(pred)!=expected[kind]:raise ValueError('Benchmark state/prediction changed')
            row={'round':j+1,'variant':kind,'load_seconds_excluded_from_forward':load,'parameters_unchanged':True,'prediction_replay_passed':True,**timing}
            raw.append(row);write_json(w/('benchmark-round%d-%s.json'%(j+1,kind)),row);emit('UNTIED_PAIR_BENCHMARK',**row)
            del model,pred;gc.collect()
            if args.device.type=='cuda':torch.cuda.empty_cache()
    summary={}
    for kind in ('bypass','untied'):
        rows=[r for r in raw if r['variant']==kind]
        summary[kind]={'median_model_forward_seconds':statistics.median(r['model_forward_seconds'] for r in rows),
                      'median_end_to_end_seconds':statistics.median(r['end_to_end_seconds'] for r in rows),
                      'max_peak_allocated_bytes':max(r['peak_allocated_bytes'] for r in rows),'max_peak_reserved_bytes':max(r['peak_reserved_bytes'] for r in rows),'rounds':3}
    gates=investment(metrics,summary);write_json(d/'benchmark.json',{'raw':raw,'summary':summary,'environment':i['parent']['environment']})
    write_json(w/'resource_finish.json',resources(w,i,'finish'));verify_files(i['pinned_files'],b);verify_files({str(ROOT/n):h for n,h in i['source_files'].items()},b);verify_arm(w,i,b)
    result={'passed':True,'marker':'UNTIED_PAIR_CONTROL_COMPLETE','protocol':protocol(),'metrics':metrics,'benchmark':summary,**gates,
      'B_minus_U_pp':{k:metrics['B'][k]-metrics['U'][k] for k in METRICS},'new_optimizer_updates':updates(),'new_reference_updates':0,'new_Distant_updates':0,
      'test_data_read':False,'test_evaluated':False,'automatic_followup':False,'novelty_established':False,'elapsed_seconds':time.monotonic()-b['start_monotonic'],
      'stronger_existing_ATGLsoup_Dev_metrics':i['stronger_existing_ATGLsoup_Dev_metrics']}
    write_json(d/'completion.json',result);soup.archive_reports(w,b);emit('UNTIED_PAIR_CONTROL_COMPLETE',**gates,reports=str(d/'reports.tar.gz'))


def require_lock():
    if os.name!='posix' or os.environ.get('TTMRE_UNTIED_PAIR_LOCK_FD')!='9':raise RuntimeError('Use sole U75 launcher')
    import fcntl
    a,z=os.fstat(9),(ROOT/'experiments/2026-10-08-baseline-alignment/.training.lock').stat()
    if (a.st_dev,a.st_ino)!=(z.st_dev,z.st_ino):raise RuntimeError('Wrong shared lock')
    fcntl.flock(9,fcntl.LOCK_EX|fcntl.LOCK_NB);return 9


def verify_context(w,b):
    c=read(RUN_ROOT/CLAIM)
    if (w!=w.resolve() or w.name!='workspace' or w.parent.parent!=RUN_ROOT.resolve() or b['hard_seconds']!=HARD_SECONDS
      or not math.isclose(b['deadline_monotonic']-b['start_monotonic'],HARD_SECONDS,abs_tol=1e-6,rel_tol=0)
      or any(b[k] is not False for k in ('automatic_retry','automatic_resume','extension')) or c['workspace']!=str(w) or c['protocol']!=protocol()
      or c['budget_sha256']!=digest(w/'budget.json') or c['identity_sha256']!=digest(w/'identity.json')):raise ValueError('Changed immutable2h context')
    check_budget(b)


def worker(w,phase):
    require_lock();b=read(w/'budget.json');verify_context(w,b);i=read(w/'identity.json')
    PROGRESS.update(phase=phase,formal_optimizer_updates=updates() if phase=='finish' else 0)
    def stop(*_):raise TimeoutError('Interrupted; no resume')
    signal.signal(signal.SIGTERM,stop)
    try:
        if i['protocol']!=protocol():raise ValueError('Protocol changed')
        verify_files(i['pinned_files'],b);verify_files({str(ROOT/n):h for n,h in i['source_files'].items()},b)
        if phase=='finish':finish(w,i,b)
        else:execute(w,i,b)
    except BaseException as error:
        write_json(w/('failure-'+phase+'.json'),{'error':str(error),'type':type(error).__name__,'complete':False,'progress':PROGRESS,'automatic_retry':False,'resumable':False})
        try:
            if not (w/'comparison/reports.tar.gz').exists():soup.archive_reports(w,b)
        except BaseException:pass
        raise


def start(w):
    fd=require_lock()
    if w.name!='workspace' or w.parent.parent!=RUN_ROOT.resolve() or w.exists():raise ValueError('Use new U75 workspace')
    if (RUN_ROOT/CLAIM).exists():raise FileExistsError('Already claimed; no retry')
    now=time.monotonic();b={'start_monotonic':now,'deadline_monotonic':now+HARD_SECONDS,'hard_seconds':HARD_SECONDS,
      'soft_reserve_seconds':SOFT_SECONDS,'term_reserve_seconds':TERM_SECONDS,'start_utc':datetime.now(timezone.utc).isoformat(),
      'boot_id':Path('/proc/sys/kernel/random/boot_id').read_text().strip(),'automatic_retry':False,'automatic_resume':False,'extension':False}
    def stop(*_):raise TimeoutError('U75 supervisor interrupted')
    signal.signal(signal.SIGTERM,stop)
    try:
        emit('UNTIED_PAIR_BUDGET_STARTED',**b);i=validate(b);emit('UNTIED_PAIR_SOURCES_VERIFIED',resources=resources(w.parent,i))
        old.supervise([sys.executable,'-B',str(ROOT/FILES[1])],b,fd);check_budget(b)
        w.mkdir();write_json(w/'budget.json',b);write_json(w/'identity.json',i)
        with (RUN_ROOT/CLAIM).open('x',encoding='utf-8') as f:
            json.dump({'workspace':str(w),'protocol':protocol(),'budget_sha256':digest(w/'budget.json'),'identity_sha256':digest(w/'identity.json'),'authorization':'user confirmed one U75 Human control,total2h'},f,indent=2);f.flush();os.fsync(f.fileno())
        command=[sys.executable,'-u','-B',str(Path(__file__).resolve()),'--mode','worker','--run-dir',str(w)]
        for phase in ('train','finish'):old.supervise(command+['--phase',phase],b,fd)
    except BaseException as error:
        write_json(w.parent/'failure-supervisor.json',{'error':str(error),'type':type(error).__name__,'complete':False,'elapsed_seconds':time.monotonic()-now,'automatic_retry':False});raise


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--mode',choices=('validate','storage','start','worker'),default='validate')
    p.add_argument('--run-dir',type=Path);p.add_argument('--phase',choices=('train','finish'));a=p.parse_args()
    if a.mode=='storage':print(json.dumps(storage_inventory(),indent=2))
    elif a.mode=='validate':print(json.dumps(validate(),indent=2))
    elif a.run_dir is None or (a.mode=='worker' and a.phase is None):raise ValueError('Use single Bash entry')
    elif a.mode=='start':start(a.run_dir.resolve())
    else:worker(a.run_dir.resolve(),a.phase)


if __name__=='__main__':main()
