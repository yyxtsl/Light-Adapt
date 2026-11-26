"""One approved no-TTM source control: seed75 Distant2 S-PU -> Human30 ATGL, cap8h.

Default validate is read-only. No Test, old-model optimizer, tuning, automatic
retry/resume/extension/follow-up, cache generation, download or deletion.
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
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import train_ttmre_pair_replacement as prior

core,base,old,soup=prior.core,prior.base,prior.old,prior.soup
read,write_json,digest,emit=prior.read,prior.write_json,prior.digest,prior.emit
RUN_ROOT=ROOT/'experiments/2026-10-22-bypass-source-control'
CLAIM='bypass-source-control-once-claim.json'
FILES=tuple('scripts/'+n for n in ('train_ttmre_bypass_source_control.py','check_ttmre_bypass_source_control.py','run_ttmre_bypass_source_control.sh'))
HARD_SECONDS,SOFT_SECONDS,TERM_SECONDS=28800,120,30
DISTANT_DOCS,HUMAN_DOCS,DEV_DOCS=101873,3053,500
DISTANT_EPOCHS,HUMAN_EPOCHS=2,30
SEED,GIB=75,1024**3
METRICS=prior.METRICS
SPU_FORWARD=core.training_forward
GENERIC_INITIAL='ec1e6d610d6b9a650185dda8b2c05dcedb1d6defbc34ab88613e3f2b34f5f160'
PAIR_PINS={'identity.json':'b08ca33dcb53325f5865718403b267426ffcd6fe881f82c611e7bdd2913ffd96',
 'comparison/completion.json':'52ea8e62de575ce0622a94ab7123069b2cadb49deec5c478538a40fd43e0c51b',
 'bypass/completion.json':'c322d5a2a6248d1c491826aac8b3c8a7f46b6900fe8699b59d45a38c0400520f'}
DISTANT_PINS={'initial_model.json':'83c5adf524f670922338b83b2110b95bbcb89be8257779f330727e2353564b43',
 'distant/artifacts/manifest.json':'af475c1286703b71e9b3edfbe4fd41a4df46b4e737ed749369db221a06c27f8e',
 'distant/artifacts/completion.json':'01bec4148696a838f61dc5b8037835ad83f66dfcd00c383c38c7b120502b8e6a',
 'distant/artifacts/training_order.json':'bbea89e485b041d5ae7a25bbe6a1c902de0e0bdff8aa6c7883989c827c81fe9c'}
PROGRESS={'phase':'not-started','formal_optimizer_updates':0}


def settings(stage):
    if stage=='distant':return DISTANT_DOCS,DISTANT_EPOCHS,5e-5
    if stage=='human-frozen':return HUMAN_DOCS,HUMAN_EPOCHS,1e-5
    raise ValueError('Unknown stage')


def updates(stage):
    n,e,_=settings(stage);return n//4*e


def protocol():
    return {'version':'bypass-source-control-v1','seed':SEED,'Distant_epochs':DISTANT_EPOCHS,'Human_epochs':HUMAN_EPOCHS,
      'new_Distant_updates':updates('distant'),'new_Human_updates':updates('human-frozen'),
      'new_optimizer_updates':updates('distant')+updates('human-frozen'),'batch_size':4,'precision':'float32',
      'Distant_loss':'original S-PU','Human_loss':'original ATGL alpha16.5','variant':'bypass in BOTH training stages and inference',
      'initialization':'pinned generic RoBERTa and original task seed75; no trained effective parameter reuse for Distant',
      'backbone_lr':3e-5,'Distant_head_lr':5e-5,'Human_head_lr':1e-5,'warmup_fraction':.06,'eps':1e-6,'weight_decay':.01,'clip':1.,
      'selection':'penultimate batch raw Dev F1; earliest tie','decoder':'all relation logits > TH; no top-k cap',
      'no_TTM_forward_or_optimizer_parameters':True,'native_ATLOP':False,'new_parameters':0,'new_soup':False,
      'source_gate_B_minus_D_F1_min_pp':.20,'source_gate_B_minus_D_IgnA_min_pp':0.,
      'test_data_read':False,'old_model_optimizer_updates':0,'hard_seconds':HARD_SECONDS,
      'soft_reserve_seconds':SOFT_SECONDS,'term_reserve_seconds':TERM_SECONDS,
      'automatic_retry':False,'automatic_resume':False,'extension':False,'automatic_followup':False}


def check_budget(budget,reserve=SOFT_SECONDS):
    if budget is None:return
    if time.monotonic()>=budget['deadline_monotonic']-reserve:raise TimeoutError('Source-control eight-hour reserve reached')
    if os.name=='posix' and Path('/proc/sys/kernel/random/boot_id').read_text().strip()!=budget['boot_id']:
        raise TimeoutError('Boot changed; budget cannot reset')


def verify_files(files,budget=None):
    for p,h in files.items():
        check_budget(budget)
        if digest(Path(p),budget)!=h:raise ValueError('Pinned file changed: '+p)


def validate(budget=None):
    w=Path(read(prior.RUN_ROOT/prior.CLAIM)['workspace'])
    if w!=w.resolve() or w.name!='workspace' or w.parent.parent!=prior.RUN_ROOT.resolve():raise ValueError('Unexpected completed B75 workspace')
    pinned={str(w/n):h for n,h in PAIR_PINS.items()};verify_files(pinned,budget)
    origin,done,bstage=read(w/'identity.json'),read(w/'comparison/completion.json'),read(w/'bypass/completion.json')
    if not done['passed'] or done['marker']!='PAIR_REPLACEMENT_COMPLETE' or done['recommended_next_candidate']!='bypass':raise ValueError('B75 screen incomplete')
    # Only required source assets, not stopped MLP/noKL weights.
    pinned.update(origin['pinned_files'])
    for n in ('predictions.json','frequency_diagnostics.json','training_order.json','manifest.json','best_model.pth'):
        h=bstage['checkpoint_sha256'] if n.endswith('.pth') else read(w/'bypass/frequency_diagnostics.json')['predictions_sha256'] if n=='predictions.json' else None
        # Metadata byte hashes below are bound to the audited archive.
        if h is None:h=B_PINS[n]
        pinned[str(w/'bypass'/n)]=h
    original=Path(origin['distant_checkpoint']).parents[2]
    for n,h in DISTANT_PINS.items():pinned[str(original/n)]=h
    verify_files(pinned,budget)
    dm=read(original/'distant/artifacts/manifest.json');dc=read(original/'distant/artifacts/completion.json')
    do=read(original/'distant/artifacts/training_order.json');bm=read(w/'bypass/manifest.json');bo=read(w/'bypass/training_order.json')
    if (not dc['checkpoint_replay_passed'] or not dc['frequency_diagnostics_passed']
        or dc['optimizer_steps']!=updates('distant') or dc['best_checkpoint_sha256']!=base.PARENT_SHA
        or read(original/'initial_model.json')['state_sha256']!=GENERIC_INITIAL or bo['seed']!=SEED
        or bstage['optimizer_steps']!=updates('human-frozen')):raise ValueError('Original stages or generic initialization mismatch')
    cache=Path(origin['parent']['baseline_workspace'])/'cache/distant'
    metadata=read(cache/'index.json')
    if metadata['documents']!=DISTANT_DOCS or not metadata['with_labels']:raise ValueError('Wrong preexisting Distant cache')
    pinned[str(cache/'features.bin')]=metadata['features_sha256']
    verify_files({str(cache/'features.bin'):metadata['features_sha256']},budget)
    sources=dict(origin['source_files']);verify_files({str(ROOT/n):h for n,h in sources.items()},budget)
    sources.update({n:digest(ROOT/n,budget) for n in FILES})
    return {'protocol':protocol(),'parent':origin['parent'],'pinned_files':pinned,'source_files':sources,
       'distant_checkpoint':origin['distant_checkpoint'],'generic_initial_sha256':GENERIC_INITIAL,
       'B75':{'checkpoint':str(w/'bypass/best_model.pth'),'predictions':str(w/'bypass/predictions.json'),'metrics':{k:bstage['best'][k] for k in METRICS}},
       'C75':{'predictions':origin['reference_predictions'],'metrics':origin['reference_metrics']},
       'pairing':{'distant':{'initial_rng_sha256':dm['initial_rng_sha256'],'sample_order_sha256':do['sample_order_sha256'],'final_rng_sha256':do['final_rng_sha256'],'optimizer_groups':dm['optimizer_groups']},
                  'human-frozen':{'initial_rng_sha256':bm['initial_rng_sha256'],'sample_order_sha256':bo['sample_order_sha256'],'final_rng_sha256':bo['final_rng_sha256'],'optimizer_groups':bm['optimizer_groups']}},
       'stronger_existing_ATGLsoup_Dev_metrics':origin['stronger_existing_ATGLsoup_Dev_metrics'],'test_data_read':False}


B_PINS={'frequency_diagnostics.json':'e91cce0f8d8d6f21e43093357c8de661d03f7d21a0a28ec242c324bc14d98878',
        'training_order.json':'4e402941275b236aad417c53cadafaca4d6b12edff81b47dacf80c6fcd9af8ec',
        'manifest.json':'a72e7662a7f05d67cf3ffa6be8a8f619ea343e561b09473bdcb4ee05d1362808'}


def storage_inventory():
    """Read-only exact two stopped N1 Weighted candidates; no apply operation."""
    workspace=ROOT/'experiments/2026-10-18-n1-weighted-completion/weighted-completion-launch-20261018-121010-EkYgSZ/workspace'
    specs=(('distant','25e28e9d6dabac95066e49ddf55b9faf1bb9fdbb7ddb61fe2ba6dfa536e1c5e4','5d710475bfda537865167446063816194c56f90ef3150f2f949ca1ff0d3ce776'),
           ('human-frozen','215fb8a27d895d353ba3aa1d661927c28d9eb702aba0608d4baf48416186043b','b5d26d6c37b24b0a5948dbff70da620777ae70b2dcbe6181075271b5054f5ff8'))
    rows=[]
    for stage,metadata_sha,weight_sha in specs:
        art=workspace/'weighted'/stage/'artifacts';verify_files({str(art/'completion.json'):metadata_sha})
        record=read(art/'completion.json')
        if record['best_checkpoint_sha256']!=weight_sha or not record['checkpoint_replay_passed']:raise ValueError('Stopped candidate metadata mismatch')
        path=art/'best_model.pth'
        if not path.exists():rows.append({'path':str(path),'already_absent':True});continue
        st=path.lstat()
        if path.resolve()!=path or path.is_symlink() or not stat.S_ISREG(st.st_mode) or st.st_nlink!=1 or st.st_dev!=ROOT.stat().st_dev:raise ValueError('Noncanonical/aliased stopped weight')
        verify_files({str(path):weight_sha})
        rows.append({'path':str(path),'size':st.st_size,'allocated_bytes':getattr(st,'st_blocks',0)*512,'sha256':weight_sha,'nlink':st.st_nlink})
    free=shutil.disk_usage(ROOT).free
    return {'mode':'read_only','files_deleted':0,'free_bytes':free,'candidates':rows,
            'allocated_bytes_potentially_releasable':sum(r.get('allocated_bytes',0) for r in rows),
            'impact_if_deleted':'Exact stopped N1 Weighted tensor replay unavailable; reports/predictions/claims and all original Control/B/C/source models preserved',
            'deletion_authorized':False,'training_started':False}


def resources(path,identity,phase='start'):
    count=2 if phase in ('start','distant') else 1 if phase=='human-frozen' else 0
    required=max(4*GIB if phase=='start' else 512*1024**2,count*Path(identity['distant_checkpoint']).stat().st_size+512*1024**2)
    free=shutil.disk_usage(path).free
    if free<required:raise RuntimeError('Project disk free %.3fGiB < required %.3fGiB; exact cleanup approval or expansion needed'%(free/GIB,required/GIB))
    return {'phase':phase,'free_bytes':free,'required_bytes':required,'recommended_available_RAM_GiB':32,'old_files_deleted':False,'CPU_best_single_write':True}


def data_context(w,identity,budget):
    args,human,dev,gold,annotated,distant=prior.data_context(w,identity,budget)
    cache=Path(args.baseline_workspace)/'cache/distant'
    data=core.baseline.data.DiskFeatures(cache,read(cache/'index.json'),True)
    if (len(data),len(human),len(dev))!=(DISTANT_DOCS,HUMAN_DOCS,DEV_DOCS):raise ValueError('Dataset size changed')
    return args,data,human,dev,gold,annotated,distant


def construct(w,identity,args,stage):
    import torch
    import ttmre_pair_replacement as method
    model=core.load_generic_model_seed75(args,identity['parent']['baseline_identity'])
    generic=None
    if stage=='distant':
        # Pooler is unused by every task forward. Canonicalize solely for the
        # full initial digest, then restore raw bytes before any task forward.
        raw_digest=core.preflight.state_digest(model)
        names=[n for n in model.state_dict() if n.startswith('model.pooler.')]
        if set(names) not in (set(),{'model.pooler.dense.weight','model.pooler.dense.bias'}):raise ValueError('Unexpected pooler state')
        raw={n:model.state_dict()[n].detach().clone() for n in names}
        reference=torch.load(identity['distant_checkpoint'],map_location='cpu',weights_only=True) if names else {}
        try:
            with torch.no_grad():
                for n in names:model.state_dict()[n].copy_(reference[n].to(model.state_dict()[n].device))
            canonical=core.preflight.state_digest(model)
        finally:
            with torch.no_grad():
                for n,v in raw.items():model.state_dict()[n].copy_(v)
        del reference,raw
        if canonical!=identity['generic_initial_sha256'] or core.preflight.state_digest(model)!=raw_digest:raise ValueError('Generic effective initialization changed')
        generic={'passed':True,'raw_initial_sha256':raw_digest,'canonical_initial_sha256':canonical,
                 'hash_only_substitution':names,'raw_restored_before_forward':True,'trained_effective_parameters_reused':False}
    before=prior.common_digest(model);rng=core.preflight.rng_state()
    model.requires_grad_(True);installation=method.install(model,'bypass');gc.collect()
    core.preflight.assert_rng_equal(rng,core.preflight.rng_state())
    if prior.common_digest(model)!=before:raise RuntimeError('Replacement changed shared initial tensors')
    source_sha=None
    if stage=='human-frozen':
        source=verify_stage(w,identity,budget=args.budget,stage='distant')
        checkpoint=w/'distant/best_model.pth';source_sha=digest(checkpoint,args.budget)
        if source_sha!=source['checkpoint_sha256']:raise ValueError('Wrong new Distant parent')
        model.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=True),strict=True)
    model.train_mode='finetune'
    if any(n.startswith('mu_encoder.') for n,_ in model.named_parameters()):raise RuntimeError('Unexpected TTM trainable parameters')
    return model,{'installation':installation,'generic_verification':generic,'source_checkpoint_sha256':source_sha,'common_initial_state_sha256':prior.common_digest(model)}


@contextmanager
def training_scope(stage,budget):
    previous=(core.training_forward,core.emit,core.aligned.emit,core.atomic_torch_save)
    snapshot={}
    def save(value,path):
        if Path(path).name!='best_model.pth':raise ValueError('Unexpected save target')
        snapshot.clear();snapshot.update(path=Path(path),state=value)
    def event(name,**fields):
        if name=='N1_FULL_SEED75_LEARNING_UPDATE_VERIFIED':name='BYPASS_SOURCE_LEARNING_UPDATE_VERIFIED'
        if 'step' in fields:PROGRESS['formal_optimizer_updates']=(updates('distant') if stage=='human-frozen' else 0)+fields['step']
        emit(name,source_control_stage=stage,**fields)
        if name=='epoch_complete' and fields['epoch']==1:
            estimate=fields['seconds']*(settings(stage)[1]-1)+(7200 if stage=='distant' else 0)+900
            available=budget['deadline_monotonic']-time.monotonic()-SOFT_SECONDS
            emit('BYPASS_SOURCE_REMAINING_PROJECTION',stage=stage,estimate_seconds=estimate,remaining_seconds=available)
            if estimate>available:raise TimeoutError('Measured projection exceeds original eight-hour cap')
    core.training_forward=SPU_FORWARD if stage=='distant' else base.training_forward
    core.emit=core.aligned.emit=event;core.atomic_torch_save=save
    try:yield snapshot
    finally:core.training_forward,core.emit,core.aligned.emit,core.atomic_torch_save=previous


def directory_for(w,stage):return w/('distant' if stage=='distant' else 'human')


def check_selection(directory,stage):
    n,e,_=settings(stage);rows=[json.loads(l) for l in (directory/'metrics.jsonl').read_text().splitlines()]
    if len(rows)!=e or any(r['epoch']!=i or r['step']!=i*(n//4)-1 or r['batch']!=n//4-1 for i,r in enumerate(rows,1)):raise ValueError('Selection trajectory mismatch')
    best=read(directory/'best_metrics.json');selected=max(rows,key=lambda r:r['dev_F1'])
    if any(best[k]!=selected[k] for k in ('epoch','step',*METRICS)):raise ValueError('Not earliest raw-F1 maximum')


def run_stage(w,identity,budget,stage):
    import numpy as np
    import torch
    from torch.utils.data import DataLoader
    from transformers.optimization import get_linear_schedule_with_warmup
    started=time.monotonic();d=directory_for(w,stage);d.mkdir()
    write_json(d/'resource_check.json',resources(w,identity,stage))
    args,data,human,dev,gold,annotated,distant=data_context(w,identity,budget)
    dataset=data if stage=='distant' else human
    model,initial=construct(w,identity,args,stage);write_json(d/'initialization.json',initial)
    memory=model.mu_encoder.memory_tokens.detach().cpu().clone()
    hook=model.register_forward_pre_hook(lambda *_:check_budget(budget));snapshot=None
    try:
        probe=old.probe if stage=='distant' else base.probe
        write_json(d/'full_model_check.json',probe(model,dataset,args,budget))
        optimizer=core.baseline.optimizer_for_stage(model,stage);groups=core.baseline.optimizer_report(model,optimizer)
        if optimizer.state:raise ValueError('Optimizer is not fresh')
        for g,r in zip(groups,identity['pairing'][stage]['optimizer_groups']):
            if (g['parameter_names']!=[n for n in r['parameter_names'] if not n.startswith('mu_encoder.')]
                or g['lr']!=r['lr'] or g['weight_decay']!=r['weight_decay']):raise ValueError('Unmatched common optimizer groups')
        core.shared.seed_all(SEED,torch,np)
        loader=DataLoader(dataset,batch_size=4,shuffle=True,drop_last=True,collate_fn=core.shared.collate_fn)
        dev_loader=DataLoader(dev,batch_size=4,shuffle=False,collate_fn=core.shared.collate_fn)
        total=updates(stage);scheduler=get_linear_schedule_with_warmup(optimizer,int(total*.06),total)
        rng=core.rng_digest()
        if rng!=identity['pairing'][stage]['initial_rng_sha256'] or len(loader)*settings(stage)[1]!=total:raise ValueError('Initial RNG/update budget differs')
        write_json(d/'manifest.json',{'protocol':protocol(),'stage':stage,'optimizer_groups':groups,'initial_rng_sha256':rng,'total_steps':total})
        with training_scope(stage,budget) as snapshot:
            result=core.train_loop(model,loader,dev_loader,args,'bypass-source',stage,d,gold,annotated,distant,
                                   core.method.frequency_buckets(human.metadata['priors']),budget,optimizer,scheduler)
        del optimizer,scheduler
        check_selection(d,stage);pair=identity['pairing'][stage]
        order={'seed':SEED,'formal_optimizer_updates':result['optimizer_steps'],'sample_order_sha256':result['sample_order_sha256'],
               'final_rng_sha256':result['final_rng_sha256'],'same_as_reference_order':result['sample_order_sha256']==pair['sample_order_sha256'],
               'same_as_reference_final_rng':result['final_rng_sha256']==pair['final_rng_sha256']}
        write_json(d/'training_order.json',order)
        if result['optimizer_steps']!=total or not order['same_as_reference_order'] or not order['same_as_reference_final_rng']:raise ValueError('Actual learning order/RNG/count differs')
        if not snapshot or snapshot['path']!=d/'best_model.pth' or snapshot['path'].exists():raise ValueError('Missing/conflicting CPU best')
        check_budget(budget);core.atomic_torch_save(snapshot['state'],d/'best_model.pth');snapshot=None
        model.zero_grad(set_to_none=True);model.load_state_dict(torch.load(d/'best_model.pth',map_location='cpu',weights_only=True),strict=True)
        best=read(d/'best_metrics.json');replay,predictions=old.grouped_replay(model,dev,args,budget,best,memory)
        score=core.shared.score_predictions(predictions,gold,annotated,distant)
        replay['metrics']=score;replay['metric_replay_passed']=all(math.isclose(score[k],best[k],abs_tol=1e-6,rel_tol=0) for k in METRICS)
        replay['passed']=replay['passed'] and replay['metric_replay_passed'];write_json(d/'checkpoint_replay.json',replay)
        if not replay['passed'] or not torch.equal(memory,model.mu_encoder.memory_tokens.detach().cpu()):raise ValueError('Best checkpoint replay/frozen buffer failed')
        old.save_prediction_report(d,predictions,[{'title':f['title'],'hts':f['hts']} for f in dev],score,gold,human)
        result.update({'passed':True,'stage':stage,'protocol':protocol(),'best':best,'checkpoint_sha256':digest(d/'best_model.pth',budget),
                       'best_checkpoint_writes':1,'checkpoint_replay_passed':True,'no_TTM_forward_or_optimizer_parameters':True,
                       'memory_buffer_unchanged':True,'source_checkpoint_sha256':initial['source_checkpoint_sha256'],'elapsed_seconds':time.monotonic()-started})
        write_json(d/'completion.json',result);emit('BYPASS_SOURCE_STAGE_COMPLETE',stage=stage,F1=score['dev_F1'],optimizer_updates=total)
    except BaseException:
        if snapshot and not (d/'best_model.pth').exists() and time.monotonic()<budget['deadline_monotonic']-60:
            needed=sum(v.numel()*v.element_size() for v in snapshot['state'].values())+64*1024**2
            if shutil.disk_usage(d).free>=needed:
                core.atomic_torch_save(snapshot['state'],d/'best_model.pth')
                write_json(d/'incomplete_best_preserved.json',{'complete':False,'resumable':False})
        raise
    finally:
        PROGRESS['formal_optimizer_updates']=(updates('distant') if stage=='human-frozen' else 0)+int(getattr(model,'_n1_completed_optimizer_steps',0))
        hook.remove()


def verify_stage(w,identity,budget,stage):
    d=directory_for(w,stage);c=read(d/'completion.json');l=read(d/'learning_update_check.json');o=read(d/'training_order.json');r=read(d/'checkpoint_replay.json')
    if (not c['passed'] or c['protocol']!=protocol() or c['stage']!=stage or c['optimizer_steps']!=updates(stage)
        or c['best_checkpoint_writes']!=1 or not c['memory_buffer_unchanged'] or not c['no_TTM_forward_or_optimizer_parameters']
        or c['checkpoint_sha256']!=digest(d/'best_model.pth',budget) or not l['passed'] or not l['parameter_changed']
        or l['lr']<=0 or l['optimizer_state_step']!=l['step'] or l['extra_optimizer_updates']!=0
        or not o['same_as_reference_order'] or not o['same_as_reference_final_rng'] or not r['passed']
        or any(o[k]!=identity['pairing'][stage][k] for k in ('sample_order_sha256','final_rng_sha256'))):raise ValueError('Incomplete/changed source stage')
    check_selection(d,stage)
    if stage=='human-frozen' and c['source_checkpoint_sha256']!=digest(w/'distant/best_model.pth',budget):raise ValueError('Human not from new Distant best')
    return c


def finish(w,identity,budget):
    d=verify_stage(w,identity,budget,'distant');h=verify_stage(w,identity,budget,'human-frozen')
    args,data,human,dev,gold,annotated,distant=data_context(w,identity,budget)
    comparison=w/'comparison';comparison.mkdir();metrics={'D75':{k:h['best'][k] for k in METRICS}}
    for label in ('B75','C75'):
        raw=Path(identity[label]['predictions']).read_bytes();pred=json.loads(raw);actual=core.shared.score_predictions(pred,gold,annotated,distant)
        if any(not math.isclose(actual[k],identity[label]['metrics'][k],abs_tol=1e-6,rel_tol=0) for k in METRICS):raise ValueError('Old reference predictions differ')
        metrics[label]=actual;write_json(comparison/(label+'-predictions.json'),pred)
    own=core.shared.score_predictions(read(w/'human/predictions.json'),gold,annotated,distant)
    if any(not math.isclose(own[k],metrics['D75'][k],abs_tol=1e-6,rel_tol=0) for k in METRICS):raise ValueError('New Human score differs')
    delta={k:metrics['B75'][k]-metrics['D75'][k] for k in METRICS}
    gate=delta['dev_F1']>=.20 and delta['dev_IgnF1_annotated']>=0
    interpretation='TTM source worth further validation' if gate else 'No clear TTM source advantage; do not add seeds or tune automatically'
    write_json(w/'resource_finish.json',resources(w,identity,'finish'))
    verify_files(identity['pinned_files'],budget);verify_files({str(ROOT/n):h for n,h in identity['source_files'].items()},budget)
    result={'passed':True,'marker':'BYPASS_SOURCE_CONTROL_COMPLETE','protocol':protocol(),'metrics':metrics,'B75_minus_D75_pp':delta,
            'source_investment_gate_passed':gate,'interpretation':interpretation,'single_seed_screen_only':True,
            'Distant_best':d['best'],'new_optimizer_updates':updates('distant')+updates('human-frozen'),
            'old_model_optimizer_updates':0,'test_data_read':False,'test_evaluated':False,'automatic_followup':False,
            'elapsed_seconds':time.monotonic()-budget['start_monotonic'],'novelty_established':False,
            'existing_stronger_ATGLsoup_Dev_metrics':identity['stronger_existing_ATGLsoup_Dev_metrics']}
    write_json(comparison/'completion.json',result);soup.archive_reports(w,budget)
    emit('BYPASS_SOURCE_CONTROL_COMPLETE',source_investment_gate_passed=gate,reports=str(comparison/'reports.tar.gz'))


def require_lock():
    if os.name!='posix' or os.environ.get('TTMRE_BYPASS_SOURCE_LOCK_FD')!='9':raise RuntimeError('Use sole source-control Bash launcher')
    import fcntl
    a,b=os.fstat(9),(ROOT/'experiments/2026-10-08-baseline-alignment/.training.lock').stat()
    if (a.st_dev,a.st_ino)!=(b.st_dev,b.st_ino):raise RuntimeError('Wrong shared training lock')
    fcntl.flock(9,fcntl.LOCK_EX|fcntl.LOCK_NB);return 9


def verify_context(w,b):
    c=read(RUN_ROOT/CLAIM)
    if (w!=w.resolve() or w.name!='workspace' or w.parent.parent!=RUN_ROOT.resolve() or b['hard_seconds']!=HARD_SECONDS
        or not math.isclose(b['deadline_monotonic']-b['start_monotonic'],HARD_SECONDS,abs_tol=1e-6,rel_tol=0)
        or any(b[k] is not False for k in ('automatic_retry','automatic_resume','extension')) or c['workspace']!=str(w)
        or c['protocol']!=protocol() or c['budget_sha256']!=digest(w/'budget.json') or c['identity_sha256']!=digest(w/'identity.json')):raise ValueError('Changed immutable eight-hour context')
    check_budget(b)


def worker(w,phase):
    require_lock();budget=read(w/'budget.json');verify_context(w,budget);identity=read(w/'identity.json')
    PROGRESS.update(phase=phase,formal_optimizer_updates=0 if phase=='distant' else updates('distant') if phase=='human-frozen' else updates('distant')+updates('human-frozen'))
    def stop(*_):raise TimeoutError('Interrupted; no resume')
    signal.signal(signal.SIGTERM,stop)
    try:
        if identity['protocol']!=protocol():raise ValueError('Protocol changed')
        verify_files(identity['pinned_files'],budget);verify_files({str(ROOT/n):h for n,h in identity['source_files'].items()},budget)
        if phase=='finish':finish(w,identity,budget)
        else:run_stage(w,identity,budget,phase)
    except BaseException as error:
        write_json(w/('failure-'+phase+'.json'),{'error':str(error),'type':type(error).__name__,'complete':False,'progress':PROGRESS,'resumable':False,'automatic_retry':False})
        try:
            if not (w/'comparison/reports.tar.gz').exists():soup.archive_reports(w,budget)
        except BaseException:pass
        raise


def start(w):
    lock=require_lock()
    if w.name!='workspace' or w.parent.parent!=RUN_ROOT.resolve() or w.exists():raise ValueError('Use a fresh source-control workspace')
    if (RUN_ROOT/CLAIM).exists():raise FileExistsError('Already claimed; no retry')
    now=time.monotonic();budget={'start_monotonic':now,'deadline_monotonic':now+HARD_SECONDS,'hard_seconds':HARD_SECONDS,
        'soft_reserve_seconds':SOFT_SECONDS,'term_reserve_seconds':TERM_SECONDS,'start_utc':datetime.now(timezone.utc).isoformat(),
        'boot_id':Path('/proc/sys/kernel/random/boot_id').read_text().strip(),'automatic_retry':False,'automatic_resume':False,'extension':False}
    def stop(*_):raise TimeoutError('Source supervisor interrupted')
    signal.signal(signal.SIGTERM,stop)
    try:
        emit('BYPASS_SOURCE_BUDGET_STARTED',**budget)
        identity=validate(budget);emit('BYPASS_SOURCE_SOURCES_VERIFIED',resources=resources(w.parent,identity))
        old.supervise([sys.executable,'-B',str(ROOT/FILES[1])],budget,lock)
        check_budget(budget);w.mkdir();write_json(w/'budget.json',budget);write_json(w/'identity.json',identity)
        with (RUN_ROOT/CLAIM).open('x',encoding='utf-8') as f:
            json.dump({'workspace':str(w),'protocol':protocol(),'budget_sha256':digest(w/'budget.json'),'identity_sha256':digest(w/'identity.json'),
                       'authorization':'user explicitly confirmed one source control, total8h'},f,indent=2);f.flush();os.fsync(f.fileno())
        command=[sys.executable,'-u','-B',str(Path(__file__).resolve()),'--mode','worker','--run-dir',str(w)]
        for phase in ('distant','human-frozen','finish'):old.supervise(command+['--phase',phase],budget,lock)
    except BaseException as error:
        write_json(w.parent/'failure-supervisor.json',{'error':str(error),'type':type(error).__name__,'complete':False,'elapsed_seconds':time.monotonic()-now,'automatic_retry':False});raise


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--mode',choices=('validate','storage','start','worker'),default='validate')
    p.add_argument('--run-dir',type=Path);p.add_argument('--phase',choices=('distant','human-frozen','finish'));a=p.parse_args()
    if a.mode=='storage':print(json.dumps(storage_inventory(),indent=2))
    elif a.mode=='validate':print(json.dumps(validate(),indent=2))
    elif a.run_dir is None or (a.mode=='worker' and a.phase is None):raise ValueError('Use sole Bash entry point')
    elif a.mode=='start':start(a.run_dir.resolve())
    else:worker(a.run_dir.resolve(),a.phase)


if __name__=='__main__':main()
