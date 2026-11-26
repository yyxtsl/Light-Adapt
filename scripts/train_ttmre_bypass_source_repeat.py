"""Approved D Human76/77 only from fixed no-TTM Distant; total cap3h.

Default validate/storage read-only. No new Distant/B/C training, benchmark,
Test, soup, tuning, retry/resume/extension/follow-up or deletion.
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
import train_ttmre_bypass_source_control as source
import train_ttmre_bypass_repeat as bref

core,base,old,soup=source.core,source.base,source.old,source.soup
read,write_json,digest,emit=source.read,source.write_json,source.digest,source.emit
RUN_ROOT=ROOT/'experiments/2026-10-22-bypass-source-repeat'
CLAIM='bypass-source-repeat-once-claim.json'
FILES=tuple('scripts/'+n for n in ('train_ttmre_bypass_source_repeat.py','check_ttmre_bypass_source_repeat.py','run_ttmre_bypass_source_repeat.sh'))
SEEDS=(76,77)
HUMAN_DOCS,DEV_DOCS,EPOCHS=3053,500,30
HARD_SECONDS,SOFT_SECONDS,TERM_SECONDS=10800,120,30
GIB=1024**3
METRICS=source.METRICS
DISTANT_SHA='086c57e6baa5de13ab166bf03babb7794d2591eecfcca37cc611443cae0e8c01'
SOURCE_PINS={'identity.json':'9352427e07cc0702e9b1816c2cf4afd9e2a91265e7c5c0aae788f926505c664a',
 'comparison/completion.json':'1c967e208f7d818b0895083c66b0d7e2edd45ea30667b3185ccb735b7fd2102b',
 'distant/completion.json':'2c17c2f4c2ee437356094b800bdfbca73765625b1282d0723c003bc7978202ef',
 'human/completion.json':'17db693d54733b9660ac9e2f65f3d0aa57272df344679d92b3306eb437954e9e'}
B_PINS={'identity.json':'7d347ad758f6971853d15daa20856607b920f0d911a18efbca176ea10cff64ca',
 'comparison/completion.json':'25ba9874f87e345bbde53c4c45bc93633754c179d18debcfe79feb3d869a6acc'}
ARM_PINS={76:{'completion.json':'8cac80f9ed6727d0e09d8af918a3cf8851da55fb8e1c0577b2e67fc946aa4788',
 'manifest.json':'5f4c89bf729e80a4be609ee20c455883b436aeda6c01ddf98803cd5c64abc642',
 'training_order.json':'9e53c3b879c2611dac9a58df95c7674ae34b4969f1ec0003a8a57af833cb5d39',
 'predictions.json':'a2052a39cc1edc10ce2d3f9b471304aea6a2d434728b876f280973519fc3e49e'},
 77:{'completion.json':'b711bb1b3f5caa456669155999e22890b4f7b181c9f031cf4c218a84d82977d7',
 'manifest.json':'6ee65124cc5d309d8c6f21ff85bef525412fad7a10da5367558e33ce310457f4',
 'training_order.json':'6c7a30373e1e9004bab066bb9edb680184c7d04cea0070e6a75505680a1ea5e9',
 'predictions.json':'c3fbb0e6e3281978aac5909a241ac0be883a1fa89a00b5c5f7a18a1857f2d643'}}
PROGRESS={'phase':'not-started','formal_optimizer_updates':0}


def updates():return HUMAN_DOCS//4*EPOCHS


def protocol():
    return {'version':'bypass-source-repeat-v1','new_Human_seeds':list(SEEDS),'epochs':EPOCHS,'updates_per_arm':updates(),
      'new_optimizer_updates':2*updates(),'Distant_parent_sha256':DISTANT_SHA,'variant':'fixed no-TTM source D, Human only',
      'loss':'original ATGL alpha16.5','batch_size':4,'precision':'float32','backbone_lr':3e-5,'head_lr':1e-5,
      'warmup_steps':int(updates()*.06),'eps':1e-6,'weight_decay':.01,'clip':1.,
      'selection':'penultimate batch raw Dev F1; earliest tie','decoder':'all relation logits > TH; no top-k cap',
      'fresh_gate':{'each_B_minus_D_F1_min_pp':0.,'mean_B_minus_D_F1_min_pp':.20,'mean_B_minus_D_IgnA_min_pp':0.,'each_B_minus_D_IgnA_min_pp':-.10},
      'one_Distant_source_per_structure':True,'old75_is_screening':True,'new_Distant_updates':0,'new_reference_updates':0,
      'new_benchmarks':False,'new_soup':False,'test_data_read':False,'hard_seconds':HARD_SECONDS,
      'soft_reserve_seconds':SOFT_SECONDS,'term_reserve_seconds':TERM_SECONDS,
      'automatic_retry':False,'automatic_resume':False,'extension':False,'automatic_followup':False}


def check_budget(b,reserve=SOFT_SECONDS):
    if b is None:return
    if time.monotonic()>=b['deadline_monotonic']-reserve:raise TimeoutError('Source repeat three-hour reserve reached')
    if os.name=='posix' and Path('/proc/sys/kernel/random/boot_id').read_text().strip()!=b['boot_id']:raise TimeoutError('Boot changed; no budget reset')


def verify_files(files,budget=None):
    for p,h in files.items():
        check_budget(budget)
        if digest(Path(p),budget)!=h:raise ValueError('Pinned file changed: '+p)


def completed_workspace(root,claim):
    w=Path(read(root/claim)['workspace'])
    if w!=w.resolve() or w.name!='workspace' or w.parent.parent!=root.resolve():raise ValueError('Unexpected reference workspace')
    return w


def validate(budget=None):
    w=completed_workspace(source.RUN_ROOT,source.CLAIM);bw=completed_workspace(bref.RUN_ROOT,bref.CLAIM)
    pins={**{str(w/n):h for n,h in SOURCE_PINS.items()},**{str(bw/n):h for n,h in B_PINS.items()}}
    verify_files(pins,budget)
    origin,done,dc=read(w/'identity.json'),read(w/'comparison/completion.json'),read(w/'distant/completion.json')
    bidentity,bdone=read(bw/'identity.json'),read(bw/'comparison/completion.json')
    if (not done['passed'] or done['marker']!='BYPASS_SOURCE_CONTROL_COMPLETE' or not done['source_investment_gate_passed']
        or not dc['passed'] or dc['optimizer_steps']!=50936 or dc['checkpoint_sha256']!=DISTANT_SHA
        or not bdone['passed'] or bdone['marker']!='BYPASS_REPEAT_COMPLETE'):raise ValueError('Prior completed sources mismatch')
    # Preserve legacy input/model and source bindings; omit unused Distant
    # feature bytes, which this Human-only experiment must never read.
    pins.update({p:h for p,h in origin['pinned_files'].items() if not p.endswith('/cache/distant/features.bin')})
    pins[str(w/'distant/best_model.pth')]=DISTANT_SHA
    refs={};pairing={}
    for seed in SEEDS:
        art=bw/('seed'+str(seed))/'bypass'
        pins.update({str(art/n):h for n,h in ARM_PINS[seed].items()});verify_files({str(art/n):h for n,h in ARM_PINS[seed].items()},budget)
        c,m,o=(read(art/n) for n in ('completion.json','manifest.json','training_order.json'))
        if (not c['passed'] or c['optimizer_steps']!=updates() or c['protocol']['active_Human_seed']!=seed
            or not c['memory_buffer_unchanged'] or o['seed']!=seed or c['sample_order_sha256']!=o['sample_order_sha256']):raise ValueError('Wrong same-seed B reference')
        refs[str(seed)]={'metrics':{k:c['best'][k] for k in METRICS},'predictions':str(art/'predictions.json')}
        pairing[str(seed)]={'initial_rng_sha256':m['initial_rng_sha256'],'sample_order_sha256':o['sample_order_sha256'],
                           'final_rng_sha256':o['final_rng_sha256'],'optimizer_groups':m['optimizer_groups']}
    verify_files(pins,budget)
    sources={**origin['source_files'],**bidentity['source_files']}
    verify_files({str(ROOT/n):h for n,h in sources.items()},budget)
    sources.update({n:digest(ROOT/n,budget) for n in FILES})
    return {'protocol':protocol(),'parent':origin['parent'],'distant_checkpoint':str(w/'distant/best_model.pth'),
            'references':refs,'pairing':pairing,'old75':{'B':done['metrics']['B75'],'D':done['metrics']['D75'],'is_screened':True},
            'pinned_files':pins,'source_files':sources,'existing_stronger_ATGLsoup_Dev_metrics':done['existing_stronger_ATGLsoup_Dev_metrics'],
            'test_data_read':False,'Distant_features_read':False}


def resources(path,identity,phase='start'):
    pending=2 if phase in ('start','seed76') else 1 if phase=='seed77' else 0
    required=max(4*GIB if phase=='start' else 512*1024**2,pending*Path(identity['distant_checkpoint']).stat().st_size+512*1024**2)
    free=shutil.disk_usage(path).free
    if free<required:raise RuntimeError('Project disk free %.3fGiB < required %.3fGiB; obtain exact cleanup approval or expand'%(free/GIB,required/GIB))
    return {'phase':phase,'free_bytes':free,'required_bytes':required,'recommended_available_RAM_GiB':32,'old_files_deleted':False}


def storage_inventory():
    root=ROOT/'experiments/2026-10-20-atgl-view-repeat/atgl-view-repeat-launch-20261020-120030-dKSlVa/workspace'
    specs=((76,'cc2e3c4b7899b4fedd59706abcea55a2ada7c5c17b3544ae8b175fddac961f77','ba70ced1beb4ec05a018c3f66e0a3a75620eb8ffae744d2af863ee9167d5aa83'),
           (77,'a0163da35b1795e2c008da4c144e54af407727d411c31d469a5c983347b54748','f19f3fced82ec9214d4ba1d0c6b743c3b626b9b3493c44b081cee63b97a5d45e'))
    rows=[]
    for seed,ch,wh in specs:
        art=root/('seed'+str(seed))/'nokl/workspace/atgl-nokl/human-frozen/artifacts'
        verify_files({str(art/'completion.json'):ch});c=read(art/'completion.json')
        if not c['passed'] or c['checkpoint_sha256']!=wh:raise ValueError('Stopped noKL metadata mismatch')
        path=art/'best_model.pth'
        if not path.exists():rows.append({'path':str(path),'already_absent':True});continue
        st=path.lstat()
        if path.resolve()!=path or path.is_symlink() or not stat.S_ISREG(st.st_mode) or st.st_nlink!=1 or st.st_dev!=ROOT.stat().st_dev:raise ValueError('Target is not a canonical single-link project file')
        verify_files({str(path):wh});rows.append({'path':str(path),'size':st.st_size,'allocated_bytes':getattr(st,'st_blocks',0)*512,'sha256':wh,'nlink':st.st_nlink})
    return {'mode':'read_only','free_bytes':shutil.disk_usage(ROOT).free,'candidates':rows,
            'allocated_bytes_potentially_releasable':sum(r.get('allocated_bytes',0) for r in rows),'files_deleted':0,'training_started':False,
            'deletion_authorized':False,'impact_if_deleted':'Exact stopped noKL Human76/77 tensor replay unavailable; reports/predictions/claims/C/B/D/Distant/cache preserved'}


def data_context(w,identity,budget):
    args,human,dev,gold,annotated,distant=source.prior.data_context(w,identity,budget)
    return args,None,human,dev,gold,annotated,distant


def construct(w,identity,args,stage):
    if stage!='human-frozen':raise ValueError('Human only')
    import torch
    import ttmre_pair_replacement as method
    model=core.load_generic_model_seed75(args,identity['parent']['baseline_identity']);model.requires_grad_(True)
    install=method.install(model,'bypass');gc.collect()
    checksum=digest(identity['distant_checkpoint'],args.budget)
    if checksum!=DISTANT_SHA:raise ValueError('Wrong fixed new Distant source')
    model.load_state_dict(torch.load(identity['distant_checkpoint'],map_location='cpu',weights_only=True),strict=True)
    model.train_mode='finetune'
    return model,{'installation':install,'source_checkpoint_sha256':checksum,'common_initial_state_sha256':source.prior.common_digest(model),
                  'active_Human_seed':identity['active_Human_seed'],'Human_checkpoint_used_for_initialization':False}


@contextmanager
def training_scope(seed,b):
    saved=(core.training_forward,core.emit,core.aligned.emit,core.atomic_torch_save);snapshot={}
    def save(value,path):
        if Path(path).name!='best_model.pth':raise ValueError('Unexpected checkpoint target')
        snapshot.clear();snapshot.update(path=Path(path),state=value)
    def event(name,**fields):
        if name=='N1_FULL_SEED75_LEARNING_UPDATE_VERIFIED':name='BYPASS_SOURCE_REPEAT_LEARNING_UPDATE_VERIFIED'
        if 'step' in fields:PROGRESS['formal_optimizer_updates']=SEEDS.index(seed)*updates()+fields['step']
        emit(name,human_seed=seed,**fields)
        if name=='epoch_complete' and fields['epoch']==1:
            estimate=fields['seconds']*((len(SEEDS)-SEEDS.index(seed))*EPOCHS-1)+900
            available=b['deadline_monotonic']-time.monotonic()-SOFT_SECONDS
            emit('BYPASS_SOURCE_REPEAT_PROJECTION',human_seed=seed,estimate_seconds=estimate,remaining_seconds=available)
            if estimate>available:raise TimeoutError('Measured projection exceeds original three-hour cap')
    core.training_forward=base.training_forward;core.emit=core.aligned.emit=event;core.atomic_torch_save=save
    try:yield snapshot
    finally:core.training_forward,core.emit,core.aligned.emit,core.atomic_torch_save=saved


@contextmanager
def variant_scope(seed):
    names=('SEED','HARD_SECONDS','protocol','data_context','construct','training_scope','resources','emit','check_budget','PROGRESS')
    saved={n:getattr(source,n) for n in names}
    source.SEED=seed;source.HARD_SECONDS=HARD_SECONDS;source.PROGRESS={'formal_optimizer_updates':0}
    source.protocol=lambda:{**protocol(),'active_Human_seed':seed}
    source.data_context=data_context;source.construct=construct
    source.training_scope=lambda stage,b:training_scope(seed,b)
    source.resources=lambda p,i,_phase:resources(p,i,'seed'+str(seed))
    source.check_budget=check_budget
    def event(name,**fields):
        if name=='BYPASS_SOURCE_STAGE_COMPLETE':name='BYPASS_SOURCE_REPEAT_ARM_COMPLETE'
        emit(name,human_seed=seed,**fields)
    source.emit=event
    try:yield
    finally:
        PROGRESS['formal_optimizer_updates']=max(PROGRESS['formal_optimizer_updates'],SEEDS.index(seed)*updates()+source.PROGRESS['formal_optimizer_updates'])
        for n,v in saved.items():setattr(source,n,v)


def run_seed(w,identity,b,seed):
    arm=w/('seed'+str(seed));arm.mkdir()
    info={**identity,'active_Human_seed':seed,'pairing':{'human-frozen':identity['pairing'][str(seed)]}}
    write_json(arm/'identity.json',info)
    with variant_scope(seed):source.run_stage(arm,info,b,'human-frozen')


def verify_seed(w,identity,b,seed):
    d=w/('seed'+str(seed))/'human';c=read(d/'completion.json');l=read(d/'learning_update_check.json');o=read(d/'training_order.json');r=read(d/'checkpoint_replay.json')
    pair=identity['pairing'][str(seed)]
    if (not c['passed'] or c['protocol']!={**protocol(),'active_Human_seed':seed} or c['optimizer_steps']!=updates()
        or c['source_checkpoint_sha256']!=DISTANT_SHA or c['checkpoint_sha256']!=digest(d/'best_model.pth',b)
        or c['best_checkpoint_writes']!=1 or not c['memory_buffer_unchanged'] or not c['no_TTM_forward_or_optimizer_parameters']
        or not l['passed'] or not l['parameter_changed'] or l['lr']<=0 or l['step']!=l['optimizer_state_step'] or l['extra_optimizer_updates']!=0
        or o['seed']!=seed or not o['same_as_reference_order'] or not o['same_as_reference_final_rng'] or not r['passed']
        or any(o[k]!=pair[k] for k in ('sample_order_sha256','final_rng_sha256'))):raise ValueError('Incomplete/changed paired Human stage')
    source.check_selection(d,'human-frozen');return c


def investment(rows):
    delta={s:{k:r['B'][k]-r['D'][k] for k in METRICS} for s,r in rows.items()}
    fresh={k:statistics.mean(delta[str(s)][k] for s in SEEDS) for k in METRICS}
    checks={'each_fresh_F1_nonnegative':all(delta[str(s)]['dev_F1']>=0 for s in SEEDS),
            'fresh_mean_F1_at_least_point20':fresh['dev_F1']>=.20,'fresh_mean_IgnA_nonnegative':fresh['dev_IgnF1_annotated']>=0,
            'each_fresh_IgnA_not_below_minuspoint10':all(delta[str(s)]['dev_IgnF1_annotated']>=-.10 for s in SEEDS)}
    return {'seedwise_B_minus_D_pp':delta,'fresh76_77_mean_delta_pp':fresh,
            'all75_76_77_mean_delta_pp':{k:statistics.mean(delta[str(s)][k] for s in (75,*SEEDS)) for k in METRICS},
            'fresh_gate_checks':checks,'source_repeat_investment_gate_passed':all(checks.values()),
            'single_model_statistics':{l:{'mean':{k:statistics.mean(rows[str(s)][l][k] for s in (75,*SEEDS)) for k in METRICS},
                                         'sample_SD':{k:statistics.stdev(rows[str(s)][l][k] for s in (75,*SEEDS)) for k in METRICS}} for l in ('B','D')}}


def finish(w,identity,b):
    rows={'75':identity['old75']};directory=w/'comparison';directory.mkdir()
    args,_,human,dev,gold,annotated,distant=data_context(w,identity,b)
    for seed in SEEDS:
        s=str(seed);c=verify_seed(w,identity,b,seed);ref=identity['references'][s]
        d=w/('seed'+s)/'human'
        for label,path,expected in (('B',Path(ref['predictions']),ref['metrics']),('D',d/'predictions.json',c['best'])):
            pred=read(path);actual=core.shared.score_predictions(pred,gold,annotated,distant)
            if any(not math.isclose(actual[k],expected[k],abs_tol=1e-6,rel_tol=0) for k in METRICS):raise ValueError('Paired reference/own score differs')
            if label=='B':write_json(directory/('B'+s+'-predictions.json'),pred)
        rows[s]={'B':ref['metrics'],'D':{k:c['best'][k] for k in METRICS},'is_screened':False,'checkpoint_sha256':c['checkpoint_sha256']}
    summary=investment(rows);write_json(w/'resource_finish.json',resources(w,identity,'finish'))
    verify_files(identity['pinned_files'],b);verify_files({str(ROOT/n):h for n,h in identity['source_files'].items()},b)
    result={'passed':True,'marker':'BYPASS_SOURCE_REPEAT_COMPLETE','protocol':protocol(),'rows':rows,**summary,
            'new_optimizer_updates':2*updates(),'new_Distant_updates':0,'new_reference_updates':0,
            'test_data_read':False,'test_evaluated':False,'automatic_followup':False,'novelty_established':False,
            'one_Distant_source_per_structure':True,'elapsed_seconds':time.monotonic()-b['start_monotonic'],
            'existing_stronger_ATGLsoup_Dev_metrics':identity['existing_stronger_ATGLsoup_Dev_metrics']}
    write_json(directory/'completion.json',result);soup.archive_reports(w,b)
    emit('BYPASS_SOURCE_REPEAT_COMPLETE',source_repeat_investment_gate_passed=summary['source_repeat_investment_gate_passed'],reports=str(directory/'reports.tar.gz'))


def require_lock():
    if os.name!='posix' or os.environ.get('TTMRE_SOURCE_REPEAT_LOCK_FD')!='9':raise RuntimeError('Use single source-repeat launcher')
    import fcntl
    a,b=os.fstat(9),(ROOT/'experiments/2026-10-08-baseline-alignment/.training.lock').stat()
    if (a.st_dev,a.st_ino)!=(b.st_dev,b.st_ino):raise RuntimeError('Wrong shared lock')
    fcntl.flock(9,fcntl.LOCK_EX|fcntl.LOCK_NB);return 9


def verify_context(w,b):
    c=read(RUN_ROOT/CLAIM)
    if (w!=w.resolve() or w.name!='workspace' or w.parent.parent!=RUN_ROOT.resolve() or b['hard_seconds']!=HARD_SECONDS
        or not math.isclose(b['deadline_monotonic']-b['start_monotonic'],HARD_SECONDS,abs_tol=1e-6,rel_tol=0)
        or any(b[k] is not False for k in ('automatic_retry','automatic_resume','extension')) or c['workspace']!=str(w)
        or c['protocol']!=protocol() or c['budget_sha256']!=digest(w/'budget.json') or c['identity_sha256']!=digest(w/'identity.json')):raise ValueError('Changed immutable3h context')
    check_budget(b)


def worker(w,phase):
    require_lock();b=read(w/'budget.json');verify_context(w,b);identity=read(w/'identity.json')
    PROGRESS.update(phase=phase,formal_optimizer_updates={'seed76':0,'seed77':updates(),'finish':2*updates()}[phase])
    def stop(*_):raise TimeoutError('Interrupted; no resume')
    signal.signal(signal.SIGTERM,stop)
    try:
        if identity['protocol']!=protocol():raise ValueError('Protocol changed')
        verify_files(identity['pinned_files'],b);verify_files({str(ROOT/n):h for n,h in identity['source_files'].items()},b)
        if phase=='finish':finish(w,identity,b)
        else:run_seed(w,identity,b,int(phase[4:]))
    except BaseException as error:
        write_json(w/('failure-'+phase+'.json'),{'error':str(error),'type':type(error).__name__,'complete':False,'progress':PROGRESS,'resumable':False,'automatic_retry':False})
        try:
            if not (w/'comparison/reports.tar.gz').exists():soup.archive_reports(w,b)
        except BaseException:pass
        raise


def start(w):
    fd=require_lock()
    if w.name!='workspace' or w.parent.parent!=RUN_ROOT.resolve() or w.exists():raise ValueError('Use new source-repeat workspace')
    if (RUN_ROOT/CLAIM).exists():raise FileExistsError('Already claimed; no retry')
    now=time.monotonic();b={'start_monotonic':now,'deadline_monotonic':now+HARD_SECONDS,'hard_seconds':HARD_SECONDS,
        'soft_reserve_seconds':SOFT_SECONDS,'term_reserve_seconds':TERM_SECONDS,'start_utc':datetime.now(timezone.utc).isoformat(),
        'boot_id':Path('/proc/sys/kernel/random/boot_id').read_text().strip(),'automatic_retry':False,'automatic_resume':False,'extension':False}
    def stop(*_):raise TimeoutError('Source-repeat supervisor interrupted')
    signal.signal(signal.SIGTERM,stop)
    try:
        emit('BYPASS_SOURCE_REPEAT_BUDGET_STARTED',**b)
        identity=validate(b);emit('BYPASS_SOURCE_REPEAT_SOURCES_VERIFIED',resources=resources(w.parent,identity))
        old.supervise([sys.executable,'-B',str(ROOT/FILES[1])],b,fd);check_budget(b)
        w.mkdir();write_json(w/'budget.json',b);write_json(w/'identity.json',identity)
        with (RUN_ROOT/CLAIM).open('x',encoding='utf-8') as f:
            json.dump({'workspace':str(w),'protocol':protocol(),'budget_sha256':digest(w/'budget.json'),'identity_sha256':digest(w/'identity.json'),
                       'authorization':'user confirmed D Human76/77 only,total3h'},f,indent=2);f.flush();os.fsync(f.fileno())
        command=[sys.executable,'-u','-B',str(Path(__file__).resolve()),'--mode','worker','--run-dir',str(w)]
        for phase in ('seed76','seed77','finish'):old.supervise(command+['--phase',phase],b,fd)
    except BaseException as error:
        write_json(w.parent/'failure-supervisor.json',{'error':str(error),'type':type(error).__name__,'complete':False,'elapsed_seconds':time.monotonic()-now,'automatic_retry':False});raise


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--mode',choices=('validate','storage','start','worker'),default='validate')
    p.add_argument('--run-dir',type=Path);p.add_argument('--phase',choices=('seed76','seed77','finish'));a=p.parse_args()
    if a.mode=='storage':print(json.dumps(storage_inventory(),indent=2))
    elif a.mode=='validate':print(json.dumps(validate(),indent=2))
    elif a.run_dir is None or (a.mode=='worker' and a.phase is None):raise ValueError('Use sole Bash entry point')
    elif a.mode=='start':start(a.run_dir.resolve())
    else:worker(a.run_dir.resolve(),a.phase)


if __name__=='__main__':main()
