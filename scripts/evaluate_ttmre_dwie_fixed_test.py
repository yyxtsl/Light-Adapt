"""One approved DWIE Test: 100 official documents, C/B/D/G75, no training.

One 3000s clock, CPU <=1200s, GPU/report <=1800s. Separate preparation,
Dev replay, prediction and sealed scoring processes. Default validate is metadata only.
"""
from __future__ import annotations
import argparse
import gc
import hashlib
import http.client
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import socket
import subprocess
import sys
import tarfile
import time
import urllib.error
import urllib.request
import zipfile

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
DOC=ROOT/'refine-logs/DWIE_TEST_PREPARATION_20261026'
AUTH_SHA='434d34c92b423e04a3e365cb3aca424ed7d56486ac4a335c6a32637bf32a949d'
REF_SHA='2278f3897aff947a42b31bcc88389096baa8eb7760f81799425dc3d84385b4b2'
CHECKER_SHA='a081d02c22d4c5d5a9d8258fc0e25b1de28bbc17f1bd6cfffa0ddcf0b253578c'
NONRUNTIME_VARIANT={'scripts/prepare_ttmre_assets.py':'93c67b4105b9d173507effa3c5e6e3ddc3442a82ac4fc396050adb2f8f792a43'}
RUN_ROOT=ROOT/'experiments/2026-10-26-dwie-fixed-test'
DATA_ROOT=ROOT/'dataset/dwie-test-20261026'
CLAIM=RUN_ROOT/'dwie-fixed-test-once-claim.json'
SYSTEMS=('C75','B75','D75','G75')
HARD,CPU,GPU,RESERVE,TERM=3000,1200,1800,60,15
MIB=1024**2

def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def write(p,x):
    with Path(p).open('x',encoding='utf-8') as f:
        json.dump(x,f,ensure_ascii=False,allow_nan=False,separators=(',',':'));f.write('\n')
def progress(p,x):
    # Only this run's progress ledger is mutable; initial clock/identity stay immutable.
    Path(p).write_text(json.dumps(x,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')
def emit(event,**x):print(json.dumps({'event':event,**x},allow_nan=False),flush=True)
def check(b,stage=None,reserve=RESERVE):
    if b is None:return
    now=time.monotonic()
    if now>=b['deadline_monotonic']-reserve:raise TimeoutError('Original 3000s deadline/reserve; no reset')
    if Path('/proc/sys/kernel/random/boot_id').exists() and Path('/proc/sys/kernel/random/boot_id').read_text().strip()!=b['boot_id']:
        raise RuntimeError('Original boot changed')
    if stage is not None and now>=stage-reserve:raise TimeoutError('Original stage ceiling/reserve reached')
def sha(p,b=None):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda:f.read(4*MIB),b''):check(b);h.update(block)
    return h.hexdigest()
def stamp(p):
    s=Path(p).stat();return [s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns]
def tree_size(p):return sum(v.stat().st_size for v in Path(p).rglob('*') if v.is_file())
def authorize():
    if sha(DOC/'AUTHORIZATION_APPROVED.json')!=AUTH_SHA:raise PermissionError('Authorization digest changed')
    a=read(DOC/'AUTHORIZATION_APPROVED.json')
    if not a['approved'] or a['systems']!=list(SYSTEMS) or (a['hard_seconds'],a['CPU_seconds'],a['GPU_report_seconds'])!=(HARD,CPU,GPU):
        raise PermissionError('Not the approved four conditions/3000s')
    if any(a[k] is not False for k in ('model_writes','model_downloads','installs','deletions','extension','Test_based_tuning','new_job_retry')):
        raise PermissionError('Scope changed')
    if a['optimizer_updates']!=0 or a['new_seeds']!=0 or a['article_attempts_max']!=2:
        raise PermissionError('Training/seed/request scope changed')
    if (a['network_bytes_cap'],a['new_dataset_write_cap'],a['new_report_bytes_cap'])!=(100*MIB,200*MIB,500*MIB):
        raise PermissionError('Byte caps changed')
    if sha(DOC/'TEST_PLAN.md')!=a['plan_SHA256'] or sha(DOC/'CANDIDATE_FREEZE.json')!=a['candidate_freeze_SHA256']:
        raise PermissionError('Approved plan/freeze changed')
    return a
def validate():
    a=authorize()
    if sha(ROOT/'scripts/check_ttmre_dwie_fixed_test.py')!=CHECKER_SHA:raise ValueError('New checker changed')
    if sha(DOC/'REFERENCE.json')!=REF_SHA:raise ValueError('Reference changed')
    r=read(DOC/'REFERENCE.json')
    if [v['id'] for v in r['models']]!=list(SYSTEMS) or len(r['Test_headers'])!=100 or len(r['rel2id'])!=63:
        raise ValueError('Fixed condition/document/vocabulary count changed')
    variants={}
    for name,digest in r['sources'].items():
        actual=sha(ROOT/name)
        if actual!=digest:
            if NONRUNTIME_VARIANT.get(name)!=actual:raise ValueError('Original source changed: '+name)
            variants[name]={'local_snapshot_SHA256':digest,'existing_cloud_SHA256':actual,'executed':False}
    if 'prepare_ttmre_assets' in sys.modules:raise RuntimeError('Unrelated asset downloader must not be imported/executed')
    if r['candidate_freeze_SHA256']!=a['candidate_freeze_SHA256']:raise ValueError('Wrong freeze')
    return {'reference':r,'authorization':a,'entry_SHA256':sha(__file__),'reference_SHA256':REF_SHA,
            'authorization_SHA256':AUTH_SHA,'actual_Test_records_read':0,'CUDA_initialized':False,'known_nonruntime_source_variants':variants}

def test_convert(annotation,content,expected_sha1):
    """Explicit Test-only conversion; never retag Test to reuse a train reader."""
    if 'test' not in annotation.get('tags',[]) or 'train' in annotation['tags']:raise ValueError('Test-only converter refuses non-Test')
    if hashlib.sha1(content.encode('utf-8')).hexdigest()!=expected_sha1:raise ValueError('Official article SHA1 mismatch: '+annotation['id'])
    concepts={c['concept']:c for c in annotation['concepts']}
    if not concepts or len(concepts)!=len(annotation['concepts']) or any(type(k) is not int for k in concepts):
        raise ValueError('Empty/duplicate/noninteger concept IDs')
    mentions={k:[] for k in concepts}
    for m in annotation['mentions']:
        key,begin,end=m['concept'],m['begin'],m['end']
        if key not in concepts or type(begin) is not int or type(end) is not int or not 0<=begin<end<=len(content):
            raise ValueError('Invalid mention span/concept')
        if content[begin:end]!=m['text'].replace('\u00a0',' ').replace('\u200b',''):
            raise ValueError('Official mention span mismatch: '+annotation['id'])
        mentions[key].append({'begin':begin,'end':end,'text':content[begin:end]})
    entities=[{'id':k,'mentions':sorted(mentions[k],key=lambda m:(m['begin'],m['end']))} for k in sorted(concepts)]
    triples=set()
    for x in annotation['relations']:
        h,t,r=x['s'],x['o'],x['p']
        if h not in concepts or t not in concepts or not isinstance(r,str) or not r:raise ValueError('Invalid relation')
        triples.add((h,r,t))
    public={'id':annotation['id'],'content':content,'entities':entities,'relations':[]}
    gold={'id':annotation['id'],'entities':entities,'relations':[{'h':h,'r':r,'t':t} for h,r,t in sorted(triples)]}
    return public,gold

class OfficialRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self,req,fp,code,msg,headers,newurl):
        # Redirects are not extra hidden article requests or alternate sources.
        raise ValueError('Article redirect refused; use frozen official endpoint')

class Fetcher:
    def __init__(self,root,b,stage,opener=None):
        self.root,self.b,self.stage=root,b,stage
        self.opener=opener or urllib.request.build_opener(OfficialRedirect()).open
        self.ledger={'network_bytes':0,'network_bytes_cap':100*MIB,'attempts':[]}
    def persist(self):progress(self.root/'manifests/network.json',self.ledger)
    def consume(self,response):
        parts=[]
        while True:
            check(self.b,self.stage)
            chunk=response.read(min(65536,max(1,100*MIB-self.ledger['network_bytes'])))
            if not chunk:break
            self.ledger['network_bytes']+=len(chunk)
            if self.ledger['network_bytes']>=100*MIB:raise ValueError('Network cap reached; no further requests')
            parts.append(chunk)
            if sum(map(len,parts))>2*MIB:raise ValueError('Article response >2MiB')
        return b''.join(parts)
    def fetch(self,row):
        if not re.fullmatch(r'https://api\.dw\.com/api/detail/article/[0-9]+',row['url']):raise ValueError('Unfrozen URL')
        for attempt in (1,2):
            check(self.b,self.stage)
            entry={'id':row['id'],'url':row['url'],'attempt':attempt,'ok':False}
            self.ledger['attempts'].append(entry);self.persist()
            try:
                req=urllib.request.Request(row['url'],headers={'User-Agent':'TTMRE-DWIE-fixed-Test/1.0'})
                with self.opener(req,timeout=8) as response:
                    entry['status']=response.status
                    raw=self.consume(response)
                    if response.status!=200:raise ValueError('Unexpected HTTP status')
                entry.update(ok=True,bytes=len(raw),SHA256=hashlib.sha256(raw).hexdigest())
                with (self.root/'raw'/(row['id']+'-article-response.json')).open('xb') as f:f.write(raw)
                return raw
            except urllib.error.HTTPError as error:
                entry.update(status=error.code,error=str(error))
                # Count unsuccessful response payloads in the same aggregate budget.
                raw=self.consume(error)
                if raw:
                    with (self.root/'raw'/(row['id']+'-attempt'+str(attempt)+'-HTTP-error.bin')).open('xb') as f:f.write(raw)
                if error.code not in (429,502,503,504) or attempt==2:raise
            except (urllib.error.URLError,socket.timeout,TimeoutError,ConnectionError,http.client.IncompleteRead) as error:
                entry['error']=str(error)
                if isinstance(error,urllib.error.URLError) and not isinstance(error.reason,(socket.timeout,TimeoutError,ConnectionError,OSError)):
                    raise
                if attempt==2:raise
            finally:self.persist()
            check(self.b,self.stage);time.sleep(1)
        raise RuntimeError('Unreachable request state')

def data_guard(d,b,stage):
    check(b,stage)
    # ZIP has its separately approved fixed allowance.
    if tree_size(d)-sum(p.stat().st_size for p in (d/'source').glob('*.zip'))>200*MIB:raise ValueError('Dataset write cap exceeded')
def prepare(w,i,b,stage):
    r=i['reference'];d=Path(i['data_root'])
    for n in ('source','raw','converted','features','manifests'):(d/n).mkdir()
    source=DOC/'official-source.zip'
    if sha(source,b)!=r['official_archive_SHA256'] or source.stat().st_size!=r['official_archive_bytes']:raise ValueError('Official ZIP mismatch')
    dest=d/'source'/source.name
    with source.open('rb') as src,dest.open('xb') as out:shutil.copyfileobj(src,out)
    from prepare_dwie_cpu_admission import reconstruct
    import ttmre_dwie_input as dwie
    from tokenizers import Tokenizer
    tokenizer=Tokenizer.from_file(str(ROOT/r['model_config_dir']/'tokenizer.json'))
    fetcher=Fetcher(d,b,stage)
    features=[];gold=[];audits=[];inputs=[]
    split=read(ROOT/r['target_split'])
    headers=r['Test_headers'];ids=[x['id'] for x in headers]
    if len(set(ids))!=100 or set(ids)&(set(split['train'])|set(split['dev'])):raise ValueError('Test split overlap/duplicates')
    with zipfile.ZipFile(dest) as z:
        for index,row in enumerate(headers):
            data_guard(d,b,stage)
            annotation=json.loads(z.read(row['archive_member']))
            if annotation['id']!=row['id'] or annotation['tags']!=row['tags']:raise ValueError('Official header changed')
            raw=fetcher.fetch(row)
            content=reconstruct(json.loads(raw),row['id'])
            public,scoring=test_convert(annotation,content,row['expected_article_SHA1'])
            feature,audit=dwie.make_feature(public,tokenizer,r['rel2id'])
            if feature['gold'] or audit['min_coverage']<1 or audit['mentions_mapped']!=sum(len(e['mentions']) for e in public['entities']):
                raise ValueError('Feature coverage/gold isolation failed')
            write(d/'converted'/(row['id']+'-public.json'),public)
            features.append(feature);gold.append(scoring);audits.append({'id':row['id'],**audit})
            inputs.append({'id':row['id'],'content_SHA1':hashlib.sha1(content.encode()).hexdigest(),
                           'annotation_SHA256':hashlib.sha256(z.read(row['archive_member'])).hexdigest(),
                           'response_SHA256':hashlib.sha256(raw).hexdigest()})
            emit('DWIE_TEST_DATA_PROGRESS',completed=index+1,total=100,network_bytes=fetcher.ledger['network_bytes'],metrics_computed=False)
    if len(features)!=100 or [x['id'] for x in features]!=ids:raise ValueError('Require 100/100; no subset')
    write(d/'features/Test_inputs.json',features);write(d/'converted/Test_gold.json',gold)
    write(d/'manifests/coverage.json',audits);write(d/'manifests/sources.json',inputs)
    for name,value in [('candidate_freeze_SHA256',r['candidate_freeze_SHA256'])]:
        if len(value)!=64:raise ValueError(name)
    hashes={str(p.relative_to(d)):sha(p,b) for p in d.rglob('*') if p.is_file()}
    result={'passed':True,'documents':100,'features':100,'model_gold_rows':0,'network_bytes':fetcher.ledger['network_bytes'],
            'files':hashes,'dataset_written_bytes':tree_size(d),'label_independent_features':True,'metrics_computed':False}
    write(d/'manifests/admission.json',result);data_guard(d,b,stage)
    write(w/'prepared.json',{'data_root':str(d),'admission_SHA256':sha(d/'manifests/admission.json'),
                            'inputs_SHA256':sha(d/'features/Test_inputs.json'),'gold_SHA256':sha(d/'converted/Test_gold.json'),
                            'documents':100,'pairs':sum(x['pairs'] for x in audits),'all100_required':True})

def build(m,r,device,factory=None,state_loader=None):
    import torch
    import ttmre_dwie_gpu_probe_v4 as probe
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(75)
        net,execution=(factory or probe.build_model)(ROOT/r['model_config_dir'])
        if m['id']!='C75':adapter=probe.replacement.install(net,'bypass')
        else:adapter=None
        probe.set_classifier(net,63)
    # Constructor's random tensors are overwritten by the complete trained target state.
    state=(state_loader or (lambda p:torch.load(p,map_location='cpu',weights_only=True)))(ROOT/m['checkpoint'])
    loaded=probe.strict_load(net,state,m['id']+' fixed trained DWIE target')
    del state
    if sum(p.numel() for p in net.parameters())!=m['expected_parameters']:raise ValueError('Parameter count changed')
    net.requires_grad_(False);net.args.device=device;net.train_mode='finetune';net.to(device);net.eval()
    return net,{'id':m['id'],'checkpoint':m['checkpoint'],'checkpoint_SHA256':m['checkpoint_SHA256'],
                'strict_load':loaded,'execution':execution,'replacement':adapter,'parameters':m['expected_parameters'],
                'trained_target_head_loaded':True,'optimizer_updates':0}

def assert_scores(actual,expected,path='metric'):
    if isinstance(expected,dict):
        for k,v in expected.items():
            if k not in actual:raise ValueError('Missing '+path+'.'+k)
            assert_scores(actual[k],v,path+'.'+k)
    elif isinstance(expected,float):
        if not math.isclose(actual,expected,abs_tol=1e-12,rel_tol=0):raise ValueError('Dev replay changed '+path)
    elif actual!=expected:raise ValueError('Dev replay changed '+path)
def get_features(p):
    if str(p).endswith('.jsonl'):
        with Path(p).open(encoding='utf-8') as f:return [json.loads(line) for line in f]
    return read(p)
def run_features(net,features,rel2id,device,b,stage,system,phase,logits_fn=None):
    import train_ttmre_dwie_adaptation as base
    import torch
    result=[];timings=[];t0=time.monotonic()
    for index,f in enumerate(features):
        check(b,stage)
        if device.type=='cuda':torch.cuda.synchronize(device)
        t=time.monotonic()
        one=base.predict_documents(net,[f],rel2id,device,logits_fn,lambda:check(b,stage))
        if device.type=='cuda':torch.cuda.synchronize(device)
        timings.append({'id':f['id'],'seconds':time.monotonic()-t,'pairs':len(f['inputs']['entity_ids'])**2,
                        'content_tokens':len(f['inputs']['content_ids'])})
        result.extend(one)
        if (index+1)%10==0 or index==0:emit('DWIE_FIXED_PROGRESS',system=system,phase=phase,documents=index+1,total=len(features),metrics_computed=False)
        # Conservative per-condition completion gate; do not truncate a corpus to fit.
        remaining=len(features)-index-1
        projected=(time.monotonic()-t0)/(index+1)*remaining*1.5+RESERVE
        if b is not None and projected>min(stage,b['deadline_monotonic'])-time.monotonic():
            raise TimeoutError('Insufficient original time for full condition; no subset')
    return sorted(result),timings
def gpu_worker(w,i,b,stage,phase,factory=None,logits_fn=None):
    import torch
    import train_ttmre_dwie_adaptation as base
    import ttmre_dwie_input as dwie
    r=i['reference'];rel2id=r['rel2id'];device=torch.device('cuda' if factory is None else 'cpu')
    if phase=='dev':
        features=get_features(ROOT/r['dev_features']);documents=read(ROOT/r['dev_scoring_metadata'])['documents']
        expected_ids=read(ROOT/r['target_split'])['dev']
        if [f['id'] for f in features]!=expected_ids or len(features)!=100:raise ValueError('Wrong Dev corpus/order')
    else:
        if not (w/'all_Dev_replayed.json').is_file():raise ValueError('All four Dev replays required before any Test prediction')
        d=Path(i['data_root']);prepared=read(w/'prepared.json')
        if sha(d/'features/Test_inputs.json')!=prepared['inputs_SHA256']:raise ValueError('Test inputs changed')
        features=get_features(d/'features/Test_inputs.json')
        if len(features)!=100 or any(f['gold'] for f in features):raise ValueError('Full label-free Test required')
    # Largest memory geometry first, independent of relation labels.
    features=sorted(features,key=lambda f:(-len(f['inputs']['content_ids']),-len(f['inputs']['entity_ids']),f['id']))
    finished=[]
    for model_index,m in enumerate(r['models']):
        check(b,stage);start=time.monotonic()
        frozen_stamp=stamp(ROOT/m['checkpoint'])
        net,recipe=build(m,r,device,factory)
        output=w/m['id'];output.mkdir(exist_ok=True)
        try:
            predictions,timing=run_features(net,features,rel2id,device,b,stage,m['id'],phase,logits_fn)
            if phase=='dev':
                reference=read(ROOT/m['Dev_predictions'])
                if predictions!=sorted(map(tuple,reference)):raise ValueError('Full Dev prediction set differs: '+m['id'])
                score=dwie.score_documents(documents,predictions,[n for n,v in rel2id.items() if v>0],ROOT/r['official_metric'])
                assert_scores(score,m['Dev_reference'])
                write(output/'Dev_replay.json',{'passed':True,'full_prediction_set_identical':True,'metrics':score,'recipe':recipe,
                                                'elapsed_seconds':time.monotonic()-start,'timing':timing})
            else:
                write(output/'Test_predictions.json',predictions)
                write(output/'Test_recipe.json',{**recipe,'documents':len(features),'elapsed_seconds':time.monotonic()-start,
                                                'timing':timing,'document_order':[f['id'] for f in features],
                                                'single_timing_not_repeated_benchmark':True,'Test_metrics_computed':False})
        finally:
            del net;gc.collect()
            if device.type=='cuda':torch.cuda.empty_cache()
        if stamp(ROOT/m['checkpoint'])!=frozen_stamp:raise ValueError('Frozen checkpoint modified')
        finished.append(m['id']);elapsed=time.monotonic()-start
        remaining=len(SYSTEMS)-model_index-1
        # Dev must also leave enough time for all four Test passes.
        passes=remaining+(4 if phase=='dev' else 0)
        if b is not None and 1.5*elapsed*passes+RESERVE>min(stage,b['deadline_monotonic'])-time.monotonic():
            raise TimeoutError('Insufficient original time for remaining full-model passes')
    if phase=='dev':write(w/'all_Dev_replayed.json',{'systems':finished,'all_full_predictions_identical':True})
    else:seal(w,i,b)

def seal(w,i,b=None):
    d=Path(i['data_root']);prepared=read(w/'prepared.json')
    names=['all_Dev_replayed.json','prepared.json']+[m+'/'+n for m in SYSTEMS for n in ('Test_predictions.json','Test_recipe.json','Dev_replay.json')]
    if read(w/'all_Dev_replayed.json')['systems']!=list(SYSTEMS):raise ValueError('Incomplete Dev gate')
    files={name:sha(w/name,b) for name in names}
    for name,digest in [('features/Test_inputs.json',prepared['inputs_SHA256']),('converted/Test_gold.json',prepared['gold_SHA256'])]:
        if sha(d/name,b)!=digest:raise ValueError('Prepared Test source changed')
    write(w/'global_predictions_sealed.json',{'systems':list(SYSTEMS),'documents_each':100,'files':files,
        'Test_inputs_SHA256':prepared['inputs_SHA256'],'Test_gold_SHA256':prepared['gold_SHA256'],
        'sealed_monotonic':time.monotonic(),'all_predictions_before_scoring':True,'metrics_computed':False})
    emit('DWIE_ALL_FOUR_PREDICTIONS_SEALED',metrics_computed=False)
def score(w,i,b=None):
    # Do not open gold until this complete global seal has been verified.
    s=read(w/'global_predictions_sealed.json')
    if s['systems']!=list(SYSTEMS) or s['documents_each']!=100 or not s['all_predictions_before_scoring']:raise ValueError('Incomplete seal')
    for name,digest in s['files'].items():
        if sha(w/name,b)!=digest:raise ValueError('Sealed file changed: '+name)
    d=Path(i['data_root'])
    if sha(d/'converted/Test_gold.json',b)!=s['Test_gold_SHA256']:raise ValueError('Gold source changed')
    write(w/'scoring_started.json',{'seal_SHA256':sha(w/'global_predictions_sealed.json'),'started_monotonic':time.monotonic()})
    documents=read(d/'converted/Test_gold.json')
    if len(documents)!=100 or [v['id'] for v in documents]!=[v['id'] for v in i['reference']['Test_headers']]:raise ValueError('Wrong full Test corpus')
    import ttmre_dwie_input as dwie
    r=i['reference'];known=[n for n,v in r['rel2id'].items() if v>0]
    rows={}
    for m in SYSTEMS:
        check(b)
        pred=list(map(tuple,read(w/m/'Test_predictions.json')))
        rows[m]=dwie.score_documents(documents,pred,known,ROOT/r['official_metric'])
    diff={x+'-'+y:{metric:100*(rows[x]['primary_full_id'][metric]-rows[y]['primary_full_id'][metric]) for metric in ('f1','precision','recall')}
          for x,y in [('B75','C75'),('B75','D75'),('B75','G75'),('D75','G75')]}
    write(w/'Test_metrics.json',{'metrics':rows,'differences_pp':diff,'primary':'B75-C75 full-ID micro-F1',
            'single_fixed_seed':75,'official_metrics_supplemental':True,'label_inventory_train_only':True})
    write(w/'completion.json',{'passed':True,'systems':list(SYSTEMS),'documents_each':100,'optimizer_updates':0,'new_seeds':0,
             'checkpoint_writes':0,'global_seal_SHA256':sha(w/'global_predictions_sealed.json'),
             'elapsed_seconds':time.monotonic()-b['start_monotonic'] if b else None})
    emit('DWIE_FIXED_TEST_SCORED',systems=4,metrics_path=str(w/'Test_metrics.json'))

def require_lock():
    import fcntl
    fd=int(os.environ['TTMRE_DWIE_TEST_LOCK_FD']);os.fstat(fd)
    fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB);return fd
def resources():
    from inventory_ttmre_redocred_generic_control import memory_metadata
    mem=memory_metadata();free=shutil.disk_usage(ROOT).free
    if free<2*1024**3 or mem.get('effective_available_bytes',0)<32*1024**3:raise RuntimeError('Need 2GiB free disk and32GiB effective RAM')
    return {'free_bytes':free,'memory':mem}
def admit(w,b):
    i=validate();r=i['reference'];i['resources']=resources()
    for name,digest in r['assets'].items():
        if sha(ROOT/name,b)!=digest:raise ValueError('Asset changed: '+name)
    for m in r['models']:
        for name,digest in [(m['checkpoint'],m['checkpoint_SHA256']),(m['Dev_predictions'],m['Dev_predictions_SHA256'])]:
            if sha(ROOT/name,b)!=digest:raise ValueError('Checkpoint/Dev predictions changed: '+name)
    from ttmre_nvidia_smi_preflight import inspect_gpu
    i['GPU_before_CUDA']=inspect_gpu()
    if not i['GPU_before_CUDA']['passed']:raise RuntimeError('GPU occupied before CUDA')
    i['data_root']=str(DATA_ROOT/w.parent.name)
    return i
def verify(w,b):
    c=read(CLAIM)
    if (w.name!='workspace' or w.parent.parent!=RUN_ROOT.resolve() or c['workspace']!=str(w)
        or sha(w/'budget.json')!=c['budget_SHA256'] or sha(w/'identity.json')!=c['identity_SHA256']
        or b['hard_seconds']!=HARD or b['deadline_monotonic']!=b['start_monotonic']+HARD):raise ValueError('Original run context changed')
    current=validate()
    if current['entry_SHA256']!=read(w/'identity.json')['entry_SHA256']:raise ValueError('Entry changed after launch')
    check(b)
def worker(w,phase,stage):
    authorize();require_lock();b=read(w/'budget.json');verify(w,b);i=read(w/'identity.json')
    try:
        allowed=b['start_monotonic']+CPU if phase=='prepare' else read(w/'GPU_stage.json')['deadline_monotonic']
        if stage!=allowed or stage>b['deadline_monotonic']:raise ValueError('Worker stage clock changed')
        if phase=='prepare':prepare(w,i,b,stage)
        elif phase in ('dev','predict'):
            resources()
            from ttmre_nvidia_smi_preflight import inspect_gpu
            external=inspect_gpu()
            if not external['passed']:raise RuntimeError('GPU occupied before CUDA')
            from ttmre_dwie_runtime_fix import runtime_identity
            write(w/('runtime-'+phase+'.json'),runtime_identity())
            for name,digest in i['reference']['assets'].items():
                if sha(ROOT/name,b)!=digest:raise ValueError('GPU asset changed')
            for m in i['reference']['models']:
                if sha(ROOT/m['checkpoint'],b)!=m['checkpoint_SHA256']:raise ValueError('Target checkpoint changed')
            gpu_worker(w,i,b,stage,phase)
        elif phase=='score':score(w,i,b)
        else:raise ValueError('Unknown phase')
        check(b,stage)
    except BaseException as error:
        write(w/('failure-'+phase+'.json'),{'error':str(error),'type':type(error).__name__,'phase':phase,
                'optimizer_updates':0,'elapsed_seconds':time.monotonic()-b['start_monotonic']})
        raise
def supervise(command,b,stage,fd):
    child=subprocess.Popen(command,start_new_session=True,pass_fds=(fd,));term=False
    try:
        while True:
            check(b,stage,reserve=0);code=child.poll()
            if code is not None:
                if code or term:raise RuntimeError('Phase failed; no second launch/budget')
                return
            if time.monotonic()>=min(stage,b['deadline_monotonic'])-TERM and not term:
                os.killpg(child.pid,signal.SIGTERM);term=True
            time.sleep(.2)
    finally:
        if child.poll() is None:os.killpg(child.pid,signal.SIGKILL);child.wait()
def archive(w,i,b,stage=None):
    check(b,stage,reserve=10)
    if tree_size(w)>250*MIB:raise ValueError('Report source exceeds archive allowance')
    with tarfile.open(w/'reports.tar.gz','x:gz') as t:
        for p in sorted(w.rglob('*')):
            check(b,stage,reserve=10)
            if p.is_file() and p.name!='reports.tar.gz':t.add(p,arcname='workspace/'+str(p.relative_to(w)))
        d=Path(i['data_root'])
        if d.exists():
            for p in sorted((d/'manifests').glob('*.json')):t.add(p,arcname='data_manifests/'+p.name)
            for name in ('converted/Test_gold.json','features/Test_inputs.json'):
                if (d/name).is_file():t.add(d/name,arcname='data/'+name)
    if tree_size(w)>500*MIB:raise ValueError('Report cap exceeded')
    emit('DWIE_REPORT_READY',bytes=(w/'reports.tar.gz').stat().st_size,SHA256=sha(w/'reports.tar.gz',b))
def start(w):
    authorize();fd=require_lock()
    start=float(os.environ['TTMRE_DWIE_TEST_START']);now=time.monotonic()
    if not math.isfinite(start) or start>now:raise ValueError('Invalid original clock')
    b={'start_monotonic':start,'deadline_monotonic':start+HARD,'hard_seconds':HARD,
       'boot_id':os.environ['TTMRE_DWIE_TEST_BOOT'],'start_utc':os.environ['TTMRE_DWIE_TEST_UTC']}
    if w.name!='workspace' or w.parent.parent!=RUN_ROOT.resolve() or w.exists() or CLAIM.exists():raise ValueError('One new task only')
    w.mkdir();write(w/'budget.json',b)
    # Claim immediately, including failed admission; a failed check cannot silently create another clock.
    write(CLAIM,{'workspace':str(w),'budget_SHA256':sha(w/'budget.json'),'authorization_SHA256':AUTH_SHA,'identity_SHA256':None})
    i=None
    try:
        check(b,start+CPU);i=admit(w,b);write(w/'identity.json',i)
        c=read(CLAIM);c['identity_SHA256']=sha(w/'identity.json');progress(CLAIM,c)
        data=Path(i['data_root']);data.mkdir(parents=True,exist_ok=False)
        write(data/'README.json',{'purpose':'Approved100-document DWIE Test only','subdirs':['source','raw','converted','features','manifests'],
                                 'gold_separate_from_model_inputs':True,'original_budget_SHA256':sha(w/'budget.json')})
        checker=ROOT/'scripts/check_ttmre_dwie_fixed_test.py'
        supervise([sys.executable,'-B',str(checker)],b,start+CPU,fd)
        cmd=[sys.executable,'-u','-B',str(Path(__file__).resolve()),'--mode','worker','--run-dir',str(w)]
        supervise(cmd+['--phase','prepare','--stage-deadline',str(start+CPU)],b,start+CPU,fd)
        gpu_started=time.monotonic();gpu_deadline=min(start+HARD,gpu_started+GPU)
        write(w/'GPU_stage.json',{'start_monotonic':gpu_started,'deadline_monotonic':gpu_deadline,'maximum_seconds':GPU,
                                  'original_total_deadline':start+HARD})
        for phase in ('dev','predict','score'):
            supervise(cmd+['--phase',phase,'--stage-deadline',str(gpu_deadline)],b,gpu_deadline,fd)
        archive(w,i,b,gpu_deadline);check(b,gpu_deadline,reserve=0)
    except BaseException as error:
        write(w/'supervisor-failure.json',{'error':str(error),'type':type(error).__name__,'budget':b,
              'elapsed_seconds':time.monotonic()-start,'optimizer_updates':0,'second_launch_allowed':False})
        if i is not None:
            try:archive(w,i,b)
            except Exception as archive_error:emit('REPORT_ARCHIVE_FAILED',error=str(archive_error))
        raise
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode',choices=('validate','authorize','start','worker'),default='validate')
    p.add_argument('--run-dir',type=Path);p.add_argument('--phase',choices=('prepare','dev','predict','score'))
    p.add_argument('--stage-deadline',type=float);a=p.parse_args()
    if a.mode=='authorize':print(json.dumps(authorize()))
    elif a.mode=='validate':
        v=validate();print(json.dumps({'passed':True,'source_count':len(v['reference']['sources']),'Test_records_read':0,'CUDA_initialized':False}))
    elif a.run_dir is None:raise ValueError('Use sole launcher')
    elif a.mode=='start':start(a.run_dir.resolve())
    elif a.phase is None or a.stage_deadline is None:raise ValueError('Use fixed worker phase and original deadline')
    else:worker(a.run_dir.resolve(),a.phase,a.stage_deadline)
if __name__=='__main__':main()
