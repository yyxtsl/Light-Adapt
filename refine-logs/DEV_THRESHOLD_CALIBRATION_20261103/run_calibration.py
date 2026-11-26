"""One fixed C/B/C-off Dev-only calibration; no model writes or optimizer."""
from __future__ import annotations
import argparse,csv,gc,hashlib,json,math,os,shutil,signal,subprocess,sys,time,traceback
from pathlib import Path

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path[:0]=[str(HERE),str(ROOT),str(ROOT/'scripts')]
KINDS=('C','B','C-off')
HARD_SECONDS=1200
DEADLINE=None


def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))


def write(p,value):
    with Path(p).open('x',encoding='utf-8') as f:
        json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')


def check():
    if DEADLINE is not None and time.monotonic()>=DEADLINE-30:
        raise TimeoutError('Original 1200-second envelope exhausted; no extension')


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(4*1024**2),b''):
            check();h.update(b)
    return h.hexdigest()


def emit(event,**kw):print(json.dumps({'event':event,**kw},allow_nan=False),flush=True)


def validate():
    manifest=read(HERE/'PACKAGE_MANIFEST.json')
    for name,expected in manifest['package_files'].items():
        p=(HERE/name).resolve()
        if not p.is_relative_to(HERE) or sha(p)!=expected:raise ValueError('Changed package: '+name)
    if manifest['scope']!={'systems':list(KINDS),'seed':75,'Dev_documents':500,'Dev_pairs':193232,
            'hard_seconds':1200,'Test':False,'training':False,'checkpoint_writes':False,
            'grid_min':-100,'grid_max':100,'grid_step':0.05,'grid_points':4001}:
        raise ValueError('Scope changed')
    return manifest


def local_contract():
    manifest=validate()
    from check_calibration import run
    run()
    emit('LOCAL_CONTRACT_PASS',scope=manifest['scope'],real_checkpoint_opens=0,GPU_forwards=0)


def inputs(ref):
    result=dict(ref['input_files'])
    result.update({ref['model_dir']+'/'+n:h for n,h in ref['generic_files'].items()})
    result.update({v['path']:v['sha256'] for v in ref['checkpoints'].values()})
    return result


def independently_score(predictions,gold,annotated,shared,curve):
    metrics=shared.score_predictions(predictions,gold,annotated,None)
    for native,field in [('dev_precision','precision'),('dev_recall','recall'),('dev_F1','F1')]:
        if not math.isclose(metrics[native],curve[field],abs_tol=1e-9,rel_tol=0):
            raise ValueError('Independent exported-prediction scoring disagreement: '+native)
    return metrics


def experiment(output):
    global DEADLINE
    start=float(os.environ['TTMRE_DEV_CALIBRATION_START'])
    if not 0<=time.monotonic()-start<60:raise ValueError('Invalid launch clock')
    DEADLINE=start+HARD_SECONDS
    output=output.resolve();runroot=ROOT/'experiments/2026-11-03-dev-threshold-calibration'
    if output.parent.parent!=runroot or output.name!='workspace':raise ValueError('Wrong output scope')
    output.mkdir(exist_ok=False)
    budget={'start_monotonic':start,'deadline_monotonic':DEADLINE,'hard_seconds':1200,
            'boot_id':Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
            'start_utc':os.environ['TTMRE_DEV_CALIBRATION_UTC'],'Test':False,'retry':False,'extension':False}
    write(output/'budget.json',budget)
    try:
        signal.signal(signal.SIGTERM,lambda *_:(_ for _ in ()).throw(TimeoutError('External timeout')))
        manifest=validate();ref=read(HERE/'REFERENCE.json')
        import fcntl
        lock=(ROOT/'experiments/2026-10-08-baseline-alignment/.training.lock').open('rb')
        fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        # Only this task's claim is touched; all historical task claims are read-only.
        write(runroot/'dev-calibration-once-claim.json',{'workspace':str(output),'budget':budget,
              'manifest_sha256':sha(HERE/'PACKAGE_MANIFEST.json')})
        if shutil.disk_usage(ROOT).free<1024**3:raise RuntimeError('Need 1 GiB free; no deletion')
        # External inspection must occur before import/runtime calls into Torch CUDA.
        from ttmre_nvidia_smi_preflight import inspect_gpu
        gpu=inspect_gpu()
        if not gpu['passed']:raise RuntimeError('GPU not idle; no takeover')
        write(output/'GPU_admission.json',gpu)
        from inventory_ttmre_redocred_generic_control import memory_metadata
        memory=memory_metadata()
        if memory.get('effective_available_bytes') is None or memory['effective_available_bytes']<16*1024**3:
            raise RuntimeError('Insufficient effective RAM')
        pins=dict(ref['source_files']);pins.update(manifest['project_sources']);pins.update(inputs(ref))
        for name,expected in pins.items():
            p=(ROOT/name).resolve()
            if not p.is_relative_to(ROOT) or sha(p)!=expected:raise ValueError('Input/source mismatch: '+name)
        write(output/'INPUT_SHA256.json',pins)
        write(output/'identity.json',{'reference_sha256':sha(HERE/'REFERENCE.json'),
              'manifest_sha256':sha(HERE/'PACKAGE_MANIFEST.json'),'scope':manifest['scope'],
              'gpu':gpu,'memory':memory})
        import numpy as np
        import torch
        import evaluate_ttmre_c_off_control as old
        from calibration_core import fixed_grid,make_gold_mask,scan,prediction_records
        _,generic,core=old.modules()
        args,_,dev,gold,annotated,_=generic.data_context(output,{'reference':ref},budget)
        if len(dev)!=500 or len(gold)!=500:raise ValueError('Need complete Dev')
        order=[{'title':f['title'],'hts':f['hts']} for f in dev]
        if sum(len(d['hts']) for d in order)!=193232:raise ValueError('Unexpected candidate count')
        rel2id=read(ROOT/'meta/rel2id.json');id2rel={v:k for k,v in rel2id.items()}
        mask=make_gold_mask(order,gold,rel2id)
        if mask.shape!=(193232,96):raise ValueError('Unexpected relation geometry')
        write(output/'Dev_pair_order.json',order)
        write(output/'runtime.json',{'python':sys.version,'torch':torch.__version__,
              'numpy':np.__version__,'device':str(args.device),'Test':False})
        expected_env=ref['environment']['versions']
        import transformers
        for name,actual in [('torch',torch.__version__),('numpy',np.__version__),('transformers',transformers.__version__)]:
            if actual!=expected_env[name]:raise ValueError('Runtime differs: '+name)
        if args.device.type!='cuda':raise RuntimeError('CPU inference not authorized by this plan')
        immutable={name:(ROOT/name).stat() for name in inputs(ref)}
        summaries={}
        for kind in KINDS:
            check();started=time.monotonic()
            with old.no_training():
                model,recipe=old.build(args,ref,kind)
                before=core.preflight.state_digest(model);rng=core.preflight.rng_state()
                blocks=[]
                with torch.no_grad():
                    for offset in range(0,len(dev),4):
                        check()
                        features=dev[offset:offset+4]
                        feed=core.shared.inputs_from_batch(core.shared.collate_fn(features),args.device,labels=False)
                        logits=model(**feed)
                        if not torch.isfinite(logits).all():raise ValueError('Nonfinite logits')
                        blocks.append(logits.detach().cpu().numpy().copy())
                        del logits,feed
                if before!=core.preflight.state_digest(model) or any(p.grad is not None for p in model.parameters()):
                    raise ValueError('Model state/gradient changed')
                core.preflight.assert_rng_equal(rng,core.preflight.rng_state())
                values=np.concatenate(blocks);del blocks
                if values.shape!=(193232,97) or values.dtype!=np.float32:raise ValueError('Logit geometry/precision differs')
                with (output/(kind+'-Dev-logits.npy')).open('xb') as f:np.save(f,values,allow_pickle=False)
                native=prediction_records(values,0,order,id2rel)
                expected=read(HERE/('reference_'+kind+'_predictions.json'))
                if old.facts(native)!=old.facts(expected):raise ValueError('delta=0 archived replay differs: '+kind)
                baseline=core.shared.score_predictions(native,gold,annotated,None)
                for key,value in ref['reference_metrics'][kind].items():
                    if key!='dev_IgnF1_distant' and not math.isclose(baseline[key],value,abs_tol=1e-6,rel_tol=0):
                        raise ValueError('Baseline score differs: '+kind+' '+key)
                del model;gc.collect();torch.cuda.empty_cache()
                check();result=scan(values,mask,fixed_grid());check()
                selected=prediction_records(values,result['selected']['delta'],order,id2rel)
                selected_metrics=independently_score(selected,gold,annotated,core.shared,result['selected'])
                zero=next(v for v in result['rows'] if v['delta']==0)
                independently_score(native,gold,annotated,core.shared,zero)
                write(output/(kind+'-zero-predictions.json'),native)
                write(output/(kind+'-selected-predictions.json'),selected)
                write(output/(kind+'-curve.json'),result)
                with (output/(kind+'-curve.csv')).open('x',newline='',encoding='utf-8') as f:
                    writer=csv.DictWriter(f,fieldnames=list(result['rows'][0]));writer.writeheader();writer.writerows(result['rows'])
                summary={'recipe':recipe,'zero':zero,'selected':result['selected'],
                         'zero_metrics':baseline,'selected_metrics':selected_metrics,
                         'boundary_limited':result['boundary_limited'],
                         'logits_sha256':sha(output/(kind+'-Dev-logits.npy')),
                         'elapsed_seconds':time.monotonic()-started,'archived_prediction_replay':True,
                         'model_state_unchanged':True,'RNG_unchanged':True}
                write(output/(kind+'-summary.json'),summary);summaries[kind]=summary
                emit('DEV_CALIBRATION_MODEL_COMPLETE',model=kind,zero_F1=zero['F1'],
                     delta=summary['selected']['delta'],selected_F1=summary['selected']['F1'],
                     boundary_limited=summary['boundary_limited'])
                del values,native,expected,selected,result;gc.collect()
        for name,stat in immutable.items():
            now=(ROOT/name).stat()
            if (now.st_ino,now.st_size,now.st_mtime_ns)!=(stat.st_ino,stat.st_size,stat.st_mtime_ns):
                raise ValueError('Original input changed: '+name)
        # Rehash both original trained weights at completion, not just metadata.
        for record in ref['checkpoints'].values():
            if sha(ROOT/record['path'])!=record['sha256']:raise ValueError('Original checkpoint changed')
        completion={'passed':True,'systems':summaries,'scope':manifest['scope'],
            'Coff_calibration_gain_pp':summaries['C-off']['selected']['F1']-summaries['C-off']['zero']['F1'],
            'B_minus_Coff_zero_pp':summaries['B']['zero']['F1']-summaries['C-off']['zero']['F1'],
            'B_minus_Coff_selected_pp':summaries['B']['selected']['F1']-summaries['C-off']['selected']['F1'],
            'elapsed_seconds':time.monotonic()-start,'optimizer_updates':0,'checkpoint_writes':0,
            'Test':False,'selection_scope':'Dev-optimized diagnostic, not held-out validation',
            'original_weights_rehashed_unchanged':True}
        check();write(output/'completion.json',completion)
        emit('DEV_THRESHOLD_CALIBRATION_COMPLETE',elapsed_seconds=completion['elapsed_seconds'])
    except BaseException as exc:
        failure={'error':repr(exc),'traceback':traceback.format_exc(),'elapsed_seconds':time.monotonic()-start,
                 'Test':False,'optimizer_updates':0,'no_automatic_retry':True}
        if not (output/'failure.json').exists():write(output/'failure.json',failure)
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--mode',choices=['validate','run'],default='validate')
    parser.add_argument('--output',type=Path);args=parser.parse_args()
    if args.mode=='validate':local_contract()
    elif args.output is None:parser.error('run requires a new output directory')
    else:experiment(args.output)
