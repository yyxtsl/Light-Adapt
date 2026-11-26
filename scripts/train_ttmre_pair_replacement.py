"""Authorized bypass/MLP Human screen, original ATGL, Dev only, total6h.

Default validate is read-only. Use the one-use launcher; no Test/new Distant,
reference optimization, retries, resume, tuning, extension or auto follow-up.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import shutil
import signal
import statistics
import sys
import time
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"scripts"))
import train_ttmre_spu_soup_control as old

base,core,soup=old.base,old.core,old.soup
read,digest,write_json,emit=old.read,old.digest,old.write_json,old.emit
RUN_ROOT=ROOT/"experiments/2026-10-21-pair-replacement"
CLAIM="pair-replacement-once-claim.json"
FILES=tuple("scripts/"+n for n in ("ttmre_pair_replacement.py","train_ttmre_pair_replacement.py",
                                   "check_ttmre_pair_replacement.py","run_ttmre_pair_replacement.sh"))
HARD_SECONDS,SOFT_SECONDS,TERM_SECONDS=21600,120,30
TRAIN_DOCS,DEV_DOCS,EPOCHS,SEED=3053,500,30,75
KINDS=("bypass","mlp")
MEMORY=soup.MEMORY
GIB=1024**3
METRICS=soup.METRICS
BENCHMARK_ROUNDS=3
WARMUP_BATCHES=3
CONTROL_PINS={"manifest.json":"6b88ab11e9b6464c3c74a71118f2c4ba8742d754ed1065ab5da8d270f7d23d8e",
              "training_order.json":"83b4564c58d6bb891c0ed05bbde93e9851f13d8bcaa881d5b480dbd958e821cf",
              "initialization.json":"f7cbafa5706033805f955bf75d1237b9de5f74a2bd32ddd400bd3e150aefd4b6",
              "completion.json":"2ddf5c10d65a13c4054d175c937ccbc3ad62cbd4e2e2e98e98f04ce337f8c8e2"}
PROGRESS={"phase":"not-started","formal_optimizer_updates":0}


def updates():return (TRAIN_DOCS//4)*EPOCHS


def protocol():
    return {"version":"pair-replacement-v1","seed":SEED,"kinds":list(KINDS),"epochs":EPOCHS,
            "updates_per_arm":updates(),"new_optimizer_updates":2*updates(),"parent_sha256":base.PARENT_SHA,
            "reference_checkpoint_sha256":soup.CHECKPOINTS[75],"loss":"ATGL alpha16.5, original implementation",
            "precision":"float32","batch_size":4,"backbone_and_new_adapter_lr":3e-5,"head_lr":1e-5,
            "warmup_steps":int(updates()*.06),"clip":1.,"eps":1e-6,"weight_decay":.01,
            "selection":"penultimate batch raw Dev F1; earliest tie","decoder":soup.protocol()["decoder"],
            "new_module":"pair residual MLP 1536-256-1536; zero up; private CPU seed75007501",
            "memory":"immutable provenance buffer only; unused in B/M forward and optimizer",
            "checkpoint_policy":"CPU best snapshot during each arm; one final best write, no overwritten old artifacts",
            "benchmark":{"rounds":BENCHMARK_ROUNDS,"warmup_batches":WARMUP_BATCHES,
                         "order":[["control","bypass","mlp"],["bypass","mlp","control"],["mlp","control","bypass"]],
                         "primary":"median full-Dev synchronized model-forward seconds; host input transfer excluded",
                         "secondary":"full pass including collate/transfer/decode, loading/preprocessing separate"},
            "quality_min_F1_gain_vs_C_and_B_pp":.30,"efficiency_F1_max_drop_pp":.10,"efficiency_IgnA_max_drop_pp":.10,
            "efficiency_min_latency_reduction":.20,"efficiency_max_peakmemory_ratio":1.,
            "prefer_B_if_B_efficient_and_M_gain_below_pp":.20,"test_data_read":False,"new_Distant_updates":0,
            "reference_optimizer_updates":0,"hard_seconds":HARD_SECONDS,"soft_reserve_seconds":SOFT_SECONDS,
            "term_reserve_seconds":TERM_SECONDS,"automatic_retry":False,"automatic_resume":False,
            "extension":False,"automatic_followup":False}


def check_budget(budget,reserve=SOFT_SECONDS):
    if budget is None:return
    if time.monotonic()>=budget["deadline_monotonic"]-reserve:raise TimeoutError("Pair screen six-hour reserve reached; no extension")
    if os.name=="posix" and Path("/proc/sys/kernel/random/boot_id").read_text().strip()!=budget["boot_id"]:
        raise TimeoutError("Boot changed; do not reset budget")


def verify_files(files,budget):
    for name,h in files.items():
        check_budget(budget)
        if digest(Path(name),budget)!=h:raise ValueError("Pinned input/source missing or changed: "+name)


def validate(budget=None):
    prior_root=Path(read(old.RUN_ROOT/old.CLAIM)["workspace"])
    if prior_root!=prior_root.resolve() or prior_root.name!="workspace" or prior_root.parent.parent!=old.RUN_ROOT.resolve():
        raise ValueError("Unexpected completed control workspace")
    pinned={str(prior_root/"identity.json"):"a45042c20c95e91cadd44c6980c676f1fd7e2937c2020487d5a9c32315eb9c35"}
    verify_files(pinned,budget)
    prior=read(prior_root/"identity.json")
    parent=prior["parent"]
    original_root=Path(prior["ATGL_soup"]["predictions"]).parent
    for name in ("identity.json","comparison/completion.json"):
        pinned[str(original_root/name)]=old.diagnostic.OLD_PINS[name]
    verify_files(pinned,budget)
    origin=read(original_root/"identity.json")
    source=next(s for s in origin["sources"] if s["seed"]==75)
    reference=Path(source["checkpoint"]);art=reference.parent
    for name,h in CONTROL_PINS.items():pinned[str(art/name)]=h
    pinned[str(reference)]=soup.CHECKPOINTS[75]
    distant=Path(parent["distant_checkpoint"]);pinned[str(distant)]=base.PARENT_SHA
    reference_predictions=Path(origin["reference_predictions"]["75"])
    pinned[str(reference_predictions)]=origin["pinned_files"][str(reference_predictions)]
    cache=Path(parent["baseline_workspace"])/"cache"
    wanted=[cache/split/name for split in ("human","dev") for name in ("index.json","features.bin")]
    wanted+=[cache/"distant/index.json"]  # Precomputed Dev matching metadata, no Distant features/labels.
    wanted+=[Path(parent["data_dir"])/name for name in ("train_revised.json","dev_revised.json")]
    model_dir=Path(parent["model_dir"])
    wanted+=[Path(p) for p in parent["input_files"] if Path(p).is_relative_to(model_dir)]
    for p in wanted:
        if not p.resolve().is_relative_to(ROOT.resolve()):raise ValueError("Asset outside project")
        pinned[str(p)]=prior["pinned_files"][str(p)]
    verify_files(pinned,budget)
    stage,manifest,initial,order=(read(art/n) for n in ("completion.json","manifest.json","initialization.json","training_order.json"))
    if (not stage["passed"] or stage["optimizer_steps"]!=updates() or stage["parent_sha256"]!=base.PARENT_SHA
            or stage["protocol"]["seed"]!=75 or stage["protocol"]["alpha"]!=16.5 or not stage["memory_unchanged"]
            or stage["checkpoint_sha256"]!=soup.CHECKPOINTS[75] or order["sample_order_sha256"]!=base.CONTROL_ORDER):
        raise ValueError("Reference C is not the original accepted ATGL75")
    sources=dict(prior["source_files"])
    verify_files({str(ROOT/n):h for n,h in sources.items()},budget)
    sources.update({n:digest(ROOT/n,budget) for n in FILES})
    return {"protocol":protocol(),"parent":parent,"pinned_files":pinned,"source_files":sources,
            "distant_checkpoint":str(distant),"reference_checkpoint":str(reference),"reference_predictions":str(reference_predictions),
            "reference_metrics":{k:stage["best"][k] for k in METRICS},"reference_optimizer_groups":manifest["optimizer_groups"],
            "expected_initial_state_sha256":initial["state_sha256"],"expected_initial_rng_sha256":manifest["initial_rng_sha256"],
            "expected_sample_order_sha256":order["sample_order_sha256"],"reference_final_rng_sha256":stage["final_rng_sha256"],
            "stronger_existing_ATGLsoup_Dev_metrics":read(original_root/"comparison/completion.json")["metrics"],
            "test_data_read":False,"historical_reference_is_conditional_on_fixed_Distant_parent":True}


def resources(path,identity,phase="start"):
    weight=Path(identity["distant_checkpoint"]).stat().st_size
    pending=2 if phase in ("start","bypass") else 1 if phase=="mlp" else 0
    reserve=512*1024**2
    required=max(4*GIB if phase=="start" else reserve,pending*weight+reserve)
    free=shutil.disk_usage(path).free
    if free<required:raise RuntimeError("Project disk free %.3fGiB < required %.3fGiB (%s)"%(free/GIB,required/GIB,phase))
    return {"phase":phase,"free_bytes":free,"required_bytes":required,"recommended_available_RAM_GiB":32,
            "CPU_best_snapshot_approx_bytes":weight,"best_checkpoint_writes_per_arm":1,"old_files_deleted":False}


def common_digest(model):
    return base.frozen.tensor_digest({n:v for n,v in model.state_dict().items() if not n.startswith("mu_encoder.")})


def data_context(workspace,identity,budget):
    return old.data_context(workspace,identity,budget)


def construct(args,identity,kind,checkpoint=None):
    import gc
    import torch
    import ttmre_pair_replacement as method
    model=core.load_generic_model_seed75(args,identity["parent"]["baseline_identity"])
    if kind=="control":
        model.load_state_dict(torch.load(identity["reference_checkpoint"],map_location="cpu",weights_only=True),strict=True)
        return model,{"kind":"control","parameters":sum(p.numel() for p in model.parameters())}
    model.load_state_dict(torch.load(identity["distant_checkpoint"],map_location="cpu",weights_only=True),strict=True)
    if core.preflight.state_digest(model)!=identity["expected_initial_state_sha256"]:
        raise ValueError("Distant initialization differs from original C")
    before=common_digest(model);rng=core.preflight.rng_state()
    model.requires_grad_(True)
    report=method.install(model,kind)
    gc.collect()  # Original checkpoint wrappers can retain self-cycles on the removed TTM.
    core.preflight.assert_rng_equal(rng,core.preflight.rng_state())
    if before!=common_digest(model):raise RuntimeError("Common initialization changed during replacement")
    report["common_initial_state_sha256"]=before
    report["private_initialization_preserved_all_RNG"]=True
    if checkpoint is not None:model.load_state_dict(torch.load(checkpoint,map_location="cpu",weights_only=True),strict=True)
    return model,report


def optimizer_for(model,identity):
    optimizer=core.baseline.optimizer_for_stage(model,"human-frozen")
    groups=core.baseline.optimizer_report(model,optimizer)
    expected=[{**g,"parameter_names":[n for n in g["parameter_names"] if not n.startswith("mu_encoder.")]} for g in identity["reference_optimizer_groups"]]
    for actual,ref in zip(groups,expected):
        common=[n for n in actual["parameter_names"] if not n.startswith("mu_encoder.")]
        if common!=ref["parameter_names"] or actual["lr"]!=ref["lr"] or actual["weight_decay"]!=ref["weight_decay"]:
            raise ValueError("Common optimizer groups changed")
    if optimizer.state or any("token_turing_machine" in n for g in groups for n in g["parameter_names"]):
        raise ValueError("Optimizer is not fresh or contains dormant TTM parameters")
    return optimizer,groups


def probe(model,human,args,budget,kind):
    # Existing real-batch ATGL probe includes reserved Adam moments, RNG/state
    # restoration and backbone/head connectivity; provenance memory is a buffer.
    check_budget(budget)
    report=base.probe(model,human,args,budget)
    report.update({"kind":kind,"memory_buffer_not_optimizer_parameter":MEMORY not in dict(model.named_parameters())})
    return report


@contextmanager
def training_scope(kind,budget):
    previous=(core.training_forward,core.emit,core.aligned.emit,core.atomic_torch_save)
    snapshot={}
    def cache_best(value,path):
        if Path(path).name!="best_model.pth":raise ValueError("Unexpected checkpoint target")
        snapshot.clear();snapshot.update({"path":Path(path),"state":value})
    def event(name,**fields):
        if name=="N1_FULL_SEED75_LEARNING_UPDATE_VERIFIED":name="PAIR_REPLACEMENT_LEARNING_UPDATE_VERIFIED"
        if name in ("PAIR_REPLACEMENT_LEARNING_UPDATE_VERIFIED","epoch_complete"):
            PROGRESS["formal_optimizer_updates"]=KINDS.index(kind)*updates()+fields["step"]
        emit(name,variant=kind,**fields)
        if name=="epoch_complete" and fields["epoch"]==1:
            estimate=fields["seconds"]*((len(KINDS)-KINDS.index(kind))*EPOCHS-1)+1200
            available=budget["deadline_monotonic"]-time.monotonic()-SOFT_SECONDS
            emit("PAIR_REPLACEMENT_PROJECTION",variant=kind,estimate_seconds=estimate,available_seconds=available)
            if estimate>available:raise TimeoutError("Measured projection exceeds six-hour budget")
    core.training_forward=base.training_forward
    core.emit=core.aligned.emit=event
    core.atomic_torch_save=cache_best
    try:yield snapshot
    finally:core.training_forward,core.emit,core.aligned.emit,core.atomic_torch_save=previous


def run_arm(workspace,identity,budget,kind):
    import numpy as np
    import torch
    from torch.utils.data import DataLoader
    from transformers.optimization import get_linear_schedule_with_warmup
    directory=workspace/kind;directory.mkdir()
    write_json(directory/"resource_check.json",resources(workspace,identity,kind))
    args,human,dev,gold,annotated,distant=data_context(workspace,identity,budget)
    model,installation=construct(args,identity,kind)
    memory=model.mu_encoder.memory_tokens.detach().cpu().clone()
    adapter_before={n:p.detach().cpu().clone() for n,p in model.named_parameters() if n.startswith("mu_encoder.")}
    write_json(directory/"installation.json",installation)
    hook=model.register_forward_pre_hook(lambda *_:check_budget(budget))
    snapshot=None
    try:
        write_json(directory/"full_model_check.json",probe(model,human,args,budget,kind))
        optimizer,groups=optimizer_for(model,identity)
        core.shared.seed_all(SEED,torch,np)
        loader=DataLoader(human,batch_size=4,shuffle=True,drop_last=True,collate_fn=core.shared.collate_fn)
        dev_loader=DataLoader(dev,batch_size=4,shuffle=False,collate_fn=core.shared.collate_fn)
        scheduler=get_linear_schedule_with_warmup(optimizer,int(updates()*.06),updates())
        initial_rng=core.rng_digest()
        if len(loader)*EPOCHS!=updates() or initial_rng!=identity["expected_initial_rng_sha256"]:
            raise ValueError("Original update count/training RNG differs")
        write_json(directory/"manifest.json",{"protocol":protocol(),"variant":kind,"optimizer_groups":groups,
                   "initial_rng_sha256":initial_rng,"identity_sha256":digest(workspace/"identity.json"),"total_steps":updates()})
        with training_scope(kind,budget) as snapshot:
            result=core.train_loop(model,loader,dev_loader,args,kind,"human-frozen",directory,gold,annotated,distant,
                core.method.frequency_buckets(human.metadata["priors"]),budget,optimizer,scheduler)
        del optimizer,scheduler
        write_json(directory/"training_order.json",{"seed":SEED,"formal_optimizer_updates":result["optimizer_steps"],
                   "sample_order_sha256":result["sample_order_sha256"],"final_rng_sha256":result["final_rng_sha256"],
                   "same_as_reference_order":result["sample_order_sha256"]==identity["expected_sample_order_sha256"],
                   "same_as_reference_final_rng":result["final_rng_sha256"]==identity["reference_final_rng_sha256"]})
        if (result["optimizer_steps"]!=updates() or not result["learning_update_check_passed"]
                or result["sample_order_sha256"]!=identity["expected_sample_order_sha256"]):
            raise ValueError("Effective learning/count/original sample order failed")
        old.check_selection(directory)
        changed={n:not torch.equal(before,dict(model.named_parameters())[n].detach().cpu()) for n,before in adapter_before.items()}
        if kind=="mlp" and (not changed or not all(changed.values())):
            raise RuntimeError("New MLP parameters did not actually learn")
        write_json(directory/"adapter_learning.json",{"passed":True,"changed_parameters":changed,
                   "zero_down_gradient_at_zero_up_initialization_is_expected":True,"extra_optimizer_updates":0})
        check_budget(budget)
        if not snapshot or snapshot["path"]!=directory/"best_model.pth" or snapshot["path"].exists():
            raise ValueError("Missing or conflicting in-memory best snapshot")
        core.atomic_torch_save(snapshot["state"],directory/"best_model.pth")
        snapshot=None
        model.zero_grad(set_to_none=True)
        model.load_state_dict(torch.load(directory/"best_model.pth",map_location="cpu",weights_only=True),strict=True)
        best=read(directory/"best_metrics.json")
        replay,predictions=old.grouped_replay(model,dev,args,budget,best,memory)
        score=core.shared.score_predictions(predictions,gold,annotated,distant)
        replay["metrics"]=score
        replay["metric_replay_passed"]=all(math.isclose(score[k],best[k],abs_tol=1e-6,rel_tol=0) for k in METRICS)
        replay["passed"]=replay["passed"] and replay["metric_replay_passed"]
        write_json(directory/"checkpoint_replay.json",replay)
        if not replay["passed"]:raise RuntimeError("Best replacement checkpoint replay failed")
        old.save_prediction_report(directory,predictions,[{"title":f["title"],"hts":f["hts"]} for f in dev],score,gold,human)
        result.update({"passed":True,"variant":kind,"protocol":protocol(),"best":best,"checkpoint_sha256":digest(directory/"best_model.pth",budget),
                       "installation":installation,"checkpoint_replay_passed":True,"memory_buffer_unchanged":True,
                       "adapter_learning_passed":True,"best_checkpoint_writes":1})
        write_json(directory/"completion.json",result)
        emit("PAIR_REPLACEMENT_ARM_COMPLETE",variant=kind,F1=score["dev_F1"],optimizer_updates=updates())
    except BaseException:
        # A non-resumable failure may still preserve the selected snapshot once,
        # if the original deadline and actual disk leave enough room.
        if snapshot and not (directory/"best_model.pth").exists() and time.monotonic()<budget["deadline_monotonic"]-60:
            needed=sum(v.numel()*v.element_size() for v in snapshot["state"].values())+64*1024**2
            if shutil.disk_usage(directory).free>=needed:
                try:
                    core.atomic_torch_save(snapshot["state"],directory/"best_model.pth")
                    write_json(directory/"incomplete_best_preserved.json",{"complete":False,"resumable":False,
                               "checkpoint_sha256":digest(directory/"best_model.pth"),"reason":"retain selected snapshot after failure"})
                except BaseException:pass
        raise
    finally:
        PROGRESS["formal_optimizer_updates"]=KINDS.index(kind)*updates()+int(getattr(model,"_n1_completed_optimizer_steps",0))
        hook.remove()


def verify_arm(workspace,kind,identity,budget):
    directory=workspace/kind
    complete=read(directory/"completion.json");learning=read(directory/"learning_update_check.json")
    order=read(directory/"training_order.json");replay=read(directory/"checkpoint_replay.json")
    if (not complete["passed"] or complete["protocol"]!=protocol() or complete["variant"]!=kind
            or complete["optimizer_steps"]!=updates() or not complete["memory_buffer_unchanged"]
            or not complete["adapter_learning_passed"] or complete["best_checkpoint_writes"]!=1
            or not learning["passed"] or not learning["parameter_changed"] or learning["lr"]<=0
            or learning["optimizer_state_step"]!=learning["step"] or learning["extra_optimizer_updates"]!=0
            or order["sample_order_sha256"]!=identity["expected_sample_order_sha256"]
            or not replay["passed"] or not replay["metric_replay_passed"]
            or digest(directory/"best_model.pth",budget)!=complete["checkpoint_sha256"]):
        raise ValueError("Incomplete original-loss replacement arm: "+kind)
    old.check_selection(directory)
    return complete


def synchronize(device):
    if device.type=="cuda":__import__("torch").cuda.synchronize(device)


def benchmark_pass(model,dev,args,budget):
    import numpy as np
    import torch
    from evaluation import to_official
    model.eval();predictions=[]
    with torch.no_grad():
        for offset in range(0,min(len(dev),4*WARMUP_BATCHES),4):
            check_budget(budget)
            model(**core.shared.inputs_from_batch(core.shared.collate_fn(dev[offset:offset+4]),args.device,labels=False))
        synchronize(args.device)
        if args.device.type=="cuda":
            torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats(args.device)
        elapsed=0.;end_to_end_start=time.perf_counter()
        for offset in range(0,len(dev),4):
            check_budget(budget)
            features=dev[offset:offset+4]
            inputs=core.shared.inputs_from_batch(core.shared.collate_fn(features),args.device,labels=False)
            synchronize(args.device)
            started=time.perf_counter()
            logits=model(**inputs)
            synchronize(args.device)
            elapsed+=time.perf_counter()-started
            if not torch.isfinite(logits).all():raise RuntimeError("Nonfinite benchmark logits")
            values=logits.detach().cpu().numpy();flags=np.zeros_like(values)
            flags[:,1:]=values[:,1:]>values[:,:1];flags[:,0]=flags[:,1:].sum(1)==0
            predictions.extend(to_official(flags,features))
            del logits,inputs
        synchronize(args.device)
        e2e=time.perf_counter()-end_to_end_start
        allocated=torch.cuda.max_memory_allocated(args.device) if args.device.type=="cuda" else 0
        reserved=torch.cuda.max_memory_reserved(args.device) if args.device.type=="cuda" else 0
    return {"model_forward_seconds":elapsed,"end_to_end_seconds":e2e,"documents":len(dev),
            "milliseconds_per_document":1000*elapsed/len(dev),"documents_per_second":len(dev)/elapsed,
            "peak_allocated_bytes":allocated,"peak_reserved_bytes":reserved,
            "parameter_elements":sum(p.numel() for p in model.parameters()),"device_type":args.device.type},predictions


def investment(metrics,benchmark):
    c,b,m=(metrics[k] for k in ("control","bypass","mlp"))
    quality=all(m["dev_F1"]-r["dev_F1"]>=.30 and m["dev_IgnF1_annotated"]>=r["dev_IgnF1_annotated"] for r in (c,b))
    efficiency={}
    for k in KINDS:
        ratio=benchmark[k]["median_model_forward_seconds"]/benchmark["control"]["median_model_forward_seconds"]
        efficiency[k]=bool(metrics[k]["dev_F1"]>=c["dev_F1"]-.10 and metrics[k]["dev_IgnF1_annotated"]>=c["dev_IgnF1_annotated"]-.10
            and ratio<=.80 and benchmark[k]["max_peak_allocated_bytes"]<=benchmark["control"]["max_peak_allocated_bytes"])
    prefer_b=efficiency["bypass"] and m["dev_F1"]-b["dev_F1"]<.20
    recommended="bypass" if prefer_b else "mlp" if quality or efficiency["mlp"] else "bypass" if efficiency["bypass"] else "stop"
    return {"quality_gate_M_vs_C_and_B":bool(quality),"efficiency_gate":efficiency,"recommended_next_candidate":recommended,
            "screen_passed":bool(quality or any(efficiency.values())),"interpretation":"single-seed Dev screen; no noninferiority or publication claim; no automatic followup"}


def finish(workspace,identity,budget):
    import gc
    import torch
    complete={kind:verify_arm(workspace,kind,identity,budget) for kind in KINDS}
    args,human,dev,gold,annotated,distant=data_context(workspace,identity,budget)
    write_json(workspace/"resource_finish.json",resources(workspace,identity,"finish"))
    expected={"control":soup.facts(read(Path(identity["reference_predictions"]))),
              **{k:soup.facts(read(workspace/k/"predictions.json")) for k in KINDS}}
    raw=[]
    orders=protocol()["benchmark"]["order"]
    for round_index,order in enumerate(orders):
        for kind in order:
            check_budget(budget)
            load_start=time.perf_counter()
            checkpoint=None if kind=="control" else workspace/kind/"best_model.pth"
            model,installed=construct(args,identity,kind,checkpoint)
            model.requires_grad_(False);model.eval()
            synchronize(args.device)
            load_seconds=time.perf_counter()-load_start
            before=core.preflight.state_digest(model)
            timing,predictions=benchmark_pass(model,dev,args,budget)
            if before!=core.preflight.state_digest(model) or soup.facts(predictions)!=expected[kind]:
                raise RuntimeError("Benchmark changed frozen state or selected predictions: "+kind)
            row={"round":round_index+1,"variant":kind,"load_seconds_excluded_from_forward":load_seconds,
                 "prediction_replay_passed":True,"parameters_unchanged":True,**timing}
            raw.append(row)
            write_json(workspace/("benchmark-round%d-%s.json"%(round_index+1,kind)),row)
            emit("PAIR_REPLACEMENT_BENCHMARK",**row)
            del model,predictions
            gc.collect()
            if args.device.type=="cuda":torch.cuda.empty_cache()
    summaries={}
    for kind in ("control",*KINDS):
        rows=[r for r in raw if r["variant"]==kind]
        summaries[kind]={"median_model_forward_seconds":statistics.median(r["model_forward_seconds"] for r in rows),
                         "median_end_to_end_seconds":statistics.median(r["end_to_end_seconds"] for r in rows),
                         "max_peak_allocated_bytes":max(r["peak_allocated_bytes"] for r in rows),
                         "max_peak_reserved_bytes":max(r["peak_reserved_bytes"] for r in rows),
                         "parameters":rows[0]["parameter_elements"],"rounds":len(rows)}
    metrics={"control":identity["reference_metrics"],**{k:{m:complete[k]["best"][m] for m in METRICS} for k in KINDS}}
    for kind in KINDS:
        score=core.shared.score_predictions(read(workspace/kind/"predictions.json"),gold,annotated,distant)
        if any(not math.isclose(score[k],metrics[kind][k],abs_tol=1e-6,rel_tol=0) for k in METRICS):raise ValueError("Final score mismatch")
    c_score=core.shared.score_predictions(read(Path(identity["reference_predictions"])),gold,annotated,distant)
    if any(not math.isclose(c_score[k],metrics["control"][k],abs_tol=1e-6,rel_tol=0) for k in METRICS):raise ValueError("Old C prediction metrics changed")
    decision=investment(metrics,summaries)
    directory=workspace/"comparison";directory.mkdir()
    write_json(directory/"benchmark.json",{"protocol":protocol()["benchmark"],"raw":raw,"summary":summaries,
               "runtime":identity["parent"]["environment"],"preprocessing_and_model_loading_excluded_from_primary":True})
    verify_files(identity["pinned_files"],budget)
    verify_files({str(ROOT/n):h for n,h in identity["source_files"].items()},budget)
    for kind in KINDS:verify_arm(workspace,kind,identity,budget)
    result={"passed":True,"marker":"PAIR_REPLACEMENT_COMPLETE","protocol":protocol(),"metrics":metrics,
            "benchmark":summaries,**decision,"deltas_vs_C_pp":{k:{m:metrics[k][m]-metrics["control"][m] for m in METRICS} for k in KINDS},
            "existing_stronger_ATGLsoup_Dev_reference":identity["stronger_existing_ATGLsoup_Dev_metrics"],
            "formal_optimizer_updates_new":2*updates(),"reference_optimizer_updates_new":0,"new_Distant_updates":0,
            "test_data_read":False,"test_evaluated":False,"automatic_followup":False,"novelty_established":False,
            "elapsed_seconds":time.monotonic()-budget["start_monotonic"]}
    write_json(directory/"completion.json",result)
    soup.archive_reports(workspace,budget)
    emit("PAIR_REPLACEMENT_COMPLETE",screen_passed=decision["screen_passed"],recommended=decision["recommended_next_candidate"],
         reports=str(directory/"reports.tar.gz"),automatic_followup=False)


def require_lock():
    if os.name!="posix" or os.environ.get("TTMRE_PAIR_REPLACEMENT_LOCK_FD")!="9":raise RuntimeError("Use the one-use pair replacement launcher")
    import fcntl
    a,b=os.fstat(9),(ROOT/"experiments/2026-10-08-baseline-alignment/.training.lock").stat()
    if (a.st_dev,a.st_ino)!=(b.st_dev,b.st_ino):raise RuntimeError("Wrong shared training lock")
    fcntl.flock(9,fcntl.LOCK_EX|fcntl.LOCK_NB)
    return 9


def verify_context(workspace,budget):
    claim=read(RUN_ROOT/CLAIM)
    if (workspace!=workspace.resolve() or workspace.name!="workspace" or workspace.parent.parent!=RUN_ROOT.resolve()
            or budget["hard_seconds"]!=HARD_SECONDS or budget["deadline_monotonic"]-budget["start_monotonic"]!=HARD_SECONDS
            or any(budget[k] is not False for k in ("automatic_retry","automatic_resume","extension"))
            or claim["workspace"]!=str(workspace) or claim["protocol"]!=protocol()
            or claim["budget_sha256"]!=digest(workspace/"budget.json") or claim["identity_sha256"]!=digest(workspace/"identity.json")):
        raise ValueError("Invalid immutable six-hour pair claim/context")
    check_budget(budget)


def worker(workspace,phase):
    require_lock();budget=read(workspace/"budget.json");verify_context(workspace,budget)
    identity=read(workspace/"identity.json")
    PROGRESS.update({"phase":phase,"formal_optimizer_updates":{"bypass":0,"mlp":updates(),"finish":2*updates()}[phase]})
    def stop(*_):raise TimeoutError("Pair worker interrupted; preserve incomplete reports")
    signal.signal(signal.SIGTERM,stop)
    try:
        if identity["protocol"]!=protocol():raise ValueError("Approved pair protocol changed")
        verify_files(identity["pinned_files"],budget);verify_files({str(ROOT/n):h for n,h in identity["source_files"].items()},budget)
        if phase=="finish":finish(workspace,identity,budget)
        else:run_arm(workspace,identity,budget,phase)
    except BaseException as error:
        write_json(workspace/("failure-"+phase+".json"),{"error":str(error),"type":type(error).__name__,"complete":False,
                   "progress":PROGRESS,"automatic_retry":False,"resumable":False})
        try:
            if not (workspace/"comparison/reports.tar.gz").exists():soup.archive_reports(workspace,budget)
        except BaseException:pass
        raise


def start(workspace):
    lock=require_lock()
    if workspace.name!="workspace" or workspace.parent.parent!=RUN_ROOT.resolve() or workspace.exists():raise ValueError("Use a new pair-screen workspace")
    claim_path=RUN_ROOT/CLAIM
    if claim_path.exists():raise FileExistsError("Pair screen already claimed; no retry")
    now=time.monotonic();budget={"start_monotonic":now,"deadline_monotonic":now+HARD_SECONDS,"hard_seconds":HARD_SECONDS,
        "soft_reserve_seconds":SOFT_SECONDS,"term_reserve_seconds":TERM_SECONDS,"start_utc":datetime.now(timezone.utc).isoformat(),
        "boot_id":Path("/proc/sys/kernel/random/boot_id").read_text().strip(),"automatic_retry":False,"automatic_resume":False,"extension":False}
    def stop(*_):raise TimeoutError("Pair supervisor interrupted")
    signal.signal(signal.SIGTERM,stop)
    try:
        emit("PAIR_REPLACEMENT_BUDGET_STARTED",**budget)
        identity=validate(budget)
        emit("PAIR_REPLACEMENT_SOURCES_VERIFIED",resources=resources(workspace.parent,identity))
        old.supervise([sys.executable,"-B",str(ROOT/FILES[2])],budget,lock)
        check_budget(budget);workspace.mkdir()
        write_json(workspace/"budget.json",budget);write_json(workspace/"identity.json",identity)
        with claim_path.open("x",encoding="utf-8") as handle:
            json.dump({"workspace":str(workspace),"protocol":protocol(),"budget_sha256":digest(workspace/"budget.json"),
                       "identity_sha256":digest(workspace/"identity.json"),"authorization":"user confirmed B/M Human screen once, cap6h"},handle,indent=2)
            handle.flush();os.fsync(handle.fileno())
        command=[sys.executable,"-u","-B",str(Path(__file__).resolve()),"--mode","worker","--run-dir",str(workspace)]
        for phase in (*KINDS,"finish"):old.supervise(command+["--phase",phase],budget,lock)
    except BaseException as error:
        write_json(workspace.parent/"failure-supervisor.json",{"error":str(error),"type":type(error).__name__,"complete":False,
                   "elapsed_seconds":time.monotonic()-now,"automatic_retry":False})
        raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode",choices=("validate","start","worker"),default="validate")
    parser.add_argument("--run-dir",type=Path);parser.add_argument("--phase",choices=(*KINDS,"finish"))
    args=parser.parse_args()
    if args.mode=="validate":print(json.dumps(validate(),indent=2))
    elif args.run_dir is None or (args.mode=="worker" and args.phase is None):raise ValueError("Use the single Bash launcher")
    elif args.mode=="start":start(args.run_dir.resolve())
    else:worker(args.run_dir.resolve(),args.phase)


if __name__=="__main__":main()
