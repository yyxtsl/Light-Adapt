"""Authorized bypass-only Human seeds76/77, total3h; original fixed ATGL.

Default validate is read-only. New jobs require the single Bash entry point.
No MLP/C training, soup, Distant, Test, tuning, retry/resume or auto follow-up.
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

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"scripts"))
import train_ttmre_pair_replacement as prior

base,core,soup,old=prior.base,prior.core,prior.soup,prior.old
read,write_json,digest,emit=prior.read,prior.write_json,prior.digest,prior.emit
RUN_ROOT=ROOT/"experiments/2026-10-22-bypass-repeat"
CLAIM="bypass-repeat-once-claim.json"
FILES=tuple("scripts/"+n for n in ("train_ttmre_bypass_repeat.py","check_ttmre_bypass_repeat.py","run_ttmre_bypass_repeat.sh"))
SEEDS=(76,77)
TRAIN_DOCS,DEV_DOCS,EPOCHS=3053,500,30
HARD_SECONDS,SOFT_SECONDS,TERM_SECONDS=10800,120,30
GIB=1024**3
METRICS=soup.METRICS
PAIR_PINS={"identity.json":"b08ca33dcb53325f5865718403b267426ffcd6fe881f82c611e7bdd2913ffd96",
           "comparison/completion.json":"52ea8e62de575ce0622a94ab7123069b2cadb49deec5c478538a40fd43e0c51b",
           "bypass/completion.json":"c322d5a2a6248d1c491826aac8b3c8a7f46b6900fe8699b59d45a38c0400520f",
           "comparison/benchmark.json":"ba60b8a467ce86d43001ef0502e9c4d7a7b1240088c7e839a0d08b77630c40fb"}
PROGRESS={"phase":"not-started","seed":None,"formal_optimizer_updates":0}


def updates():return (TRAIN_DOCS//4)*EPOCHS


def protocol():
    return {"version":"bypass-repeat-v1","new_Human_seeds":list(SEEDS),"reused_screen_seed":75,
            "epochs":EPOCHS,"updates_per_arm":updates(),"new_optimizer_updates":2*updates(),
            "parent_sha256":base.PARENT_SHA,"variant":"bypass only; same accepted pair-replacement implementation",
            "loss":"original ATGL alpha16.5","precision":"float32","batch_size":4,
            "decoder":soup.protocol()["decoder"],"warmup_steps":int(updates()*.06),"clip":1.,
            "new_MLP_training":False,"reference_optimizer_updates":0,"new_soup":False,"test_data_read":False,
            "new_seed_gate":{"each_F1_delta_min_pp":-.10,"each_IgnA_delta_min_pp":-.10,
                             "each_latency_reduction_min":.20,"each_peakallocated_ratio_max":1.},
            "benchmark_rounds":3,"benchmark_pair_order":[["control","bypass"],["bypass","control"],["control","bypass"]],
            "old_seed75_is_screening_not_independent_confirmation":True,
            "hard_seconds":HARD_SECONDS,"soft_reserve_seconds":SOFT_SECONDS,"term_reserve_seconds":TERM_SECONDS,
            "automatic_retry":False,"automatic_resume":False,"extension":False,"automatic_followup":False}


def arm_protocol(seed):return {**protocol(),"active_Human_seed":seed}


def check_budget(budget,reserve=SOFT_SECONDS):
    if budget is None:return
    if time.monotonic()>=budget["deadline_monotonic"]-reserve:raise TimeoutError("Three-hour bypass budget reserve reached; no extension")
    if os.name=="posix" and Path("/proc/sys/kernel/random/boot_id").read_text().strip()!=budget["boot_id"]:
        raise TimeoutError("Boot changed; budget cannot reset")


def verify_files(files,budget):
    for p,h in files.items():
        check_budget(budget)
        if digest(Path(p),budget)!=h:raise ValueError("Pinned file missing/changed: "+p)


def validate(budget=None):
    previous=Path(read(prior.RUN_ROOT/prior.CLAIM)["workspace"])
    if previous!=previous.resolve() or previous.name!="workspace" or previous.parent.parent!=prior.RUN_ROOT.resolve():
        raise ValueError("Unexpected completed pair-screen workspace")
    pinned={str(previous/n):h for n,h in PAIR_PINS.items()};verify_files(pinned,budget)
    origin,done=read(previous/"identity.json"),read(previous/"comparison/completion.json")
    if not done["passed"] or done["marker"]!="PAIR_REPLACEMENT_COMPLETE" or done["recommended_next_candidate"]!="bypass":
        raise ValueError("Previous screen is not the accepted B recommendation")
    # Keep source hashes and necessary data, but do not require stopped M weights.
    for p,h in origin["pinned_files"].items():pinned[p]=h
    soup_root=Path(next(p for p,h in origin["pinned_files"].items() if h==old.diagnostic.OLD_PINS["identity.json"])).parent
    soup_origin=read(soup_root/"identity.json")
    references={};pairing={}
    for seed in SEEDS:
        source=next(s for s in soup_origin["sources"] if s["seed"]==seed)
        checkpoint=Path(source["checkpoint"]);art=checkpoint.parent
        for name in ("completion.json","initialization.json"):
            pinned[str(art/name)]=soup_origin["pinned_files"][str(art/name)]
        for name,h in old.ATGL_PINS[seed].items():pinned[str(art/name)]=h
        pred=Path(soup_origin["reference_predictions"][str(seed)])
        pinned[str(pred)]=soup_origin["pinned_files"][str(pred)]
        pinned[str(checkpoint)]=soup.CHECKPOINTS[seed]
        verify_files({str(art/n):pinned[str(art/n)] for n in ("completion.json","initialization.json",*old.ATGL_PINS[seed])},budget)
        stage,manifest,init,order=(read(art/n) for n in ("completion.json","manifest.json","initialization.json","training_order.json"))
        if (not stage["passed"] or stage["optimizer_steps"]!=updates() or stage["protocol"]["seed"]!=seed
                or stage["parent_sha256"]!=base.PARENT_SHA or not stage["memory_unchanged"]
                or order["sample_order_sha256"]!=stage["sample_order_sha256"]):raise ValueError("Mismatched same-seed C reference")
        references[str(seed)]={"checkpoint":str(checkpoint),"checkpoint_sha256":soup.CHECKPOINTS[seed],
                               "predictions":str(pred),"metrics":{k:stage["best"][k] for k in METRICS}}
        pairing[str(seed)]={"initial_state_sha256":init["state_sha256"],"initial_rng_sha256":manifest["initial_rng_sha256"],
                           "sample_order_sha256":stage["sample_order_sha256"],"final_rng_sha256":stage["final_rng_sha256"],
                           "optimizer_groups":manifest["optimizer_groups"]}
    old_B=read(previous/"bypass/completion.json")
    B75_prediction=previous/"bypass/predictions.json"
    # The SHA is read from the pinned existing artifact's source-independent export metadata.
    fd=previous/"bypass/frequency_diagnostics.json"
    # Pin this export using the audited user report's exact hash, not self-certification.
    pinned[str(fd)]=OLD_B_EXPORT_PINS["frequency_diagnostics.json"]
    pinned[str(B75_prediction)]=OLD_B_EXPORT_PINS["predictions.json"]
    verify_files(pinned,budget)
    sources=dict(origin["source_files"])
    verify_files({str(ROOT/n):h for n,h in sources.items()},budget)
    sources.update({n:digest(ROOT/n,budget) for n in FILES})
    return {"protocol":protocol(),"parent":origin["parent"],"pinned_files":pinned,"source_files":sources,
            "distant_checkpoint":origin["distant_checkpoint"],"references":references,"pairing":pairing,
            "old75":{"B":old_B["best"],"C":origin["reference_metrics"],"B_predictions":str(B75_prediction),
                     "benchmark":done["benchmark"],"is_screened":True},
            "existing_stronger_ATGLsoup_Dev_metrics":origin["stronger_existing_ATGLsoup_Dev_metrics"],"test_data_read":False}


# Audited reports9 values, filled from exact raw bytes before local acceptance.
OLD_B_EXPORT_PINS={"frequency_diagnostics.json":"e91cce0f8d8d6f21e43093357c8de661d03f7d21a0a28ec242c324bc14d98878",
                   "predictions.json":"778288ca19911726f1128c3c96b4ec13a70533cbdd7cfc65e3a17c442d20862d"}


def resources(path,identity,phase="start"):
    source_size=Path(identity["distant_checkpoint"]).stat().st_size
    pending=2 if phase in ("start","seed76") else 1 if phase=="seed77" else 0
    reserve=512*1024**2
    required=max(4*GIB if phase=="start" else reserve,pending*source_size+reserve)
    free=shutil.disk_usage(path).free
    if free<required:raise RuntimeError("Project disk free %.3fGiB < required %.3fGiB; obtain exact cleanup approval or expand storage"%(free/GIB,required/GIB))
    return {"phase":phase,"free_bytes":free,"required_bytes":required,"recommended_available_RAM_GiB":32,
            "checkpoint_policy":"CPU best; final single write; no old files removed"}


def arm_identity(identity,seed):
    pair,ref=identity["pairing"][str(seed)],identity["references"][str(seed)]
    return {"protocol":arm_protocol(seed),"parent":identity["parent"],"distant_checkpoint":identity["distant_checkpoint"],
            "reference_checkpoint":ref["checkpoint"],"reference_predictions":ref["predictions"],"reference_metrics":ref["metrics"],
            "reference_optimizer_groups":pair["optimizer_groups"],"expected_initial_state_sha256":pair["initial_state_sha256"],
            "expected_initial_rng_sha256":pair["initial_rng_sha256"],"expected_sample_order_sha256":pair["sample_order_sha256"],
            "reference_final_rng_sha256":pair["final_rng_sha256"],"pinned_files":identity["pinned_files"],"source_files":identity["source_files"],
            "stronger_existing_ATGLsoup_Dev_metrics":identity["existing_stronger_ATGLsoup_Dev_metrics"]}


@contextmanager
def training_scope(seed,budget):
    saved=(core.training_forward,core.emit,core.aligned.emit,core.atomic_torch_save)
    snapshot={}
    def best(value,path):
        if Path(path).name!="best_model.pth":raise ValueError("Unexpected checkpoint target")
        snapshot.clear();snapshot.update({"path":Path(path),"state":value})
    def event(name,**fields):
        if name=="N1_FULL_SEED75_LEARNING_UPDATE_VERIFIED":name="BYPASS_REPEAT_LEARNING_UPDATE_VERIFIED"
        if name in ("BYPASS_REPEAT_LEARNING_UPDATE_VERIFIED","epoch_complete"):
            PROGRESS["formal_optimizer_updates"]=SEEDS.index(seed)*updates()+fields["step"]
        emit(name,human_seed=seed,**fields)
        if name=="epoch_complete" and fields["epoch"]==1:
            estimate=fields["seconds"]*((len(SEEDS)-SEEDS.index(seed))*EPOCHS-1)+900
            available=budget["deadline_monotonic"]-time.monotonic()-SOFT_SECONDS
            emit("BYPASS_REPEAT_PROJECTION",human_seed=seed,estimate_seconds=estimate,available_seconds=available)
            if estimate>available:raise TimeoutError("Measured remaining projection exceeds original3h cap")
    core.training_forward=base.training_forward;core.emit=core.aligned.emit=event;core.atomic_torch_save=best
    try:yield snapshot
    finally:core.training_forward,core.emit,core.aligned.emit,core.atomic_torch_save=saved


@contextmanager
def variant_scope(seed,budget):
    names=("SEED","HARD_SECONDS","KINDS","PROGRESS","protocol","training_scope","resources","emit")
    saved={name:getattr(prior,name) for name in names}
    def event(name,**fields):
        if name=="PAIR_REPLACEMENT_ARM_COMPLETE":name="BYPASS_REPEAT_ARM_COMPLETE"
        emit(name,human_seed=seed,**fields)
    prior.SEED=seed;prior.HARD_SECONDS=HARD_SECONDS;prior.KINDS=("bypass",);prior.PROGRESS={"formal_optimizer_updates":0}
    prior.protocol=lambda:arm_protocol(seed)
    prior.training_scope=lambda _kind,b:training_scope(seed,b)
    prior.resources=lambda p,i,_kind:resources(p,i,"seed"+str(seed))
    prior.emit=event
    try:yield
    finally:
        PROGRESS["formal_optimizer_updates"]=max(PROGRESS["formal_optimizer_updates"],SEEDS.index(seed)*updates()+prior.PROGRESS["formal_optimizer_updates"])
        for name,value in saved.items():setattr(prior,name,value)


def run_seed(workspace,identity,budget,seed):
    arm=workspace/("seed"+str(seed));arm.mkdir()
    info=arm_identity(identity,seed);write_json(arm/"identity.json",info)
    with variant_scope(seed,budget):
        prior.run_arm(arm,info,budget,"bypass")
        result=prior.verify_arm(arm,"bypass",info,budget)
    order=read(arm/"bypass/training_order.json")
    if order["seed"]!=seed or not order["same_as_reference_final_rng"]:
        raise ValueError("Actual training seed/final RNG differs from paired C")
    return result


def verify_seed(workspace,identity,budget,seed):
    arm=workspace/("seed"+str(seed));info=arm_identity(identity,seed)
    if read(arm/"identity.json")!=info:raise ValueError("Arm identity changed")
    with variant_scope(seed,budget):result=prior.verify_arm(arm,"bypass",info,budget)
    record=read(arm/"bypass/training_order.json")
    if (record["seed"]!=seed or record["formal_optimizer_updates"]!=updates()
            or record["final_rng_sha256"]!=info["reference_final_rng_sha256"] or not record["same_as_reference_final_rng"]):
        raise ValueError("Bypass repeat training pairing incomplete")
    return result


def investment(rows):
    checks={}
    for seed in SEEDS:
        row=rows[str(seed)];b,c=row["B"],row["C"];bench=row["benchmark"]
        checks[str(seed)]={"F1_within_tolerance":b["dev_F1"]>=c["dev_F1"]-.10,
                           "IgnA_within_tolerance":b["dev_IgnF1_annotated"]>=c["dev_IgnF1_annotated"]-.10,
                           "latency_reduction_at_least20pct":bench["bypass"]["median_model_forward_seconds"]<=.80*bench["control"]["median_model_forward_seconds"],
                           "peak_allocated_not_increased":bench["bypass"]["max_peak_allocated_bytes"]<=bench["control"]["max_peak_allocated_bytes"]}
    return {"fresh_seed_checks":checks,"efficiency_repeat_screen_passed":all(all(x.values()) for x in checks.values()),
            "fresh_seeds_both_F1_positive":all(rows[str(s)]["B"]["dev_F1"]>rows[str(s)]["C"]["dev_F1"] for s in SEEDS),
            "interpretation":"Dev candidate replication only; no statistical noninferiority or independent Distant/heldout claim"}


def finish(workspace,identity,budget):
    import gc
    import torch
    results={str(seed):verify_seed(workspace,identity,budget,seed) for seed in SEEDS}
    args,human,dev,gold,annotated,distant=prior.data_context(workspace,identity,budget)
    write_json(workspace/"resource_finish.json",resources(workspace,identity,"finish"))
    rows={"75":identity["old75"]};all_raw=[]
    for seed in SEEDS:
        info=arm_identity(identity,seed);arm=workspace/("seed"+str(seed))
        expected={"control":soup.facts(read(Path(info["reference_predictions"]))),"bypass":soup.facts(read(arm/"bypass/predictions.json"))}
        raw=[]
        for index,order in enumerate(protocol()["benchmark_pair_order"]):
            for kind in order:
                check_budget(budget);start_load=time.perf_counter()
                model,_=prior.construct(args,info,kind,None if kind=="control" else arm/"bypass/best_model.pth")
                model.requires_grad_(False);model.eval();prior.synchronize(args.device)
                load_seconds=time.perf_counter()-start_load;before=core.preflight.state_digest(model)
                timing,pred=prior.benchmark_pass(model,dev,args,budget)
                if before!=core.preflight.state_digest(model) or soup.facts(pred)!=expected[kind]:
                    raise RuntimeError("Fixed benchmark state/predictions changed")
                row={"seed":seed,"round":index+1,"variant":kind,"load_seconds_excluded_from_forward":load_seconds,
                     "parameters_unchanged":True,"prediction_replay_passed":True,**timing}
                raw.append(row);all_raw.append(row)
                write_json(workspace/("benchmark-seed%d-round%d-%s.json"%(seed,index+1,kind)),row)
                emit("BYPASS_REPEAT_BENCHMARK",**row)
                del model,pred;gc.collect()
                if args.device.type=="cuda":torch.cuda.empty_cache()
        summary={}
        for kind in ("control","bypass"):
            group=[r for r in raw if r["variant"]==kind]
            summary[kind]={"median_model_forward_seconds":statistics.median(r["model_forward_seconds"] for r in group),
                           "median_end_to_end_seconds":statistics.median(r["end_to_end_seconds"] for r in group),
                           "max_peak_allocated_bytes":max(r["peak_allocated_bytes"] for r in group),
                           "max_peak_reserved_bytes":max(r["peak_reserved_bytes"] for r in group),"rounds":len(group)}
        b=read(arm/"bypass/metrics.json");c=info["reference_metrics"]
        for label,source,expected_metrics in (("B",arm/"bypass/predictions.json",b),("C",Path(info["reference_predictions"]),c)):
            actual=core.shared.score_predictions(read(source),gold,annotated,distant)
            if any(not math.isclose(actual[k],expected_metrics[k],abs_tol=1e-6,rel_tol=0) for k in METRICS):raise ValueError("Final paired score differs")
        rows[str(seed)]={"B":b,"C":c,"benchmark":summary,"checkpoint_sha256":results[str(seed)]["checkpoint_sha256"],"is_screened":False}
    gate=investment(rows)
    deltas={seed:{k:row["B"][k]-row["C"][k] for k in METRICS} for seed,row in rows.items()}
    summary={"seedwise_B_minus_C_pp":deltas,
             "fresh76_77_mean_delta_pp":{k:statistics.mean(deltas[str(s)][k] for s in SEEDS) for k in METRICS},
             "all75_76_77_mean_delta_pp":{k:statistics.mean(deltas[str(s)][k] for s in (75,76,77)) for k in METRICS},
             "single_model_statistics":{label:{"mean":{k:statistics.mean(rows[str(s)][label][k] for s in (75,76,77)) for k in METRICS},
                                                "sample_SD":{k:statistics.stdev(rows[str(s)][label][k] for s in (75,76,77)) for k in METRICS}}
                                         for label in ("B","C")},**gate}
    directory=workspace/"comparison";directory.mkdir()
    write_json(directory/"benchmark.json",{"protocol":protocol()["benchmark_pair_order"],"raw":all_raw,"environment":identity["parent"]["environment"]})
    write_json(directory/"paired_metrics.json",{"rows":rows,**summary})
    verify_files(identity["pinned_files"],budget);verify_files({str(ROOT/n):h for n,h in identity["source_files"].items()},budget)
    for seed in SEEDS:verify_seed(workspace,identity,budget,seed)
    completion={"passed":True,"marker":"BYPASS_REPEAT_COMPLETE","protocol":protocol(),"rows":rows,**summary,
                "formal_optimizer_updates_new":2*updates(),"old_model_optimizer_updates_new":0,"test_data_read":False,"test_evaluated":False,
                "automatic_followup":False,"novelty_established":False,"elapsed_seconds":time.monotonic()-budget["start_monotonic"],
                "existing_stronger_ATGLsoup_Dev_metrics":identity["existing_stronger_ATGLsoup_Dev_metrics"]}
    write_json(directory/"completion.json",completion);soup.archive_reports(workspace,budget)
    emit("BYPASS_REPEAT_COMPLETE",efficiency_repeat_screen_passed=gate["efficiency_repeat_screen_passed"],
         reports=str(directory/"reports.tar.gz"),automatic_followup=False)


def require_lock():
    if os.name!="posix" or os.environ.get("TTMRE_BYPASS_REPEAT_LOCK_FD")!="9":raise RuntimeError("Use one-use bypass repeat launcher")
    import fcntl
    a,b=os.fstat(9),(ROOT/"experiments/2026-10-08-baseline-alignment/.training.lock").stat()
    if (a.st_dev,a.st_ino)!=(b.st_dev,b.st_ino):raise RuntimeError("Wrong shared training lock")
    fcntl.flock(9,fcntl.LOCK_EX|fcntl.LOCK_NB);return 9


def verify_context(workspace,budget):
    c=read(RUN_ROOT/CLAIM)
    if (workspace!=workspace.resolve() or workspace.name!="workspace" or workspace.parent.parent!=RUN_ROOT.resolve()
            or budget["hard_seconds"]!=HARD_SECONDS
            or not math.isclose(budget["deadline_monotonic"]-budget["start_monotonic"],HARD_SECONDS,rel_tol=0.,abs_tol=1e-6)
            or any(budget[k] is not False for k in ("automatic_retry","automatic_resume","extension"))
            or c["workspace"]!=str(workspace) or c["protocol"]!=protocol() or c["budget_sha256"]!=digest(workspace/"budget.json")
            or c["identity_sha256"]!=digest(workspace/"identity.json")):raise ValueError("Invalid immutable3h bypass context")
    check_budget(budget)


def worker(workspace,phase):
    require_lock();budget=read(workspace/"budget.json");verify_context(workspace,budget);identity=read(workspace/"identity.json")
    PROGRESS.update({"phase":phase,"seed":int(phase[4:]) if phase!="finish" else None,
                     "formal_optimizer_updates":{"seed76":0,"seed77":updates(),"finish":2*updates()}[phase]})
    def stop(*_):raise TimeoutError("Bypass repeat interrupted; no resume")
    signal.signal(signal.SIGTERM,stop)
    try:
        if identity["protocol"]!=protocol():raise ValueError("Approved protocol changed")
        verify_files(identity["pinned_files"],budget);verify_files({str(ROOT/n):h for n,h in identity["source_files"].items()},budget)
        if phase=="finish":finish(workspace,identity,budget)
        else:run_seed(workspace,identity,budget,int(phase[4:]))
    except BaseException as error:
        write_json(workspace/("failure-"+phase+".json"),{"error":str(error),"type":type(error).__name__,"complete":False,
                   "progress":PROGRESS,"automatic_retry":False,"resumable":False})
        try:
            if not (workspace/"comparison/reports.tar.gz").exists():soup.archive_reports(workspace,budget)
        except BaseException:pass
        raise


def start(workspace):
    lock=require_lock()
    if workspace.name!="workspace" or workspace.parent.parent!=RUN_ROOT.resolve() or workspace.exists():raise ValueError("Use new bypass repeat workspace")
    claim=RUN_ROOT/CLAIM
    if claim.exists():raise FileExistsError("Repeat already claimed; no retry")
    now=time.monotonic();budget={"start_monotonic":now,"deadline_monotonic":now+HARD_SECONDS,"hard_seconds":HARD_SECONDS,
        "soft_reserve_seconds":SOFT_SECONDS,"term_reserve_seconds":TERM_SECONDS,"start_utc":datetime.now(timezone.utc).isoformat(),
        "boot_id":Path("/proc/sys/kernel/random/boot_id").read_text().strip(),"automatic_retry":False,"automatic_resume":False,"extension":False}
    def stop(*_):raise TimeoutError("Bypass supervisor interrupted")
    signal.signal(signal.SIGTERM,stop)
    try:
        emit("BYPASS_REPEAT_BUDGET_STARTED",**budget)
        identity=validate(budget);emit("BYPASS_REPEAT_SOURCES_VERIFIED",resources=resources(workspace.parent,identity))
        old.supervise([sys.executable,"-B",str(ROOT/FILES[1])],budget,lock)
        check_budget(budget);workspace.mkdir();write_json(workspace/"budget.json",budget);write_json(workspace/"identity.json",identity)
        with claim.open("x",encoding="utf-8") as h:
            json.dump({"workspace":str(workspace),"protocol":protocol(),"budget_sha256":digest(workspace/"budget.json"),
                       "identity_sha256":digest(workspace/"identity.json"),"authorization":"user confirmed B76/77 Human once,total3h"},h,indent=2)
            h.flush();os.fsync(h.fileno())
        command=[sys.executable,"-u","-B",str(Path(__file__).resolve()),"--mode","worker","--run-dir",str(workspace)]
        for phase in ("seed76","seed77","finish"):old.supervise(command+["--phase",phase],budget,lock)
    except BaseException as error:
        write_json(workspace.parent/"failure-supervisor.json",{"error":str(error),"type":type(error).__name__,"complete":False,
                   "elapsed_seconds":time.monotonic()-now,"automatic_retry":False});raise


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--mode",choices=("validate","start","worker"),default="validate")
    p.add_argument("--run-dir",type=Path);p.add_argument("--phase",choices=("seed76","seed77","finish"));a=p.parse_args()
    if a.mode=="validate":print(json.dumps(validate(),indent=2))
    elif a.run_dir is None or (a.mode=="worker" and a.phase is None):raise ValueError("Use single Bash launcher")
    elif a.mode=="start":start(a.run_dir.resolve())
    else:worker(a.run_dir.resolve(),a.phase)


if __name__=="__main__":main()
