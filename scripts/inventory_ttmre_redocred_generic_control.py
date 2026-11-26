"""Read-only assets/resources for a proposed Re-DocRED generic-only control.

No model tensors, feature records, Test labels, optimizer, CUDA initialization,
downloads, writes or deletion. Exact Human/Dev cache paths only; no old-weight
directory scan. Presence is not final training admission or model provenance.
"""
from pathlib import Path
import hashlib
import json
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "models/hf-cache/models--FacebookAI--roberta-large/snapshots/722cf37b1afa9454edce342e7895e588b6ff1d59"
BASELINE = ROOT / "experiments/2026-10-11-human-distant-baseline/two-stage-paper-freeze-seed74-launch-20261011-190240-rjeB6g/workspace"
MODEL_NAMES = ("config.json", "model.safetensors", "merges.txt", "tokenizer.json", "tokenizer_config.json", "vocab.json")
CHECKPOINT_REFERENCE_BOUND = 1453845935
REPORT_MARGIN = 512 * 1024**2


def metadata(path):
    if not path.is_file():
        return {"path": str(path), "present": False}
    info = path.stat()
    return {"path": str(path), "present": True, "bytes": info.st_size,
            "allocated_bytes": getattr(info, "st_blocks", 0) * 512,
            "nlink": info.st_nlink, "content_read": False}


def memory_metadata():
    values, constraints = {}, []
    path = Path("/proc/meminfo")
    if path.is_file():
        for line in path.read_text().splitlines():
            key, value = line.split(":", 1)
            if key == "MemAvailable":
                values["host_MemAvailable_bytes"] = int(value.split()[0]) * 1024
    for limit, current in (("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory.current"),
                           ("/sys/fs/cgroup/memory/memory.limit_in_bytes", "/sys/fs/cgroup/memory/memory.usage_in_bytes")):
        a, b = Path(limit), Path(current)
        if a.is_file() and b.is_file():
            cap, used = a.read_text().strip(), b.read_text().strip()
            if cap.isdigit() and used.isdigit():
                constraints.append({"limit_file": limit, "limit_bytes": int(cap), "current_bytes": int(used),
                                    "remaining_bytes": max(0, int(cap) - int(used))})
    candidates = [r["remaining_bytes"] for r in constraints]
    if "host_MemAvailable_bytes" in values:
        candidates.append(values["host_MemAvailable_bytes"])
    return {**values, "cgroup_constraints": constraints,
            "effective_available_bytes": min(candidates) if candidates else None,
            "host_total_is_not_container_quota": True}


def inventory():
    disk = shutil.disk_usage(ROOT)
    caches = {}
    for split, expected in (("human", 3053), ("dev", 500)):
        folder = BASELINE / "cache" / split
        caches[split] = {"expected_documents_from_existing_protocol": expected,
                         "files_metadata_only": [metadata(folder / n) for n in ("index.json", "features.bin")]}
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,memory.used,utilization.gpu", "--format=csv,noheader"],
                           capture_output=True, text=True, timeout=10, check=False)
        gpu = {"source": "external_nvidia_smi_metadata_only", "exit_code": r.returncode,
               "output": r.stdout.strip()[:1500], "error": r.stderr.strip()[:400]}
    except (OSError, subprocess.TimeoutExpired) as error:
        gpu = {"source": "external_nvidia_smi_metadata_only", "unavailable": type(error).__name__}
    required = CHECKPOINT_REFERENCE_BOUND + REPORT_MARGIN
    return {"mode": "read_only", "purpose": "Re-DocRED generic-only G proposal admission, not a training command",
        "project": str(ROOT), "python_version": sys.version.split()[0],
        "generic_model_dir": str(MODEL), "model_files_metadata_only": [metadata(MODEL / n) for n in MODEL_NAMES],
        "Human_Dev_caches": caches, "memory": memory_metadata(), "GPU_metadata": gpu,
        "disk": {"free_bytes": disk.free, "free_GiB": disk.free / 1024**3,
                 "reference_single97class_checkpoint_plus_margin_bytes": required,
                 "reference_single97class_checkpoint_plus_margin_GiB": required / 1024**3,
                 "sufficient_for_reference_bound": disk.free >= required},
        "proposed_target_updates_not_authorized": 22890, "proposed_target_seed": 75,
        "full_model_or_feature_identity_verified": False, "feature_records_read": 0, "tensor_loads": 0,
        "CUDA_initialized_by_script": False, "training_started": False,
        "files_written": 0, "files_deleted": 0, "Test_gold_or_content_read": False,
        "new_budget_approved": False}


if __name__ == "__main__":
    print(json.dumps(inventory(), ensure_ascii=False, indent=2))
