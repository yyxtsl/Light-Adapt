"""Standard-library-only GPU idleness check that does not initialize CUDA."""
from __future__ import annotations

import csv
import io
import os
import subprocess

MIB = 1024 * 1024
DEFAULT_LIMIT_BYTES = 500 * MIB
QUERY = (
    "nvidia-smi", "--query-gpu=index,name,memory.used,memory.total,utilization.gpu,uuid",
    "--format=csv,noheader,nounits",
)


def parse_rows(stdout):
    rows = []
    for row in csv.reader(io.StringIO(stdout)):
        if not row or all(not item.strip() for item in row):
            continue
        if len(row) != 6:
            raise ValueError("Unexpected nvidia-smi GPU row: " + repr(row))
        index, name, used, total, utilization, uuid = (item.strip() for item in row)
        try:
            used_mib = float(used)
            total_mib = float(total)
            utilization_pct = float(utilization)
            physical_index = int(index)
        except ValueError as error:
            raise ValueError("nvidia-smi returned a nonnumeric GPU field: " + repr(row)) from error
        if min(used_mib, total_mib, utilization_pct) < 0 or total_mib <= 0:
            raise ValueError("nvidia-smi returned an invalid GPU metric: " + repr(row))
        rows.append({"index": physical_index, "name": name, "uuid": uuid,
                     "memory_used_bytes": int(used_mib * MIB),
                     "memory_total_bytes": int(total_mib * MIB),
                     "utilization_percent": utilization_pct})
    if not rows:
        raise RuntimeError("nvidia-smi returned no GPU rows")
    return rows


def select_visible_gpu(rows, visible_devices=None):
    if visible_devices is None:
        visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible_devices is None or visible_devices.strip() == "":
        wanted = "0"
    else:
        wanted = visible_devices.split(",", 1)[0].strip()
        if wanted == "-1":
            raise RuntimeError("CUDA_VISIBLE_DEVICES disables CUDA")
    for row in rows:
        if wanted in (str(row["index"]), row["uuid"]):
            return row
    raise RuntimeError("Cannot map first CUDA_VISIBLE_DEVICES entry %r to an nvidia-smi GPU" % wanted)


def inspect_gpu(memory_limit_bytes=DEFAULT_LIMIT_BYTES, visible_devices=None, run=None):
    """Query the externally visible pre-process GPU state without importing torch."""
    runner = subprocess.run if run is None else run
    result = runner(QUERY, capture_output=True, text=True, check=True, timeout=10)
    rows = parse_rows(result.stdout)
    selected = select_visible_gpu(rows, visible_devices)
    return {"source": "nvidia-smi-before-torch-cuda-init",
            "index": selected["index"], "name": selected["name"], "uuid": selected["uuid"],
            "memory_used_bytes": selected["memory_used_bytes"],
            "memory_total_bytes": selected["memory_total_bytes"],
            "utilization_percent": selected["utilization_percent"],
            "idle_threshold_bytes": memory_limit_bytes,
            "passed": selected["memory_used_bytes"] < memory_limit_bytes,
            "all_gpu_rows": rows}
