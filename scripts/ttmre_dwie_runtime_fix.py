"""Audit the actual AdamW implementation in the fixed PyTorch 2.7.1 layout.

The failed runner looked in torch.optim.adamw for _single_tensor_adamw.
In 2.7.1 AdamW inherits Adam.step and uses adam._single_tensor_adam with
decoupled_weight_decay=True. This module records that implementation without
constructing an optimizer, initializing CUDA or changing the training method.
"""
from __future__ import annotations

import ast
import hashlib
import importlib
import inspect
from pathlib import Path


def adamw_implementation_identity():
    import torch
    adam_module = importlib.import_module("torch.optim.adam")
    step = inspect.unwrap(torch.optim.AdamW.step)
    if (step is not inspect.unwrap(torch.optim.Adam.step)
            or step.__module__ != "torch.optim.adam"):
        raise RuntimeError("Unexpected AdamW step layout; fixed 2.7.1 admission refuses")
    single = getattr(adam_module, "_single_tensor_adam", None)
    functional = getattr(adam_module, "adam", None)
    if (not callable(single) or not callable(functional)
            or "decoupled_weight_decay" not in inspect.signature(single).parameters):
        raise RuntimeError("Missing decoupled AdamW scalar implementation; admission refuses")
    class_source = inspect.getsource(torch.optim.AdamW)
    tree = ast.parse(class_source)
    decoupled = any(isinstance(node, ast.keyword)
                    and node.arg == "decoupled_weight_decay"
                    and isinstance(node.value, ast.Constant) and node.value.value is True
                    for node in ast.walk(tree))
    if not decoupled:
        raise RuntimeError("AdamW class does not establish decoupled weight decay")
    sources = {"AdamW_class": class_source, "Adam_step": inspect.getsource(step),
               "adam_functional": inspect.getsource(functional),
               "single_tensor_adam": inspect.getsource(single)}
    hashes = {name: hashlib.sha256(source.encode()).hexdigest() for name, source in sources.items()}
    return {"AdamW_step_module": step.__module__, "AdamW_step_qualname": step.__qualname__,
            "AdamW_single_tensor_implementation": "torch.optim.adam._single_tensor_adam",
            "AdamW_decoupled_weight_decay": True,
            "AdamW_single_tensor_source_sha256": hashes["single_tensor_adam"],
            "AdamW_implementation_source_sha256": hashes,
            "foreach": False, "fused": False,
            "optimizer_created_for_audit": False, "optimizer_updates_for_audit": 0}


def runtime_identity():
    import sys
    import torch
    import numpy as np
    import transformers
    if torch.__version__ != "2.7.1+cu128":
        raise RuntimeError("Expected verified cloud torch 2.7.1+cu128; no install/version fallback")
    if np.__version__ != "1.24.3" or transformers.__version__ != "4.34.0":
        raise RuntimeError("Expected verified cloud numpy 1.24.3 / transformers 4.34.0; no install/version fallback")
    implementation = adamw_implementation_identity()
    limits = {}
    for name in ("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory.current",
                 "/sys/fs/cgroup/memory/memory.limit_in_bytes", "/sys/fs/cgroup/memory/memory.usage_in_bytes"):
        path = Path(name)
        if path.is_file():
            limits[name] = path.read_text().strip()
    return {"python": sys.version, "torch": torch.__version__, "numpy": np.__version__,
            "transformers": transformers.__version__, "cuda_runtime": torch.version.cuda,
            "cgroup_memory": limits, "host_meminfo_is_not_container_limit": True,
            **implementation}
