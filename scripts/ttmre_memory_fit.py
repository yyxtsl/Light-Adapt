"""Execution-only memory savings for the pinned single-step TTM path."""

from types import MethodType

import torch
from torch import nn


def _require_single_step(module, inputs):
    tokens = inputs[0]
    if tokens.ndim != 4 or tokens.shape[1] != 1:
        raise ValueError("Unused-write bypass requires exactly one TTM step")


def _unused_write(self, memory, controls, train=False):
    # The single-step encoder returns only controls; this memory is discarded.
    return memory


def install_unused_write_bypass(model):
    encoder = model.mu_encoder
    if getattr(encoder, "_ttmre_unused_write_bypassed", False):
        raise ValueError("Unused-write bypass already installed")
    writer = encoder.token_turing_machine_unit.token_add_erase_write
    if any(isinstance(module, nn.Dropout) and module.p != 0 for module in writer.modules()):
        raise ValueError("Unused-write bypass requires zero write dropout to preserve RNG state")
    # Retain the original module and all parameter/checkpoint/optimizer paths.
    writer.forward = MethodType(_unused_write, writer)
    encoder.register_forward_pre_hook(_require_single_step)
    encoder._ttmre_unused_write_bypassed = True
    return {"unused_write_bypassed": True, "required_ttm_steps": 1,
            "write_parameter_tensors_retained": sum(1 for _ in writer.parameters()),
            "parameter_paths_preserved": True}


def _checkpoint_ttm_forward(self, input_tokens, train=False):
    original = self._ttmre_uncheckpointed_forward
    if self.training and torch.is_grad_enabled():
        from torch.utils.checkpoint import checkpoint
        return checkpoint(original, input_tokens, train,
                          use_reentrant=False, preserve_rng_state=True)
    return original(input_tokens, train)


def install_ttm_checkpoint(model):
    encoder = model.mu_encoder
    if not getattr(encoder, "_ttmre_unused_write_bypassed", False):
        raise ValueError("Install the guarded single-step write bypass first")
    if hasattr(encoder, "_ttmre_uncheckpointed_forward"):
        raise ValueError("TTM checkpoint already installed")
    encoder._ttmre_uncheckpointed_forward = encoder.forward
    encoder.forward = MethodType(_checkpoint_ttm_forward, encoder)
    return {"ttm_activation_checkpointing": True, "use_reentrant": False,
            "preserve_rng_state": True, "pair_chunk_size_unchanged": 256,
            "scope": "TTM only during training with gradients; eval forward unchanged"}


def reserve_adam_moments(model):
    # Upper bound: even disconnected parameters are reserved. No optimizer or
    # updates, no random sampling. Actual AdamW lazily allocates only for grads.
    buffers = []
    for parameter in model.parameters():
        if parameter.requires_grad:
            buffers.extend((torch.zeros_like(parameter), torch.zeros_like(parameter)))
    size = sum(buffer.numel() * buffer.element_size() for buffer in buffers)
    return buffers, {"adam_moments_reserved_GiB": size / 1024**3,
                     "scope": "two parameter-shaped moment buffers; excludes optimizer-step temporary allocations",
                     "optimizer_steps": 0}
