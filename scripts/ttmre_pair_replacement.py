"""Known residual pair MLP / identity substitute for conditional TTM adaptation.

The original model2 still applies output/2 + input/2. Returning input+2*residual
therefore implements the proposed input+residual, retaining the upstream head.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

KINDS = ("bypass", "mlp")
BOTTLENECK = 256
PRIVATE_SEED = 75007501


class PairReplacement(nn.Module):
    def __init__(self, width, kind, memory):
        super().__init__()
        if kind not in KINDS or width <= 0:
            raise ValueError("Unknown pair replacement or width")
        self.width, self.kind = width, kind
        # Provenance-only immutable buffer: never a trainable parameter/read input.
        self.register_buffer("memory_tokens", memory.detach().clone())
        if kind == "mlp":
            # Only CPU RNG is used/restored; CUDA/Python/NumPy streams untouched.
            with torch.random.fork_rng(devices=[]):
                torch.random.default_generator.manual_seed(PRIVATE_SEED)
                self.down = nn.Linear(2*width, BOTTLENECK)
                self.up = nn.Linear(BOTTLENECK, 2*width)
                nn.init.zeros_(self.up.weight)
                nn.init.zeros_(self.up.bias)
            self.to(device=memory.device, dtype=memory.dtype)

    def forward(self, input_tokens, train=False):
        if input_tokens.ndim != 4 or input_tokens.shape[1:] != (1, 2, self.width):
            raise ValueError("Replacement requires independent pair, one step, two entity tokens")
        if self.kind == "bypass":
            return input_tokens
        x = input_tokens[:, 0].reshape(len(input_tokens), 2*self.width)
        residual = self.up(F.gelu(self.down(x)))
        return (x + 2*residual).reshape_as(input_tokens)


def install(model, kind):
    if isinstance(model.mu_encoder, PairReplacement):
        raise ValueError("Replacement already installed")
    old = model.mu_encoder
    before = {n: id(p) for n,p in model.named_parameters() if not n.startswith("mu_encoder.")}
    old_parameters = sum(p.numel() for p in old.parameters())
    replacement = PairReplacement(model.emb_size, kind, old.memory_tokens)
    model.mu_encoder = replacement
    after = dict(model.named_parameters())
    if any(n not in after or id(after[n]) != identity for n,identity in before.items()):
        raise RuntimeError("Replacement changed common parameter identities")
    names = [n for n in after if n.startswith("mu_encoder.")]
    allowed = [] if kind == "bypass" else ["mu_encoder.down.weight", "mu_encoder.down.bias", "mu_encoder.up.weight", "mu_encoder.up.bias"]
    if names != allowed or any(p.requires_grad is False for p in model.parameters()):
        raise RuntimeError("Unexpected replacement optimizer/freeze scope")
    return {"kind": kind, "removed_TTM_parameter_elements": old_parameters,
            "new_parameter_elements": sum(p.numel() for p in replacement.parameters()),
            "total_parameter_elements": sum(p.numel() for p in model.parameters()),
            "provenance_memory_buffer_elements": replacement.memory_tokens.numel(),
            "memory_participates_in_forward": False, "common_parameter_objects_preserved": True,
            "bottleneck": BOTTLENECK if kind == "mlp" else None,
            "formula": "x+U*GELU(V*x) after original model2 half fusion" if kind == "mlp" else "identity entity-pair enhancement",
            "private_initialization_seed": PRIVATE_SEED if kind == "mlp" else None}
