"""Pair-conditioned redistribution inside the original TTM memory read."""

from __future__ import annotations

import math

import torch
from torch import nn


class PairSlotRead(nn.Module):
    def __init__(self, original, memory_size=200, rank=64, alpha_init=0.01):
        super().__init__()
        if memory_size < 1 or rank < 1 or not 0 < alpha_init < 1:
            raise ValueError("Invalid memory size, rank or initial gate")
        if original.mlp1.linear_2.out_features != 2:
            raise ValueError("Expected exactly two read outputs")
        if any(isinstance(module, nn.Dropout) and module.p != 0 for module in original.modules()):
            raise ValueError("The pinned read has zero dropout")
        # Reuse modules at their original paths; shared state_dict keys stay intact.
        self.layernorm = original.layernorm
        self.mlp1 = original.mlp1
        self.memory_size, self.rank, self.enabled = memory_size, rank, True
        dimension = original.mlp1.linear_1.in_features
        self.query_self = nn.Linear(dimension, rank, bias=False)
        self.query_other = nn.Linear(dimension, rank, bias=False)
        self.memory_key = nn.Linear(dimension, rank, bias=False)
        for module in (self.query_self, self.query_other, self.memory_key):
            nn.init.xavier_uniform_(module.weight)
        self.eta = nn.Parameter(torch.tensor(math.log(alpha_init / (1 - alpha_init))))
        self.to(device=original.layernorm.weight.device, dtype=original.layernorm.weight.dtype)

    def new_parameters(self):
        return {"query_self.weight": self.query_self.weight,
                "query_other.weight": self.query_other.weight,
                "memory_key.weight": self.memory_key.weight, "eta": self.eta}

    def distributions(self, inputs):
        if inputs.ndim != 3 or inputs.shape[1] != self.memory_size + 2:
            raise ValueError("Expected [pairs, memory_size + 2, hidden]")
        normalized = self.layernorm(inputs)
        scores = self.mlp1(normalized).transpose(1, 2)
        original = scores.softmax(-1)
        memory_scores = scores[:, :, :self.memory_size]
        rho = memory_scores.softmax(-1)
        head, tail = normalized[:, -2], normalized[:, -1]
        queries = torch.stack((self.query_self(head) + self.query_other(tail),
                               self.query_self(tail) + self.query_other(head)), dim=1)
        keys = self.memory_key(normalized[:, :self.memory_size])
        delta = torch.einsum("bir,bmr->bim", queries, keys) / math.sqrt(self.rank)
        conditional = (memory_scores + delta).softmax(-1)
        mass = original[:, :, :self.memory_size].sum(-1, keepdim=True)
        alpha = self.eta.sigmoid()
        mixed = (1 - alpha) * rho + alpha * conditional
        weights = torch.cat((mass * mixed, original[:, :, self.memory_size:]), dim=-1)
        return weights, original, rho, conditional, mixed, mass

    def forward(self, inputs):
        if not self.enabled:
            from ttm import TokenLearnerModuleV11
            return TokenLearnerModuleV11.forward(self, inputs)
        weights = self.distributions(inputs)[0]
        return torch.einsum("bsi,bid->bsd", weights, inputs)

    @torch.no_grad()
    def diagnostics(self, inputs):
        weights, original, rho, conditional, mixed, mass = self.distributions(inputs)
        eps = torch.finfo(inputs.dtype).tiny

        def entropy(values):
            return -(values * values.clamp_min(eps).log()).sum(-1).mean().item()

        def pair_distance(values):
            if len(values) < 2:
                return 0.0
            return torch.pdist(values.flatten(1), p=2).mean().item()

        summaries = torch.einsum("bim,bmd->bid", mixed, inputs[:, :self.memory_size])
        ranks = []
        for index in range(2):
            singular = torch.linalg.svdvals(summaries[:, index].float())
            if singular.sum().item() == 0:
                ranks.append(0.0)
            else:
                probabilities = singular / singular.sum()
                ranks.append((-(probabilities * probabilities.clamp_min(eps).log()).sum()).exp().item())
        return {"sample_pairs": len(inputs), "alpha": self.eta.sigmoid().item(),
                "memory_mass_mean": mass.mean().item(),
                "memory_mass_max_delta": (weights[:, :, :self.memory_size].sum(-1, keepdim=True) - mass).abs().max().item(),
                "entity_weights_max_delta": (weights[:, :, self.memory_size:] - original[:, :, self.memory_size:]).abs().max().item(),
                "original_entropy": entropy(rho), "conditional_entropy": entropy(conditional),
                "mixed_entropy": entropy(mixed), "original_pair_distance": pair_distance(rho),
                "conditional_pair_distance": pair_distance(conditional), "mixed_pair_distance": pair_distance(mixed),
                "mixed_to_original_KL_mean": (mixed * (mixed.clamp_min(eps).log() - rho.clamp_min(eps).log())).sum(-1).mean().item(),
                "memory_summary_effective_ranks": ranks}


def install_pair_read(model, rank=64, alpha_init=0.01):
    unit = model.mu_encoder.token_turing_machine_unit
    if isinstance(unit.token_learner_model, PairSlotRead):
        raise ValueError("Pair read is already installed")
    original_parameters = {name: id(parameter) for name, parameter in model.named_parameters()}
    unit.token_learner_model = PairSlotRead(unit.token_learner_model, model.mu_encoder.memory_size,
                                           rank, alpha_init)
    current = dict(model.named_parameters())
    if any(name not in current or id(current[name]) != identity for name, identity in original_parameters.items()):
        raise RuntimeError("Installation changed a shared parameter or its path")
    reader = unit.token_learner_model
    return {"shared_parameters_preserved": True,
            "new_parameter_count": sum(parameter.numel() for parameter in reader.new_parameters().values())}


def new_gradient_report(model):
    reader = model.mu_encoder.token_turing_machine_unit.token_learner_model
    report = {}
    for name, parameter in reader.new_parameters().items():
        gradient = parameter.grad
        if gradient is None or not torch.isfinite(gradient).all().item() or gradient.abs().sum().item() == 0:
            raise RuntimeError(f"Missing, nonfinite or zero new gradient: {name}")
        report[name] = {"norm": gradient.norm().item(), "max_abs": gradient.abs().max().item()}
    return report
