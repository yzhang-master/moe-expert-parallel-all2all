import math

import torch
from torch import nn
from torch.nn import functional as F

from .router import Route, load_coefficient_of_variation


class Expert(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int) -> None:
        super().__init__()
        self.gate = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.silu(self.gate(x)) * self.up(x))


def combine_experts(
    x: torch.Tensor,
    route: Route,
    experts: nn.ModuleList,
    capacity: int | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    output = torch.zeros_like(x)
    dropped = 0
    for expert_index, expert in enumerate(experts):
        token_indices, slots = torch.where(route.indices == expert_index)
        if capacity is not None:
            dropped += max(0, len(token_indices) - capacity)
            token_indices, slots = token_indices[:capacity], slots[:capacity]
        if len(token_indices):
            values = expert(x[token_indices]) * route.weights[token_indices, slots, None]
            output.index_add_(0, token_indices, values)
    metrics = {
        "dropped_assignments": float(dropped),
        "dropped_fraction": dropped / route.indices.numel(),
        "load_cv": load_coefficient_of_variation(route.counts),
    }
    return output, metrics


class DenseMoE(nn.Module):
    def __init__(
        self, router: nn.Module, experts: nn.ModuleList, capacity_factor: float | None = None
    ) -> None:
        super().__init__()
        self.router = router
        self.experts = experts
        self.capacity_factor = capacity_factor

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
        route = self.router(x)
        capacity = None
        if self.capacity_factor is not None:
            capacity = math.ceil(
                self.capacity_factor * x.shape[0] * self.router.top_k / len(self.experts)
            )
        return combine_experts(x, route, self.experts, capacity)
