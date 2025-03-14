from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class Route:
    indices: torch.Tensor
    weights: torch.Tensor
    counts: torch.Tensor


def load_coefficient_of_variation(counts: torch.Tensor) -> float:
    counts = counts.float()
    return float(counts.std(unbiased=False) / counts.mean().clamp_min(1e-12))


class TopKRouter(nn.Module):
    """Softmax routing weights; a gradient-free bias affects expert selection only."""

    def __init__(self, hidden_size: int, num_experts: int, top_k: int = 2) -> None:
        super().__init__()
        if not 1 <= top_k <= num_experts:
            raise ValueError("top_k must be between one and the number of experts")
        self.num_experts = num_experts
        self.top_k = top_k
        self.gate = nn.Linear(hidden_size, num_experts, bias=False)
        self.register_buffer("balance_bias", torch.zeros(num_experts))

    def forward(self, x: torch.Tensor) -> Route:
        probabilities = self.gate(x).softmax(dim=-1)
        _, indices = torch.topk(probabilities + self.balance_bias, self.top_k, dim=-1)
        weights = probabilities.gather(-1, indices)
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        counts = torch.zeros(self.num_experts, dtype=torch.long, device=x.device)
        counts.scatter_add_(0, indices.flatten(), torch.ones_like(indices.flatten()))
        return Route(indices, weights, counts)

    @torch.no_grad()
    def update_balance_bias(self, global_counts: torch.Tensor, rate: float = 0.05) -> None:
        """Move selection bias away from overload; no auxiliary gradient is involved."""
        load = global_counts.float() / global_counts.sum().clamp_min(1)
        target = 1.0 / self.num_experts
        self.balance_bias.add_(rate * torch.sign(target - load))
