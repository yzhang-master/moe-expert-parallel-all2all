import math

import torch
import torch.distributed as dist
from torch import nn

from .moe import DenseMoE
from .router import load_coefficient_of_variation


class AllToAll(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, input_splits, output_splits):
        ctx.input_splits = input_splits
        ctx.output_splits = output_splits
        output = x.new_empty((sum(output_splits), *x.shape[1:]))
        dist.all_to_all_single(
            output, x, output_split_sizes=output_splits, input_split_sizes=input_splits
        )
        return output

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.new_empty((sum(ctx.input_splits), *grad_output.shape[1:]))
        dist.all_to_all_single(
            grad_input,
            grad_output.contiguous(),
            output_split_sizes=ctx.input_splits,
            input_split_sizes=ctx.output_splits,
        )
        return grad_input, None, None


def all_to_all(x: torch.Tensor, input_splits: list[int], output_splits: list[int]) -> torch.Tensor:
    return AllToAll.apply(x, tuple(input_splits), tuple(output_splits))


class ExpertParallelMoE(nn.Module):
    def __init__(
        self,
        router: nn.Module,
        local_experts: nn.ModuleList,
        expert_start: int,
        capacity_factor: float | None = None,
    ) -> None:
        super().__init__()
        self.router = router
        self.local_experts = local_experts
        self.expert_start = expert_start
        self.capacity_factor = capacity_factor

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
        if not dist.is_initialized():
            if dist.is_available() and self.expert_start == 0:
                return DenseMoE(self.router, self.local_experts, self.capacity_factor)(x)
            raise RuntimeError("initialize a process group for expert parallelism")
        rank, world_size = dist.get_rank(), dist.get_world_size()
        route = self.router(x)
        flat_indices = route.indices.flatten()
        token_ids = torch.arange(x.shape[0], device=x.device).repeat_interleave(
            self.router.top_k
        )
        slots = torch.arange(self.router.top_k, device=x.device).repeat(x.shape[0])
        destination = flat_indices // len(self.local_experts)
        send_order = torch.cat([torch.where(destination == dest)[0] for dest in range(world_size)])
        send_counts = torch.bincount(destination, minlength=world_size).long()
        recv_counts = torch.empty_like(send_counts)
        dist.all_to_all_single(recv_counts, send_counts, output_split_sizes=[1] * world_size)
        send_meta = torch.stack(
            (torch.full_like(token_ids, rank), token_ids, slots, flat_indices), dim=1
        )[send_order]
        recv_meta = send_meta.new_empty((int(recv_counts.sum()), 4))
        dist.all_to_all_single(
            recv_meta,
            send_meta,
            output_split_sizes=recv_counts.tolist(),
            input_split_sizes=send_counts.tolist(),
        )
        sent_x = x[token_ids][send_order]
        received_x = all_to_all(sent_x, send_counts.tolist(), recv_counts.tolist())
        sent_weights = route.weights.flatten()[send_order, None]
        received_weights = all_to_all(
            sent_weights, send_counts.tolist(), recv_counts.tolist()
        )
        global_tokens = torch.tensor(x.shape[0], device=x.device)
        dist.all_reduce(global_tokens)
        capacity = None
        if self.capacity_factor is not None:
            capacity = math.ceil(
                self.capacity_factor
                * int(global_tokens)
                * self.router.top_k
                / self.router.num_experts
            )
        local_outputs = torch.zeros_like(received_x)
        dropped = 0
        for local_index, expert in enumerate(self.local_experts):
            expert_index = self.expert_start + local_index
            positions = torch.where(recv_meta[:, 3] == expert_index)[0]
            if capacity is not None:
                dropped += max(0, len(positions) - capacity)
                positions = positions[:capacity]
            if len(positions):
                local_outputs[positions] = expert(received_x[positions])
        returned = all_to_all(
            local_outputs, recv_counts.tolist(), send_counts.tolist()
        )
        returned_weights = all_to_all(
            received_weights, recv_counts.tolist(), send_counts.tolist()
        )
        output = torch.zeros_like(x)
        output.index_add_(0, send_meta[:, 1], returned * returned_weights)
        dropped_tensor = torch.tensor(dropped, device=x.device)
        global_counts = route.counts.clone()
        dist.all_reduce(dropped_tensor)
        dist.all_reduce(global_counts)
        total_assignments = int(global_tokens) * self.router.top_k
        metrics = {
            "dropped_assignments": float(dropped_tensor),
            "dropped_fraction": float(dropped_tensor) / total_assignments,
            "load_cv": load_coefficient_of_variation(global_counts),
            "received_tokens": float(len(received_x)),
        }
        return output, metrics
