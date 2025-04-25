import argparse
import os
import socket
import time

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn

from moe_ep.expert_parallel import ExpertParallelMoE
from moe_ep.moe import DenseMoE, Expert
from moe_ep.router import TopKRouter


def run(rank: int, world_size: int, check: bool, capacity_factor: float | None) -> None:
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(backend, rank=rank, world_size=world_size)
    try:
        device = torch.device("cpu")
        if torch.cuda.is_available():
            device = torch.device("cuda", int(os.environ.get("LOCAL_RANK", rank)))
            torch.cuda.set_device(device)
        torch.manual_seed(31)
        router = TopKRouter(8, 4, 2)
        reference = DenseMoE(
            router, nn.ModuleList(Expert(8, 16) for _ in range(4)), capacity_factor
        )
        experts_per_rank = 4 // world_size
        local_experts = nn.ModuleList(Expert(8, 16) for _ in range(experts_per_rank))
        model = ExpertParallelMoE(
            TopKRouter(8, 4, 2), local_experts, rank * experts_per_rank, capacity_factor
        )
        model.router.load_state_dict(reference.router.state_dict())
        for local, source in zip(
            model.local_experts,
            reference.experts[rank * experts_per_rank : (rank + 1) * experts_per_rank],
            strict=True,
        ):
            local.load_state_dict(source.state_dict())
        model, reference = model.to(device), reference.to(device)
        generator = torch.Generator().manual_seed(41)
        x = torch.randn(6, 8, generator=generator).to(device).requires_grad_(True)
        x_reference = x.detach().repeat(world_size, 1).requires_grad_(True)
        actual, metrics = model(x)
        expected_all, expected_metrics = reference(x_reference)
        expected = expected_all[rank * len(x) : (rank + 1) * len(x)]
        if check:
            torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
            assert metrics["dropped_assignments"] == expected_metrics["dropped_assignments"]
        actual.square().mean().backward()
        expected.square().mean().backward()
        if check:
            expected_grad = x_reference.grad[rank * len(x) : (rank + 1) * len(x)]
            torch.testing.assert_close(x.grad, expected_grad, rtol=1e-4, atol=1e-6)
            assert model.local_experts[0].down.weight.grad is not None
        route = model.router(x.detach())
        dist.all_reduce(route.counts)
        old_bias = model.router.balance_bias.clone()
        model.router.update_balance_bias(route.counts)
        assert not torch.equal(old_bias, model.router.balance_bias)
        benchmark_steps = int(os.environ.get("MOE_STEPS", "1"))
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        dist.barrier()
        started = time.perf_counter()
        for _ in range(benchmark_steps):
            model.zero_grad()
            if x.grad is not None:
                x.grad = None
            values, metrics = model(x)
            values.square().mean().backward()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        dist.barrier()
        seconds = time.perf_counter() - started
        tokens_per_second = benchmark_steps * len(x) / seconds
        print(
            f"rank={rank} load_cv={metrics['load_cv']:.6f} "
            f"dropped={metrics['dropped_assignments']:.0f} "
            f"received={metrics['received_tokens']:.0f} grad_norm={x.grad.norm():.6f} "
            f"tokens/s/rank={tokens_per_second:.2f}",
            flush=True,
        )
    finally:
        dist.destroy_process_group()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--capacity-factor", type=float)
    parser.add_argument("--steps", type=int, default=1)
    args = parser.parse_args()
    os.environ["MOE_STEPS"] = str(args.steps)
    world_size = int(os.environ.get("WORLD_SIZE", "2"))
    if "RANK" in os.environ:
        run(int(os.environ["RANK"]), world_size, args.check, args.capacity_factor)
    else:
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        if "MASTER_PORT" not in os.environ:
            with socket.socket() as sock:
                sock.bind(("127.0.0.1", 0))
                os.environ["MASTER_PORT"] = str(sock.getsockname()[1])
        mp.start_processes(
            run,
            args=(world_size, args.check, args.capacity_factor),
            nprocs=world_size,
            start_method="spawn",
            join=True,
        )


if __name__ == "__main__":
    main()
