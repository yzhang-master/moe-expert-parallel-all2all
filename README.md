# moe-expert-parallel-all2all

> **Era:** 2025+ · Staff AI Engineer training systems. Technique from 2024–2025, reimplemented in 2026 on PyTorch 2.x.

## Problem

Expert parallelism sends each token to experts owned by different workers, then returns and combines the results. A correct implementation needs variable-sized communication, gradient flow, bounded expert capacity, and balanced routing without letting an auxiliary loss distort the task objective.

## Method

**Top-k routing.** For token hidden state `x`, router probabilities are `p = softmax(Wx)`. Selection uses `topk(p + b, k)`, while combine weights are gathered from the unmodified `p` and renormalized over selected experts. Each expert is a SwiGLU MLP.

**Auxiliary-loss-free balancing.** After globally reducing assignment counts, the gradient-free selection bias is updated as

```text
load_i = count_i / sum(count)
b_i <- b_i + rate * sign(1/num_experts - load_i)
```

Overloaded experts receive lower future selection bias. `balance_bias` is a buffer, not a trainable loss term.

**Real expert parallel all-to-all.** Each rank owns a disjoint expert slice. The router sends hidden states, gates and `(source_rank, token, slot, expert)` metadata with variable-sized `all_to_all_single`. Local experts process their arrivals, and reverse all-to-all returns outputs and gates for weighted `index_add` combination. A custom autograd function performs the reverse collective in backward, so input, router and expert gradients are real.

Capacity is global: `ceil(capacity_factor * global_tokens * k / num_experts)`. Overflow assignments receive zero expert output. Metrics report globally dropped assignments/fraction, expert-load coefficient of variation `std(count)/mean(count)`, received tokens and per-rank token throughput.

## Results

Local CPU verification: **7 tests passed**. Two real two-rank Gloo runs compare expert-parallel forward outputs and input gradients against a dense global reference, both without capacity and with `capacity_factor=0.5`. They also check expert gradients, global drop counts and a nonzero bias update. This is correctness evidence, not a large-model speed result.

| Model | Backend | Load CV | Dropped tokens | Throughput |
|---|---|---|---|---|
| Tiny 4-expert proxy | 2×CPU/Gloo | printed by command | printed by command | printed by command |
| Large MoE | multi-GPU/NCCL | **pending** | **pending** | **pending** |

The included tiny benchmark also runs on CUDA/NCCL:

```bash
# CPU smoke and global-capacity correctness
python scripts/smoke_distributed.py --check
python scripts/smoke_distributed.py --check --capacity-factor 0.5

# Tiny timing; not a large-model benchmark
python scripts/smoke_distributed.py --steps 100 --capacity-factor 1.0

# Executable GPU harness; measurements pending on this machine
torchrun --nproc-per-node=4 scripts/smoke_distributed.py --steps 100 --capacity-factor 1.0
```

The current harness fixes four small experts and six tokens/rank. Scaling to a production model requires replacing those shapes and adding grouped/fused expert kernels; no large-MoE throughput is claimed.

## Reproduce

```bash
python -m pip install -e ".[dev]"
ruff check . && pytest -q
python scripts/smoke_distributed.py --check --steps 3
```

## Limitations

- The smoke uses identical inputs on ranks; the implementation supports variable per-rank send counts, but heterogeneous multi-node workloads remain unbenchmarked.
- Capacity overflow is deterministic arrival-order dropping and dropped tokens have no residual fallback inside this isolated MoE layer.
- Python loops over local experts and routing metadata are intentionally readable, not fused GPU inference kernels.
- Large-model memory, NCCL throughput and balancing-over-training curves are pending.

## References

- D. Lepikhin et al. *GShard: Scaling Giant Models with Conditional Computation and Automatic Sharding.* 2020. arXiv:2006.16668
- A. Fedus, B. Zoph, N. Shazeer. *Switch Transformers.* 2022. arXiv:2101.03961
- DeepSeek-AI. *DeepSeek-V3 Technical Report.* 2024. arXiv:2412.19437 (auxiliary-loss-free load balancing)

## License

MIT
