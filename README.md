# moe-expert-parallel-all2all

> **Era:** 2025+; technique reimplemented in 2026.

Tiny CPU/Gloo proxy for top-k MoE routing, auxiliary-loss-free router bias and distributed expert parallelism. GPU throughput is pending.

## Reproduce

```bash
pip install -e ".[dev]"
ruff check . && pytest -q
```

## Limitations

No large model or GPU benchmark has run.

## References

- DeepSeek-V3 auxiliary-loss-free balancing: https://arxiv.org/abs/2412.19437
- GShard: https://arxiv.org/abs/2006.16668

## License

MIT
