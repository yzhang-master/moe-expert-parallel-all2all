import torch

from moe_ep.moe import DenseMoE, Expert
from moe_ep.router import TopKRouter, load_coefficient_of_variation


def test_topk_weights_and_selection_bias_are_distinct():
    router = TopKRouter(2, 3, 2)
    with torch.no_grad():
        router.gate.weight.copy_(torch.tensor([[4.0, 0.0], [3.0, 0.0], [0.0, 1.0]]))
        router.balance_bias.copy_(torch.tensor([0.0, 0.0, 2.0]))
    route = router(torch.tensor([[1.0, 0.0], [1.0, 0.0]]))
    assert route.indices[:, 0].tolist() == [2, 2]
    assert torch.allclose(route.weights.sum(-1), torch.ones(2))
    assert route.counts.tolist() == [2, 0, 2]


def test_auxiliary_loss_free_bias_update_pushes_overloaded_expert_down():
    router = TopKRouter(2, 2, 1)
    counts = torch.tensor([90, 10])
    before = router.balance_bias.clone()
    router.update_balance_bias(counts, rate=0.1)
    assert router.balance_bias[0] < before[0]
    assert router.balance_bias[1] > before[1]
    assert router.gate.weight.grad is None


def test_capacity_drops_overflow_and_preserves_output_shape():
    torch.manual_seed(1)
    router = TopKRouter(4, 2, 1)
    model = DenseMoE(router, torch.nn.ModuleList(Expert(4, 8) for _ in range(2)), 0.5)
    output, metrics = model(torch.randn(8, 4))
    assert output.shape == (8, 4)
    assert metrics["dropped_assignments"] > 0
    assert 0 < metrics["dropped_fraction"] <= 1


def test_load_cv_formula():
    counts = torch.tensor([2.0, 4.0, 6.0])
    expected = counts.std(unbiased=False) / counts.mean()
    assert load_coefficient_of_variation(counts) == float(expected)
