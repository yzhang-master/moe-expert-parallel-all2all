import torch


def test_all_to_all_is_available():
    assert torch.distributed.is_gloo_available()
    assert callable(torch.distributed.all_to_all_single)
