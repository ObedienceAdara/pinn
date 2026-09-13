import torch

from pinn.core.causal import CausalWeighter


def test_epsilon_zero_recovers_uniform_weighting() -> None:
    weighter = CausalWeighter(n_bins=5, epsilon=0.0)
    times = torch.linspace(0, 1, 50).unsqueeze(1)
    residual_sq = torch.rand(50, 1)
    weights = weighter.weight(residual_sq, times)
    assert torch.allclose(weights, torch.ones_like(weights))


def test_large_early_residual_suppresses_later_bins() -> None:
    weighter = CausalWeighter(n_bins=5, epsilon=2.0)
    times = torch.linspace(0, 1, 100).unsqueeze(1)
    residual_sq = torch.ones(100, 1)
    residual_sq[:20] *= 5.0  # large residual concentrated in the earliest bin

    weights = weighter.weight(residual_sq, times)
    per_bin = weights.reshape(-1)[::20]  # one representative point per bin
    assert per_bin[0] == 1.0  # the first bin is never suppressed
    assert torch.all(per_bin[1:] < per_bin[:-1])  # strictly decaying afterward


def test_degenerate_time_range_returns_ones() -> None:
    weighter = CausalWeighter()
    times = torch.zeros(10, 1)  # every point at the same instant
    residual_sq = torch.rand(10, 1)
    weights = weighter.weight(residual_sq, times)
    assert torch.allclose(weights, torch.ones_like(weights))


def test_weight_output_is_detached_from_the_graph() -> None:
    weighter = CausalWeighter(n_bins=4, epsilon=1.0)
    times = torch.linspace(0, 1, 20).unsqueeze(1).requires_grad_(True)
    residual_sq = (times**2).requires_grad_(True)
    weights = weighter.weight(residual_sq, times)
    assert not weights.requires_grad
