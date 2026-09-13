import pytest
import torch

from pinn.core.weighting import FixedWeights, GradNormWeights, SelfAdaptiveWeights


def _dummy_model() -> torch.nn.Module:
    torch.manual_seed(0)
    return torch.nn.Sequential(torch.nn.Linear(2, 8), torch.nn.Tanh(), torch.nn.Linear(8, 1))


def test_fixed_weights_applies_given_scalars() -> None:
    weighter = FixedWeights({"physics": 1.0, "boundary": 10.0})
    terms = {"physics": torch.tensor([[2.0]]), "boundary": torch.tensor([[3.0]])}
    total = weighter.combine(terms, _dummy_model())
    assert torch.isclose(total, torch.tensor(1.0 * 2.0 + 10.0 * 3.0))


def test_fixed_weights_uses_default_for_unnamed_terms() -> None:
    weighter = FixedWeights({}, default=5.0)
    terms = {"anything": torch.tensor([[1.0]])}
    total = weighter.combine(terms, _dummy_model())
    assert torch.isclose(total, torch.tensor(5.0))


def test_gradnorm_weights_requires_the_anchor_term() -> None:
    weighter = GradNormWeights(anchor="physics")
    with pytest.raises(KeyError):
        weighter.combine({"boundary": torch.tensor([[1.0]])}, _dummy_model())


def test_gradnorm_weights_runs_and_upweights_the_smaller_gradient_term() -> None:
    model = _dummy_model()
    points = torch.rand(16, 2, requires_grad=True)
    out = model(points)

    # "boundary" has a much smaller gradient magnitude than "physics" here,
    # since it is scaled down before squaring.
    terms = {"physics": out.square(), "boundary": (0.01 * out).square()}
    weighter = GradNormWeights(anchor="physics", update_every=1)
    total = weighter.combine(terms, model)
    assert torch.isfinite(total)
    assert weighter._weights["boundary"] > 1.0  # rebalanced up from the implicit weight of 1


def test_self_adaptive_weights_increase_for_the_higher_residual_term() -> None:
    model = _dummy_model()
    weighter = SelfAdaptiveWeights(lr=0.5)
    # Two terms with very different fixed residual magnitudes.
    small = torch.full((4, 1), 0.01)
    large = torch.full((4, 1), 10.0)

    for _ in range(5):
        total = weighter.combine({"small": small, "large": large}, model)
        total.backward()
        weighter.after_step()

    small_weight = torch.nn.functional.softplus(weighter._lambdas["small"]).mean()
    large_weight = torch.nn.functional.softplus(weighter._lambdas["large"]).mean()
    assert large_weight > small_weight  # ascent pushes weight toward the harder term


def test_self_adaptive_weights_reinitializes_on_point_count_change() -> None:
    model = _dummy_model()
    weighter = SelfAdaptiveWeights()
    weighter.combine({"physics": torch.zeros(10, 1)}, model)
    assert weighter._lambdas["physics"].shape == (10, 1)
    weighter.combine({"physics": torch.zeros(25, 1)}, model)  # e.g. after adaptive resampling
    assert weighter._lambdas["physics"].shape == (25, 1)
