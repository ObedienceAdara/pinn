import pytest
import torch

from pinn.core import Domain, GeneralMLP, HardDirichletAnsatz
from pinn.core.networks import FourierFeatures, FourierMLP, PeriodicEmbedding


def test_general_mlp_shape_and_input_validation() -> None:
    d = Domain(spatial_bounds=[(-1.0, 1.0), (0.0, 2.0)], time_bounds=(0.0, 1.0))
    model = GeneralMLP(d, hidden_dim=8, hidden_layers=2)
    out = model(torch.rand(5, 3))
    assert out.shape == (5, 1)
    with pytest.raises(ValueError):
        model(torch.rand(5, 2))  # wrong number of input columns


def test_general_mlp_normalizes_bounds_to_unit_cube() -> None:
    d = Domain(spatial_bounds=[(-2.0, 2.0)])
    model = GeneralMLP(d)
    corners = torch.tensor([[-2.0], [2.0], [0.0]])
    normalized = model.normalize_inputs(corners)
    assert torch.allclose(normalized, torch.tensor([[-1.0], [1.0], [0.0]]))


def test_fourier_features_output_shape() -> None:
    ff = FourierFeatures(input_dim=2, mapping_size=16)
    out = ff(torch.rand(10, 2))
    assert out.shape == (10, 32)  # sin + cos, each mapping_size wide


def test_fourier_mlp_runs_on_transient_domain() -> None:
    d = Domain(spatial_bounds=[(-1.0, 1.0)], time_bounds=(0.0, 1.0))
    model = FourierMLP(d, hidden_dim=8, hidden_layers=2, mapping_size=8)
    out = model(torch.rand(5, 2))
    assert out.shape == (5, 1)


def test_periodic_embedding_matches_at_period_boundary() -> None:
    embed = PeriodicEmbedding(input_dim=2, periodic_dim=0, period=2.0)
    a = torch.tensor([[-1.0, 0.5]])
    b = torch.tensor([[1.0, 0.5]])  # one full period away
    assert torch.allclose(embed(a), embed(b), atol=1e-5)


def test_periodic_embedding_rejects_bad_dim_index() -> None:
    with pytest.raises(ValueError):
        PeriodicEmbedding(input_dim=2, periodic_dim=5, period=1.0)


def test_hard_ansatz_satisfies_initial_condition_and_boundary_exactly() -> None:
    d = Domain(spatial_bounds=[(-1.0, 1.0)], time_bounds=(0.0, 1.0))
    base = GeneralMLP(d, hidden_dim=8, hidden_layers=2)
    h = lambda x: torch.sin(torch.pi * x)
    ansatz = HardDirichletAnsatz(base, d, initial_condition_fn=h)

    x = torch.linspace(-1, 1, 25).unsqueeze(1)
    at_t0 = torch.cat([x, torch.zeros_like(x)], dim=1)
    assert torch.allclose(ansatz(at_t0), h(x), atol=1e-5)

    t = torch.rand(25, 1)
    left = torch.cat([-torch.ones_like(t), t], dim=1)
    right = torch.cat([torch.ones_like(t), t], dim=1)
    assert torch.allclose(ansatz(left), torch.zeros_like(t), atol=1e-5)
    assert torch.allclose(ansatz(right), torch.zeros_like(t), atol=1e-5)


def test_hard_ansatz_steady_boundary_extension() -> None:
    d = Domain(spatial_bounds=[(0.0, 1.0), (0.0, 1.0)])
    base = GeneralMLP(d, hidden_dim=8, hidden_layers=2)
    ansatz = HardDirichletAnsatz(base, d, boundary_value_fn=lambda p: torch.zeros(p.shape[0], 1))
    edges = torch.tensor([[0.0, 0.3], [1.0, 0.3], [0.3, 0.0], [0.3, 1.0]])
    assert torch.allclose(ansatz(edges), torch.zeros(4, 1), atol=1e-5)


def test_hard_ansatz_rejects_incompatible_argument_combinations() -> None:
    transient = Domain(spatial_bounds=[(-1.0, 1.0)], time_bounds=(0.0, 1.0))
    steady = Domain(spatial_bounds=[(-1.0, 1.0)])

    with pytest.raises(NotImplementedError):
        HardDirichletAnsatz(
            GeneralMLP(transient),
            transient,
            boundary_value_fn=lambda p: torch.zeros(p.shape[0], 1),
        )
    with pytest.raises(ValueError):
        HardDirichletAnsatz(GeneralMLP(transient), transient)  # missing initial_condition_fn
    with pytest.raises(ValueError):
        HardDirichletAnsatz(
            GeneralMLP(steady), steady, initial_condition_fn=lambda x: torch.zeros(x.shape[0], 1)
        )
