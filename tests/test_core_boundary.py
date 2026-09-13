import torch

from pinn.core.boundary import DirichletBC, NeumannBC, PeriodicBC


class ConstantModel(torch.nn.Module):
    """Returns a fixed value regardless of input — enough to test BC wiring
    without needing a trained network."""

    def __init__(self, value: float) -> None:
        super().__init__()
        self.value = value

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        return torch.full((points.shape[0], 1), self.value)


class QuadraticModel(torch.nn.Module):
    """u(x) = x^2, so du/dx = 2x — a known derivative to check NeumannBC against."""

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        return points[:, 0:1] ** 2


def test_dirichlet_bc_reports_zero_error_when_prediction_matches_target() -> None:
    points = torch.zeros(10, 1)
    bc = DirichletBC(points, value_fn=lambda p: torch.full((p.shape[0], 1), 3.0), name="left")
    error = bc.per_point_error(ConstantModel(3.0))
    assert torch.allclose(error, torch.zeros_like(error))
    assert bc.name == "left"


def test_dirichlet_bc_reports_nonzero_error_on_mismatch() -> None:
    points = torch.zeros(10, 1)
    bc = DirichletBC(points, value_fn=lambda p: torch.zeros(p.shape[0], 1))
    error = bc.per_point_error(ConstantModel(5.0))
    assert torch.allclose(error, torch.full_like(error, 5.0))


def test_neumann_bc_matches_known_derivative() -> None:
    points = torch.tensor([[1.0], [2.0], [3.0]])
    bc = NeumannBC(points, normal_dim=0, value_fn=lambda p: 2 * p[:, 0:1])
    error = bc.per_point_error(QuadraticModel())
    assert torch.allclose(error, torch.zeros_like(error), atol=1e-5)


def test_periodic_bc_value_only() -> None:
    low = torch.tensor([[-1.0, 0.5]])
    high = torch.tensor([[1.0, 0.5]])
    bc = PeriodicBC(low, high, periodic_dim=0, match_derivative=False)
    error = bc.per_point_error(ConstantModel(1.0))
    assert error.shape == (1, 1)
    assert torch.allclose(error, torch.zeros_like(error))


def test_periodic_bc_rejects_mismatched_point_counts() -> None:
    low = torch.zeros(5, 2)
    high = torch.zeros(3, 2)
    try:
        PeriodicBC(low, high, periodic_dim=0)
        assert False, "expected ValueError for mismatched point counts"
    except ValueError:
        pass
