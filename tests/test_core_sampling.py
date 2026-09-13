import torch

from pinn.core import Domain
from pinn.core.autodiff import grad, laplacian
from pinn.core.sampling import (
    AdaptiveResampler,
    sample_boundary_faces,
    sample_initial,
    sample_interior,
)


def _within_bounds(points: torch.Tensor, domain: Domain) -> bool:
    lower = torch.as_tensor(domain.lower())
    upper = torch.as_tensor(domain.upper())
    return bool(torch.all(points >= lower) and torch.all(points <= upper))


def test_sample_interior_shapes_and_bounds_for_every_method() -> None:
    d = Domain(spatial_bounds=[(-1.0, 1.0), (0.0, 2.0)], time_bounds=(0.0, 1.0))
    for method in ("uniform", "lhs", "sobol"):
        points = sample_interior(d, 200, method=method, seed=1)
        assert points.shape == (200, 3)
        assert _within_bounds(points, d)


def test_sample_initial_fixes_time_column_at_t0() -> None:
    d = Domain(spatial_bounds=[(-1.0, 1.0)], time_bounds=(0.0, 1.0))
    points = sample_initial(d, 50, method="lhs")
    assert points.shape == (50, 2)
    assert torch.all(points[:, 1] == 0.0)


def test_sample_boundary_faces_pins_the_correct_column() -> None:
    d = Domain(spatial_bounds=[(-1.0, 1.0), (0.0, 2.0)])
    faces = sample_boundary_faces(d, 30, method="lhs")
    assert len(faces) == 4
    for face in faces:
        assert torch.allclose(face.points[:, face.dim_index], torch.full((30,), face.value))
        assert _within_bounds(face.points, d)


def test_adaptive_resampler_favors_higher_residual_region() -> None:
    d = Domain(spatial_bounds=[(-1.0, 1.0)])

    class LinearModel(torch.nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x

    def residual_fn(model: torch.nn.Module, points: torch.Tensor) -> torch.Tensor:
        # Residual magnitude is largest near the domain edges (|x| close to 1),
        # so resampling should concentrate there regardless of sign (the
        # resampler ranks by squared residual).
        return points

    resampler = AdaptiveResampler(domain=d, pool_size=2000, method="sobol", seed=0)
    selected = resampler.refresh(residual_fn, LinearModel(), n_select=50)
    assert selected.shape == (50, 1)
    assert not selected.requires_grad
    assert selected.abs().mean().item() > 0.9  # concentrated toward the high-|residual| edges


def test_laplacian_matches_known_analytic_function() -> None:
    # u = x^2 + y^2 has Laplacian == 4 everywhere.
    points = torch.rand(20, 2, requires_grad=True)
    u = points[:, 0:1] ** 2 + points[:, 1:2] ** 2
    lap = laplacian(u, points, spatial_indices=[0, 1])
    assert torch.allclose(lap, torch.full_like(lap, 4.0), atol=1e-4)


def test_grad_rejects_non_scalar_output_and_missing_requires_grad() -> None:
    points = torch.rand(5, 2, requires_grad=True)
    two_column_output = points  # shape (5, 2), not (5, 1)
    try:
        grad(two_column_output, points)
        assert False, "expected ValueError for non-scalar output"
    except ValueError:
        pass

    no_grad_points = torch.rand(5, 2)
    u = no_grad_points.sum(dim=1, keepdim=True)
    try:
        grad(u, no_grad_points)
        assert False, "expected ValueError for inputs without requires_grad"
    except ValueError:
        pass
