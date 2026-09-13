"""1D heat equation, rebuilt on ``pinn.core`` instead of the hardcoded baseline.

This is deliberately the same problem as ``pinn.physics``/``pinn.trainer``
(same domain, same alpha=0.1 default, same initial/boundary data) so its
results can be checked against the original baseline and against the exact
solution — proof that generalizing the framework didn't change what the
original problem solves, only how much of the code is problem-specific.
"""

from __future__ import annotations

import torch
from torch import nn

from ..core.autodiff import grad, laplacian
from ..core.boundary import DirichletBC
from ..core.domain import Domain
from ..core.pde import PDE
from ..core.problem import Problem
from ..core.sampling import AdaptiveResampler, sample_boundary_faces, sample_initial, sample_interior

Tensor = torch.Tensor


class Heat1D(PDE):
    """u_t - alpha * u_xx = 0."""

    def __init__(self, alpha: float = 0.1) -> None:
        if alpha <= 0:
            raise ValueError("alpha must be > 0")
        self.alpha = alpha

    def residual(self, model: nn.Module, points: Tensor) -> Tensor:
        u = model(points)
        u_t = grad(u, points)[:, 1:2]
        u_xx = laplacian(u, points, spatial_indices=[0])
        return u_t - self.alpha * u_xx


def initial_condition(x: Tensor) -> Tensor:
    return torch.sin(torch.pi * x)


def exact_solution(x: Tensor, t: Tensor, alpha: float = 0.1) -> Tensor:
    return torch.exp(-alpha * torch.pi**2 * t) * torch.sin(torch.pi * x)


def domain() -> Domain:
    return Domain(spatial_bounds=[(-1.0, 1.0)], time_bounds=(0.0, 1.0))


def build_problem(
    alpha: float = 0.1,
    n_interior: int = 10_000,
    n_boundary: int = 2_000,
    n_initial: int = 2_000,
    sampling_method: str = "lhs",
    seed: int = 42,
    hard_constraint: bool = False,
    with_resampler: bool = False,
) -> Problem:
    """Build the standard heat-equation problem through the generalized framework.

    With ``hard_constraint=True`` the returned ``Problem`` has NO boundary
    conditions at all — pair it with ``pinn.core.HardDirichletAnsatz`` (see
    ``examples/solve_heat1d_general.py``), which satisfies both the IC and
    the boundary condition exactly, so there is nothing left to penalize.
    """
    d = domain()
    interior = sample_interior(d, n_interior, method=sampling_method, seed=seed)

    boundary_conditions: list[DirichletBC] = []
    if not hard_constraint:
        initial_points = sample_initial(d, n_initial, method=sampling_method, seed=seed + 1)
        boundary_conditions.append(
            DirichletBC(initial_points, lambda p: initial_condition(p[:, 0:1]), name="initial")
        )
        for face in sample_boundary_faces(d, n_boundary, method=sampling_method, seed=seed + 2):
            side_name = "left" if face.side < 0 else "right"
            boundary_conditions.append(
                DirichletBC(face.points, lambda p: torch.zeros(p.shape[0], 1), name=f"{side_name}_boundary")
            )

    resampler = AdaptiveResampler(domain=d, seed=seed + 3) if with_resampler else None

    return Problem(
        domain=d,
        pde=Heat1D(alpha=alpha),
        boundary_conditions=boundary_conditions,
        interior_points=interior,
        resampler=resampler,
        exact_solution=lambda p: exact_solution(p[:, 0:1], p[:, 1:2], alpha),
    )
