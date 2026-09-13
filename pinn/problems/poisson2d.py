"""2D Poisson equation: -Laplacian(u) = f, u = 0 on the boundary of [0,1]^2.

The manufactured solution u*(x, y) = sin(pi*x) * sin(pi*y) is chosen because
it is exactly zero on all four edges of the unit square, which lets the
homogeneous-Dirichlet hard-constraint ansatz apply directly, and because its
source term f is known in closed form, which lets a trained network be
checked against ground truth rather than only against itself. This is the
smallest problem in the repo that a 1D-only implementation genuinely could
not express — proof the Domain/PDE/sampling generalization actually buys
something, not just an API reshuffle.
"""

from __future__ import annotations

import torch
from torch import nn

from ..core.autodiff import laplacian
from ..core.boundary import DirichletBC
from ..core.domain import Domain
from ..core.pde import PDE
from ..core.problem import Problem
from ..core.sampling import sample_boundary_faces, sample_interior

Tensor = torch.Tensor


class Poisson2D(PDE):
    """-Laplacian(u) - f = 0."""

    def residual(self, model: nn.Module, points: Tensor) -> Tensor:
        u = model(points)
        lap = laplacian(u, points, spatial_indices=[0, 1])
        return -lap - source_term(points)


def source_term(points: Tensor) -> Tensor:
    x, y = points[:, 0:1], points[:, 1:2]
    return 2 * torch.pi**2 * torch.sin(torch.pi * x) * torch.sin(torch.pi * y)


def exact_solution(points: Tensor) -> Tensor:
    x, y = points[:, 0:1], points[:, 1:2]
    return torch.sin(torch.pi * x) * torch.sin(torch.pi * y)


def domain() -> Domain:
    return Domain(spatial_bounds=[(0.0, 1.0), (0.0, 1.0)])


def build_problem(
    n_interior: int = 4_000,
    n_boundary_per_face: int = 400,
    sampling_method: str = "sobol",
    seed: int = 0,
    hard_constraint: bool = True,
) -> Problem:
    d = domain()
    interior = sample_interior(d, n_interior, method=sampling_method, seed=seed)

    boundary_conditions: list[DirichletBC] = []
    if not hard_constraint:
        for i, face in enumerate(sample_boundary_faces(d, n_boundary_per_face, method=sampling_method, seed=seed + 1)):
            boundary_conditions.append(
                DirichletBC(face.points, lambda p: torch.zeros(p.shape[0], 1), name=f"edge_{i}")
            )

    return Problem(
        domain=d,
        pde=Poisson2D(),
        boundary_conditions=boundary_conditions,
        interior_points=interior,
        resampler=None,
        exact_solution=exact_solution,
    )
