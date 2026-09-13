"""1D viscous Burgers equation: u_t + u*u_x - viscosity*u_xx = 0.

Same equation and initial/boundary data as ``use_cases/viscous_flow``,
rebuilt on the generalized framework: that use case hand-wrote its own
training loop because the original ``pinn.trainer.PINNTrainer`` was locked
to the heat equation's loss terms. Here the same physics plugs into
``GeneralTrainer`` directly, and can use adaptive resampling, causal
weighting, or a hard-constrained ansatz with no extra training-loop code.
"""

from __future__ import annotations

import torch
from torch import nn

from ..core.autodiff import grad
from ..core.boundary import DirichletBC
from ..core.domain import Domain
from ..core.pde import PDE
from ..core.problem import Problem
from ..core.sampling import AdaptiveResampler, sample_boundary_faces, sample_initial, sample_interior

Tensor = torch.Tensor


class Burgers1D(PDE):
    def __init__(self, viscosity: float = 0.01) -> None:
        if viscosity <= 0:
            raise ValueError("viscosity must be > 0")
        self.viscosity = viscosity

    def residual(self, model: nn.Module, points: Tensor) -> Tensor:
        u = model(points)
        derivatives = grad(u, points)
        u_x = derivatives[:, 0:1]
        u_t = derivatives[:, 1:2]
        u_xx = grad(u_x, points)[:, 0:1]
        return u_t + u * u_x - self.viscosity * u_xx


def initial_condition(x: Tensor) -> Tensor:
    return -torch.sin(torch.pi * x)


def domain() -> Domain:
    return Domain(spatial_bounds=[(-1.0, 1.0)], time_bounds=(0.0, 1.0))


def build_problem(
    viscosity: float = 0.01,
    n_interior: int = 8_000,
    n_boundary: int = 800,
    n_initial: int = 800,
    sampling_method: str = "lhs",
    seed: int = 11,
    with_resampler: bool = True,
) -> Problem:
    d = domain()
    interior = sample_interior(d, n_interior, method=sampling_method, seed=seed)

    initial_points = sample_initial(d, n_initial, method=sampling_method, seed=seed + 1)
    boundary_conditions: list[DirichletBC] = [
        DirichletBC(initial_points, lambda p: initial_condition(p[:, 0:1]), name="initial")
    ]
    for face in sample_boundary_faces(d, n_boundary, method=sampling_method, seed=seed + 2):
        side_name = "left" if face.side < 0 else "right"
        boundary_conditions.append(
            DirichletBC(face.points, lambda p: torch.zeros(p.shape[0], 1), name=f"{side_name}_boundary")
        )

    resampler = AdaptiveResampler(domain=d, seed=seed + 3) if with_resampler else None

    return Problem(
        domain=d,
        pde=Burgers1D(viscosity=viscosity),
        boundary_conditions=boundary_conditions,
        interior_points=interior,
        resampler=resampler,
        exact_solution=None,  # Burgers has no simple closed form for this IC/viscosity
    )
