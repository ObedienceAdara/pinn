"""Boundary and initial conditions as first-class, swappable objects.

The baseline trainer computed exactly two boundary terms and one initial
term inline, hardcoded to the heat equation's zero-Dirichlet setup. Every
``BoundaryCondition`` here exposes the same ``per_point_error`` method, so
``GeneralTrainer`` can sum an arbitrary list of them — 2 faces for a 1D
problem, 4 for 2D, 6 for 3D, any mix of Dirichlet/Neumann/periodic — without
knowing anything about the specific PDE.

Note on initial conditions: mathematically, "u(x, 0) = h(x)" is a Dirichlet
condition on the domain's ``t = t0`` face. ``DirichletBC`` is used for both;
give it a descriptive ``name`` ("initial", "left_wall", "inlet", ...) and it
reports under that name in training history and adaptive-weighting output.
"""

from __future__ import annotations

import abc
from typing import Callable

import torch
from torch import nn

from . import autodiff

Tensor = torch.Tensor
ValueFn = Callable[[Tensor], Tensor]


class BoundaryCondition(abc.ABC):
    """A named constraint evaluated on a fixed set of points."""

    name: str

    @abc.abstractmethod
    def per_point_error(self, model: nn.Module) -> Tensor:
        """Raw (unsquared) per-point violation of the condition, shape (N, k)."""
        raise NotImplementedError


class DirichletBC(BoundaryCondition):
    """Pins the model's output to ``value_fn(points)`` on a fixed point set.

    Used for ordinary spatial Dirichlet boundaries ("u = 0 at the wall") and
    equally for initial conditions ("u = h(x) at t = 0") — see module note.
    """

    def __init__(self, points: Tensor, value_fn: ValueFn, name: str = "dirichlet") -> None:
        self.points = points
        self.value_fn = value_fn
        self.name = name

    def per_point_error(self, model: nn.Module) -> Tensor:
        prediction = model(self.points)
        target = self.value_fn(self.points)
        return prediction - target


class NeumannBC(BoundaryCondition):
    """Pins the derivative of the model's output along one input axis.

    ``normal_dim`` is the column of the domain's coordinates the derivative
    is taken with respect to (usually the outward-normal spatial axis of the
    face the points lie on).
    """

    def __init__(self, points: Tensor, normal_dim: int, value_fn: ValueFn, name: str = "neumann") -> None:
        self.points = points.clone().detach().requires_grad_(True)
        self.normal_dim = normal_dim
        self.value_fn = value_fn
        self.name = name

    def per_point_error(self, model: nn.Module) -> Tensor:
        u = model(self.points)
        derivative = autodiff.grad(u, self.points)[:, self.normal_dim : self.normal_dim + 1]
        target = self.value_fn(self.points)
        return derivative - target


class PeriodicBC(BoundaryCondition):
    """Soft periodicity: matches value (and optionally derivative) between
    two point sets that agree on every coordinate except the periodic axis.

    This is the general, works-on-any-domain fallback. When a problem's
    *only* nonstandard condition is periodicity, ``core.networks.PeriodicEmbedding``
    is the better choice — it makes periodicity exact by construction instead
    of penalizing mismatches, at the cost of only applying to that one axis.
    """

    def __init__(
        self,
        points_low: Tensor,
        points_high: Tensor,
        periodic_dim: int,
        match_derivative: bool = False,
        name: str = "periodic",
    ) -> None:
        if points_low.shape != points_high.shape:
            raise ValueError("points_low and points_high must be paired one-to-one")
        needs_grad = match_derivative
        self.points_low = points_low.clone().detach().requires_grad_(needs_grad)
        self.points_high = points_high.clone().detach().requires_grad_(needs_grad)
        self.periodic_dim = periodic_dim
        self.match_derivative = match_derivative
        self.name = name

    def per_point_error(self, model: nn.Module) -> Tensor:
        u_low = model(self.points_low)
        u_high = model(self.points_high)
        value_error = u_low - u_high
        if not self.match_derivative:
            return value_error
        d_low = autodiff.grad(u_low, self.points_low)[:, self.periodic_dim : self.periodic_dim + 1]
        d_high = autodiff.grad(u_high, self.points_high)[:, self.periodic_dim : self.periodic_dim + 1]
        return torch.cat([value_error, d_low - d_high], dim=1)
