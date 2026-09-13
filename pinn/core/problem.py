"""Bundles everything ``GeneralTrainer`` needs for one PDE problem.

Living in ``pinn.core`` (rather than ``pinn.problems``) keeps the dependency
direction one-way: ``pinn.problems.*`` modules import from ``pinn.core``,
never the reverse.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch

from .boundary import BoundaryCondition
from .domain import Domain
from .pde import PDE
from .sampling import AdaptiveResampler

Tensor = torch.Tensor


@dataclass
class Problem:
    """Everything needed to train and evaluate one PDE problem."""

    domain: Domain
    pde: PDE
    boundary_conditions: list[BoundaryCondition]
    interior_points: Tensor
    resampler: AdaptiveResampler | None = None
    exact_solution: Callable[[Tensor], Tensor] | None = None
