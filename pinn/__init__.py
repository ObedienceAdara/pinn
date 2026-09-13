"""Physics-informed neural network components.

Two layers live in this package:

- The original 1D heat-equation baseline (``MLP``, ``heat_residual``,
  ``sample_heat_equation``, ``PINNTrainer``, ...), re-exported here
  unchanged for backward compatibility.
- ``pinn.core``, a generalized framework (arbitrary-dimension domains,
  swappable PDE/boundary-condition/weighting/sampling strategies) that the
  baseline is one small instance of, plus ``pinn.problems`` with concrete
  equations (heat, Poisson, Burgers) built on it. See ``ARCHITECTURE.md``.
"""

from . import core, problems
from .models import MLP
from .physics import heat_residual, heat_exact_solution
from .sampling import HeatEquationPoints, sample_heat_equation
from .trainer import PINNConfig, PINNTrainer

__all__ = [
    "MLP",
    "heat_residual",
    "heat_exact_solution",
    "HeatEquationPoints",
    "sample_heat_equation",
    "PINNConfig",
    "PINNTrainer",
    "core",
    "problems",
]
