"""Standard physics-informed neural network components."""

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
]
