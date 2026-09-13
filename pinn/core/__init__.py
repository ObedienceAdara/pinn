"""Generalized, dimension-agnostic PINN building blocks.

This package is the framework the ``pinn`` baseline generalizes into:
arbitrary-dimension domains, a swappable PDE/boundary-condition interface,
Latin Hypercube/Sobol/adaptive sampling, hard-constraint and Fourier-feature
network options, gradient-aware and self-adaptive loss weighting, and
causal training for time-dependent PDEs. See ``pinn.problems`` for concrete
equations built on top of it, and ``ARCHITECTURE.md`` at the repo root for
the design rationale and known limitations.
"""

from . import autodiff
from .boundary import BoundaryCondition, DirichletBC, NeumannBC, PeriodicBC
from .causal import CausalWeighter
from .domain import Domain
from .networks import FourierFeatures, FourierMLP, GeneralMLP, HardDirichletAnsatz, PeriodicEmbedding
from .pde import PDE
from .problem import Problem
from .sampling import AdaptiveResampler, FaceSample, sample_boundary_faces, sample_initial, sample_interior
from .trainer import GeneralPINNConfig, GeneralTrainer
from .weighting import FixedWeights, GradNormWeights, LossWeighter, SelfAdaptiveWeights

__all__ = [
    "autodiff",
    "BoundaryCondition",
    "DirichletBC",
    "NeumannBC",
    "PeriodicBC",
    "CausalWeighter",
    "Domain",
    "FourierFeatures",
    "FourierMLP",
    "GeneralMLP",
    "HardDirichletAnsatz",
    "PeriodicEmbedding",
    "PDE",
    "Problem",
    "AdaptiveResampler",
    "FaceSample",
    "sample_boundary_faces",
    "sample_initial",
    "sample_interior",
    "GeneralPINNConfig",
    "GeneralTrainer",
    "FixedWeights",
    "GradNormWeights",
    "LossWeighter",
    "SelfAdaptiveWeights",
]
