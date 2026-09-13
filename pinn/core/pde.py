"""The interface any PDE plugs into the generalized trainer through.

The baseline had one free function, ``heat_residual``, wired directly into
``PINNTrainer``. Adding a second equation (as the viscous-flow use case
already needed to) meant copy-pasting the training loop. ``PDE`` is the
seam that removes that duplication: implement ``residual`` once and
``GeneralTrainer`` can train it regardless of how many spatial dimensions
or output components the equation has.
"""

from __future__ import annotations

import abc

import torch
from torch import nn

Tensor = torch.Tensor


class PDE(abc.ABC):
    """A partial differential equation expressed as a residual.

    ``residual`` should return a tensor that is exactly zero when ``model``
    is the true solution, evaluated pointwise. Shape is ``(N, k)`` where
    ``k`` is the number of scalar equations being enforced (1 for a scalar
    equation like heat or Burgers, more for a coupled system like
    Navier-Stokes continuity + momentum).

    Implementations should NOT set ``requires_grad`` on their input
    themselves — the trainer is responsible for that, so the same points
    can be reused across several loss terms and resampling logic.
    """

    @abc.abstractmethod
    def residual(self, model: nn.Module, points: Tensor) -> Tensor:
        raise NotImplementedError
