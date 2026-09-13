"""Small autograd helpers shared by every PDE residual.

The original ``heat_residual`` inlined its ``torch.autograd.grad`` calls
because it only ever needed u_x, u_t and u_xx on a fixed 2-column input.
Once a PDE can live in 1D, 2D or 3D, that inlining has to happen for every
new equation. These two functions are the reusable core of it.
"""

from __future__ import annotations

from typing import Sequence

import torch

Tensor = torch.Tensor


def grad(outputs: Tensor, inputs: Tensor) -> Tensor:
    """First derivative of a scalar-per-row output with respect to ``inputs``.

    ``outputs`` must have shape ``(N, 1)`` and ``inputs`` must require grad.
    Returns a tensor shaped like ``inputs``, so column ``j`` of the result is
    d(outputs)/d(inputs[:, j]).
    """
    if outputs.shape[-1] != 1:
        raise ValueError("grad() expects a scalar output per row; got shape " f"{tuple(outputs.shape)}")
    if not inputs.requires_grad:
        raise ValueError("inputs must have requires_grad=True before calling grad()")
    return torch.autograd.grad(
        outputs,
        inputs,
        grad_outputs=torch.ones_like(outputs),
        create_graph=True,
        retain_graph=True,
    )[0]


def laplacian(u: Tensor, coords: Tensor, spatial_indices: Sequence[int]) -> Tensor:
    """sum_i d^2u/dx_i^2 over the given column indices of ``coords``.

    This is the piece that used to be hand-written per equation (u_xx for
    heat, u_xx for Burgers). Passing ``spatial_indices=[0]`` reproduces the
    1D case; ``[0, 1]`` gives the 2D Laplacian used by ``Poisson2D``;
    ``[0, 1, 2]`` extends to 3D with no change to the calling code.
    """
    first = grad(u, coords)
    total = torch.zeros_like(u)
    for i in spatial_indices:
        second = grad(first[:, i : i + 1], coords)[:, i : i + 1]
        total = total + second
    return total
