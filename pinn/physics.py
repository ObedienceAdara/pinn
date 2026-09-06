from __future__ import annotations

import torch
from torch import nn


def heat_residual(
    model: nn.Module,
    xt: torch.Tensor,
    alpha: float,
) -> torch.Tensor:
    """Return the 1D heat-equation residual u_t - alpha * u_xx.

    ``xt`` must have columns ``[x, t]`` and requires gradients because the
    residual is evaluated through PyTorch automatic differentiation.
    """
    if xt.ndim != 2 or xt.shape[1] != 2:
        raise ValueError("xt must have shape (N, 2) with columns [x, t]")
    if not xt.requires_grad:
        xt = xt.clone().detach().requires_grad_(True)

    u = model(xt)
    if u.shape[-1] != 1:
        raise ValueError("heat_equation model must return one scalar per point")

    grad_u = torch.autograd.grad(
        u,
        xt,
        grad_outputs=torch.ones_like(u),
        create_graph=True,
        retain_graph=True,
    )[0]
    u_x = grad_u[:, 0:1]
    u_t = grad_u[:, 1:2]
    u_xx = torch.autograd.grad(
        u_x,
        xt,
        grad_outputs=torch.ones_like(u_x),
        create_graph=True,
        retain_graph=True,
    )[0][:, 0:1]
    return u_t - alpha * u_xx


def heat_initial_condition(x: torch.Tensor) -> torch.Tensor:
    """Initial condition u(x, 0) = sin(pi*x)."""
    return torch.sin(torch.pi * x)


def heat_boundary_condition(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """Homogeneous Dirichlet boundary values at x = +/- 1."""
    del x
    return torch.zeros_like(t)


def heat_exact_solution(x: torch.Tensor, t: torch.Tensor, alpha: float) -> torch.Tensor:
    """Analytical solution for the reference heat-equation problem."""
    return torch.exp(-alpha * torch.pi**2 * t) * torch.sin(torch.pi * x)
