"""PINN example for a 1D viscous convection-diffusion flow.

This case uses the public ``pinn.MLP`` model and builds a custom residual for
Burgers' equation because the baseline trainer intentionally targets the heat
equation. It demonstrates the extension path for fluid and aerospace models.

PDE:
    u_t + u * u_x = nu * u_xx
"""

from __future__ import annotations

import torch
from torch import Tensor

from pinn import MLP


def burgers_residual(model: MLP, xt: Tensor, nu: float) -> Tensor:
    """Return the Burgers-equation residual at collocation points."""
    if xt.ndim != 2 or xt.shape[1] != 2:
        raise ValueError("xt must have shape (N, 2) with columns [x, t]")
    xt = xt.clone().detach().requires_grad_(True)
    u = model(xt)

    grad_u = torch.autograd.grad(
        u,
        xt,
        grad_outputs=torch.ones_like(u),
        create_graph=True,
    )[0]
    u_x = grad_u[:, 0:1]
    u_t = grad_u[:, 1:2]
    u_xx = torch.autograd.grad(
        u_x,
        xt,
        grad_outputs=torch.ones_like(u_x),
        create_graph=True,
    )[0][:, 0:1]

    return u_t + u * u_x - nu * u_xx


def main() -> None:
    torch.manual_seed(11)
    nu = 0.01

    model = MLP(input_dim=2, output_dim=1, hidden_dim=64, hidden_layers=4)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Collocation points in a normalized spatial/time domain.
    n_interior = 4_000
    interior = torch.rand(n_interior, 2)
    interior[:, 0] = 2.0 * interior[:, 0] - 1.0

    x_initial = torch.linspace(-1.0, 1.0, 400).unsqueeze(1)
    t_initial = torch.zeros_like(x_initial)
    initial = torch.cat([x_initial, t_initial], dim=1)
    initial_target = -torch.sin(torch.pi * x_initial)

    # Dirichlet boundaries are held at zero.
    t_boundary = torch.rand(400, 1)
    left = torch.cat([-torch.ones_like(t_boundary), t_boundary], dim=1)
    right = torch.cat([torch.ones_like(t_boundary), t_boundary], dim=1)

    model.train()
    for epoch in range(1, 1_501):
        optimizer.zero_grad(set_to_none=True)

        residual = burgers_residual(model, interior, nu)
        physics_loss = residual.square().mean()

        initial_loss = (model(initial) - initial_target).square().mean()
        boundary_loss = 0.5 * (
            model(left).square().mean() + model(right).square().mean()
        )

        loss = physics_loss + 10.0 * initial_loss + 10.0 * boundary_loss
        loss.backward()
        optimizer.step()

        if epoch in {1, 500, 1_000, 1_500}:
            print(
                f"epoch={epoch:4d} "
                f"total={loss.item():.3e} "
                f"physics={physics_loss.item():.3e} "
                f"initial={initial_loss.item():.3e} "
                f"boundary={boundary_loss.item():.3e}"
            )

    # Vectorized inference on a space/time grid.
    with torch.no_grad():
        query = torch.stack(
            torch.meshgrid(
                torch.linspace(-1.0, 1.0, 41),
                torch.linspace(0.0, 1.0, 21),
                indexing="ij",
            ),
            dim=-1,
        ).reshape(-1, 2)
        field = model(query).reshape(41, 21)

    print("Viscous-flow PINN")
    print(f"predicted velocity-field shape: {tuple(field.shape)}")
    print(f"predicted velocity range: [{field.min().item():+.6f}, {field.max().item():+.6f}]")


if __name__ == "__main__":
    main()
