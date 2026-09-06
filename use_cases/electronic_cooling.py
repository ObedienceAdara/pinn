"""Transient thermal diffusion example for an electronics substrate.

The PDE is the same heat equation used by the baseline library, but the
application framing changes the interpretation: temperature is the state,
thermal diffusivity represents an effective substrate property, and the PINN
provides a differentiable temperature field that can be queried without a
fresh numerical time integration for every query point.
"""

from __future__ import annotations

import torch

from pinn import MLP, PINNConfig, PINNTrainer, sample_heat_equation


def main() -> None:
    alpha = 0.15
    points = sample_heat_equation(
        n_interior=2_500,
        n_initial=400,
        n_boundary=400,
        x_min=-1.0,
        x_max=1.0,
        t_max=1.0,
        seed=21,
    )

    model = MLP(input_dim=2, output_dim=1, hidden_dim=64, hidden_layers=4)
    trainer = PINNTrainer(
        model,
        PINNConfig(
            alpha=alpha,
            physics_weight=1.0,
            initial_weight=15.0,
            boundary_weight=15.0,
            learning_rate=1e-3,
            epochs=1_500,
            log_every=500,
            use_lbfgs=True,
            lbfgs_steps=100,
            seed=21,
        ),
    )

    history = trainer.train(points)

    # Treat the coordinate as a normalized location through the substrate.
    # The learned field is queried in a vectorized batch, which is the useful
    # inference pattern for downstream monitoring or optimization loops.
    query = torch.stack(
        torch.meshgrid(
            torch.linspace(-1.0, 1.0, 21),
            torch.linspace(0.0, 1.0, 11),
            indexing="ij",
        ),
        dim=-1,
    ).reshape(-1, 2)
    temperature_field = trainer.predict(query).reshape(21, 11)

    print("Electronics-cooling PINN")
    print(f"final Adam loss: {history.total[-1]:.3e}")
    print(f"predicted field shape: {tuple(temperature_field.shape)}")
    print(
        "temperature range: "
        f"[{temperature_field.min().item():+.6f}, "
        f"{temperature_field.max().item():+.6f}]"
    )


if __name__ == "__main__":
    main()
