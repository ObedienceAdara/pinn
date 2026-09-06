"""Electronics-cooling example using the public pinn package API."""

from __future__ import annotations

import torch

from pinn import MLP, PINNConfig, PINNTrainer, sample_heat_equation


def main() -> None:
    alpha = 0.15
    points = sample_heat_equation(
        n_interior=2_500,
        n_initial=400,
        n_boundary=400,
        seed=21,
    )

    model = MLP(input_dim=2, output_dim=1, hidden_dim=64, hidden_layers=4)
    trainer = PINNTrainer(
        model,
        PINNConfig(
            alpha=alpha,
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

    trainer.train(points)

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
    print(f"predicted field shape: {tuple(temperature_field.shape)}")
    print(
        f"temperature range: [{temperature_field.min().item():+.6f}, "
        f"{temperature_field.max().item():+.6f}]"
    )


if __name__ == "__main__":
    main()
