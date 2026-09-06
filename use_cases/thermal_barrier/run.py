"""Thermal-barrier example using the public pinn package API."""

from __future__ import annotations

import torch

from pinn import MLP, PINNConfig, PINNTrainer, sample_heat_equation


def main() -> None:
    alpha = 0.05
    points = sample_heat_equation(
        n_interior=2_000,
        n_initial=300,
        n_boundary=300,
        seed=7,
    )

    model = MLP(input_dim=2, output_dim=1, hidden_dim=64, hidden_layers=4)
    trainer = PINNTrainer(
        model,
        PINNConfig(
            alpha=alpha,
            epochs=1_500,
            learning_rate=1e-3,
            initial_weight=20.0,
            boundary_weight=20.0,
            log_every=500,
        ),
    )

    history = trainer.train(points)

    query = torch.tensor(
        [
            [-0.75, 0.10],
            [0.00, 0.10],
            [0.75, 0.10],
            [0.00, 0.50],
            [0.00, 1.00],
        ],
        dtype=torch.float32,
    )
    temperatures = trainer.predict(query).squeeze(-1)

    print("Thermal-barrier PINN")
    print(f"final loss: {history.total[-1]:.3e}")
    for (x, t), temperature in zip(query.tolist(), temperatures.tolist()):
        print(f"x={x:+.2f}, t={t:.2f} -> T={temperature:+.6f}")


if __name__ == "__main__":
    main()
