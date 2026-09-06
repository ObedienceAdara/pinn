"""Transient conduction through a simplified thermal-barrier layer.

Engineering interpretation:
    A thin insulation/thermal-protection layer receives a temperature pulse on
    one face while both faces are constrained by an idealized thermal boundary.

The implementation deliberately uses the public PINN API just as an application
would: the domain is sampled, the model is created, the trainer is configured,
and the trained network is queried for temperatures.
"""

from __future__ import annotations

import torch

from pinn import MLP, PINNConfig, PINNTrainer, sample_heat_equation


def main() -> None:
    # Dimensionless x in [-1, 1] and t in [0, 1]. The diffusivity is a
    # dimensionless effective value; mapping to physical units is application
    # specific and belongs in the surrounding engineering model.
    alpha = 0.05
    points = sample_heat_equation(
        n_interior=2_000,
        n_initial=300,
        n_boundary=300,
        x_min=-1.0,
        x_max=1.0,
        t_max=1.0,
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

    # Query the learned temperature field at selected locations/times.
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
    print(f"final training loss: {history.total[-1]:.3e}")
    for (x, t), temperature in zip(query.tolist(), temperatures.tolist()):
        print(f"x={x:+.2f}, t={t:.2f} -> T={temperature:+.6f}")


if __name__ == "__main__":
    main()
