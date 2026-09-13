"""Solve the 1D viscous Burgers equation through the generalized framework.

``use_cases/viscous_flow/run.py`` hand-wrote its own training loop because
the original ``PINNTrainer`` was locked to the heat equation's three loss
terms. Here the same equation plugs into ``GeneralTrainer`` directly, and
gets adaptive resampling, causal weighting and L-BFGS refinement with no
extra training-loop code — only ``pinn/problems/burgers1d.py``, which is
~25 lines of PDE-specific logic.

Burgers with a low viscosity is a genuinely harder problem than the heat
equation: it develops a near-shock (a thin region of very steep gradient)
around t ~ 0.3-0.4, which is exactly the kind of advection-dominated
behavior causal training was built for. Compare the printed initial-
condition error with and without ``causal=CausalWeighter(...)`` to see the
effect directly.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import torch

from pinn.core import CausalWeighter, FixedWeights, GeneralMLP, GeneralPINNConfig, GeneralTrainer
from pinn.problems import burgers1d


def main() -> None:
    torch.set_default_dtype(torch.float32)
    seed = 1

    problem = burgers1d.build_problem(
        viscosity=0.01, n_interior=8_000, n_boundary=800, n_initial=800, sampling_method="lhs", seed=seed
    )
    model = GeneralMLP(problem.domain, hidden_dim=32, hidden_layers=4)
    weighter = FixedWeights(
        {"physics": 1.0, "initial": 10.0, "left_boundary": 10.0, "right_boundary": 10.0}
    )
    config = GeneralPINNConfig(
        epochs=5_000,
        learning_rate=2e-3,
        log_every=500,
        seed=seed,
        resample_every=500,
        n_resample=2_000,
        causal=CausalWeighter(n_bins=10, epsilon=1.0),
        use_lbfgs=True,
        lbfgs_steps=300,
    )
    trainer = GeneralTrainer(model, problem, weighter, config)

    def log(epoch: int, history: dict) -> None:
        print(
            f"epoch={epoch:5d} total={history['total'][-1]:.4e} "
            f"physics={history['physics'][-1]:.4e} initial={history['initial'][-1]:.4e}"
        )

    trainer.train(callback=log)

    x = torch.linspace(-1.0, 1.0, 40).unsqueeze(1)
    t0 = torch.zeros_like(x)
    predicted_ic = trainer.predict(torch.cat([x, t0], dim=1))
    target_ic = burgers1d.initial_condition(x)
    ic_error = float((predicted_ic - target_ic).abs().max())
    print(f"initial-condition max error after training: {ic_error:.4e}")

    n = 200
    x_grid = torch.linspace(-1.0, 1.0, n)
    t_grid = torch.linspace(0.0, 1.0, n)
    xx, tt = torch.meshgrid(x_grid, t_grid, indexing="ij")
    xt = torch.stack((xx.reshape(-1), tt.reshape(-1)), dim=1)
    prediction = trainer.predict(xt).reshape(n, n).detach().numpy()

    fig, ax = plt.subplots(figsize=(6, 5), constrained_layout=True)
    image = ax.imshow(prediction, origin="lower", aspect="auto", extent=(0, 1, -1, 1))
    ax.set_xlabel("t")
    ax.set_ylabel("x")
    ax.set_title("Burgers' equation — PINN solution")
    fig.colorbar(image, ax=ax)

    output = Path("artifacts/burgers1d_general_solution.png")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)
    print(f"saved figure to {output}")


if __name__ == "__main__":
    main()
