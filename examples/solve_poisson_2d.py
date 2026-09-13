"""Solve the 2D Poisson equation -Laplacian(u) = f on the unit square.

This is the smallest problem the original 1D-only baseline could not
express at all: two spatial dimensions, four boundary edges instead of two
endpoints, and a Laplacian built from two second-partials instead of one.
Nothing in ``pinn.core`` needed to change to add it — only a new ``PDE``
subclass and a domain with two spatial bounds instead of one
(``pinn/problems/poisson2d.py``).

The manufactured solution sin(pi x) sin(pi y) is known in closed form, so
both the hard-constrained and soft-constrained runs are checked against
ground truth, not just against each other.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import torch

from pinn.core import FixedWeights, GeneralMLP, GeneralPINNConfig, GeneralTrainer, HardDirichletAnsatz
from pinn.problems import poisson2d


def _train(hard_constraint: bool, seed: int = 0) -> tuple[GeneralTrainer, float]:
    problem = poisson2d.build_problem(
        n_interior=4_000, n_boundary_per_face=400, sampling_method="sobol", seed=seed, hard_constraint=hard_constraint
    )

    if hard_constraint:
        base = GeneralMLP(problem.domain, hidden_dim=32, hidden_layers=3)
        model: torch.nn.Module = HardDirichletAnsatz(
            base, problem.domain, boundary_value_fn=lambda p: torch.zeros(p.shape[0], 1)
        )
        weighter = FixedWeights({"physics": 1.0})
    else:
        model = GeneralMLP(problem.domain, hidden_dim=32, hidden_layers=3)
        weights = {"physics": 1.0}
        for bc in problem.boundary_conditions:
            weights[bc.name] = 20.0
        weighter = FixedWeights(weights)

    config = GeneralPINNConfig(epochs=2_000, learning_rate=2e-3, log_every=500, use_lbfgs=True, seed=seed)
    trainer = GeneralTrainer(model, problem, weighter, config)

    tag = "hard" if hard_constraint else "soft"

    def log(epoch: int, history: dict) -> None:
        print(f"[{tag}] epoch={epoch:5d} total={history['total'][-1]:.4e} physics={history['physics'][-1]:.4e}")

    trainer.train(callback=log)

    grid = torch.linspace(0.0, 1.0, 100)
    xx, yy = torch.meshgrid(grid, grid, indexing="ij")
    xy = torch.stack((xx.reshape(-1), yy.reshape(-1)), dim=1)
    prediction = trainer.predict(xy).reshape(100, 100)
    exact = poisson2d.exact_solution(xy).reshape(100, 100)
    relative_l2 = float(torch.linalg.vector_norm(prediction - exact) / torch.linalg.vector_norm(exact))
    return trainer, relative_l2


def main() -> None:
    torch.set_default_dtype(torch.float32)

    hard_trainer, hard_error = _train(hard_constraint=True)
    print(f"[hard] relative L2 error: {hard_error:.4e}")

    soft_trainer, soft_error = _train(hard_constraint=False)
    print(f"[soft] relative L2 error: {soft_error:.4e}")
    print(f"improvement factor: {soft_error / hard_error:.1f}x")

    grid = torch.linspace(0.0, 1.0, 100)
    xx, yy = torch.meshgrid(grid, grid, indexing="ij")
    xy = torch.stack((xx.reshape(-1), yy.reshape(-1)), dim=1)
    exact = poisson2d.exact_solution(xy).reshape(100, 100).detach().numpy()
    hard_pred = hard_trainer.predict(xy).reshape(100, 100).detach().numpy()
    soft_pred = soft_trainer.predict(xy).reshape(100, 100).detach().numpy()

    fig, axes = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)
    for ax, field, title in zip(
        axes,
        (exact, hard_pred, soft_pred),
        ("Exact solution", f"Hard-constrained (rel L2={hard_error:.2e})", f"Soft-constrained (rel L2={soft_error:.2e})"),
    ):
        image = ax.imshow(field, origin="lower", extent=(0, 1, 0, 1))
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title(title)
        fig.colorbar(image, ax=ax)

    output = Path("artifacts/poisson2d_solution.png")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)
    print(f"saved figure to {output}")


if __name__ == "__main__":
    main()
