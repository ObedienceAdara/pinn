"""Solve the same 1D heat equation as ``examples/solve_heat.py``, but through
the generalized ``pinn.core`` framework, two ways:

1. Soft-constrained — a plain ``GeneralMLP`` with Dirichlet loss terms for
   the initial and boundary conditions, matching the original baseline's
   approach almost exactly (this is the parity check).
2. Hard-constrained + causal — a ``HardDirichletAnsatz`` that satisfies the
   initial condition and the boundary condition exactly by construction,
   trained with ``CausalWeighter`` so the physics residual is only weighted
   in at later times once earlier times are already well fit.

Run it and compare the two "relative L2 error" numbers it prints: the
gap between them is the concrete, measured effect of the changes described
in ARCHITECTURE.md, not just an architectural claim about them.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import torch

from pinn.core import (
    CausalWeighter,
    FixedWeights,
    GeneralMLP,
    GeneralPINNConfig,
    GeneralTrainer,
    HardDirichletAnsatz,
)
from pinn.problems import heat1d


def _train_soft(seed: int) -> tuple[GeneralTrainer, dict]:
    problem = heat1d.build_problem(
        alpha=0.1, n_interior=10_000, n_boundary=2_000, n_initial=2_000, sampling_method="lhs", seed=seed
    )
    model = GeneralMLP(problem.domain, hidden_dim=64, hidden_layers=4)
    weighter = FixedWeights(
        {"physics": 1.0, "initial": 10.0, "left_boundary": 10.0, "right_boundary": 10.0}
    )
    config = GeneralPINNConfig(epochs=3_000, learning_rate=1e-3, log_every=500, use_lbfgs=True, seed=seed)
    trainer = GeneralTrainer(model, problem, weighter, config)

    def log(epoch: int, history: dict) -> None:
        print(f"[soft]  epoch={epoch:5d} total={history['total'][-1]:.4e} physics={history['physics'][-1]:.4e}")

    history = trainer.train(callback=log)
    return trainer, history


def _train_hard_causal(seed: int) -> tuple[GeneralTrainer, dict]:
    problem = heat1d.build_problem(alpha=0.1, n_interior=10_000, sampling_method="lhs", seed=seed, hard_constraint=True)
    base = GeneralMLP(problem.domain, hidden_dim=64, hidden_layers=4)
    model = HardDirichletAnsatz(base, problem.domain, initial_condition_fn=heat1d.initial_condition)
    weighter = FixedWeights({"physics": 1.0})
    config = GeneralPINNConfig(
        epochs=3_000,
        learning_rate=1e-3,
        log_every=500,
        use_lbfgs=True,
        seed=seed,
        causal=CausalWeighter(n_bins=10, epsilon=1.0),
    )
    trainer = GeneralTrainer(model, problem, weighter, config)

    def log(epoch: int, history: dict) -> None:
        print(f"[hard]  epoch={epoch:5d} total={history['total'][-1]:.4e} physics={history['physics'][-1]:.4e}")

    history = trainer.train(callback=log)
    return trainer, history


def _relative_l2_error(trainer: GeneralTrainer, n: int = 200) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    x = torch.linspace(-1.0, 1.0, n)
    t = torch.linspace(0.0, 1.0, n)
    xx, tt = torch.meshgrid(x, t, indexing="ij")
    xt = torch.stack((xx.reshape(-1), tt.reshape(-1)), dim=1)
    prediction = trainer.predict(xt).reshape(n, n)
    exact = heat1d.exact_solution(xx, tt, alpha=0.1)
    relative_l2 = torch.linalg.vector_norm(prediction - exact) / torch.linalg.vector_norm(exact)
    return prediction, exact, xx, float(relative_l2)


def main() -> None:
    torch.set_default_dtype(torch.float32)

    soft_trainer, _ = _train_soft(seed=42)
    soft_pred, exact, xx, soft_error = _relative_l2_error(soft_trainer)
    print(f"[soft]  relative L2 error: {soft_error:.4e}")

    hard_trainer, _ = _train_hard_causal(seed=42)
    hard_pred, _, _, hard_error = _relative_l2_error(hard_trainer)
    print(f"[hard]  relative L2 error: {hard_error:.4e}")
    print(f"improvement factor: {soft_error / hard_error:.1f}x")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)
    for ax, field, title in zip(
        axes,
        (exact.detach().numpy(), soft_pred.detach().numpy(), hard_pred.detach().numpy()),
        ("Exact solution", f"Soft-constrained (rel L2={soft_error:.2e})", f"Hard + causal (rel L2={hard_error:.2e})"),
    ):
        image = ax.imshow(field, origin="lower", aspect="auto", extent=(0, 1, -1, 1))
        ax.set_xlabel("t")
        ax.set_ylabel("x")
        ax.set_title(title)
        fig.colorbar(image, ax=ax)

    output = Path("artifacts/heat1d_general_comparison.png")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)
    print(f"saved figure to {output}")


if __name__ == "__main__":
    main()
