from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from pinn import MLP, PINNConfig, PINNTrainer, heat_exact_solution, sample_heat_equation


def main() -> None:
    torch.set_default_dtype(torch.float32)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = MLP(
        input_dim=2,
        output_dim=1,
        hidden_dim=64,
        hidden_layers=4,
        input_lower=(-1.0, 0.0),
        input_upper=(1.0, 1.0),
    ).to(device)

    config = PINNConfig(
        alpha=0.1,
        physics_weight=1.0,
        initial_weight=10.0,
        boundary_weight=10.0,
        learning_rate=1e-3,
        epochs=3_000,
        log_every=250,
        use_lbfgs=True,
        lbfgs_steps=250,
        seed=42,
    )
    points = sample_heat_equation(
        n_interior=10_000,
        n_initial=2_000,
        n_boundary=2_000,
        seed=config.seed,
        device=device,
    )

    trainer = PINNTrainer(model, config)

    def log(epoch: int, history) -> None:
        print(
            f"epoch={epoch:5d} "
            f"total={history.total[-1]:.4e} "
            f"physics={history.physics[-1]:.4e} "
            f"ic={history.initial[-1]:.4e} "
            f"bc={history.boundary[-1]:.4e}"
        )

    history = trainer.train(points, callback=log)

    n = 200
    x = torch.linspace(-1.0, 1.0, n, device=device)
    t = torch.linspace(0.0, 1.0, n, device=device)
    xx, tt = torch.meshgrid(x, t, indexing="ij")
    xt = torch.stack((xx.reshape(-1), tt.reshape(-1)), dim=1)

    prediction = trainer.predict(xt).reshape(n, n)
    exact = heat_exact_solution(xx, tt, config.alpha)
    error = torch.abs(prediction - exact)
    relative_l2 = torch.linalg.vector_norm(prediction - exact) / torch.linalg.vector_norm(exact)

    print(f"relative L2 error: {relative_l2.item():.4e}")

    x_np = x.cpu().numpy()
    t_np = t.cpu().numpy()
    pred_np = prediction.detach().cpu().numpy()
    exact_np = exact.detach().cpu().numpy()
    err_np = error.detach().cpu().numpy()

    fig, axes = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)
    for ax, field, title in zip(
        axes,
        (pred_np, exact_np, err_np),
        ("PINN prediction", "Exact solution", "Absolute error"),
    ):
        image = ax.imshow(
            field,
            extent=(t_np.min(), t_np.max(), x_np.min(), x_np.max()),
            origin="lower",
            aspect="auto",
        )
        ax.set_xlabel("t")
        ax.set_ylabel("x")
        ax.set_title(title)
        fig.colorbar(image, ax=ax)

    output = Path("artifacts/heat_equation_solution.png")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)
    print(f"saved figure to {output}")


if __name__ == "__main__":
    main()
