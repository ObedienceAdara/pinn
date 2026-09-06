from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import torch
from torch import nn

from .physics import heat_boundary_condition, heat_initial_condition, heat_residual
from .sampling import HeatEquationPoints


@dataclass
class PINNConfig:
    """Training hyperparameters for the standard PINN baseline."""

    alpha: float = 0.1
    physics_weight: float = 1.0
    initial_weight: float = 10.0
    boundary_weight: float = 10.0
    learning_rate: float = 1e-3
    epochs: int = 3_000
    log_every: int = 250
    use_lbfgs: bool = False
    lbfgs_steps: int = 250
    seed: int = 42


@dataclass
class TrainingHistory:
    """Scalar loss history recorded during training."""

    total: list[float] = field(default_factory=list)
    physics: list[float] = field(default_factory=list)
    initial: list[float] = field(default_factory=list)
    boundary: list[float] = field(default_factory=list)


class PINNTrainer:
    """Optimize a neural-network PDE solution with physics and constraints."""

    def __init__(self, model: nn.Module, config: PINNConfig | None = None) -> None:
        self.model = model
        self.config = config or PINNConfig()
        if self.config.alpha <= 0:
            raise ValueError("alpha must be > 0")
        if self.config.epochs < 1:
            raise ValueError("epochs must be >= 1")
        if min(
            self.config.physics_weight,
            self.config.initial_weight,
            self.config.boundary_weight,
        ) < 0:
            raise ValueError("loss weights must be non-negative")
        torch.manual_seed(self.config.seed)

    def loss_components(self, points: HeatEquationPoints) -> tuple[torch.Tensor, ...]:
        """Compute total, physics, initial and boundary losses."""
        interior = points.interior.clone().detach().requires_grad_(True)
        residual = heat_residual(self.model, interior, self.config.alpha)
        physics_loss = torch.mean(residual.square())

        initial_pred = self.model(points.initial)
        initial_target = heat_initial_condition(points.initial[:, 0:1])
        initial_loss = torch.mean((initial_pred - initial_target).square())

        left_pred = self.model(points.left_boundary)
        right_pred = self.model(points.right_boundary)
        left_target = heat_boundary_condition(
            points.left_boundary[:, 0:1], points.left_boundary[:, 1:2]
        )
        right_target = heat_boundary_condition(
            points.right_boundary[:, 0:1], points.right_boundary[:, 1:2]
        )
        boundary_loss = torch.mean((left_pred - left_target).square())
        boundary_loss = boundary_loss + torch.mean((right_pred - right_target).square())
        boundary_loss = 0.5 * boundary_loss

        total = (
            self.config.physics_weight * physics_loss
            + self.config.initial_weight * initial_loss
            + self.config.boundary_weight * boundary_loss
        )
        return total, physics_loss, initial_loss, boundary_loss

    def train(
        self,
        points: HeatEquationPoints,
        callback: Callable[[int, TrainingHistory], None] | None = None,
    ) -> TrainingHistory:
        """Train with Adam and optionally an L-BFGS refinement stage."""
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.learning_rate)
        history = TrainingHistory()

        self.model.train()
        for epoch in range(1, self.config.epochs + 1):
            optimizer.zero_grad(set_to_none=True)
            total, physics, initial, boundary = self.loss_components(points)
            total.backward()
            optimizer.step()

            history.total.append(float(total.detach()))
            history.physics.append(float(physics.detach()))
            history.initial.append(float(initial.detach()))
            history.boundary.append(float(boundary.detach()))

            if callback is not None and (
                epoch == 1 or epoch % self.config.log_every == 0 or epoch == self.config.epochs
            ):
                callback(epoch, history)

        if self.config.use_lbfgs:
            self._lbfgs_refinement(points)

        return history

    def _lbfgs_refinement(self, points: HeatEquationPoints) -> None:
        """Optional quasi-Newton refinement commonly used after Adam in PINNs."""
        optimizer = torch.optim.LBFGS(
            self.model.parameters(),
            max_iter=self.config.lbfgs_steps,
            tolerance_grad=1e-9,
            tolerance_change=1e-11,
            history_size=50,
            line_search_fn="strong_wolfe",
        )

        def closure() -> torch.Tensor:
            optimizer.zero_grad(set_to_none=True)
            total, *_ = self.loss_components(points)
            total.backward()
            return total

        optimizer.step(closure)

    @torch.no_grad()
    def predict(self, xt: torch.Tensor) -> torch.Tensor:
        """Evaluate the trained solution network."""
        self.model.eval()
        return self.model(xt)
