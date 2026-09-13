"""The generalized training loop.

Same overall shape as the baseline ``PINNTrainer`` (Adam, then optional
L-BFGS refinement, with periodic logging) but assembled from swappable
pieces instead of the heat equation's three hardcoded terms: any ``PDE``,
any list of ``BoundaryCondition`` objects, any ``LossWeighter``, and
optionally a ``CausalWeighter`` and an ``AdaptiveResampler``.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Callable

import torch
from torch import nn

from .causal import CausalWeighter
from .problem import Problem
from .weighting import LossWeighter, PerPointTerms

Tensor = torch.Tensor
History = dict[str, list[float]]


@dataclass
class GeneralPINNConfig:
    """Training hyperparameters for the generalized trainer."""

    epochs: int = 3_000
    learning_rate: float = 1e-3
    log_every: int = 250
    use_lbfgs: bool = False
    lbfgs_steps: int = 250
    seed: int = 42
    resample_every: int | None = None
    n_resample: int = 0
    causal: CausalWeighter | None = None

    def __post_init__(self) -> None:
        if self.epochs < 1:
            raise ValueError("epochs must be >= 1")
        if self.resample_every is not None and self.n_resample < 1:
            raise ValueError("n_resample must be >= 1 when resample_every is set")


class GeneralTrainer:
    """Optimize a network against an arbitrary PDE + boundary-condition set."""

    def __init__(
        self,
        model: nn.Module,
        problem: Problem,
        weighter: LossWeighter,
        config: GeneralPINNConfig | None = None,
    ) -> None:
        self.model = model
        self.problem = problem
        self.weighter = weighter
        self.config = config or GeneralPINNConfig()
        torch.manual_seed(self.config.seed)
        self.interior = problem.interior_points.clone()

    def _per_point_terms(self) -> PerPointTerms:
        interior = self.interior.clone().detach().requires_grad_(True)
        residual = self.problem.pde.residual(self.model, interior)
        residual_sq = residual.square().sum(dim=1, keepdim=True)

        causal = self.config.causal
        if causal is not None and self.problem.domain.is_time_dependent:
            time_index = self.problem.domain.time_index
            times = interior[:, time_index : time_index + 1]
            weight = causal.weight(residual_sq, times)
            residual_sq = weight * residual_sq

        terms: PerPointTerms = {"physics": residual_sq}
        for bc in self.problem.boundary_conditions:
            error = bc.per_point_error(self.model)
            terms[bc.name] = error.square().sum(dim=1, keepdim=True)
        return terms

    def train(self, callback: Callable[[int, History], None] | None = None) -> History:
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.learning_rate)
        history: History = defaultdict(list)

        self.model.train()
        for epoch in range(1, self.config.epochs + 1):
            if (
                self.config.resample_every is not None
                and self.problem.resampler is not None
                and epoch % self.config.resample_every == 0
            ):
                self.interior = self.problem.resampler.refresh(
                    self.problem.pde.residual, self.model, self.config.n_resample
                )

            optimizer.zero_grad(set_to_none=True)
            terms = self._per_point_terms()
            total = self.weighter.combine(terms, self.model)
            total.backward()
            optimizer.step()
            self.weighter.after_step()

            history["total"].append(float(total.detach()))
            for name, value in terms.items():
                history[name].append(float(value.mean().detach()))

            if callback is not None and (
                epoch == 1 or epoch % self.config.log_every == 0 or epoch == self.config.epochs
            ):
                callback(epoch, history)

        if self.config.use_lbfgs:
            self._lbfgs_refinement()

        return dict(history)

    def _lbfgs_refinement(self) -> None:
        optimizer = torch.optim.LBFGS(
            self.model.parameters(),
            max_iter=self.config.lbfgs_steps,
            tolerance_grad=1e-9,
            tolerance_change=1e-11,
            history_size=50,
            line_search_fn="strong_wolfe",
        )

        def closure() -> Tensor:
            optimizer.zero_grad(set_to_none=True)
            terms = self._per_point_terms()
            total = self.weighter.combine(terms, self.model)
            total.backward()
            return total

        optimizer.step(closure)

    @torch.no_grad()
    def predict(self, points: Tensor) -> Tensor:
        self.model.eval()
        return self.model(points)
