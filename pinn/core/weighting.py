"""Ways to combine per-term losses into one scalar to backpropagate.

The baseline used three fixed numbers (``physics_weight=1``,
``initial_weight=10``, ``boundary_weight=10``) tuned by hand for the heat
equation. That stops working once terms have genuinely different gradient
scales — which is the normal case, not the exception, once the PDE isn't a
toy 1D problem. All three strategies below share one interface, so
``GeneralTrainer`` can use any of them without knowing which is active:

    weighter.combine(per_point_terms, model) -> scalar loss

``per_point_terms`` maps a term name ("physics", "initial", "left_wall", ...)
to its *unreduced* per-point squared error, shape (N, 1) — unreduced because
``SelfAdaptiveWeights`` needs to apply a different weight to every point,
not just every term.
"""

from __future__ import annotations

import abc
from typing import Dict

import torch
from torch import nn

Tensor = torch.Tensor
PerPointTerms = Dict[str, Tensor]


class LossWeighter(abc.ABC):
    @abc.abstractmethod
    def combine(self, per_point_terms: PerPointTerms, model: nn.Module) -> Tensor:
        raise NotImplementedError

    def after_step(self) -> None:
        """Optional hook the trainer calls after the model's optimizer step.

        Only ``SelfAdaptiveWeights`` uses this, to run its own gradient
        ASCENT step on the per-point weights once the model's descent step
        for this iteration is done.
        """
        return None


class FixedWeights(LossWeighter):
    """The baseline's behavior: constant, hand-set scalar weight per term."""

    def __init__(self, weights: dict[str, float] | None = None, default: float = 1.0) -> None:
        self.weights = weights or {}
        self.default = default

    def combine(self, per_point_terms: PerPointTerms, model: nn.Module) -> Tensor:
        total: Tensor | float = 0.0
        for name, value in per_point_terms.items():
            total = total + self.weights.get(name, self.default) * torch.mean(value)
        return total


class GradNormWeights(LossWeighter):
    """Rebalances every term so its parameter-gradient norm matches the
    anchor term's, refreshed every ``update_every`` steps (Wang, Teng &
    Perdikaris, 2021 — "Understanding and mitigating gradient pathologies
    in physics-informed neural networks", simplified to a single anchor).

    A term whose gradient is much smaller than the anchor's barely moves
    the network even with weight 1; this scales it up until its influence
    on the parameters matches the anchor's, instead of leaving that to
    hand-tuning.
    """

    def __init__(self, anchor: str = "physics", update_every: int = 100, momentum: float = 0.9) -> None:
        self.anchor = anchor
        self.update_every = update_every
        self.momentum = momentum
        self._weights: dict[str, float] = {}
        self._step = 0

    def combine(self, per_point_terms: PerPointTerms, model: nn.Module) -> Tensor:
        if self.anchor not in per_point_terms:
            raise KeyError(f"anchor term {self.anchor!r} not present in loss terms {list(per_point_terms)}")
        self._step += 1
        if self._step == 1 or self._step % self.update_every == 0:
            self._refresh(per_point_terms, model)

        total = torch.mean(per_point_terms[self.anchor])
        for name, value in per_point_terms.items():
            if name == self.anchor:
                continue
            total = total + self._weights.get(name, 1.0) * torch.mean(value)
        return total

    def _refresh(self, per_point_terms: PerPointTerms, model: nn.Module) -> None:
        params = [p for p in model.parameters() if p.requires_grad]

        def grad_norm(term_loss: Tensor) -> float:
            grads = torch.autograd.grad(term_loss, params, retain_graph=True, allow_unused=True)
            squares = [g.detach().square().sum() for g in grads if g is not None]
            if not squares:
                return 1e-12
            return float(torch.sqrt(torch.stack(squares).sum())) + 1e-12

        anchor_norm = grad_norm(torch.mean(per_point_terms[self.anchor]))
        for name, value in per_point_terms.items():
            if name == self.anchor:
                continue
            term_norm = grad_norm(torch.mean(value))
            target = anchor_norm / term_norm
            previous = self._weights.get(name, target)
            self._weights[name] = self.momentum * previous + (1 - self.momentum) * target


class SelfAdaptiveWeights(LossWeighter):
    """Trainable per-point weights, updated by gradient ASCENT (McClenny &
    Braga-Neto, 2020 — "Self-adaptive physics-informed neural networks").

    The network minimizes the weighted residual while the weights maximize
    it: points the network is currently fitting poorly get an increasing
    weight, which concentrates training effort on them automatically
    instead of relying on where collocation points happened to land.
    """

    def __init__(self, lr: float = 1e-2) -> None:
        self.lr = lr
        self._lambdas: dict[str, nn.Parameter] = {}
        self._optimizer: torch.optim.Optimizer | None = None

    def parameters(self) -> list[nn.Parameter]:
        return list(self._lambdas.values())

    def combine(self, per_point_terms: PerPointTerms, model: nn.Module) -> Tensor:
        total: Tensor | float = 0.0
        rebuild = False
        for name, value in per_point_terms.items():
            existing = self._lambdas.get(name)
            if existing is None or existing.shape[0] != value.shape[0]:
                # First time seeing this term, or its point count changed
                # (e.g. after adaptive resampling) — (re)initialize at 1.0.
                self._lambdas[name] = nn.Parameter(torch.ones_like(value))
                rebuild = True
        if rebuild:
            self._optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, maximize=True)

        for name, value in per_point_terms.items():
            weight = torch.nn.functional.softplus(self._lambdas[name])
            total = total + torch.mean(weight * value)
        return total

    def after_step(self) -> None:
        if self._optimizer is not None:
            self._optimizer.step()
            self._optimizer.zero_grad(set_to_none=True)
