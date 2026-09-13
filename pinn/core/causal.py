"""Causal (time-respecting) weighting for transient PDE residuals.

A PINN trained on the whole time domain at once is free to fit late-time
points before it has actually satisfied the equation at early times — which
is backwards, since a transient PDE's solution at time t is causally
determined by its solution at times before t. Wang, Sankaran & Perdikaris
("Respecting causality for training physics-informed neural networks",
2022) fix this by down-weighting the residual at each time until earlier
times are already well fit. This matters once problems go past the heat
equation's short, diffusive time horizon — advection-dominated PDEs and
long rollouts are exactly where a non-causal PINN tends to fail to
converge at all, not just converge slowly.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

Tensor = torch.Tensor


@dataclass
class CausalWeighter:
    """Reweights a physics-residual term by time bin.

    ``epsilon`` controls how strictly later times are held back: 0 recovers
    ordinary (non-causal) uniform weighting; larger values enforce causality
    more strictly but can slow early training since later times are almost
    fully suppressed until earlier ones converge.
    """

    n_bins: int = 10
    epsilon: float = 1.0

    def weight(self, residual_sq: Tensor, times: Tensor) -> Tensor:
        """Per-point weight, shape matching ``residual_sq``, detached from the graph.

        ``residual_sq`` is the current per-point squared physics residual;
        ``times`` is the matching time coordinate for each point. Weights are
        computed from the *current* per-bin loss and then detached, so they
        modulate this step's gradient without being differentiated through
        themselves — consistent with how the original paper applies them.
        """
        if residual_sq.shape[0] != times.shape[0]:
            raise ValueError("residual_sq and times must have the same number of rows")
        times = times.detach()
        t_min = float(times.min())
        t_max = float(times.max())
        if t_max <= t_min:
            return torch.ones_like(residual_sq)

        flat_residual = residual_sq.detach().reshape(-1)
        flat_times = times.reshape(-1).contiguous()
        edges = torch.linspace(t_min, t_max, self.n_bins + 1, device=times.device, dtype=times.dtype)
        bin_idx = torch.bucketize(flat_times, edges[1:-1], right=False)

        bin_losses = torch.zeros(self.n_bins, device=times.device, dtype=times.dtype)
        for b in range(self.n_bins):
            mask = bin_idx == b
            if torch.any(mask):
                bin_losses[b] = flat_residual[mask].mean()

        # Cumulative loss strictly *before* each bin — bin 0 always gets weight 1.
        cumulative_before = torch.cumsum(bin_losses, dim=0) - bin_losses
        bin_weights = torch.exp(-self.epsilon * cumulative_before)
        return bin_weights[bin_idx].reshape(residual_sq.shape)
