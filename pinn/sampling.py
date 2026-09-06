from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class HeatEquationPoints:
    """Collocation, initial-condition and boundary-condition points."""

    interior: torch.Tensor
    initial: torch.Tensor
    left_boundary: torch.Tensor
    right_boundary: torch.Tensor


def _uniform(
    generator: torch.Generator,
    shape: tuple[int, ...],
    low: float,
    high: float,
    dtype: torch.dtype,
) -> torch.Tensor:
    return low + (high - low) * torch.rand(shape, generator=generator, dtype=dtype)


def sample_heat_equation(
    n_interior: int = 10_000,
    n_initial: int = 2_000,
    n_boundary: int = 2_000,
    seed: int = 42,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str = "cpu",
) -> HeatEquationPoints:
    """Sample training points for x in [-1,1], t in [0,1].

    Sampling is performed on CPU with a local generator, so calling this
    function is deterministic without changing PyTorch's global RNG state.
    """
    for name, value in (
        ("n_interior", n_interior),
        ("n_initial", n_initial),
        ("n_boundary", n_boundary),
    ):
        if value < 1:
            raise ValueError(f"{name} must be >= 1")

    generator = torch.Generator(device="cpu").manual_seed(seed)

    x_f = _uniform(generator, (n_interior, 1), -1.0, 1.0, dtype)
    t_f = _uniform(generator, (n_interior, 1), 0.0, 1.0, dtype)
    interior = torch.cat((x_f, t_f), dim=1)

    x_0 = _uniform(generator, (n_initial, 1), -1.0, 1.0, dtype)
    initial = torch.cat((x_0, torch.zeros_like(x_0)), dim=1)

    t_b = _uniform(generator, (n_boundary, 1), 0.0, 1.0, dtype)
    left_boundary = torch.cat((-torch.ones_like(t_b), t_b), dim=1)
    right_boundary = torch.cat((torch.ones_like(t_b), t_b), dim=1)

    return HeatEquationPoints(
        interior=interior.to(device=device),
        initial=initial.to(device=device),
        left_boundary=left_boundary.to(device=device),
        right_boundary=right_boundary.to(device=device),
    )
