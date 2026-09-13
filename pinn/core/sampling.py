"""Collocation-point sampling for an arbitrary-dimension ``Domain``.

The baseline sampler drew ``x`` and ``t`` independently with ``torch.rand``.
That is fine in 2D but wastes points fast as dimension grows — plain uniform
random sampling leaves visible gaps in a cube that it wouldn't leave in a
square. Latin Hypercube and Sobol sampling both spread points far more
evenly per dimension, which is the standard fix and the reason every
serious PINN implementation offers them as options.

This module also generalizes the "resample where the residual is worst"
idea from the project's 1D notebook into ``AdaptiveResampler``, which works
for any PDE/domain pair.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Literal

import torch
from torch.quasirandom import SobolEngine

from .domain import Domain

Tensor = torch.Tensor
SamplingMethod = Literal["uniform", "lhs", "sobol"]


def _unit_uniform(n: int, dim: int, generator: torch.Generator, dtype: torch.dtype) -> Tensor:
    return torch.rand((n, dim), generator=generator, dtype=dtype)


def _unit_lhs(n: int, dim: int, generator: torch.Generator, dtype: torch.dtype) -> Tensor:
    """Latin Hypercube samples on [0, 1)^dim.

    Each dimension is split into ``n`` equal strata; one sample is drawn
    (uniformly, jittered) from a random permutation of the strata for every
    dimension independently. Every dimension is therefore covered exactly
    once per stratum, which is what keeps LHS from leaving the sampling
    holes that plain uniform random sampling does in higher dimension.
    """
    strata = torch.arange(n, dtype=dtype).unsqueeze(1).expand(n, dim).clone()
    for j in range(dim):
        perm = torch.randperm(n, generator=generator)
        strata[:, j] = strata[perm, j]
    jitter = torch.rand((n, dim), generator=generator, dtype=dtype)
    return (strata + jitter) / n


def _unit_sobol(n: int, dim: int, seed: int) -> Tensor:
    """Sobol' low-discrepancy samples on [0, 1)^dim, scrambled for randomization."""
    engine = SobolEngine(dimension=dim, scramble=True, seed=seed)
    return engine.draw(n, dtype=torch.get_default_dtype())


def _sample_unit_box(
    n: int,
    dim: int,
    method: SamplingMethod,
    generator: torch.Generator,
    seed: int,
    dtype: torch.dtype,
) -> Tensor:
    if method == "uniform":
        return _unit_uniform(n, dim, generator, dtype)
    if method == "lhs":
        return _unit_lhs(n, dim, generator, dtype)
    if method == "sobol":
        return _unit_sobol(n, dim, seed)
    raise ValueError(f"unknown sampling method {method!r}; expected 'uniform', 'lhs' or 'sobol'")


def _scale(unit_points: Tensor, lower: tuple[float, ...], upper: tuple[float, ...]) -> Tensor:
    low = torch.as_tensor(lower, dtype=unit_points.dtype)
    high = torch.as_tensor(upper, dtype=unit_points.dtype)
    return low + (high - low) * unit_points


def sample_interior(
    domain: Domain,
    n: int,
    method: SamplingMethod = "lhs",
    seed: int = 42,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str = "cpu",
) -> Tensor:
    """Sample ``n`` collocation points from the interior of ``domain``."""
    if n < 1:
        raise ValueError("n must be >= 1")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    unit = _sample_unit_box(n, domain.input_dim, method, generator, seed, dtype)
    points = _scale(unit, domain.lower(), domain.upper())
    return points.to(device=device)


def sample_initial(
    domain: Domain,
    n: int,
    method: SamplingMethod = "lhs",
    seed: int = 42,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str = "cpu",
) -> Tensor:
    """Sample ``n`` points on the initial-time slice ``t = t0``.

    Only valid for a time-dependent domain. Spatial coordinates are sampled
    across the full spatial box; the time column is fixed at ``t0``.
    """
    if not domain.is_time_dependent:
        raise ValueError("sample_initial requires a time-dependent domain")
    if n < 1:
        raise ValueError("n must be >= 1")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    unit = _sample_unit_box(n, domain.spatial_dim, method, generator, seed, dtype)
    spatial_lower = tuple(low for low, _ in domain.spatial_bounds)
    spatial_upper = tuple(high for _, high in domain.spatial_bounds)
    spatial = _scale(unit, spatial_lower, spatial_upper)
    t0, _ = domain.time_bounds  # type: ignore[misc]
    time_col = torch.full((n, 1), float(t0), dtype=dtype)
    return torch.cat((spatial, time_col), dim=1).to(device=device)


@dataclass(frozen=True)
class FaceSample:
    """Sampled points lying on one boundary face of a box domain."""

    dim_index: int
    side: int  # -1 for the lower face, +1 for the upper face
    value: float
    points: Tensor


def sample_boundary_faces(
    domain: Domain,
    n_per_face: int,
    method: SamplingMethod = "lhs",
    seed: int = 42,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str = "cpu",
) -> list[FaceSample]:
    """Sample ``n_per_face`` points on every spatial face of ``domain``.

    A 1D interval yields 2 faces (left/right, matching the original
    baseline exactly), a 2D rectangle yields 4 edges, a 3D box yields 6
    faces — the loop is identical either way, which is the point of
    routing sampling through ``Domain`` instead of hardcoding endpoints.
    Other spatial dimensions and (if present) time are sampled freely
    across their own ranges; the face's own dimension is pinned.
    """
    if n_per_face < 1:
        raise ValueError("n_per_face must be >= 1")

    faces: list[FaceSample] = []
    free_dim = domain.input_dim - 1  # every column except the pinned face dimension
    for face_index, (dim_index, side, value) in enumerate(domain.faces()):
        face_seed = seed + 1000 * (face_index + 1)
        generator = torch.Generator(device="cpu").manual_seed(face_seed)
        if free_dim == 0:
            # A 1D domain's faces are single points in space (plus time).
            unit = torch.zeros((n_per_face, 0), dtype=dtype)
        else:
            unit = _sample_unit_box(n_per_face, free_dim, method, generator, face_seed, dtype)

        free_lower = [low for i, (low, _) in enumerate(domain.spatial_bounds) if i != dim_index]
        free_upper = [high for i, (_, high) in enumerate(domain.spatial_bounds) if i != dim_index]
        if domain.is_time_dependent:
            free_lower.append(domain.time_bounds[0])  # type: ignore[index]
            free_upper.append(domain.time_bounds[1])  # type: ignore[index]
        free_values = _scale(unit, tuple(free_lower), tuple(free_upper)) if free_dim > 0 else unit

        columns = []
        free_cursor = 0
        for col in range(domain.input_dim):
            if col == dim_index:
                columns.append(torch.full((n_per_face, 1), float(value), dtype=dtype))
            else:
                columns.append(free_values[:, free_cursor : free_cursor + 1])
                free_cursor += 1
        points = torch.cat(columns, dim=1)
        faces.append(FaceSample(dim_index=dim_index, side=side, value=value, points=points.to(device=device)))
    return faces


PdeResidualFn = Callable[[torch.nn.Module, Tensor], Tensor]


@dataclass
class AdaptiveResampler:
    """Residual-based adaptive collocation-point resampling.

    Generalizes the project's original 1D notebook (draw a candidate pool,
    keep the points with the largest PDE residual) to any dimension: a PDE
    residual is just a function of ``(model, points)`` regardless of how
    many columns ``points`` has, so nothing here is 1D-specific.
    """

    domain: Domain
    pool_size: int = 20_000
    method: SamplingMethod = "sobol"
    seed: int = 0
    _step: int = field(default=0, init=False, repr=False)

    def refresh(self, residual_fn: PdeResidualFn, model: torch.nn.Module, n_select: int) -> Tensor:
        """Draw a candidate pool and keep the ``n_select`` worst-residual points."""
        if n_select < 1:
            raise ValueError("n_select must be >= 1")
        self._step += 1
        pool = sample_interior(
            self.domain,
            self.pool_size,
            method=self.method,
            seed=self.seed + self._step,
        ).requires_grad_(True)
        with torch.enable_grad():
            residual = residual_fn(model, pool)
            score = residual.detach().square().sum(dim=1)
        n_select = min(n_select, self.pool_size)
        _, top_indices = torch.topk(score, n_select)
        return pool[top_indices].detach().clone()
