"""Dimension-agnostic box domains.

The original baseline hardcoded x in [-1, 1] and t in [0, 1]. ``Domain``
replaces that with an axis-aligned box of arbitrary spatial dimension, with
an optional trailing time axis for transient problems. Every other module in
``pinn.core`` (sampling, boundary conditions, hard-constraint ansatzes) is
written against this abstraction instead of against raw tuples, so a new
problem only has to describe its domain once.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Sequence


Bounds = tuple[float, float]


@dataclass(frozen=True)
class Domain:
    """An axis-aligned box: ``spatial_bounds`` spatial dimensions, plus an
    optional time dimension appended last.

    Parameters
    ----------
    spatial_bounds:
        One ``(low, high)`` pair per spatial dimension. Length sets the
        spatial dimensionality (1 for a line, 2 for a plane, 3 for a volume,
        and so on — nothing downstream assumes a specific count).
    time_bounds:
        ``(t0, t1)`` if the problem is transient, ``None`` for a steady-state
        problem. When present, every point tensor has time as its last
        column.
    """

    spatial_bounds: Sequence[Bounds]
    time_bounds: Bounds | None = None

    def __post_init__(self) -> None:
        if len(self.spatial_bounds) < 1:
            raise ValueError("a domain needs at least one spatial dimension")
        for bounds in self.spatial_bounds:
            low, high = bounds
            if high <= low:
                raise ValueError(f"invalid spatial bounds {bounds!r}: high must exceed low")
        if self.time_bounds is not None:
            low, high = self.time_bounds
            if high <= low:
                raise ValueError(f"invalid time bounds {self.time_bounds!r}: high must exceed low")

    @property
    def spatial_dim(self) -> int:
        return len(self.spatial_bounds)

    @property
    def is_time_dependent(self) -> bool:
        return self.time_bounds is not None

    @property
    def input_dim(self) -> int:
        """Total number of input columns a network for this domain must accept."""
        return self.spatial_dim + (1 if self.is_time_dependent else 0)

    @property
    def time_index(self) -> int | None:
        """Column index of the time coordinate, or ``None`` for steady problems."""
        return self.spatial_dim if self.is_time_dependent else None

    def full_bounds(self) -> list[Bounds]:
        """All bounds in column order: spatial dimensions, then time if present."""
        bounds = list(self.spatial_bounds)
        if self.time_bounds is not None:
            bounds.append(self.time_bounds)
        return bounds

    def lower(self) -> tuple[float, ...]:
        return tuple(low for low, _ in self.full_bounds())

    def upper(self) -> tuple[float, ...]:
        return tuple(high for _, high in self.full_bounds())

    def faces(self) -> Iterator[tuple[int, int, float]]:
        """Yield every spatial boundary face as ``(dim_index, side, value)``.

        ``side`` is ``-1`` for the lower face of that dimension and ``+1``
        for the upper face. A box in ``d`` spatial dimensions has ``2 * d``
        faces; a 1D interval has the familiar left/right pair, a 2D
        rectangle has four edges, a 3D box has six faces, and so on.
        """
        for i, (low, high) in enumerate(self.spatial_bounds):
            yield (i, -1, low)
            yield (i, +1, high)
