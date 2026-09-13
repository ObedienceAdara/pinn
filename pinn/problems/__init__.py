"""Concrete PDE problems built on ``pinn.core``.

Each submodule exposes a ``build_problem(...) -> pinn.core.Problem`` and,
where one exists in closed form, an ``exact_solution`` function for
validation. Add a new physics problem by writing a new module here that
implements ``pinn.core.PDE`` — the domain, sampling, boundary conditions,
network, weighting and training loop are all reusable as-is.
"""

from . import burgers1d, heat1d, poisson2d

__all__ = ["heat1d", "poisson2d", "burgers1d"]
