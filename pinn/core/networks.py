"""Network architectures for the generalized framework.

``GeneralMLP`` is the original ``pinn.models.MLP`` with its bounds pulled
from a ``Domain`` instead of two raw tuples, so it works unmodified at any
dimension. The rest of this module is new capability: Fourier features for
high-frequency solutions, a periodic input embedding that makes periodicity
exact instead of penalized, and a hard-constraint ansatz that does the same
for Dirichlet boundaries and initial conditions.
"""

from __future__ import annotations

from typing import Callable, Sequence

import torch
from torch import nn

from .domain import Domain

Tensor = torch.Tensor
ValueFn = Callable[[Tensor], Tensor]


class GeneralMLP(nn.Module):
    """Fully-connected PINN ansatz for a domain of any dimension.

    Behaves exactly like the original ``pinn.models.MLP``: inputs are
    normalized to roughly [-1, 1] using the domain's own bounds, and Tanh
    activations keep the network's second derivatives smooth, which the PDE
    residual needs.
    """

    def __init__(
        self,
        domain: Domain,
        output_dim: int = 1,
        hidden_dim: int = 64,
        hidden_layers: int = 4,
    ) -> None:
        super().__init__()
        if hidden_layers < 1:
            raise ValueError("hidden_layers must be >= 1")
        if hidden_dim < 1:
            raise ValueError("hidden_dim must be >= 1")

        self.domain = domain
        input_dim = domain.input_dim
        lower = torch.as_tensor(domain.lower(), dtype=torch.get_default_dtype())
        upper = torch.as_tensor(domain.upper(), dtype=torch.get_default_dtype())
        self.register_buffer("input_lower", lower)
        self.register_buffer("input_upper", upper)

        layers: list[nn.Module] = [nn.Linear(input_dim, hidden_dim), nn.Tanh()]
        for _ in range(hidden_layers - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.Tanh()])
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.network = nn.Sequential(*layers)
        self._initialize()

    def _initialize(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                nn.init.zeros_(module.bias)

    def normalize_inputs(self, inputs: Tensor) -> Tensor:
        lower = self.input_lower.to(dtype=inputs.dtype, device=inputs.device)
        upper = self.input_upper.to(dtype=inputs.dtype, device=inputs.device)
        return 2.0 * (inputs - lower) / (upper - lower) - 1.0

    def forward(self, inputs: Tensor) -> Tensor:
        if inputs.ndim != 2 or inputs.shape[-1] != self.input_lower.numel():
            raise ValueError(
                f"expected inputs with shape (N, {self.input_lower.numel()}), got {tuple(inputs.shape)}"
            )
        return self.network(self.normalize_inputs(inputs))


class FourierFeatures(nn.Module):
    """Random Fourier feature embedding (Tancik et al., 2020).

    Plain coordinate inputs bias an MLP toward learning low-frequency
    functions ("spectral bias"), which shows up in PINNs as solutions that
    look over-smoothed near sharp gradients (boundary layers, shocks, steep
    initial conditions). Projecting inputs through a fixed random Gaussian
    matrix before the sine/cosine embedding gives the network direct access
    to higher-frequency components and measurably speeds up convergence on
    that class of problem.
    """

    def __init__(self, input_dim: int, mapping_size: int = 64, scale: float = 4.0, seed: int = 0) -> None:
        super().__init__()
        generator = torch.Generator().manual_seed(seed)
        projection = torch.randn((input_dim, mapping_size), generator=generator) * scale
        self.register_buffer("projection", projection)

    @property
    def output_dim(self) -> int:
        return 2 * self.projection.shape[1]

    def forward(self, inputs: Tensor) -> Tensor:
        projected = 2 * torch.pi * inputs @ self.projection.to(dtype=inputs.dtype)
        return torch.cat([torch.sin(projected), torch.cos(projected)], dim=-1)


class FourierMLP(nn.Module):
    """``GeneralMLP`` with a Fourier-feature front end on the spatial inputs.

    Only the spatial columns are embedded; a time column (if present) is
    passed straight through and concatenated, since time in a PINN is
    usually smooth even when the spatial solution is not.
    """

    def __init__(
        self,
        domain: Domain,
        output_dim: int = 1,
        hidden_dim: int = 64,
        hidden_layers: int = 4,
        mapping_size: int = 64,
        fourier_scale: float = 4.0,
        seed: int = 0,
    ) -> None:
        super().__init__()
        self.domain = domain
        lower = torch.as_tensor(domain.lower(), dtype=torch.get_default_dtype())
        upper = torch.as_tensor(domain.upper(), dtype=torch.get_default_dtype())
        self.register_buffer("input_lower", lower)
        self.register_buffer("input_upper", upper)

        self.fourier = FourierFeatures(domain.spatial_dim, mapping_size, fourier_scale, seed)
        mlp_input_dim = self.fourier.output_dim + (1 if domain.is_time_dependent else 0)

        layers: list[nn.Module] = [nn.Linear(mlp_input_dim, hidden_dim), nn.Tanh()]
        for _ in range(hidden_layers - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.Tanh()])
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.network = nn.Sequential(*layers)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, inputs: Tensor) -> Tensor:
        lower = self.input_lower.to(dtype=inputs.dtype, device=inputs.device)
        upper = self.input_upper.to(dtype=inputs.dtype, device=inputs.device)
        normalized = 2.0 * (inputs - lower) / (upper - lower) - 1.0
        spatial_dim = self.domain.spatial_dim
        embedded = self.fourier(normalized[:, :spatial_dim])
        if self.domain.is_time_dependent:
            embedded = torch.cat([embedded, normalized[:, spatial_dim:]], dim=-1)
        return self.network(embedded)


class PeriodicEmbedding(nn.Module):
    """Encodes one input axis so periodicity is exact instead of penalized.

    Replacing ``x`` with ``[cos(2*pi*x/period), sin(2*pi*x/period)]`` makes
    the embedded coordinate identical at ``x`` and ``x + period`` for any
    downstream network — periodicity holds by construction, not because a
    ``PeriodicBC`` loss term pushed it there. Non-periodic columns (other
    spatial axes, time) pass through unchanged.
    """

    def __init__(self, input_dim: int, periodic_dim: int, period: float) -> None:
        super().__init__()
        if not 0 <= periodic_dim < input_dim:
            raise ValueError("periodic_dim must be a valid column index")
        self.input_dim = input_dim
        self.periodic_dim = periodic_dim
        self.period = period

    @property
    def output_dim(self) -> int:
        return self.input_dim + 1  # one column becomes two

    def forward(self, inputs: Tensor) -> Tensor:
        x_p = inputs[:, self.periodic_dim : self.periodic_dim + 1]
        angle = 2 * torch.pi * x_p / self.period
        others = torch.cat(
            [inputs[:, : self.periodic_dim], inputs[:, self.periodic_dim + 1 :]], dim=1
        )
        return torch.cat([others, torch.cos(angle), torch.sin(angle)], dim=1)


class HardDirichletAnsatz(nn.Module):
    """Wraps a base network so Dirichlet boundaries/initial conditions hold exactly.

    Two composable, individually-correct cases are supported (see the
    project's ARCHITECTURE.md for why a single closed form covering both at
    once is not implemented):

    Steady, homogeneous or given boundary value ``g``:
        u(x) = g(x) + D(x) * N(x)
    Transient, homogeneous spatial Dirichlet BC with initial condition ``h``:
        u(x, t) = h(x) + (t - t0) * D(x) * N(x, t)

    ``D(x) = prod_i (x_i - lo_i) * (hi_i - x_i)`` vanishes on every spatial
    boundary face of the box, so the boundary term drops out there
    regardless of what the inner network outputs — the constraint is exact
    by construction, not driven there by a loss term.
    """

    def __init__(
        self,
        network: nn.Module,
        domain: Domain,
        boundary_value_fn: ValueFn | None = None,
        initial_condition_fn: ValueFn | None = None,
    ) -> None:
        super().__init__()
        if domain.is_time_dependent:
            if boundary_value_fn is not None:
                raise NotImplementedError(
                    "combining a nonzero spatial boundary extension with an initial "
                    "condition in one exact ansatz needs a spacetime-consistent "
                    "extension of the boundary data and is not implemented here; "
                    "the transient case only hard-constrains a homogeneous (zero) "
                    "spatial Dirichlet BC. Use a soft DirichletBC for a nonzero "
                    "spatial boundary on a transient problem instead."
                )
            if initial_condition_fn is None:
                raise ValueError("a time-dependent domain needs initial_condition_fn")
        elif initial_condition_fn is not None:
            raise ValueError("initial_condition_fn only applies to a time-dependent domain")

        self.network = network
        self.domain = domain
        self.boundary_value_fn = boundary_value_fn
        self.initial_condition_fn = initial_condition_fn
        lower = torch.as_tensor([low for low, _ in domain.spatial_bounds], dtype=torch.get_default_dtype())
        upper = torch.as_tensor([high for _, high in domain.spatial_bounds], dtype=torch.get_default_dtype())
        self.register_buffer("_spatial_lower", lower)
        self.register_buffer("_spatial_upper", upper)

    def _distance(self, x_spatial: Tensor) -> Tensor:
        lower = self._spatial_lower.to(dtype=x_spatial.dtype, device=x_spatial.device)
        upper = self._spatial_upper.to(dtype=x_spatial.dtype, device=x_spatial.device)
        return torch.prod((x_spatial - lower) * (upper - x_spatial), dim=1, keepdim=True)

    def forward(self, points: Tensor) -> Tensor:
        spatial_dim = self.domain.spatial_dim
        x_spatial = points[:, :spatial_dim]
        distance = self._distance(x_spatial)
        n_out = self.network(points)

        if not self.domain.is_time_dependent:
            boundary = self.boundary_value_fn(points) if self.boundary_value_fn else torch.zeros_like(n_out)
            return boundary + distance * n_out

        t = points[:, spatial_dim : spatial_dim + 1]
        t0, _ = self.domain.time_bounds  # type: ignore[misc]
        initial = self.initial_condition_fn(x_spatial) if self.initial_condition_fn else torch.zeros_like(n_out)
        return initial + (t - t0) * distance * n_out
