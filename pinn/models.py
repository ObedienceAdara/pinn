from __future__ import annotations

import torch
from torch import nn


class MLP(nn.Module):
    """Fully-connected network used as the PINN solution ansatz.

    Coordinates are normalized from the supplied physical-domain bounds to
    approximately [-1, 1] before entering the network. Tanh activations are
    used because PINNs require smooth derivatives with respect to inputs.
    """

    def __init__(
        self,
        input_dim: int = 2,
        output_dim: int = 1,
        hidden_dim: int = 64,
        hidden_layers: int = 4,
        input_lower: tuple[float, ...] = (-1.0, 0.0),
        input_upper: tuple[float, ...] = (1.0, 1.0),
    ) -> None:
        super().__init__()
        if hidden_layers < 1:
            raise ValueError("hidden_layers must be >= 1")
        if hidden_dim < 1:
            raise ValueError("hidden_dim must be >= 1")
        if len(input_lower) != input_dim or len(input_upper) != input_dim:
            raise ValueError("input bounds must match input_dim")

        lower = torch.as_tensor(input_lower, dtype=torch.get_default_dtype())
        upper = torch.as_tensor(input_upper, dtype=torch.get_default_dtype())
        if torch.any(upper <= lower):
            raise ValueError("each input_upper value must exceed input_lower")

        self.register_buffer("input_lower", lower)
        self.register_buffer("input_upper", upper)

        layers: list[nn.Module] = [nn.Linear(input_dim, hidden_dim), nn.Tanh()]
        for _ in range(hidden_layers - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.Tanh()])
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.network = nn.Sequential(*layers)
        self._initialize()

    def _initialize(self) -> None:
        """Xavier initialization is a stable default for tanh PINNs."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                nn.init.zeros_(module.bias)

    def normalize_inputs(self, inputs: torch.Tensor) -> torch.Tensor:
        """Map physical coordinates to [-1, 1]."""
        lower = self.input_lower.to(dtype=inputs.dtype, device=inputs.device)
        upper = self.input_upper.to(dtype=inputs.dtype, device=inputs.device)
        return 2.0 * (inputs - lower) / (upper - lower) - 1.0

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim != 2 or inputs.shape[-1] != self.input_lower.numel():
            raise ValueError(
                f"expected inputs with shape (N, {self.input_lower.numel()}), "
                f"got {tuple(inputs.shape)}"
            )
        return self.network(self.normalize_inputs(inputs))
