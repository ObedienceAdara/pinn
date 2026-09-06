import pytest
import torch

from pinn import MLP


def test_mlp_output_shape_and_normalization() -> None:
    model = MLP(hidden_dim=16, hidden_layers=2)
    inputs = torch.tensor([
        [-1.0, 0.0],
        [0.0, 0.5],
        [1.0, 1.0],
    ])
    outputs = model(inputs)
    assert outputs.shape == (3, 1)


def test_mlp_rejects_wrong_shape() -> None:
    model = MLP()
    with pytest.raises(ValueError):
        model(torch.zeros(4, 3))
