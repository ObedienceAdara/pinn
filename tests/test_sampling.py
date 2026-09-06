import torch

from pinn import sample_heat_equation


def test_sampling_shapes_and_domain() -> None:
    points = sample_heat_equation(
        n_interior=32,
        n_initial=16,
        n_boundary=16,
        seed=7,
    )

    assert points.interior.shape == (32, 2)
    assert points.initial.shape == (16, 2)
    assert points.left_boundary.shape == (16, 2)
    assert points.right_boundary.shape == (16, 2)

    assert torch.all((points.interior[:, 0] >= -1) & (points.interior[:, 0] <= 1))
    assert torch.all((points.interior[:, 1] >= 0) & (points.interior[:, 1] <= 1))
    assert torch.all(points.initial[:, 1] == 0)
    assert torch.all(points.left_boundary[:, 0] == -1)
    assert torch.all(points.right_boundary[:, 0] == 1)


def test_sampling_seed_is_reproducible() -> None:
    first = sample_heat_equation(8, 4, 4, seed=123)
    second = sample_heat_equation(8, 4, 4, seed=123)
    assert torch.equal(first.interior, second.interior)
    assert torch.equal(first.initial, second.initial)
    assert torch.equal(first.left_boundary, second.left_boundary)
    assert torch.equal(first.right_boundary, second.right_boundary)
