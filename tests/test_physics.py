import torch

from pinn import MLP, heat_exact_solution, heat_residual


def test_exact_solution_has_small_pde_residual() -> None:
    alpha = 0.1
    x = torch.linspace(-0.9, 0.9, 20)
    t = torch.linspace(0.05, 0.95, 20)
    xx, tt = torch.meshgrid(x, t, indexing="ij")
    xt = torch.stack((xx.reshape(-1), tt.reshape(-1)), dim=1).requires_grad_(True)

    class ExactModel(torch.nn.Module):
        def forward(self, inputs: torch.Tensor) -> torch.Tensor:
            return heat_exact_solution(inputs[:, 0:1], inputs[:, 1:2], alpha)

    residual = heat_residual(ExactModel(), xt, alpha)
    assert torch.max(torch.abs(residual)).item() < 1e-5


def test_heat_residual_is_differentiable() -> None:
    model = MLP(hidden_dim=12, hidden_layers=2)
    xt = torch.rand(32, 2)
    xt[:, 0] = 2.0 * xt[:, 0] - 1.0

    residual = heat_residual(model, xt, alpha=0.1)
    loss = residual.square().mean()

    assert residual.requires_grad
    loss.backward()

    gradients = [parameter.grad for parameter in model.parameters()]

    # A PDE derivative loss does not need every parameter to have a gradient.
    # In particular, a constant output bias can disappear under input
    # differentiation. The residual must still backpropagate into the model.
    assert any(gradient is not None for gradient in gradients)
    assert any(
        gradient is not None and torch.any(torch.abs(gradient) > 0)
        for gradient in gradients
    )
    assert all(
        gradient is None or torch.isfinite(gradient).all()
        for gradient in gradients
    )
