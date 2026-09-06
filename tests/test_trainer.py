import torch

from pinn import MLP, PINNConfig, PINNTrainer, sample_heat_equation


def test_pinn_loss_is_finite_and_train_step_updates_model() -> None:
    model = MLP(hidden_dim=8, hidden_layers=2)
    trainer = PINNTrainer(
        model,
        PINNConfig(epochs=1, seed=0),
    )
    points = sample_heat_equation(
        n_interior=16,
        n_initial=8,
        n_boundary=8,
        seed=0,
    )

    before = [parameter.detach().clone() for parameter in model.parameters()]
    history = trainer.train(points)

    assert len(history.total) == 1
    assert torch.isfinite(torch.tensor(history.total[0]))
    assert torch.isfinite(torch.tensor(history.physics[0]))
    assert torch.isfinite(torch.tensor(history.initial[0]))
    assert torch.isfinite(torch.tensor(history.boundary[0]))
    assert any(
        not torch.equal(old, new)
        for old, new in zip(before, model.parameters())
    )
