import torch

from pinn.core import (
    Domain,
    FixedWeights,
    GeneralMLP,
    GeneralPINNConfig,
    GeneralTrainer,
    HardDirichletAnsatz,
)
from pinn.problems import heat1d


def test_training_step_is_finite_and_updates_parameters() -> None:
    problem = heat1d.build_problem(n_interior=32, n_boundary=8, n_initial=8, sampling_method="uniform")
    model = GeneralMLP(problem.domain, hidden_dim=8, hidden_layers=2)
    weighter = FixedWeights({"physics": 1.0, "initial": 10.0, "left_boundary": 10.0, "right_boundary": 10.0})
    trainer = GeneralTrainer(model, problem, weighter, GeneralPINNConfig(epochs=1, seed=0))

    before = [p.detach().clone() for p in model.parameters()]
    history = trainer.train()

    assert len(history["total"]) == 1
    assert torch.isfinite(torch.tensor(history["total"][0]))
    assert torch.isfinite(torch.tensor(history["physics"][0]))
    assert "initial" in history and "left_boundary" in history and "right_boundary" in history
    assert any(not torch.equal(old, new) for old, new in zip(before, model.parameters()))


def test_hard_constrained_problem_has_no_boundary_terms() -> None:
    problem = heat1d.build_problem(n_interior=32, hard_constraint=True, sampling_method="uniform")
    assert problem.boundary_conditions == []

    base = GeneralMLP(problem.domain, hidden_dim=8, hidden_layers=2)
    model = HardDirichletAnsatz(base, problem.domain, initial_condition_fn=heat1d.initial_condition)
    trainer = GeneralTrainer(model, problem, FixedWeights({"physics": 1.0}), GeneralPINNConfig(epochs=1, seed=0))
    history = trainer.train()
    assert list(history.keys()) == ["total", "physics"]


def test_resample_every_replaces_interior_points() -> None:
    problem = heat1d.build_problem(n_interior=32, with_resampler=True, sampling_method="uniform")
    model = GeneralMLP(problem.domain, hidden_dim=8, hidden_layers=2)
    weighter = FixedWeights({"physics": 1.0, "initial": 10.0, "left_boundary": 10.0, "right_boundary": 10.0})
    config = GeneralPINNConfig(epochs=2, seed=0, resample_every=1, n_resample=16)
    trainer = GeneralTrainer(model, problem, weighter, config)

    original_points = trainer.interior.clone()
    trainer.train()
    assert trainer.interior.shape == (16, 2)
    assert not torch.equal(original_points, trainer.interior)


def test_predict_runs_in_eval_mode_without_grad() -> None:
    problem = heat1d.build_problem(n_interior=16, sampling_method="uniform")
    model = GeneralMLP(problem.domain, hidden_dim=8, hidden_layers=2)
    trainer = GeneralTrainer(model, problem, FixedWeights({"physics": 1.0}), GeneralPINNConfig(epochs=1, seed=0))
    trainer.train()
    points = torch.rand(5, 2)
    prediction = trainer.predict(points)
    assert prediction.shape == (5, 1)
    assert not prediction.requires_grad
