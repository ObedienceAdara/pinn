import torch

from pinn.core import FixedWeights, GeneralMLP, GeneralPINNConfig, GeneralTrainer, HardDirichletAnsatz
from pinn.problems import burgers1d, heat1d, poisson2d


def test_heat1d_soft_constraint_has_four_boundary_terms() -> None:
    problem = heat1d.build_problem(n_interior=16, n_boundary=4, n_initial=4, sampling_method="uniform")
    names = sorted(bc.name for bc in problem.boundary_conditions)
    assert names == ["initial", "left_boundary", "right_boundary"]


def test_heat1d_exact_solution_matches_initial_condition_at_t0() -> None:
    x = torch.linspace(-1, 1, 10).unsqueeze(1)
    t0 = torch.zeros_like(x)
    points = torch.cat([x, t0], dim=1)
    exact = heat1d.exact_solution(x, t0)
    assert torch.allclose(exact, heat1d.initial_condition(x), atol=1e-6)


def test_poisson2d_exact_solution_vanishes_on_boundary() -> None:
    edges = torch.tensor([[0.0, 0.4], [1.0, 0.4], [0.4, 0.0], [0.4, 1.0]])
    assert torch.allclose(poisson2d.exact_solution(edges), torch.zeros(4, 1), atol=1e-6)


def test_poisson2d_hard_constraint_has_no_boundary_terms() -> None:
    problem = poisson2d.build_problem(n_interior=16, hard_constraint=True)
    assert problem.boundary_conditions == []


def test_poisson2d_soft_constraint_has_four_edge_terms() -> None:
    problem = poisson2d.build_problem(n_interior=16, n_boundary_per_face=4, hard_constraint=False)
    assert len(problem.boundary_conditions) == 4


def test_burgers1d_problem_has_no_closed_form_exact_solution() -> None:
    problem = burgers1d.build_problem(n_interior=16, n_boundary=4, n_initial=4)
    assert problem.exact_solution is None
    assert problem.resampler is not None  # enabled by default for this problem


def test_one_epoch_training_is_finite_for_every_problem() -> None:
    """A cheap regression guard: each problem should plug into GeneralTrainer
    without shape or autograd errors, even if one epoch won't converge."""
    heat_problem = heat1d.build_problem(n_interior=16, n_boundary=4, n_initial=4, sampling_method="uniform")
    heat_model = GeneralMLP(heat_problem.domain, hidden_dim=8, hidden_layers=2)
    heat_weights = FixedWeights({"physics": 1.0, "initial": 10.0, "left_boundary": 10.0, "right_boundary": 10.0})
    heat_history = GeneralTrainer(heat_model, heat_problem, heat_weights, GeneralPINNConfig(epochs=1, seed=0)).train()
    assert torch.isfinite(torch.tensor(heat_history["total"][0]))

    poisson_problem = poisson2d.build_problem(n_interior=16, hard_constraint=True)
    poisson_base = GeneralMLP(poisson_problem.domain, hidden_dim=8, hidden_layers=2)
    poisson_model = HardDirichletAnsatz(
        poisson_base, poisson_problem.domain, boundary_value_fn=lambda p: torch.zeros(p.shape[0], 1)
    )
    poisson_history = GeneralTrainer(
        poisson_model, poisson_problem, FixedWeights({"physics": 1.0}), GeneralPINNConfig(epochs=1, seed=0)
    ).train()
    assert torch.isfinite(torch.tensor(poisson_history["total"][0]))

    burgers_problem = burgers1d.build_problem(n_interior=16, n_boundary=4, n_initial=4, with_resampler=False)
    burgers_model = GeneralMLP(burgers_problem.domain, hidden_dim=8, hidden_layers=2)
    burgers_weights = FixedWeights(
        {"physics": 1.0, "initial": 10.0, "left_boundary": 10.0, "right_boundary": 10.0}
    )
    burgers_history = GeneralTrainer(
        burgers_model, burgers_problem, burgers_weights, GeneralPINNConfig(epochs=1, seed=0)
    ).train()
    assert torch.isfinite(torch.tensor(burgers_history["total"][0]))
