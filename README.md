# Physics-Informed Neural Network

A clean PyTorch implementation of a standard Physics-Informed Neural Network (PINN) for solving the 1D heat equation.

## Problem

The reference problem is

$$
 u_t = \alpha u_{xx}, \qquad x \in [-1,1],\; t \in [0,1],
$$

with

$$
 u(x,0)=\sin(\pi x),
$$

and homogeneous Dirichlet boundary conditions

$$
 u(-1,t)=u(1,t)=0.
$$

The analytical solution is

$$
 u(x,t)=e^{-\alpha\pi^2t}\sin(\pi x).
$$

The network learns $u_\theta(x,t)$ while automatic differentiation enforces the PDE residual:

$$
 r_\theta = \frac{\partial u_\theta}{\partial t} - \alpha\frac{\partial^2u_\theta}{\partial x^2}.
$$

The total loss is

$$
\mathcal L = \lambda_f\mathcal L_f + \lambda_{ic}\mathcal L_{ic} + \lambda_{bc}\mathcal L_{bc}.
$$

## Repository layout

```text
pinn/
├── pinn/
│   ├── __init__.py
│   ├── models.py          # neural network architecture
│   ├── physics.py         # PDE residual and exact solution
│   ├── sampling.py        # collocation / IC / BC sampling
│   └── trainer.py         # PINN optimization loop
├── examples/
│   └── solve_heat.py      # end-to-end training and visualization
├── tests/
│   ├── test_model.py
│   ├── test_physics.py
│   └── test_sampling.py
├── .github/workflows/ci.yml
├── pyproject.toml
└── README.md
```

## Install

Python 3.10+ is recommended.

```bash
python -m pip install -e .
```

For development/testing:

```bash
python -m pip install -e ".[dev]"
```

## Run

```bash
python examples/solve_heat.py
```

The script trains the PINN and writes a prediction/reference/error figure to `artifacts/heat_equation_solution.png`.

## Test

```bash
pytest
```

## What is implemented

- Fully-connected multilayer perceptron with `tanh` activations.
- Input normalization from physical coordinates to `[-1, 1]`.
- PyTorch autograd for first- and second-order PDE derivatives.
- Interior collocation points for physics loss.
- Initial-condition and boundary-condition losses.
- Configurable loss weights and training settings.
- Reproducible sampling with an explicit seed.
- Exact-solution error evaluation for the reference heat-equation problem.
- Lightweight unit tests and GitHub Actions CI.

This is intentionally a standard baseline. Advanced additions such as adaptive sampling, loss balancing, Fourier features, hard constraints, domain decomposition, or operator learning should be layered on after this baseline is validated.
