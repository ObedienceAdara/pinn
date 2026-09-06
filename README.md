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
│   ├── models.py
│   ├── physics.py
│   ├── sampling.py
│   └── trainer.py
├── examples/
│   └── solve_heat.py
├── use_cases/
│   ├── __init__.py
│   ├── README.md
│   ├── thermal_barrier/
│   │   ├── README.md
│   │   └── run.py
│   ├── electronics_cooling/
│   │   ├── README.md
│   │   └── run.py
│   └── viscous_flow/
│       ├── README.md
│       └── run.py
├── tests/
│   ├── test_model.py
│   ├── test_physics.py
│   ├── test_sampling.py
│   ├── test_trainer.py
│   └── test_use_cases.py
├── .github/workflows/ci.yml
├── pyproject.toml
├── README.md
└── BROAD.md
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

Reference benchmark:

```bash
python examples/solve_heat.py
```

Application-oriented examples:

```bash
python use_cases/thermal_barrier/run.py
python use_cases/electronics_cooling/run.py
python use_cases/viscous_flow/run.py
```

The application examples import the package through its public API or reuse its core model component, showing how the implementation can be embedded in engineering workflows.

## Test

```bash
pytest
```

## Documentation

`BROAD.md` is the detailed technical implementation guide. It documents the mathematical formulation, repository architecture, modules, classes, functions, tensor shapes, automatic-differentiation flow, loss construction, optimization, validation, and extension model.

## What is implemented

- Fully-connected multilayer perceptron with `tanh` activations.
- Input normalization from physical coordinates to `[-1, 1]`.
- PyTorch autograd for first- and second-order PDE derivatives.
- Interior collocation points for physics loss.
- Initial-condition and boundary-condition losses.
- Configurable loss weights and training settings.
- Reproducible sampling with an explicit seed.
- Optional Adam + L-BFGS optimization.
- Exact-solution error evaluation for the reference heat-equation problem.
- Unit tests and GitHub Actions CI.

## Scope

This is a standard PINN baseline for learning and extension. The thermal and viscous-flow examples are simplified technical demonstrations, not validated industrial or safety-critical solvers.
