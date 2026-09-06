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
│   │   ├── __init__.py
│   │   ├── README.md
│   │   └── run.py
│   ├── electronics_cooling/
│   │   ├── __init__.py
│   │   ├── README.md
│   │   └── run.py
│   └── viscous_flow/
│       ├── __init__.py
│       ├── README.md
│       └── run.py
├── tests/
├── .github/workflows/ci.yml
├── pyproject.toml
├── BROAD.md
└── README.md
```

For a deep implementation-level explanation, see [`BROAD.md`](BROAD.md).

## Install

Python 3.10+ is required.

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
python -m use_cases.thermal_barrier.run
python -m use_cases.electronics_cooling.run
python -m use_cases.viscous_flow.run
```

## Test

```bash
pytest
```

## CI

GitHub Actions tests Python 3.10, 3.11, and 3.12. CI uses the CPU-only PyTorch wheel index because the workflow runs on CPU runners. It also builds a wheel and validates imports from outside the repository tree so packaging regressions are detected.

## Scope

This is a standard PINN baseline for research, education, and engineering experimentation. The included use cases are reference implementations, not validated industrial or safety-critical solvers. Production use requires verification against appropriate analytical, experimental, and/or high-fidelity numerical methods.
