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
├── notebooks/
│   ├── README.md
│   ├── 01_pinn_foundations.ipynb
│   ├── 02_repository_implementation.ipynb
│   ├── 03_autograd_and_pde.ipynb
│   ├── 04_training_and_optimization.ipynb
│   ├── 05_validation_and_visualization.ipynb
│   └── 06_advanced_extensions.ipynb
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

For a complete executable learning/research walkthrough, see [`notebooks/README.md`](notebooks/README.md).

## Install

Python 3.10+ is required.

```bash
python -m pip install -e .
```

For development/testing:

```bash
python -m pip install -e ".[dev]"
```

For the notebook workflow, install Jupyter separately in your environment:

```bash
python -m pip install jupyterlab
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

Notebook suite:

```bash
jupyter lab
```

Start with `notebooks/01_pinn_foundations.ipynb` and proceed in numerical order.

## Test

```bash
pytest
```

## CI

GitHub Actions tests Python 3.10, 3.11, and 3.12. CI uses the CPU-only PyTorch wheel index because the workflow runs on CPU runners. It also builds a wheel and validates imports from outside the repository tree so packaging regressions are detected.

## Scope

This is a standard PINN baseline for research, education, and engineering experimentation. The included use cases and notebooks are reference implementations, not validated industrial or safety-critical solvers. Production use requires verification against appropriate analytical, experimental, and/or high-fidelity numerical methods.
