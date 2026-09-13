# Physics-Informed Neural Network

A clean PyTorch implementation of a standard Physics-Informed Neural Network (PINN), starting from a 1D heat-equation baseline and generalized into a dimension-agnostic framework (`pinn/core`) for building and training PINNs on new PDEs.

## Two layers

**Baseline** (`pinn/models.py`, `physics.py`, `sampling.py`, `trainer.py`) — the original 1D heat-equation implementation, unchanged. See the Problem section below and [`BROAD.md`](BROAD.md) for a line-by-line walkthrough.

**Generalized framework** (`pinn/core/`, `pinn/problems/`) — arbitrary-dimension domains, a swappable PDE/boundary-condition interface, Latin Hypercube/Sobol/adaptive sampling, hard-constraint and Fourier-feature network options, gradient-aware and self-adaptive loss weighting, and causal training for transient PDEs. Includes three concrete problems (`heat1d`, `poisson2d`, `burgers1d`) and three runnable examples that reproduce the heat-equation baseline, solve a genuinely 2D problem, and solve Burgers' equation with adaptive resampling + causal weighting. See [`ARCHITECTURE.md`](ARCHITECTURE.md) for the design, measured results, and known limitations.

## Baseline problem

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
│   ├── models.py, physics.py, sampling.py, trainer.py    # baseline (unchanged)
│   ├── core/                                              # generalized framework
│   │   ├── domain.py, autodiff.py, sampling.py
│   │   ├── pde.py, boundary.py, networks.py
│   │   ├── weighting.py, causal.py
│   │   └── trainer.py, problem.py
│   └── problems/                                          # concrete equations on core/
│       ├── heat1d.py, poisson2d.py, burgers1d.py
├── examples/
│   ├── solve_heat.py                    # baseline
│   ├── solve_heat1d_general.py          # baseline problem, generalized framework, soft vs hard+causal
│   ├── solve_poisson_2d.py              # genuinely 2D problem, hard vs soft
│   └── solve_burgers_general.py         # advection-dominated, adaptive resampling + causal weighting
├── notebooks/
│   ├── README.md
│   └── 01_pinn_foundations.ipynb ... 06_advanced_extensions.ipynb
├── use_cases/
│   ├── __init__.py, README.md
│   ├── thermal_barrier/, electronics_cooling/, viscous_flow/   # still baseline-API, not yet migrated
├── tests/                                # 8 original + 46 new, all passing
├── .github/workflows/ci.yml
├── pyproject.toml
├── BROAD.md          # baseline, line-by-line
├── ARCHITECTURE.md   # generalized framework: design, results, limitations
└── README.md
```

For a deep implementation-level explanation of the baseline, see [`BROAD.md`](BROAD.md).

For the generalized framework's design rationale and measured results, see [`ARCHITECTURE.md`](ARCHITECTURE.md).

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

Baseline reference benchmark:

```bash
python examples/solve_heat.py
```

Generalized framework examples:

```bash
python examples/solve_heat1d_general.py   # parity check + hard-constraint/causal comparison
python examples/solve_poisson_2d.py       # 2D Poisson, hard vs soft constrained
python examples/solve_burgers_general.py  # Burgers with adaptive resampling + causal weighting
```

Application-oriented use cases (still on the baseline API):

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

This is a standard PINN baseline and framework for research, education, and engineering experimentation — not a validated industrial or safety-critical solver, and not (yet) benchmarked against a real mesh-based simulation (ANSYS, OpenFOAM). See "What this does NOT do" in [`ARCHITECTURE.md`](ARCHITECTURE.md) for the current honest limitations. Production use requires verification against appropriate analytical, experimental, and/or high-fidelity numerical methods.
