# Physics-Informed Neural Network

A PyTorch implementation of Physics-Informed Neural Networks (PINNs), evolving from a transparent 1D heat-equation example into a reusable, dimension-agnostic framework for experimenting with different PDEs, domains, boundary conditions, sampling strategies, neural architectures, and training methods.

## Overview

This repository intentionally contains two layers:

1. **The original baseline** — a small, readable PINN for the transient 1D heat equation.
2. **The generalized framework** — reusable abstractions for building PINNs across dimensions and PDEs without rewriting the training loop.

The baseline remains available as a reference implementation and keeps its original public API. The generalized implementation lives alongside it rather than replacing it.

## Baseline: 1D heat equation

The reference problem is

$$
u_t = \alpha u_{xx}, \qquad x \in [-1,1],\; t \in [0,1].
$$

with

$$
u(x,0)=\sin(\pi x),
$$

and homogeneous Dirichlet boundary conditions

$$
u(-1,t)=u(1,t)=0.
$$

Its analytical solution is

$$
u(x,t)=e^{-\alpha\pi^2t}\sin(\pi x).
$$

The network represents $u_\theta(x,t)$ and automatic differentiation provides the derivatives needed for the PDE residual:

$$
r_\theta = \frac{\partial u_\theta}{\partial t} - \alpha\frac{\partial^2 u_\theta}{\partial x^2}.
$$

The baseline implementation is contained in:

```text
pinn/models.py
pinn/physics.py
pinn/sampling.py
pinn/trainer.py
```

See [`BROAD.md`](BROAD.md) for the detailed implementation walkthrough.

## Generalized framework

The generalized system is built around a reusable problem pipeline:

```text
Domain
  ↓
PDE + Boundary Conditions
  ↓
Collocation / Boundary Sampling
  ↓
Neural Network
  ↓
Loss Weighting
  ↓
GeneralTrainer
  ↓
Validation / Diagnostics
```

A new PDE is represented through a `PDE.residual(model, points)` implementation. The rest of the infrastructure can then be reused.

### Core capabilities

| Capability | Implementation |
|---|---|
| Arbitrary spatial dimensions | `Domain` |
| Optional time dimension | `Domain.time_bounds` |
| PDE abstraction | `PDE` |
| Dirichlet conditions | `DirichletBC` |
| Neumann conditions | `NeumannBC` |
| Periodic conditions | `PeriodicBC`, `PeriodicEmbedding` |
| Uniform sampling | `method="uniform"` |
| Latin Hypercube sampling | `method="lhs"` |
| Sobol sampling | `method="sobol"` |
| Residual-based adaptive sampling | `AdaptiveResampler` |
| Standard MLP | `GeneralMLP` |
| Hard Dirichlet constraints | `HardDirichletAnsatz` |
| Fourier features | `FourierFeatures`, `FourierMLP` |
| Fixed loss weighting | `FixedWeights` |
| Gradient-norm weighting | `GradNormWeights` |
| Self-adaptive pointwise weighting | `SelfAdaptiveWeights` |
| Causal transient weighting | `CausalWeighter` |
| Adam optimization | `GeneralTrainer` |
| Optional L-BFGS refinement | `GeneralPINNConfig` |

## Included PDE problems

### `heat1d`

Reimplements the original heat equation through the generalized framework, providing a direct parity check between the old and new APIs.

### `poisson2d`

A genuinely two-dimensional problem that demonstrates why the dimension-agnostic `Domain` and generalized training abstractions matter.

### `burgers1d`

A nonlinear, advection-dominated problem migrated onto the generalized framework. The example combines adaptive residual sampling with causal weighting.

## Examples

```bash
# Original baseline
python examples/solve_heat.py

# Generalized framework on the same heat problem
python examples/solve_heat1d_general.py

# Two-dimensional Poisson problem
python examples/solve_poisson_2d.py

# Burgers equation with adaptive resampling + causal weighting
python examples/solve_burgers_general.py
```

The application-oriented examples under `use_cases/` still use the original baseline API:

```bash
python -m use_cases.thermal_barrier.run
python -m use_cases.electronics_cooling.run
python -m use_cases.viscous_flow.run
```

## Measured results

The following are representative single-seed runs from this repository. They are project measurements, not general claims about PINNs or rigorous multi-seed ablations.

| Problem | Configuration | Result |
|---|---|---:|
| Heat 1D | Soft constraints, 1500 epochs | 1.26e-2 relative L2 |
| Heat 1D | Hard constraint + causal, 1500 epochs | 6.0e-4 relative L2 |
| Poisson 2D | Soft constraints, 2000 epochs | 2.89e-3 relative L2 |
| Poisson 2D | Hard constraint, 2000 epochs | 9.96e-5 relative L2 |
| Burgers 1D | Adam only, 2500 epochs | 0.20 max IC error |
| Burgers 1D | Causal weighting + L-BFGS, 3000 epochs | 0.0052 max IC error |

The heat and Poisson experiments show improved error in the hard-constraint runs. The Burgers experiment shows a much larger improvement after adding causal weighting and L-BFGS. Because these configurations are not matched, multi-seed ablation studies would be needed before drawing stronger conclusions.

See [`ARCHITECTURE.md`](ARCHITECTURE.md) for the complete experimental context and limitations.

## Repository structure

```text
pinn/
├── pinn/
│   ├── __init__.py
│   ├── models.py
│   ├── physics.py
│   ├── sampling.py
│   ├── trainer.py
│   │
│   ├── core/
│   │   ├── __init__.py
│   │   ├── autodiff.py
│   │   ├── boundary.py
│   │   ├── causal.py
│   │   ├── domain.py
│   │   ├── networks.py
│   │   ├── pde.py
│   │   ├── problem.py
│   │   ├── sampling.py
│   │   ├── trainer.py
│   │   └── weighting.py
│   │
│   └── problems/
│       ├── __init__.py
│       ├── heat1d.py
│       ├── poisson2d.py
│       └── burgers1d.py
│
├── examples/
│   ├── solve_heat.py
│   ├── solve_heat1d_general.py
│   ├── solve_poisson_2d.py
│   └── solve_burgers_general.py
│
├── notebooks/
│   ├── main.ipynb
│   ├── 01_pinn_foundations.ipynb
│   ├── 02_repository_implementation.ipynb
│   ├── 03_autograd_and_pde.ipynb
│   ├── 04_training_and_optimization.ipynb
│   ├── 05_validation_and_visualization.ipynb
│   ├── 06_advanced_extensions.ipynb
│   └── README.md
│
├── use_cases/
├── tests/
├── .github/workflows/ci.yml
├── pyproject.toml
├── BROAD.md
├── ARCHITECTURE.md
└── README.md
```

## Installation

Python 3.10+ is required.

```bash
python -m pip install -e .
```

For development and testing:

```bash
python -m pip install -e ".[dev]"
```

For the notebook workflow:

```bash
python -m pip install jupyterlab
```

## Testing

Run the complete suite with:

```bash
pytest
```

The test suite covers both the original baseline and the generalized framework, including automatic differentiation, domains, boundary conditions, sampling, network constraints, loss weighting, causal training, trainer behavior, and the included PDE problems.

## Notebooks

[`notebooks/main.ipynb`](notebooks/main.ipynb) is the canonical end-to-end notebook. It connects the PDE formulation to the PyTorch implementation, automatic differentiation, collocation sampling, composite losses, optimization, analytical validation, residual diagnostics, adaptive refinement, inverse parameter identification, and the reusable package API.

Focused notebooks provide deeper studies:

1. `01_pinn_foundations.ipynb` — PINN foundations.
2. `02_repository_implementation.ipynb` — reusable package implementation.
3. `03_autograd_and_pde.ipynb` — automatic differentiation and PDE residuals.
4. `04_training_and_optimization.ipynb` — optimization and loss weighting.
5. `05_validation_and_visualization.ipynb` — validation and visualization.
6. `06_advanced_extensions.ipynb` — adaptive collocation and inverse identification.

See [`notebooks/README.md`](notebooks/README.md) for the notebook roadmap.

## Adding a new PDE

The generalized architecture is designed so that the equation-specific part is isolated from the rest of the training system.

Conceptually:

```python
class Wave1D(PDE):
    def __init__(self, c):
        self.c = c

    def residual(self, model, points):
        u = model(points)
        # compute u_tt and u_xx with pinn.core.autodiff
        return u_tt - self.c**2 * u_xx
```

The same domain, sampling, boundary-condition, network, weighting, and trainer infrastructure can then be reused.

## Documentation

- [`BROAD.md`](BROAD.md) — detailed explanation of the original 1D baseline.
- [`ARCHITECTURE.md`](ARCHITECTURE.md) — generalized architecture, experiments, trade-offs, and limitations.
- [`notebooks/README.md`](notebooks/README.md) — notebook roadmap.

## CI

GitHub Actions tests Python 3.10, 3.11, and 3.12. The workflow uses the CPU-only PyTorch wheel index for CPU runners, builds a wheel, verifies imports outside the repository tree, and runs the test suite.

## Scope and limitations

This project is intended for research, education, and engineering experimentation. It is **not** a validated industrial or safety-critical solver.

The generalized framework is not presented as a replacement for mature mesh-based methods such as ANSYS or OpenFOAM. The included problems have analytical validation where an exact solution is available, but the project is not yet a comprehensive benchmark against high-fidelity CFD/FEA workflows.

The current hard Dirichlet ansatz also has deliberate limits: transient homogeneous spatial conditions and steady prescribed Dirichlet data have separate constructions; general transient nonzero spatial Dirichlet data is not silently approximated. Neumann conditions remain soft constraints, while periodicity uses a dedicated embedding mechanism.

Production use requires verification against appropriate analytical, experimental, and/or high-fidelity numerical references.

## Project direction

The repository is moving from a single equation implementation toward a reusable physics-informed machine-learning foundation: multiple PDEs, multiple dimensions, interchangeable constraints and samplers, more expressive neural representations, adaptive loss strategies, causal training, and a clean path for adding new equations without duplicating infrastructure.
