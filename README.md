# Physics-Informed Neural Network

A PyTorch implementation of Physics-Informed Neural Networks (PINNs), evolving from a deliberately simple 1D heat-equation baseline into a reusable, dimension-agnostic framework for solving and experimenting with new PDEs.

> The repository keeps the original implementation intact while adding a generalized architecture on top of it. This makes the project useful both as a clear PINN learning reference and as a research/engineering codebase for experimenting with different equations, domains, constraints, sampling strategies, and training methods.

## What this repository contains

There are two layers.

### 1. Original 1D PINN baseline

The original implementation solves the transient 1D heat equation:

$$
u_t = \alpha u_{xx}, \qquad x \in [-1,1],\; t \in [0,1].
$$

with

$$
u(x,0)=\sin(\pi x),
$$

and homogeneous Dirichlet conditions

$$
u(-1,t)=u(1,t)=0.
$$

The analytical solution is

$$
u(x,t)=e^{-\alpha\pi^2t}\sin(\pi x).
$$

The network learns $u_\theta(x,t)$, while automatic differentiation evaluates the PDE residual

$$
r_\theta = \frac{\partial u_\theta}{\partial t} - \alpha\frac{\partial^2u_\theta}{\partial x^2},
$$

and optimization minimizes a weighted combination of physics, initial-condition, and boundary-condition errors.

The baseline lives in:

```text
pinn/models.py
pinn/physics.py
pinn/sampling.py
pinn/trainer.py
```

These files remain the reference implementation and preserve the original public API.

### 2. Generalized PINN framework

The new framework lives in `pinn/core/` and `pinn/problems/`.

It separates the parts of a PINN that should be reusable from the definition of a particular PDE:

```text
Domain
  ↓
PDE + Boundary Conditions
  ↓
Sampling
  ↓
Network / Constraint Ansatz
  ↓
Loss Weighting
  ↓
GeneralTrainer
  ↓
Solution / Validation
```

A new PDE can therefore be implemented as a `PDE.residual(model, points)` class without copying the training loop.

## Generalized capabilities

| Capability | Implementation |
|---|---|
| Arbitrary spatial dimensions | `Domain` |
| Optional time dimension | `Domain.time_bounds` |
| PDE abstraction | `PDE` |
| Dirichlet boundary conditions | `DirichletBC` |
| Neumann boundary conditions | `NeumannBC` |
| Periodic boundary conditions | `PeriodicBC` / `PeriodicEmbedding` |
| Uniform sampling | `sample_interior(..., method="uniform")` |
| Latin Hypercube sampling | `method="lhs"` |
| Sobol sampling | `method="sobol"` |
| Residual-based adaptive sampling | `AdaptiveResampler` |
| Standard MLP | `GeneralMLP` |
| Hard Dirichlet constraints | `HardDirichletAnsatz` |
| Fourier features | `FourierFeatures`, `FourierMLP` |
| Fixed loss weights | `FixedWeights` |
| Gradient-norm weighting | `GradNormWeights` |
| Self-adaptive pointwise weighting | `SelfAdaptiveWeights` |
| Causal transient training | `CausalWeighter` |
| Adam training | `GeneralTrainer` |
| Optional L-BFGS refinement | `GeneralPINNConfig` |

## Problems included

The generalized framework currently includes three concrete equations.

### Heat equation — 1D

`pinn/problems/heat1d.py`

Rebuilds the original heat-equation problem using the generalized abstractions. This provides a parity check between the original and generalized APIs.

### Poisson equation — 2D

`pinn/problems/poisson2d.py`

A genuinely two-dimensional problem that the original 1D-specific trainer could not express without restructuring the code.

### Burgers equation — 1D

`pinn/problems/burgers1d.py`

An advection-dominated nonlinear problem migrated onto the generalized framework. The accompanying example uses adaptive residual sampling and causal weighting.

## Examples

The `examples/` directory provides runnable demonstrations:

```bash
# Original baseline
python examples/solve_heat.py

# Same heat problem through the generalized framework
python examples/solve_heat1d_general.py

# First genuinely 2D problem
python examples/solve_poisson_2d.py

# Nonlinear Burgers problem with adaptive + causal training
python examples/solve_burgers_general.py
```

The application-oriented examples under `use_cases/` are still built on the original baseline API:

```bash
python -m use_cases.thermal_barrier.run
python -m use_cases.electronics_cooling.run
python -m use_cases.viscous_flow.run
```

## Representative measured results

The generalized framework includes realistic training runs recorded in `ARCHITECTURE.md`. These are single-seed runs from this repository, so they should be treated as project measurements rather than general claims about PINNs.

| Problem | Configuration | Error / metric |
|---|---|---:|
| Heat 1D | Soft constraints, 1500 epochs | 1.26e-2 relative L2 |
| Heat 1D | Hard constraint + causal, 1500 epochs | 6.0e-4 relative L2 |
| Poisson 2D | Soft constraints, 2000 epochs | 2.89e-3 relative L2 |
| Poisson 2D | Hard constraint, 2000 epochs | 9.96e-5 relative L2 |
| Burgers 1D | Adam only, 2500 epochs | 0.20 max IC error |
| Burgers 1D | Causal weighting + L-BFGS, 3000 epochs | 0.0052 max IC error |

These experiments indicate that the hard-constraint formulation improved the heat and Poisson runs in this codebase, while causal weighting was especially useful for the more difficult Burgers case. The exact improvement factors should not be interpreted as rigorous ablation results because the runs use individual seeds and different training configurations.

For the experimental setup, design rationale, limitations, and interpretation of these results, see [`ARCHITECTURE.md`](ARCHITECTURE.md).

## Repository structure

```text
pinn/
├── pinn/
│   ├── __init__.py
│   │
│   ├── models.py
│   ├── physics.py
│   ├── sampling.py
│   └── trainer.py
│      └── Original 1D heat-equation API
│
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
│   ├── thermal_barrier/
│   ├── electronics_cooling/
│   └── viscous_flow/
│
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

Run the complete test suite with:

```bash
pytest
```

The repository currently contains the original tests plus generalized-framework tests covering domain behavior, automatic differentiation, boundary conditions, sampling, networks, weighting, causal training, trainer behavior, and the concrete PDE problems.

## Notebooks

`notebooks/main.ipynb` is the canonical end-to-end learning and implementation walkthrough. It connects the mathematical problem to the PyTorch implementation, automatic differentiation, collocation sampling, composite losses, training, L-BFGS refinement, analytical validation, residual diagnostics, adaptive refinement, inverse parameter identification, and the reusable package API.

The focused notebooks provide deeper studies of individual topics:

1. `01_pinn_foundations.ipynb` — build the method from the PDE upward.
2. `02_repository_implementation.ipynb` — connect the mathematics to the reusable package.
3. `03_autograd_and_pde.ipynb` — inspect derivatives and PDE residual construction.
4. `04_training_and_optimization.ipynb` — investigate optimization and loss weighting.
5. `05_validation_and_visualization.ipynb` — quantify solution and residual errors.
6. `06_advanced_extensions.ipynb` — adaptive collocation and inverse identification.

See [`notebooks/README.md`](notebooks/README.md) for the recommended notebook workflow.

## Adding a new PDE

The main architectural payoff is that a new equation is no longer expected to duplicate the trainer.

Conceptually, a new equation implements:

```python
class Wave1D(PDE):
    def __init__(self, c):
        self.c = c

    def residual(self, model, points):
        u = model(points)
        # compute u_tt and u_xx with pinn.core.autodiff
        return u_tt - self.c**2 * u_xx
```

The same `Domain`, sampling, boundary-condition infrastructure, network components, weighting strategies, and `GeneralTrainer` can then be reused.

This pattern is demonstrated concretely by the included `heat1d`, `poisson2d`, and `burgers1d` problems.

## Documentation

- [`BROAD.md`](BROAD.md) — detailed, line-by-line explanation of the original baseline.
- [`ARCHITECTURE.md`](ARCHITECTURE.md) — generalized framework design, measured results, engineering trade-offs, and known limitations.
- [`notebooks/README.md`](notebooks/README.md) — notebook roadmap.

## CI

GitHub Actions tests the package across Python 3.10, 3.11, and 3.12. The workflow uses the CPU-only PyTorch wheel index for CPU runners, builds a wheel, verifies imports outside the repository tree, and runs the test suite.

## Scope and limitations

This repository is intended for research, education, and engineering experimentation. It is **not** a validated industrial or safety-critical solver.

In particular, the generalized framework is not presented as a replacement for mature mesh-based methods such as ANSYS or OpenFOAM. The project has analytical validation for the included benchmark problems, but it is not yet a comprehensive benchmark against high-fidelity CFD/FEA workflows.

There are also deliberate framework limitations. For example, the current hard Dirichlet ansatz has separate constructions for transient homogeneous spatial conditions and steady prescribed Dirichlet data; the combination of transient dynamics with general nonzero spatial Dirichlet data is not silently approximated. Neumann conditions remain soft constraints, while periodicity has a dedicated embedding-based mechanism.

Production use requires verification against appropriate analytical, experimental, and/or high-fidelity numerical references.

## Project direction

The generalized architecture turns this repository from a single heat-equation implementation into a foundation for experimenting with physics-informed machine learning across different PDEs, dimensions, constraints, samplers, architectures, and optimization strategies—while keeping the original baseline available as a transparent reference implementation.
