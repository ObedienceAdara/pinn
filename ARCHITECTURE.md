# ARCHITECTURE.md — the generalized framework

`BROAD.md` documents the original 1D heat-equation baseline line by line.
This document covers what was added on top of it in `pinn/core/` and
`pinn/problems/`, why it's built the way it is, what it measurably buys
over the baseline, and — just as important — what it deliberately does not
solve yet.

The baseline is untouched. Every original test, example, and use case
still imports `pinn.MLP`, `pinn.heat_residual`, `pinn.PINNTrainer` and gets
identical behavior. The generalized framework is a second layer alongside
it, not a replacement.

## Why generalize at all

The baseline hardcoded one equation (heat), one domain (`x` in `[-1,1]`,
`t` in `[0,1]`), one network shape, three fixed loss weights, and one
training loop. Adding a second equation (`use_cases/viscous_flow`, Burgers)
already needed a hand-rolled copy of the training loop, because nothing in
`pinn.trainer.PINNTrainer` was reusable for a different residual. Extending
to 2D or 3D, or to a problem where boundary conditions aren't "zero at both
ends," meant rewriting from scratch each time.

`pinn/core` replaces the fixed pieces with five interfaces:

| Baseline (fixed)                          | Generalized (`pinn/core`)                          |
|--------------------------------------------|-----------------------------------------------------|
| `x in [-1,1]`, `t in [0,1]`                 | `Domain`: any number of spatial dims + optional time |
| `heat_residual(model, x, t)`                | `PDE.residual(model, points)` — any equation         |
| two hardcoded endpoints, zero Dirichlet     | `BoundaryCondition`: Dirichlet / Neumann / Periodic  |
| `torch.rand` only                           | uniform / Latin Hypercube / Sobol + adaptive resample |
| `physics_weight=1, initial_weight=10, ...`  | `FixedWeights` / `GradNormWeights` / `SelfAdaptiveWeights` |

A new equation is now a ~30-line `PDE` subclass (see
`pinn/problems/poisson2d.py`); the domain, sampling, network, weighting and
training loop are unchanged.

## The five changes, and what each one measurably does

### 1. Domain abstraction (`core/domain.py`)

`Domain(spatial_bounds=[...], time_bounds=(t0, t1) | None)` replaces raw
tuples. `spatial_bounds` is a list, so its length *is* the dimensionality —
1 spatial bound is the original 1D line, 2 is the plane `poisson2d` solves
on, 3 is a volume, with no code path that changes between them. Everything
downstream (`sample_interior`, `sample_boundary_faces`, `GeneralMLP`,
`HardDirichletAnsatz`) is written against `Domain`, not against literal
numbers, which is what let `poisson2d.py` exist without touching `core/` at
all.

`Domain.faces()` generalizes "the two endpoints of an interval" to "the
`2 * spatial_dim` faces of a box" — 2 for 1D, 4 for 2D, 6 for 3D.

### 2. Sampling (`core/sampling.py`)

Three sampling strategies behind one signature (`sample_interior`,
`sample_initial`, `sample_boundary_faces` all take `method="uniform" |
"lhs" | "sobol"`):

- **Uniform** — the baseline's `torch.rand`, kept for comparison.
- **Latin Hypercube** — stratifies every dimension into `n` equal bins and
  permutes them independently, so every stratum of every axis gets exactly
  one sample. This is what actually fixes the "uniform random leaves gaps
  in higher dimension" problem — LHS guarantees even per-axis coverage
  regardless of dimension.
- **Sobol** — a low-discrepancy quasi-random sequence
  (`torch.quasirandom.SobolEngine`), which tends to cover the domain even
  more evenly than LHS at moderate-to-large sample counts. Used as the
  default for `poisson2d` and for `AdaptiveResampler`'s candidate pool.

`AdaptiveResampler.refresh(residual_fn, model, n_select)` generalizes the
project's original 1D notebook (draw a pool, keep the worst-residual
points) to any `PDE`: it takes the *function* `pde.residual`, not a
1D-specific formula, so the exact same class drives resampling for
`heat1d`, `burgers1d`, or a 3D problem that doesn't exist yet.

### 3. Hard-constrained boundaries/initial conditions (`core/networks.py`)

`HardDirichletAnsatz` wraps a base network so the constraint holds by
algebraic construction instead of by loss penalty:

- Transient, homogeneous spatial BC: `u(x,t) = h(x) + (t - t0) * D(x) * N(x,t)`
- Steady, given boundary value `g`: `u(x) = g(x) + D(x) * N(x)`

where `D(x) = prod_i (x_i - lo_i)(hi_i - x_i)` vanishes on every spatial
face of the box. Because `D` is exactly zero on the boundary regardless of
what `N` outputs, the constraint cannot be violated by training — there is
nothing to converge to, it's already there. `tests/test_core_networks.py`
checks this algebraically (error `< 1e-5`, i.e. float32 noise, not
"small"), and both example scripts also check it after training.

**Honest limitation:** the two cases above are individually correct but
cannot currently be combined into one closed form — a transient problem
with a *nonzero* spatial Dirichlet BC would need a spacetime-consistent
extension of the boundary data into the interior, which is closer to an
open research problem (see Sukumar & Srivastava's approximate distance
functions for general geometries) than a one-line fix. The constructor
raises `NotImplementedError` with an explanation rather than silently
producing an ansatz that's wrong in that combination. **Neumann conditions
are soft-only** (`NeumannBC`) — hard-constraining a derivative condition
generally needs the ansatz's *derivative*, not just its value, to vanish
correctly on the boundary, which the simple product-of-distances `D(x)`
does not guarantee. **Periodicity has a real hard-constraint path**, but it
is a different mechanism: `PeriodicEmbedding` (see below), not
`HardDirichletAnsatz`.

Also new in `core/networks.py`:

- **`FourierFeatures` / `FourierMLP`** — random Fourier feature embedding
  (Tancik et al., 2020) for the spatial inputs, addressing the "spectral
  bias" that makes plain MLPs slow to learn high-frequency or sharp-gradient
  solutions (boundary layers, near-shocks).
- **`PeriodicEmbedding`** — encodes one axis as `[cos(2*pi*x/L), sin(2*pi*x/L)]`
  so periodicity holds for *any* downstream network by construction. This
  is the better tool than a soft `PeriodicBC` whenever periodicity is the
  only nonstandard condition in the problem.

### 4. Loss weighting (`core/weighting.py`)

Three interchangeable strategies behind `combine(per_point_terms, model)`:

- **`FixedWeights`** — the baseline's behavior (hand-set scalars), kept as
  the default and as a fair basis for comparison.
- **`GradNormWeights`** — every `update_every` steps, rescales each term so
  its parameter-gradient norm matches an anchor term's (simplified from
  Wang, Teng & Perdikaris, 2021). A term with a naturally small gradient
  barely moves the network at weight 1; this fixes that without hand-tuning.
- **`SelfAdaptiveWeights`** — per-*point* trainable weights updated by
  gradient **ascent** (McClenny & Braga-Neto, 2020) via
  `torch.optim.Adam(..., maximize=True)`, while the network descends on the
  same weighted loss. This is a genuine min-max game: points the network
  currently fits poorly get an automatically increasing weight.
  `tests/test_core_weighting.py::test_self_adaptive_weights_increase_for_the_higher_residual_term`
  confirms the ascent direction is correct, not just that it runs.

### 5. Causal training (`core/causal.py`)

`CausalWeighter` implements the temporal reweighting from Wang, Sankaran &
Perdikaris (2022): the physics residual at each time bin is scaled by
`exp(-epsilon * cumulative_loss_before_that_bin)`, so later times are only
weighted in once earlier times are already well fit. `epsilon=0` recovers
ordinary uniform weighting exactly (tested).

**This is the one change with a large, directly measured effect in this
repo.** The heat equation (diffusive, short time horizon) barely needs it.
Burgers with low viscosity (`viscosity=0.01`) does — it develops a
near-shock around `t ~ 0.3-0.4`, and Adam-only training without causal
weighting or L-BFGS refinement left a max initial-condition error of
**0.20** after 2500 epochs; adding `CausalWeighter` plus L-BFGS refinement
brought the same problem down to **0.0052** — a ~40x improvement, run in
this repo and reproducible via `examples/solve_burgers_general.py`.

## Measured results (this repo, this hardware)

| Problem                          | Setting                                  | Relative L2 error |
|-----------------------------------|-------------------------------------------|--------------------|
| Heat 1D (`heat1d.py`)             | Soft-constrained (parity w/ baseline)     | 1.26e-2            |
| Heat 1D (`heat1d.py`)             | Hard-constrained + causal weighting       | 6.0e-4 (~20x better) |
| Poisson 2D (`poisson2d.py`)       | Soft-constrained                          | 2.89e-3            |
| Poisson 2D (`poisson2d.py`)       | Hard-constrained                          | 9.96e-5 (~29x better) |
| Burgers 1D (`burgers1d.py`)       | Adam only, no causal weighting            | IC max error 0.20  |
| Burgers 1D (`burgers1d.py`)       | + causal weighting + L-BFGS refinement    | IC max error 0.0052 (~40x better) |

These are single runs on fixed seeds, not averaged over multiple seeds —
treat the exact multipliers as illustrative of the *direction and rough
magnitude* of each change, not as a rigorously benchmarked ablation study.

## What this does NOT do (yet)

Being direct about scope, since overclaiming here is exactly the failure
mode to avoid when the eventual goal is comparing against real CFD:

- **No Navier-Stokes / fluid problem yet.** `poisson2d` proves the
  dimension-generalization works; a genuinely fluid problem (2D lid-driven
  cavity, benchmarked against the standard Ghia et al. reference data or an
  OpenFOAM run) is the natural next step and is not in this repo.
- **No automated comparison-against-real-simulation harness.** There's no
  code here that ingests an OpenFOAM/ANSYS export and computes error/timing
  against it. The `use_cases/` directory's framing (thermal barrier,
  electronics cooling) is still aspirational in that sense — it motivates a
  physical scenario but doesn't validate against a real simulation run.
- **`use_cases/` was not migrated.** The three existing use cases still
  call the original baseline API directly and were left alone to keep this
  change reviewable; they are good candidates to rebuild on `pinn.problems`
  next, especially `viscous_flow`, which already has a generalized
  counterpart in `problems/burgers1d.py` and `examples/solve_burgers_general.py`.
- **Neumann/periodic hard constraints are partial**, as detailed above.
- **3D is supported by `Domain`/`sampling`/`autodiff` but has no example
  problem in this repo** — the pieces (`spatial_bounds` of length 3,
  `laplacian(u, x, [0,1,2])`) are dimension-agnostic and should work, but
  "should work" is not the same claim as "measured to work," and it hasn't
  been run here.

## Where to look for what

- `pinn/core/domain.py`, `sampling.py`, `autodiff.py` — the
  dimension-agnostic primitives.
- `pinn/core/pde.py`, `boundary.py` — the interfaces a new equation
  implements.
- `pinn/core/networks.py` — `GeneralMLP`, `FourierMLP`, `PeriodicEmbedding`,
  `HardDirichletAnsatz`.
- `pinn/core/weighting.py`, `causal.py` — the three weighting strategies and
  causal training.
- `pinn/core/trainer.py`, `problem.py` — `GeneralTrainer` and the `Problem`
  bundle that ties a domain, PDE, boundary conditions and sampling together.
- `pinn/problems/heat1d.py`, `poisson2d.py`, `burgers1d.py` — concrete
  equations built on the above.
- `examples/solve_heat1d_general.py`, `solve_poisson_2d.py`,
  `solve_burgers_general.py` — runnable scripts that reproduce the numbers
  in the table above and save comparison plots to `artifacts/`.
- `tests/test_core_*.py`, `test_problems.py` — the new test suite (46 new
  tests; the original 8 are untouched and still pass).
