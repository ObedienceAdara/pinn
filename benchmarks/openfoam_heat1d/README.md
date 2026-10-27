# OpenFOAM validation: transient 1-D heat diffusion

This benchmark compares the repository's physics-informed neural network (PINN) with OpenFOAM's finite-volume solution and the exact analytical solution of the same transient diffusion problem.

## Mathematical problem

\[
u_t = \alpha u_{xx},\qquad x\in[-1,1],\quad t\in[0,1],
\]

with

\[
u(x,0)=\sin(\pi x),\qquad u(-1,t)=u(1,t)=0,\qquad \alpha=0.1.
\]

The exact solution is

\[
u(x,t)=e^{-\alpha\pi^2t}\sin(\pi x).
\]

The case uses (T) as the scalar field name expected by `laplacianFoam`. The field is dimensionless and the diffusivity has dimensions of area per time. The mesh spans ([-1,1]) in (x), with one cell in each inactive direction and `empty` boundary patches.

## Requirements

- OpenFOAM **Foundation** v14 (the dictionaries use its `physicalProperties` format).
- Python 3.10 or later for the case generator and comparison scripts.
- The PINN repository's dependencies for the neural-network comparison scripts.

Source OpenFOAM before running the solver. On the setup used for the reported run:

```bash
source /opt/openfoam14/etc/bashrc
```

Adjust that path if OpenFOAM Foundation v14 is installed elsewhere.

## Generate and run the coarse case

Run from the PINN repository root:

```bash
python3 benchmarks/openfoam_heat1d/create_case.py \
  --output "$FOAM_RUN/heat1d_pinn_validation" \
  --cells 200 --alpha 0.1 --delta-t 0.001 --end-time 1 --write-interval 0.01

cd "$FOAM_RUN/heat1d_pinn_validation"
blockMesh
checkMesh
laplacianFoam
```

## Generate and run the refined case

Use a separate directory to preserve the coarse-case results:

```bash
python3 "$OLDPWD/benchmarks/openfoam_heat1d/create_case.py" \
  --output "$FOAM_RUN/heat1d_pinn_refined" \
  --cells 800 --alpha 0.1 --delta-t 0.000125 --end-time 1 --write-interval 0.01

cd "$FOAM_RUN/heat1d_pinn_refined"
blockMesh
checkMesh
laplacianFoam
```

If your shell's `OLDPWD` is not the repository root, use the absolute path to `benchmarks/openfoam_heat1d/create_case.py`. The generator refuses to overwrite a non-empty output directory; choose a new directory when regenerating.

The time step and mesh are both refined in the reported coarse-versus-refined comparison. That comparison demonstrates improvement under the combined refinement, but does not isolate spatial and temporal convergence rates individually.

## Evaluate OpenFOAM against the exact solution

From the repository root, with its virtual environment activated:

```bash
python scripts/validate_openfoam_heat.py \
  "$FOAM_RUN/heat1d_pinn_refined" --alpha 0.1
```

The validator compares cell-centred values against the exact solution at the cell centres and reports relative (L^2) error and maximum absolute error at each saved time. The current implementation assumes a uniform mesh spanning ([-1,1]), consistent with the generator.

## Compare the PINN with OpenFOAM

The common-grid comparison script supports the original baseline and generalized soft-/hard-constrained models. Examples:

```bash
# Original baseline PINN
python scripts/compare_pinn_openfoam_heat.py \
  --case "$FOAM_RUN/heat1d_pinn_refined" --model baseline

# Generalized soft-constrained PINN
python scripts/compare_pinn_openfoam_heat.py \
  --case "$FOAM_RUN/heat1d_pinn_refined" --model soft

# Hard constraints with causal weighting
python scripts/compare_pinn_openfoam_heat.py \
  --case "$FOAM_RUN/heat1d_pinn_refined" --model hard --causal

# Same hard-constrained model without causal weighting
python scripts/compare_pinn_openfoam_heat.py \
  --case "$FOAM_RUN/heat1d_pinn_refined" --model hard
```

The defaults mirror the experiments discussed in the validation report: diffusivity 0.1, 10,000 interior collocation points, seed 42, 3,000 Adam epochs and an L-BFGS refinement stage. Use `--help` to see available overrides. The hard-constrained model uses the repository's `HardDirichletAnsatz` and exact initial condition. The `--causal` option changes only whether the causal residual weighting is enabled for that hard-constrained experiment.

The scripts write generated metrics, field arrays and model weights to `artifacts/`, which is ignored by Git. They do not commit results automatically.

## Verification and interpretation

The analytical solution is the primary reference. OpenFOAM is an independent numerical comparator, not treated as exact ground truth. Keep the following distinctions explicit:

- The linear-system residual printed by OpenFOAM measures convergence of the linear solve, not the error in the physical solution.
- The initial OpenFOAM field is initialized from the analytical profile at cell centres, so its (t=0) comparison error is nearly zero by construction.
- Relative errors from different evaluation grids should not be compared as if their sampling were identical.
- The current PINN comparisons are single-seed runs. They are useful controlled observations, not statistical evidence that a training strategy is universally better.
- This benchmark validates one 1-D scalar diffusion problem; it does not by itself establish general CFD or experimental validation.

See [the recorded results](../../results/heat1d/README.md) for measured values from the runs performed during this study.
