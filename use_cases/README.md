# PINN use cases

This directory shows how the `pinn` package can be imported into application-style engineering workflows.

The examples are intentionally small enough to run on a laptop, but each is framed around a physical problem rather than a synthetic neural-network task.

## Included examples

| Example | Engineering interpretation | PINN formulation |
| --- | --- | --- |
| `thermal_barrier.py` | Transient conduction through a thermal protection / insulation layer | 1D heat equation |
| `electronic_cooling.py` | Transient temperature diffusion through a cooled electronics substrate | 1D heat equation |
| `viscous_flow.py` | Differentiable surrogate for a 1D viscous convection-diffusion flow | Burgers equation |

## Running the examples

From the repository root:

```bash
pip install -e .
python -m use_cases.thermal_barrier
python -m use_cases.electronic_cooling
python -m use_cases.viscous_flow
```

The examples import the package as a normal dependency:

```python
from pinn import MLP, PINNConfig, PINNTrainer, sample_heat_equation
```

The heat-transfer cases use the repository's standard `PINNTrainer`. The viscous-flow case demonstrates how the same `MLP` component can be reused to construct a custom PDE residual when the built-in heat-equation helper is not appropriate.

## Scope

These are reference implementations and educational engineering demonstrations. They are not validated industrial solvers and should not be used for safety-critical predictions without verification against appropriate analytical, experimental, or high-fidelity numerical results.
