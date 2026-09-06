# Electronics Cooling

## Purpose

Models transient temperature diffusion through a simplified electronics substrate.

## Physics

The example uses the same 1D heat equation as the baseline PINN:

$$
T_t = \alpha T_{xx}.
$$

The network represents a differentiable temperature field `T(x, t)`.

## PINN usage

The application imports:

```python
from pinn import MLP, PINNConfig, PINNTrainer, sample_heat_equation
```

It samples the domain, trains the PINN with physics/initial/boundary losses, and evaluates the learned field on a vectorized space-time grid.

## Run

From the repository root:

```bash
python use_cases/electronics_cooling/run.py
```

## Engineering note

The example uses normalized coordinates and an effective diffusivity. A production thermal model should map these quantities to measured geometry, material properties, heat sources, and boundary conditions.
