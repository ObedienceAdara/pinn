# Viscous Flow

## Purpose

Demonstrates reuse of the package `MLP` for a custom fluid PDE that is not handled by the baseline heat-equation trainer.

## Physics

The example uses the 1D viscous Burgers equation:

$$
u_t + 
u\nu_x = \mu\nu_{xx}$$

with an initial profile and zero Dirichlet boundary values.

## PINN usage

The example imports:

```python
from pinn import MLP
```

It then uses PyTorch automatic differentiation to build a custom PDE residual, samples collocation and constraint points, optimizes the same neural solution ansatz, and evaluates the resulting velocity field.

## Run

From the repository root:

```bash
python use_cases/viscous_flow/run.py
```

## Engineering note

This is a 1D demonstration of the PINN extension pattern. It is not a Navier-Stokes solver and should not be interpreted as a validated CFD model.
