# Thermal Barrier

## Purpose

Models transient heat conduction through a simplified thermal-barrier or insulation layer.

## Physics

The use case solves the 1D heat equation

$$
T_t = \alpha T_{xx}, \qquad x\in[-1,1],\; t\in[0,1],
$$

with the baseline initial condition

$$
T(x,0)=\sin(\pi x),
$$

and homogeneous Dirichlet boundaries

$$
T(-1,t)=T(1,t)=0.
$$

## PINN usage

The example imports the public package API:

```python
from pinn import MLP, PINNConfig, PINNTrainer, sample_heat_equation
```

The sampled collocation, initial, and boundary points are passed to `PINNTrainer`. The trained network is then queried at selected locations and times.

## Run

From the repository root:

```bash
python use_cases/thermal_barrier/run.py
```

## Engineering note

The coordinates and diffusivity are dimensionless in this example. Physical thickness, material properties, and dimensional time should be introduced by an application-specific nondimensionalization or parameter mapping.
