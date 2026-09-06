# PINN Implementation Guide

## 1. Scope

This document describes the current implementation in this repository at the module, class, function, tensor, loss, and execution-flow level.

The project implements a standard Physics-Informed Neural Network (PINN) baseline in PyTorch for the one-dimensional transient heat equation. The code is intentionally small and explicit so that the numerical method can be inspected, tested, and extended.

The current reference problem is

$$
 u_t = \alpha u_{xx},
\qquad x\in[-1,1],\quad t\in[0,1],
$$

with

$$
 u(x,0)=\sin(\pi x),
$$

and

$$
 u(-1,t)=u(1,t)=0.
$$

The analytical solution is

$$
 u(x,t)=e^{-\alpha\pi^2t}\sin(\pi x).
$$

## 2. Repository architecture

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
├── use_cases/
│   ├── __init__.py
│   ├── README.md
│   ├── thermal_barrier/
│   │   ├── README.md
│   │   └── run.py
│   ├── electronics_cooling/
│   │   ├── README.md
│   │   └── run.py
│   └── viscous_flow/
│       ├── README.md
│       └── run.py
├── tests/
├── .github/workflows/ci.yml
├── pyproject.toml
├── README.md
└── BROAD.md
```

The separation is intentional:

- `pinn/` contains reusable library code.
- `examples/` contains the canonical repository example.
- `use_cases/` contains application-oriented examples that show how a user consumes or extends the library.
- `tests/` validates the reusable implementation.
- `BROAD.md` documents the implementation.

## 3. Core mathematical model

A PINN represents an unknown physical field with a neural network:

$$
 u_\theta(x,t) \approx u(x,t).
$$

The network parameters are

$$
\theta = \{W_l,b_l\}_{l=1}^{L}.
$$

The PINN is not trained only against labeled solution values. It is trained to satisfy the governing PDE and the problem constraints.

For the heat equation, define the residual

$$
 r_\theta(x,t)=
\frac{\partial u_\theta}{\partial t}
-
\alpha\frac{\partial^2u_\theta}{\partial x^2}.
$$

For an exact solution,

$$
 r_\theta(x,t)=0
$$

throughout the domain.

The implementation therefore minimizes a composite loss

$$
\mathcal L =
\lambda_f\mathcal L_f
+
\lambda_{ic}\mathcal L_{ic}
+
\lambda_{bc}\mathcal L_{bc}.
$$

Here:

- `L_f` is the interior physics residual loss.
- `L_ic` is the initial-condition loss.
- `L_bc` is the boundary-condition loss.
- `lambda_*` are configurable weights.

## 4. `pinn/__init__.py`

Purpose: define the public package API.

The module imports and exposes:

```python
MLP
heat_residual
heat_exact_solution
HeatEquationPoints
sample_heat_equation
PINNConfig
PINNTrainer
```

The `__all__` list defines the symbols intended for normal package-level imports.

Typical consumer code is therefore:

```python
from pinn import MLP, PINNConfig, PINNTrainer, sample_heat_equation
```

This keeps application code independent from internal module paths.

## 5. `pinn/models.py`

### 5.1 `MLP`

`MLP` is the neural-network solution ansatz.

Inheritance:

```text
nn.Module
  └── MLP
```

Signature:

```python
MLP(
    input_dim=2,
    output_dim=1,
    hidden_dim=64,
    hidden_layers=4,
    input_lower=(-1.0, 0.0),
    input_upper=(1.0, 1.0),
)
```

For the heat equation:

```text
input:  [x, t]
output: [u_theta(x,t)]
```

### 5.2 `MLP.__init__`

Responsibilities:

1. Validate the architecture parameters.
2. Convert physical input bounds into tensors.
3. Register bounds as module buffers.
4. Construct the feed-forward network.
5. Initialize all linear layers.

Validation rules:

- `hidden_layers >= 1`.
- `hidden_dim >= 1`.
- `len(input_lower) == input_dim`.
- `len(input_upper) == input_dim`.
- every upper bound is greater than its matching lower bound.

### 5.3 Input bounds

The network stores:

```python
self.input_lower
self.input_upper
```

as registered buffers.

They are therefore part of the module state and automatically follow the model when it is moved between compatible devices/dtypes.

For the reference problem:

```text
input_lower = [-1, 0]
input_upper = [ 1, 1]
```

corresponding to `(x,t)`.

### 5.4 Network construction

The architecture is:

```text
Linear(input_dim, hidden_dim)
Tanh
Linear(hidden_dim, hidden_dim)
Tanh
...
Linear(hidden_dim, hidden_dim)
Tanh
Linear(hidden_dim, output_dim)
```

With the defaults, there are four hidden layers of width 64 and one scalar output.

The output layer has no activation.

### 5.5 Why `Tanh`

The PINN needs first- and second-order input derivatives. `tanh` is smooth and has well-defined derivatives of the orders required by the reference PDE.

This is a numerical design choice, not a requirement that every PINN must use `tanh`.

### 5.6 `_initialize`

Private helper:

```python
_initialize()
```

Every `nn.Linear` receives:

```python
nn.init.xavier_normal_(module.weight)
n.init.zeros_(module.bias)
```

Xavier initialization is used as the baseline initialization for the `tanh` network.

### 5.7 `normalize_inputs`

Signature:

```python
normalize_inputs(inputs: torch.Tensor) -> torch.Tensor
```

Maps each physical input from `[lower, upper]` to `[-1, 1]` using

$$
\hat{x}=2\frac{x-x_{min}}{x_{max}-x_{min}}-1.
$$

For the heat equation:

- `x=-1` maps to `-1`.
- `x=+1` maps to `+1`.
- `t=0` maps to `-1`.
- `t=1` maps to `+1`.

The operation is vectorized over a batch of points.

### 5.8 `forward`

Signature:

```python
forward(inputs: torch.Tensor) -> torch.Tensor
```

Expected shape:

```text
(N, input_dim)
```

It first validates the shape, then performs:

```text
physical coordinates
        ↓
normalize_inputs
        ↓
feed-forward network
        ↓
u_theta
```

For the heat equation:

```text
(N, 2) -> (N, 1)
```

## 6. `pinn/physics.py`

This module contains the physics-specific reference equations.

### 6.1 `heat_residual`

Signature:

```python
heat_residual(
    model: nn.Module,
    xt: torch.Tensor,
    alpha: float,
) -> torch.Tensor
```

Purpose: evaluate

$$
 u_t-\alpha u_{xx}
$$

at a batch of collocation points.

Expected input:

```text
xt.shape == (N, 2)
xt[:,0] = x
xt[:,1] = t
```

### 6.2 Gradient preparation

If `xt` does not require gradients, the function creates a detached copy and enables gradients:

```python
xt = xt.clone().detach().requires_grad_(True)
```

This is necessary because the PDE residual requires derivatives of the network output with respect to the coordinates.

### 6.3 Network evaluation

```python
u = model(xt)
```

Expected output:

```text
(N, 1)
```

A shape check rejects models that do not produce one scalar field value per point.

### 6.4 First derivative

PyTorch automatic differentiation computes

$$
\nabla_{x,t}u_	heta
$$

with:

```python
grad_u = torch.autograd.grad(
    u,
    xt,
    grad_outputs=torch.ones_like(u),
    create_graph=True,
    retain_graph=True,
)[0]
```

The resulting tensor has shape `(N, 2)`.

The columns are split as:

```text
u_x = grad_u[:, 0:1]
u_t = grad_u[:, 1:2]
```

`create_graph=True` is required because `u_x` must itself be differentiated to obtain `u_xx`, and because the residual must remain differentiable for backpropagation into the network parameters.

### 6.5 Second derivative

The implementation differentiates `u_x` once more with respect to `xt`:

```python
u_xx = torch.autograd.grad(
    u_x,
    xt,
    grad_outputs=torch.ones_like(u_x),
    create_graph=True,
    retain_graph=True,
)[0][:, 0:1]
```

The first coordinate derivative is selected because it corresponds to `x`.

### 6.6 Residual output

The final residual is:

```python
u_t - alpha * u_xx
```

Shape:

```text
(N, 1)
```

This tensor is not reduced to a scalar inside `heat_residual`. The trainer performs the squared mean so that the residual remains inspectable.

### 6.7 `heat_initial_condition`

Signature:

```python
heat_initial_condition(x: torch.Tensor) -> torch.Tensor
```

Returns:

$$
\sin(\pi x).
$$

The function is vectorized and preserves the input batch shape.

### 6.8 `heat_boundary_condition`

Signature:

```python
heat_boundary_condition(x, t)
```

Returns zero values for both Dirichlet boundaries.

The `x` argument is accepted because a generic boundary-condition interface naturally receives coordinates, but the reference boundary value is independent of coordinate.

### 6.9 `heat_exact_solution`

Signature:

```python
heat_exact_solution(x, t, alpha)
```

Returns

$$
 e^{-\alpha\pi^2t}\sin(\pi x).
$$

This function is primarily for validation and benchmarking. The PINN does not use the analytical solution in its training loss.

That separation is important: the network learns from the PDE and constraints, while the exact solution is used afterward to measure error.

## 7. `pinn/sampling.py`

This module creates the training point sets.

### 7.1 `HeatEquationPoints`

Frozen dataclass containing four tensors:

```python
interior
initial
left_boundary
right_boundary
```

Each tensor represents a different mathematical part of the problem.

### 7.2 Interior points

`interior` represents points inside the space-time domain:

$$
(x,t)\in[-1,1]\times[0,1].
$$

These points are used only for the PDE residual loss.

There is no target solution value attached to them.

### 7.3 Initial points

`initial` contains:

$$
(x,0).
$$

These points enforce the initial condition.

### 7.4 Boundary points

`left_boundary` contains:

$$
(-1,t),
$$

and `right_boundary` contains:

$$
(1,t).
$$

They enforce the two Dirichlet boundaries.

### 7.5 `_uniform`

Private helper:

```python
_uniform(generator, shape, low, high, dtype)
```

Generates uniformly distributed random tensors using a provided local random generator.

It implements:

$$
X = a+(b-a)U,
$$

where

$$
U\sim U(0,1).
$$

### 7.6 `sample_heat_equation`

Signature:

```python
sample_heat_equation(
    n_interior=10_000,
    n_initial=2_000,
    n_boundary=2_000,
    seed=42,
    dtype=torch.float32,
    device="cpu",
)
```

The function validates that each sample count is at least one, constructs a local CPU generator, samples all four point sets, and finally moves the tensors to the requested device.

The use of a local generator is deliberate: deterministic sampling should not alter PyTorch's global random-number state.

### 7.7 Sampling tensor shapes

For counts `N_f`, `N_0`, and `N_b`:

```text
interior       : (N_f, 2)
initial        : (N_0, 2)
left_boundary  : (N_b, 2)
right_boundary : (N_b, 2)
```

## 8. `pinn/trainer.py`

This module contains the optimization layer.

### 8.1 `PINNConfig`

Dataclass storing training configuration:

```text
alpha
physics_weight
initial_weight
boundary_weight
learning_rate
epochs
log_every
use_lbfgs
lbfgs_steps
seed
```

Defaults provide a baseline that can be overridden without modifying library code.

### 8.2 Configuration validation

The trainer rejects:

- non-positive diffusivity `alpha`.
- fewer than one epoch.
- negative loss weights.

The intent is to fail at configuration time rather than halfway through training.

### 8.3 `TrainingHistory`

Stores four Python lists:

```text
total
physics
initial
boundary
```

Every Adam epoch appends one scalar value to each list.

This gives the caller enough information to diagnose whether a low total loss is being achieved by all components or by one component dominating the objective.

### 8.4 `PINNTrainer.__init__`

The trainer stores the model and configuration, validates configuration values, and seeds PyTorch using `config.seed`.

It does not construct the model itself. This separation allows the caller to choose the architecture.

### 8.5 `loss_components`

Signature:

```python
loss_components(points: HeatEquationPoints)
```

Returns:

```text
(total, physics_loss, initial_loss, boundary_loss)
```

#### Physics loss

The interior points are detached and made differentiable:

```python
interior = points.interior.clone().detach().requires_grad_(True)
```

The residual is evaluated, then:

$$
\mathcal L_f
=
\operatorname{mean}(r_\theta^2).
$$

#### Initial-condition loss

Prediction:

```python
initial_pred = model(points.initial)
```

Target:

```python
sin(pi*x)
```

Loss:

$$
\mathcal L_{ic}
=
\operatorname{mean}
\left(u_\theta(x,0)-\sin(\pi x)\right)^2.
$$

#### Boundary-condition loss

The model is evaluated independently at left and right boundaries.

Both targets are zero.

The two boundary MSE values are averaged:

$$
\mathcal L_{bc}
=
\frac12
\left(
\operatorname{MSE}_{left}
+
\operatorname{MSE}_{right}
\right).
$$

#### Weighted total loss

Finally:

$$
\mathcal L
=
\lambda_f\mathcal L_f
+
\lambda_{ic}\mathcal L_{ic}
+
\lambda_{bc}\mathcal L_{bc}.
$$

This is the scalar objective passed to `backward()`.

### 8.6 `train`

Signature:

```python
train(points, callback=None) -> TrainingHistory
```

The training algorithm is:

```text
create Adam
    ↓
for each epoch
    ↓
zero gradients
    ↓
compute physics / IC / BC losses
    ↓
form weighted total loss
    ↓
backward()
    ↓
optimizer.step()
    ↓
record history
```

The optional callback receives `(epoch, history)` at epoch 1, configured logging intervals, and the final epoch.

### 8.7 Optional L-BFGS refinement

If `use_lbfgs=True`, Adam is followed by `_lbfgs_refinement`.

This uses PyTorch's L-BFGS optimizer with a closure that recomputes the complete loss and gradients.

The design is therefore:

```text
Adam
 ↓
coarse optimization
 ↓
L-BFGS
 ↓
local refinement
```

L-BFGS is optional because its closure-based optimization can be substantially more expensive per step than Adam, especially when high-order derivatives are used.

### 8.8 `_lbfgs_refinement`

Private helper that constructs:

```python
torch.optim.LBFGS(...)
```

The closure:

1. zeros gradients.
2. recomputes the loss.
3. calls `backward()`.
4. returns the scalar loss.

The optimizer then performs the configured quasi-Newton refinement.

### 8.9 `predict`

Signature:

```python
predict(xt: torch.Tensor) -> torch.Tensor
```

The method switches the model to evaluation mode, disables gradient calculation, and returns network predictions.

This path is intended for inference, visualization, downstream optimization, and repeated field queries after training.

## 9. End-to-end execution flow

A normal user workflow is:

```text
1. import public API
        ↓
2. sample points
        ↓
3. construct MLP
        ↓
4. configure PINNTrainer
        ↓
5. train
        ↓
6. evaluate prediction
        ↓
7. compare against exact solution or engineering reference
```

Concrete form:

```python
from pinn import MLP, PINNConfig, PINNTrainer, sample_heat_equation

points = sample_heat_equation(...)
model = MLP(...)
trainer = PINNTrainer(model, PINNConfig(...))
trainer.train(points)
prediction = trainer.predict(query)
```

## 10. What happens during one training epoch

For every epoch, the trainer computes three independent physics-informed objectives.

### Step 1: interior points

The network produces

$$
 u_\theta(x_i,t_i).
$$

Automatic differentiation produces

$$
 u_t(x_i,t_i),
\qquad
 u_{xx}(x_i,t_i).
$$

The residual is

$$
 r_i=u_t-\alpha u_{xx}.
$$

Then

$$
\mathcal L_f=\frac1{N_f}\sum_i r_i^2.
$$

### Step 2: initial points

Evaluate

$$
 u_\theta(x_j,0).
$$

Compare with

$$
\sin(\pi x_j).
$$

### Step 3: boundary points

Evaluate

$$
 u_\theta(-1,t_k),
\qquad
 u_\theta(1,t_k).
$$

Compare both with zero.

### Step 4: combine

Compute

$$
\mathcal L=\lambda_f\mathcal L_f+\lambda_{ic}\mathcal L_{ic}+\lambda_{bc}\mathcal L_{bc}.
$$

### Step 5: parameter update

PyTorch differentiates the scalar loss with respect to all trainable network parameters:

$$
\nabla_\theta\mathcal L.
$$

Adam updates the parameters.

## 11. Tensor and autograd flow

The important dependency graph is:

```text
x,t
 ↓
normalize_inputs(x,t)
 ↓
MLP
 ↓
u_theta
 ├──────────────┐
 ↓              ↓
∂u/∂x           ∂u/∂t
 ↓
∂²u/∂x²
 └──────┬───────┘
        ↓
 u_t - alpha*u_xx
        ↓
      MSE
        ↓
 total loss
        ↓
    backward()
        ↓
 network parameters
```

This is the essential mechanism that differentiates a PINN from an ordinary supervised network in this repository.

## 12. Training data versus collocation data

The reference implementation does not require a full labeled solution dataset.

There are three kinds of points:

| Point set | Has target value? | Used for |
|---|---:|---|
| Interior | No | PDE residual |
| Initial | Yes | Initial condition |
| Boundary | Yes | Boundary conditions |

The interior points are therefore called **collocation points**.

The physics target at an interior point is effectively zero residual, not a measured field value.

## 13. Exact solution and validation

`heat_exact_solution` is not part of the training objective.

It exists so the example can compare

$$
 u_\theta(x,t)
$$

against

$$
 u_{exact}(x,t).
$$

A standard validation metric is relative L2 error:

$$
\frac{\|u_\theta-u_{exact}\|_2}
{\|u_{exact}\|_2}.
$$

This distinction should remain explicit in future work: a PINN should not accidentally use the exact analytical answer as hidden training supervision when the purpose is to test physics-informed learning.

## 14. Numerical design choices

### Smooth network

`Tanh` is used because the current PDE requires second derivatives.

### Input normalization

Coordinates are mapped to `[-1,1]` because normalized inputs commonly make optimization better conditioned than raw heterogeneous physical scales.

### Xavier initialization

Weights are initialized for the chosen `tanh` architecture rather than relying entirely on framework defaults.

### Explicit loss components

The implementation records physics, initial, and boundary losses separately. This makes optimization failures easier to diagnose.

### Deterministic sampling

Sampling accepts an explicit seed and uses a local generator.

### Adam followed by optional L-BFGS

The implementation permits the common two-stage optimization pattern without forcing every experiment to pay the L-BFGS cost.

## 15. What this implementation does not claim to solve

The current package is a baseline, not a general-purpose PDE framework.

It does not yet provide built-in abstractions for:

- arbitrary PDE definitions.
- arbitrary domain geometries.
- Neumann or Robin boundary operators as first-class objects.
- multiple coupled fields.
- adaptive collocation.
- automatic loss balancing.
- hard constraint transformations.
- domain decomposition.
- parameter inference.
- uncertainty quantification.
- distributed training.
- operator learning.
- production CFD validation.

Those are extension points rather than missing pieces of the current baseline specification.

## 16. Extension pattern

The viscous-flow use case demonstrates the intended minimum extension pattern.

A new PDE can reuse the `MLP` solution representation while defining a new residual with automatic differentiation.

For a PDE

$$
\mathcal F[u]=0,
$$

the extension concept is:

```text
MLP
 ↓
new residual function
 ↓
physics loss
 ↓
constraint losses
 ↓
optimizer
```

A more mature package should eventually generalize this into a reusable PDE/trainer abstraction instead of duplicating training loops.

## 17. Use cases

The current application examples are intentionally simple.

### Thermal barrier

Uses the public heat-equation API to frame transient conduction as a simplified insulation or thermal-protection problem.

### Electronics cooling

Uses the same PDE workflow to represent temperature diffusion in a simplified electronics substrate and demonstrates vectorized field inference.

### Viscous flow

Imports the reusable `MLP` and creates a Burgers-equation residual manually. This demonstrates how the core neural representation can be reused for a different PDE while the baseline trainer remains heat-equation-specific.

## 18. Performance considerations

The expensive part of a PINN is not ordinary forward inference alone. During training, higher-order derivatives create additional autograd graphs and increase memory and compute requirements.

For the heat equation, every interior point requires network evaluation plus derivatives through the network sufficient to compute `u_t` and `u_xx`.

Therefore:

```text
more collocation points
        →
more autograd work
        →
more memory / compute
```

and

```text
higher PDE derivative order
        →
deeper derivative graph
        →
more expensive optimization
```

The repository therefore treats inference and training as separate computational regimes.

## 19. Reproducibility model

Reproducibility has two main controls:

1. sampled collocation points use an explicit `seed`.
2. the trainer seeds PyTorch with `PINNConfig.seed`.

Exact bitwise reproducibility across all hardware and software configurations should not be inferred from these controls alone; deterministic execution can depend on the underlying backend and operations.

## 20. Testing strategy

The repository tests the implementation at several levels.

### Model tests

Validate input shapes, normalization behavior, and output dimensions.

### Physics tests

Validate derivative calculations and the analytical heat-equation solution/reference behavior.

### Sampling tests

Validate sample shapes, domain bounds, boundary locations, and deterministic seeded sampling.

### Trainer tests

Validate loss computation and basic optimization behavior.

### Use-case tests

Validate that application examples import the package correctly and remain structurally compatible with the public API.

The goal is not to prove the PINN converges for every configuration. Tests protect the implementation contract and catch regressions.

## 21. Packaging

`pyproject.toml` defines the package as `pinn`, requires Python 3.10+, and declares PyTorch and Matplotlib as runtime dependencies. `pytest` is provided as a development dependency.

The package discovery rule includes:

```text
pinn*
```

so the reusable library is installable with:

```bash
python -m pip install -e .
```

## 22. Public API contract

The intended high-level API is small:

```python
from pinn import (
    MLP,
    heat_residual,
    heat_exact_solution,
    HeatEquationPoints,
    sample_heat_equation,
    PINNConfig,
    PINNTrainer,
)
```

The current strongest abstraction boundary is therefore:

```text
pinn/
    reusable implementation

examples/
    canonical demonstrations

use_cases/
    consumer-oriented applications and extension examples
```

## 23. Implementation summary

The complete baseline can be summarized as:

$$
\boxed{
\text{coordinates}
\rightarrow
\text{normalized MLP}
\rightarrow
u_\theta
\rightarrow
\text{autograd derivatives}
\rightarrow
\text{PDE residual}
}
$$

combined with

$$
\boxed{
\text{initial loss}
+
\text{boundary loss}
+
\text{physics loss}
\rightarrow
\text{optimizer}
}
$$

The network is therefore learning a function that simultaneously minimizes violation of the governing equation and the specified problem constraints.

That is the complete numerical idea implemented by the current repository.
