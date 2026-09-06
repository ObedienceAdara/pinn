# PINN Use Cases

Application-style examples showing how the `pinn` package is consumed and extended.

## Structure

```text
use_cases/
├── README.md
├── thermal_barrier/
│   ├── README.md
│   └── run.py
├── electronics_cooling/
│   ├── README.md
│   └── run.py
└── viscous_flow/
    ├── README.md
    └── run.py
```

## Use cases

| Directory | Application | Method |
| --- | --- | --- |
| `thermal_barrier/` | Simplified transient thermal protection / insulation | Heat-equation PINN |
| `electronics_cooling/` | Simplified transient electronics thermal diffusion | Heat-equation PINN |
| `viscous_flow/` | 1D viscous flow demonstration | Custom Burgers residual + `pinn.MLP` |

Each directory contains a small `README.md` and a runnable `run.py`.

## Run

From the repository root:

```bash
python -m pip install -e .
python use_cases/thermal_barrier/run.py
python use_cases/electronics_cooling/run.py
python use_cases/viscous_flow/run.py
```

## Scope

These are technical reference examples, not validated industrial or safety-critical solvers. Production use requires appropriate analytical, experimental, and/or high-fidelity numerical validation.
