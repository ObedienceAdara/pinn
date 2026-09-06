# PINN notebooks

The notebooks provide both a canonical end-to-end implementation and focused technical studies.

## Canonical entry point

`main.ipynb` is the primary notebook. It contains the complete implementation from the PDE definition through model construction, automatic differentiation, collocation sampling, composite loss construction, Adam training, optional L-BFGS refinement, analytical validation, visualization, residual diagnostics, adaptive refinement, inverse parameter identification, and verification of the reusable repository API.

Start here for a complete implementation.

## Focused studies

1. `01_pinn_foundations.ipynb` — build the PINN from the PDE upward.
2. `02_repository_implementation.ipynb` — map the notebook implementation to the reusable package.
3. `03_autograd_and_pde.ipynb` — inspect first/second derivatives and gradient behavior.
4. `04_training_and_optimization.ipynb` — study loss weighting, reproducibility, Adam, and L-BFGS.
5. `05_validation_and_visualization.ipynb` — quantify field, PDE, initial-condition, and boundary errors.
6. `06_advanced_extensions.ipynb` — adaptive collocation and inverse diffusion-coefficient identification.

## Environment

From the repository root:

```bash
python -m pip install -e ".[dev]"
python -m pip install jupyterlab
jupyter lab
```

The notebooks are educational/research implementations. Numerical results can vary with PyTorch versions, hardware, and optimization settings. Analytical validation is preferred whenever an exact solution is available.
