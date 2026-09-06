# PINN notebooks

A structured notebook suite for understanding, reproducing, validating, and extending the repository PINN implementation.

## Recommended sequence

1. `01_pinn_foundations.ipynb` — build the PINN from the PDE upward.
2. `02_repository_implementation.ipynb` — map the notebook implementation to the reusable package.
3. `03_autograd_and_pde.ipynb` — inspect first/second derivatives and gradient behavior.
4. `04_training_and_optimization.ipynb` — study loss weighting, reproducibility, Adam, and L-BFGS.
5. `05_validation_and_visualization.ipynb` — quantify field, PDE, initial-condition, and boundary errors.
6. `06_advanced_extensions.ipynb` — adaptive collocation and an inverse diffusion-coefficient example.

## Environment

From the repository root:

```bash
python -m pip install -e ".[dev]"
```

Then start Jupyter:

```bash
jupyter lab
```

The notebooks are educational/research implementations. Numerical results can vary with PyTorch versions, hardware, and optimization settings. Analytical validation is preferred whenever an exact solution is available.
