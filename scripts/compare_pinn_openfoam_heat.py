#!/usr/bin/env python3
"""Compare an existing heat PINN with OpenFOAM and the exact solution.

Examples
--------
python scripts/compare_pinn_openfoam_heat.py --case "$FOAM_RUN/heat1d_pinn_refined" --model hard --causal
python scripts/compare_pinn_openfoam_heat.py --case "$FOAM_RUN/heat1d_pinn_refined" --model hard
python scripts/compare_pinn_openfoam_heat.py --case "$FOAM_RUN/heat1d_pinn_refined" --model soft
python scripts/compare_pinn_openfoam_heat.py --case "$FOAM_RUN/heat1d_pinn_refined" --model baseline

The OpenFOAM parser expects ASCII scalar fields on a uniform mesh spanning
x in [-1, 1], matching benchmarks/openfoam_heat1d/create_case.py.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from pinn import MLP, PINNConfig, PINNTrainer, sample_heat_equation
from pinn.core import (
    CausalWeighter,
    FixedWeights,
    GeneralMLP,
    GeneralPINNConfig,
    GeneralTrainer,
    HardDirichletAnsatz,
)
from pinn.problems import heat1d
from validate_openfoam_heat import read_scalar_field

def exact_solution(x: np.ndarray, time: float, alpha: float) -> np.ndarray:
    return np.exp(-alpha * np.pi**2 * time) * np.sin(np.pi * x)


def relative_l2(prediction: np.ndarray, reference: np.ndarray) -> float:
    denominator = np.linalg.norm(reference.ravel())
    if denominator == 0.0:
        return float("nan")
    return float(np.linalg.norm((prediction - reference).ravel()) / denominator)


def collect_snapshots(case: Path) -> list[tuple[float, Path]]:
    snapshots = []
    for directory in case.iterdir():
        field_path = directory / "T"
        if not directory.is_dir() or not field_path.is_file():
            continue
        try:
            time = float(directory.name)
        except ValueError:
            continue
        snapshots.append((time, field_path))
    snapshots.sort(key=lambda item: item[0])
    if not snapshots:
        raise RuntimeError(f"No numeric time directories containing T were found in {case}")
    return snapshots


def _read_foam(path: Path) -> np.ndarray:
    # Keep parsing logic local so this script can be run directly by path.
    return read_scalar_field(path)


def _baseline_callback(epoch: int, history) -> None:
    print(
        f"epoch={epoch:5d} total={history.total[-1]:.4e} "
        f"physics={history.physics[-1]:.4e} "
        f"initial={history.initial[-1]:.4e} "
        f"boundary={history.boundary[-1]:.4e}"
    )


def _general_callback(epoch: int, history: dict) -> None:
    print(
        f"epoch={epoch:5d} total={history['total'][-1]:.4e} "
        f"physics={history['physics'][-1]:.4e}"
    )


def train_model(args: argparse.Namespace, device: torch.device):
    """Build and train one of the repository's existing heat PINN variants."""
    torch.set_default_dtype(torch.float32)
    torch.manual_seed(args.seed)

    if args.model == "baseline":
        model = MLP(
            input_dim=2,
            output_dim=1,
            hidden_dim=64,
            hidden_layers=4,
            input_lower=(-1.0, 0.0),
            input_upper=(1.0, 1.0),
        ).to(device)
        points = sample_heat_equation(
            n_interior=args.n_interior,
            n_initial=args.n_initial,
            n_boundary=args.n_boundary,
            seed=args.seed,
            device=device,
        )
        config = PINNConfig(
            alpha=args.alpha,
            physics_weight=1.0,
            initial_weight=10.0,
            boundary_weight=10.0,
            learning_rate=args.learning_rate,
            epochs=args.epochs,
            log_every=args.log_every,
            use_lbfgs=not args.no_lbfgs,
            lbfgs_steps=args.lbfgs_steps,
            seed=args.seed,
        )
        trainer = PINNTrainer(model, config)
        trainer.train(callback=_baseline_callback)
        return trainer

    problem = heat1d.build_problem(
        alpha=args.alpha,
        n_interior=args.n_interior,
        n_boundary=args.n_boundary,
        n_initial=args.n_initial,
        sampling_method="lhs",
        seed=args.seed,
        hard_constraint=(args.model == "hard"),
    )
    base_model = GeneralMLP(problem.domain, hidden_dim=64, hidden_layers=4)

    if args.model == "hard":
        model = HardDirichletAnsatz(
            base_model,
            problem.domain,
            initial_condition_fn=heat1d.initial_condition,
        )
        weighter = FixedWeights({"physics": 1.0})
        causal = CausalWeighter(n_bins=10, epsilon=1.0) if args.causal else None
    else:
        model = base_model
        weighter = FixedWeights({
            "physics": 1.0,
            "initial": 10.0,
            "left_boundary": 10.0,
            "right_boundary": 10.0,
        })
        causal = None

    config = GeneralPINNConfig(
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        log_every=args.log_every,
        use_lbfgs=not args.no_lbfgs,
        lbfgs_steps=args.lbfgs_steps,
        seed=args.seed,
        causal=causal,
    )
    trainer = GeneralTrainer(model, problem, weighter, config)
    trainer.train(callback=_general_callback)
    return trainer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", required=True, help="OpenFOAM case directory")
    parser.add_argument(
        "--model",
        choices=("baseline", "soft", "hard"),
        default="hard",
        help="PINN formulation to train (default: hard)",
    )
    parser.add_argument(
        "--causal",
        action="store_true",
        help="Enable causal residual weighting; supported only with --model hard",
    )
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=3000)
    parser.add_argument("--n-interior", type=int, default=10000)
    parser.add_argument("--n-initial", type=int, default=2000)
    parser.add_argument("--n-boundary", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--log-every", type=int, default=250)
    parser.add_argument("--lbfgs-steps", type=int, default=250)
    parser.add_argument("--no-lbfgs", action="store_true", help="Disable L-BFGS refinement")
    parser.add_argument(
        "--output-dir",
        default="artifacts/openfoam_heat1d",
        help="Destination for generated CSVs, JSON summary and model weights",
    )
    args = parser.parse_args()

    if args.alpha <= 0:
        parser.error("--alpha must be positive")
    if args.causal and args.model != "hard":
        parser.error("--causal is supported only with --model hard")

    case = Path(args.case).expanduser().resolve()
    if not case.is_dir():
        parser.error(f"OpenFOAM case directory does not exist: {case}")

    snapshots = collect_snapshots(case)
    foam_fields = [(time, _read_foam(path)) for time, path in snapshots]
    n_cells = int(foam_fields[0][1].size)
    if any(values.size != n_cells for _, values in foam_fields):
        raise ValueError("OpenFOAM field size differs between saved times")

    # This benchmark has a uniform mesh over [-1, 1].
    dx = 2.0 / n_cells
    x = -1.0 + (np.arange(n_cells) + 0.5) * dx

    if args.model == "baseline" and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        # The generalized heat problem currently creates its point tensors on CPU.
        # Keep those runs on CPU to avoid a model/tensor device mismatch.
        device = torch.device("cpu")
    print(
        f"Training model={args.model}, causal={args.causal}, device={device}; "
        f"epochs={args.epochs}, cells={n_cells}, snapshots={len(snapshots)}"
    )

    trainer = train_model(args, device)

    if args.model == "hard":
        tag = "hard_causal" if args.causal else "hard_no_causal"
    else:
        tag = args.model
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(trainer.model.state_dict(), output_dir / f"{tag}.pt")

    metric_rows: list[dict[str, float | int | str]] = []
    field_rows: list[np.ndarray] = []
    time_errors: list[float] = []
    time_error_times: list[float] = []

    for time, foam in foam_fields:
        x_tensor = torch.as_tensor(x, dtype=torch.float32, device=device)
        t_tensor = torch.full_like(x_tensor, float(time))
        points = torch.stack((x_tensor, t_tensor), dim=1)
        with torch.no_grad():
            pinn = trainer.predict(points).detach().cpu().numpy().reshape(-1)

        exact = exact_solution(x, time, args.alpha)
        pinn_error = relative_l2(pinn, exact)
        foam_error = relative_l2(foam, exact)
        normalized_difference = float(
            np.linalg.norm(pinn - foam)
            / max(np.linalg.norm(exact), np.finfo(float).tiny)
        )
        time_errors.append(pinn_error)
        time_error_times.append(float(time))

        metric_rows.append({
            "time": float(time),
            "pinn_vs_exact_rel_l2": pinn_error,
            "openfoam_vs_exact_rel_l2": foam_error,
            "pinn_vs_openfoam_rel_l2_normalized_exact": normalized_difference,
            "pinn_vs_exact_max_abs": float(np.max(np.abs(pinn - exact))),
            "openfoam_vs_exact_max_abs": float(np.max(np.abs(foam - exact))),
            "pinn_vs_openfoam_max_abs": float(np.max(np.abs(pinn - foam))),
        })
        field_rows.append(np.column_stack((
            np.full(n_cells, time),
            x,
            pinn,
            foam,
            exact,
        )))

    metrics_path = output_dir / f"{tag}_per_time.csv"
    with metrics_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(metric_rows[0]))
        writer.writeheader()
        writer.writerows(metric_rows)

    field_path = output_dir / f"{tag}_fields.csv"
    all_fields = np.vstack(field_rows)
    np.savetxt(
        field_path,
        all_fields,
        delimiter=",",
        comments="",
        header="time,x,pinn,openfoam,exact",
        fmt="%.12g",
    )

    pinn_all, foam_all, exact_all = all_fields[:, 2], all_fields[:, 3], all_fields[:, 4]
    denominator = max(np.linalg.norm(exact_all), np.finfo(float).tiny)
    summary = {
        "case": str(case),
        "model": args.model,
        "causal_weighting": bool(args.causal),
        "seed": args.seed,
        "epochs": args.epochs,
        "alpha": args.alpha,
        "cells": n_cells,
        "snapshots": len(snapshots),
        "time_start": float(snapshots[0][0]),
        "time_end": float(snapshots[-1][0]),
        "space_time_relative_l2_pinn_vs_exact": relative_l2(pinn_all, exact_all),
        "space_time_relative_l2_openfoam_vs_exact": relative_l2(foam_all, exact_all),
        "space_time_relative_l2_pinn_vs_openfoam_normalized_exact": float(
            np.linalg.norm(pinn_all - foam_all) / denominator
        ),
        "mean_per_time_relative_l2_pinn_vs_exact": float(np.mean(time_errors)),
        "rms_per_time_relative_l2_pinn_vs_exact": float(np.sqrt(np.mean(np.square(time_errors)))),
        "median_per_time_relative_l2_pinn_vs_exact": float(np.median(time_errors)),
        "worst_per_time_relative_l2_pinn_vs_exact": float(np.max(time_errors)),
        "worst_error_time": float(time_error_times[int(np.argmax(time_errors))]),
        "metrics_csv": str(metrics_path),
        "fields_csv": str(field_path),
        "weights_file": str(output_dir / f"{tag}.pt"),
    }
    summary_path = output_dir / f"{tag}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print("\nRelative L2 errors at selected times")
    print(
        f"{'Time':>7} {'PINN/exact':>14} "
        f"{'OpenFOAM/exact':>16} {'PINN-OF*':>14}"
    )
    times = np.asarray([row["time"] for row in metric_rows], dtype=float)
    for target in np.linspace(times[0], times[-1], 11):
        row = metric_rows[int(np.argmin(np.abs(times - target)))]
        print(
            f"{float(row['time']):7.3f} "
            f"{float(row['pinn_vs_exact_rel_l2']):14.6e} "
            f"{float(row['openfoam_vs_exact_rel_l2']):16.6e} "
            f"{float(row['pinn_vs_openfoam_rel_l2_normalized_exact']):14.6e}"
        )

    print("\nSpace-time relative L2 errors on the same grid")
    print(f"PINN vs exact:     {summary['space_time_relative_l2_pinn_vs_exact']:.6e}")
    print(f"OpenFOAM vs exact: {summary['space_time_relative_l2_openfoam_vs_exact']:.6e}")
    print(
        "PINN vs OpenFOAM:  "
        f"{summary['space_time_relative_l2_pinn_vs_openfoam_normalized_exact']:.6e}"
    )
    print("PINN time-slice error summary")
    print(f"Mean:   {summary['mean_per_time_relative_l2_pinn_vs_exact']:.6e}")
    print(f"RMS:    {summary['rms_per_time_relative_l2_pinn_vs_exact']:.6e}")
    print(f"Median: {summary['median_per_time_relative_l2_pinn_vs_exact']:.6e}")
    print(
        f"Worst:  {summary['worst_per_time_relative_l2_pinn_vs_exact']:.6e} "
        f"at t={summary['worst_error_time']:.6f}"
    )
    print(f"\nMetrics: {metrics_path}")
    print(f"Fields:  {field_path}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
