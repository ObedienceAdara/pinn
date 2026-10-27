#!/usr/bin/env python3
"""Validate an OpenFOAM 1-D scalar-diffusion run against its exact solution.

The parser targets ASCII OpenFOAM volScalarField files with a nonuniform
internalField and a uniform mesh spanning x in [-1, 1]. These assumptions
match benchmarks/openfoam_heat1d/create_case.py.
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import numpy as np


def read_scalar_field(path: Path) -> np.ndarray:
    """Read the nonuniform scalar values from an ASCII OpenFOAM field."""
    text = path.read_text(encoding="utf-8")
    match = re.search(
        r"internalField\s+nonuniform\s+List<scalar>\s+"
        r"(\d+)\s*\((.*?)\)\s*;",
        text,
        re.DOTALL,
    )
    if match is None:
        raise ValueError(
            f"Could not parse a nonuniform ASCII scalar internalField in {path}"
        )

    expected = int(match.group(1))
    values = np.fromstring(match.group(2), sep=" ")
    if values.size != expected:
        raise ValueError(
            f"{path}: expected {expected} scalar values, found {values.size}"
        )
    return values


def exact_solution(x: np.ndarray, time: float, alpha: float) -> np.ndarray:
    """Exact solution for u_t = alpha*u_xx, u(x,0)=sin(pi*x), u(±1,t)=0."""
    return np.exp(-alpha * np.pi**2 * time) * np.sin(np.pi * x)


def relative_l2(prediction: np.ndarray, reference: np.ndarray) -> float:
    """Relative Euclidean L2 error; returns NaN for a zero reference."""
    denominator = np.linalg.norm(reference.ravel())
    if denominator == 0.0:
        return float("nan")
    return float(np.linalg.norm((prediction - reference).ravel()) / denominator)


def collect_snapshots(case: Path) -> list[tuple[float, Path]]:
    """Return sorted (time, T-field path) entries from a case directory."""
    snapshots: list[tuple[float, Path]] = []
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


def validate_case(case: Path, alpha: float) -> list[dict[str, float]]:
    """Compute per-time errors for a uniform mesh on x in [-1, 1]."""
    snapshots = collect_snapshots(case)
    results: list[dict[str, float]] = []
    expected_cells: int | None = None

    for time, field_path in snapshots:
        numerical = read_scalar_field(field_path)
        if expected_cells is None:
            expected_cells = int(numerical.size)
        elif numerical.size != expected_cells:
            raise ValueError(
                f"Field cell count changed at t={time}: {numerical.size} vs {expected_cells}"
            )

        n_cells = numerical.size
        dx = 2.0 / n_cells
        x = -1.0 + (np.arange(n_cells) + 0.5) * dx
        exact = exact_solution(x, time, alpha)
        difference = numerical - exact

        results.append(
            {
                "time": float(time),
                "cells": int(n_cells),
                "relative_l2": relative_l2(numerical, exact),
                "max_abs": float(np.max(np.abs(difference))),
            }
        )
    return results


def write_csv(path: Path, rows: list[dict[str, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "case",
        help="OpenFOAM case directory containing numeric time directories and T fields",
    )
    parser.add_argument("--alpha", type=float, default=0.1, help="Diffusivity; default 0.1")
    parser.add_argument(
        "--output",
        default="artifacts/openfoam_heat1d_validation.csv",
        help="CSV output path (default is ignored by Git)",
    )
    args = parser.parse_args()

    if args.alpha <= 0:
        parser.error("--alpha must be positive")

    case = Path(args.case).expanduser().resolve()
    if not case.is_dir():
        parser.error(f"Case directory does not exist: {case}")

    rows = validate_case(case, args.alpha)
    output = Path(args.output).expanduser()
    write_csv(output, rows)

    times = np.asarray([row["time"] for row in rows])
    errors = np.asarray([row["relative_l2"] for row in rows])
    absolute = np.asarray([row["max_abs"] for row in rows])
    valid = np.isfinite(errors)

    worst_l2_index = int(np.nanargmax(errors))
    worst_abs_index = int(np.argmax(absolute))
    first, last = rows[0], rows[-1]

    print(f"Case: {case}")
    print(f"Snapshots: {len(rows)}")
    print(f"Cells: {first['cells']}")
    print(f"Time range: {times[0]:.6f} to {times[-1]:.6f}")
    print(f"Diffusivity alpha: {args.alpha:g}")
    print()
    print(f"{'Time':>10} {'Relative L2':>15} {'Max absolute error':>20}")

    targets = np.linspace(times[0], times[-1], 11)
    for target in targets:
        index = int(np.argmin(np.abs(times - target)))
        row = rows[index]
        print(
            f"{row['time']:10.4f} "
            f"{row['relative_l2']:15.6e} "
            f"{row['max_abs']:20.6e}"
        )

    print()
    print(f"Worst relative L2 error: {rows[worst_l2_index]['relative_l2']:.6e} "
          f"at t={rows[worst_l2_index]['time']:.4f}")
    print(f"Worst maximum absolute error: {rows[worst_abs_index]['max_abs']:.6e} "
          f"at t={rows[worst_abs_index]['time']:.4f}")
    print(f"CSV written to: {output.resolve()}")
    if not np.all(valid):
        print("Note: at least one relative error is undefined because the reference norm is zero.")
    print(f"Initial relative L2 error: {first['relative_l2']:.6e}")
    print(f"Final relative L2 error:   {last['relative_l2']:.6e}")


if __name__ == "__main__":
    main()
