from __future__ import annotations

import ast
import math
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "benchmarks" / "openfoam_heat1d"))
sys.path.insert(0, str(ROOT / "scripts"))

from create_case import create_case  # noqa: E402
from validate_openfoam_heat import read_scalar_field  # noqa: E402


def _read_initial_values(path: Path) -> list[float]:
    text = path.read_text(encoding="utf-8")
    match = re.search(
        r"internalField\s+nonuniform\s+List<scalar>\s+(\d+)\s*\((.*?)\)\s*;",
        text,
        re.DOTALL,
    )
    assert match is not None
    expected = int(match.group(1))
    values = [float(value) for value in match.group(2).split()]
    assert len(values) == expected
    return values


def test_case_generator_writes_matching_heat_problem(tmp_path: Path) -> None:
    cells = 8
    case = create_case(
        tmp_path / "heat_case",
        cells=cells,
        alpha=0.1,
        delta_t=0.001,
        end_time=1.0,
        write_interval=0.01,
    )

    assert (case / "0" / "T").is_file()
    assert (case / "constant" / "physicalProperties").is_file()
    assert (case / "system" / "controlDict").is_file()
    assert (case / "system" / "blockMeshDict").is_file()
    assert (case / "system" / "fvSchemes").is_file()
    assert (case / "system" / "fvSolution").is_file()

    mesh = (case / "system" / "blockMeshDict").read_text(encoding="utf-8")
    assert f"({cells} 1 1) simpleGrading" in mesh

    properties = (case / "constant" / "physicalProperties").read_text(encoding="utf-8")
    assert "DT              [0 2 -1 0 0 0 0] 0.1;" in properties

    field = (case / "0" / "T").read_text(encoding="utf-8")
    assert "left" in field and "right" in field
    assert field.count("type empty;") == 4

    values = _read_initial_values(case / "0" / "T")
    expected = [
        math.sin(math.pi * (-1.0 + (i + 0.5) * (2.0 / cells)))
        for i in range(cells)
    ]
    assert values == pytest.approx(expected, abs=1e-14)


def test_case_generator_refuses_to_overwrite_existing_case(tmp_path: Path) -> None:
    output = tmp_path / "existing_case"
    output.mkdir()
    (output / "keep_me.txt").write_text("preserve", encoding="utf-8")

    with pytest.raises(FileExistsError):
        create_case(output)

    assert (output / "keep_me.txt").read_text(encoding="utf-8") == "preserve"


@pytest.mark.parametrize(
    "kwargs",
    [
        {"cells": 1},
        {"alpha": 0},
        {"delta_t": 0},
        {"end_time": 0},
        {"write_interval": 0},
    ],
)
def test_case_generator_rejects_invalid_parameters(tmp_path: Path, kwargs: dict) -> None:
    with pytest.raises(ValueError):
        create_case(tmp_path / "invalid", **kwargs)


def test_validation_scripts_parse_and_read_generated_field(tmp_path: Path) -> None:
    case = create_case(tmp_path / "parser_case", cells=8)
    values = read_scalar_field(case / "0" / "T")
    assert len(values) == 8
    assert values[0] < 0 and values[-1] > 0

    for script in (
        ROOT / "scripts" / "validate_openfoam_heat.py",
        ROOT / "scripts" / "compare_pinn_openfoam_heat.py",
    ):
        ast.parse(script.read_text(encoding="utf-8"), filename=str(script))
