#!/usr/bin/env python3
"""Generate a reproducible OpenFOAM Foundation v14 1-D heat-diffusion case.

The generated case solves dT/dt = alpha * d2T/dx2 on x in [-1, 1],
with T(x, 0) = sin(pi*x) and T(-1, t) = T(1, t) = 0.

OpenFOAM represents the 1-D domain with a line of cells and one cell in
each inactive direction. This script writes the case dictionaries and the
cell-centred initial field; OpenFOAM itself is still required to run it.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path


def _foam_header(class_name: str, object_name: str, location: str | None = None) -> str:
    location_line = f'    location    "{location}";\n' if location else ""
    return (
        "FoamFile\n"
        "{\n"
        "    version     2.0;\n"
        "    format      ascii;\n"
        f"    class       {class_name};\n"
        f"{location_line}"
        f"    object      {object_name};\n"
        "}\n"
    )


def _block_mesh_dict(cells: int) -> str:
    return _foam_header("dictionary", "blockMeshDict") + f"""
convertToMeters 1;

vertices
(
    (-1 0 0)
    ( 1 0 0)
    ( 1 1 0)
    (-1 1 0)
    (-1 0 1)
    ( 1 0 1)
    ( 1 1 1)
    (-1 1 1)
);

blocks
(
    hex (0 1 2 3 4 5 6 7) ({cells} 1 1) simpleGrading (1 1 1)
);

edges
(
);

boundary
(
    left
    {{
        type patch;
        faces ((0 4 7 3));
    }}

    right
    {{
        type patch;
        faces ((1 2 6 5));
    }}

    front
    {{
        type empty;
        faces ((0 1 5 4));
    }}

    back
    {{
        type empty;
        faces ((3 7 6 2));
    }}

    bottom
    {{
        type empty;
        faces ((0 3 2 1));
    }}

    top
    {{
        type empty;
        faces ((4 5 6 7));
    }}
);

mergePatchPairs
(
);
"""


def _control_dict(delta_t: float, end_time: float, write_interval: float) -> str:
    return _foam_header("dictionary", "controlDict") + f"""
application     laplacianFoam;

startFrom       startTime;
startTime       0;
stopAt          endTime;
endTime         {end_time:.12g};
deltaT          {delta_t:.12g};

writeControl    runTime;
writeInterval   {write_interval:.12g};
purgeWrite      0;

writeFormat     ascii;
writePrecision  12;
writeCompression off;

timeFormat      fixed;
timePrecision   6;

runTimeModifiable true;
"""


def _fv_schemes() -> str:
    return _foam_header("dictionary", "fvSchemes") + """
ddtSchemes
{
    default Euler;
}

gradSchemes
{
    default Gauss linear;
}

divSchemes
{
    default none;
}

laplacianSchemes
{
    default none;
    laplacian(DT,T) Gauss linear corrected;
}

interpolationSchemes
{
    default linear;
}

snGradSchemes
{
    default corrected;
}
"""


def _fv_solution() -> str:
    # OpenFOAM Foundation v14 laplacianFoam requests SIMPLE controls.
    return _foam_header("dictionary", "fvSolution") + """
solvers
{
    T
    {
        solver          PCG;
        preconditioner  DIC;
        tolerance       1e-10;
        relTol          0;
    }
}

SIMPLE
{
    nNonOrthogonalCorrectors 0;
}
"""


def _physical_properties(alpha: float) -> str:
    return _foam_header("dictionary", "physicalProperties") + f"""
DT              [0 2 -1 0 0 0 0] {alpha:.12g};
"""


def _initial_field(cells: int) -> str:
    dx = 2.0 / cells
    values = [
        math.sin(math.pi * (-1.0 + (i + 0.5) * dx))
        for i in range(cells)
    ]
    body = "\n".join(f"    {value:.16g}" for value in values)
    return (
        _foam_header("volScalarField", "T", "0")
        + "dimensions      [0 0 0 0 0 0 0];\n\n"
        + f"internalField   nonuniform List<scalar>\n{cells}\n(\n"
        + body
        + "\n);\n\n"
        + """boundaryField
{
    left
    {
        type    fixedValue;
        value   uniform 0;
    }

    right
    {
        type    fixedValue;
        value   uniform 0;
    }

    front  { type empty; }
    back   { type empty; }
    bottom { type empty; }
    top    { type empty; }
}
"""
    )


def create_case(
    output: str | Path,
    *,
    cells: int = 200,
    alpha: float = 0.1,
    delta_t: float = 0.001,
    end_time: float = 1.0,
    write_interval: float = 0.01,
) -> Path:
    """Write a complete case to a new or empty directory and return its path."""
    if cells < 2:
        raise ValueError("cells must be at least 2")
    if alpha <= 0:
        raise ValueError("alpha must be positive")
    if delta_t <= 0 or end_time <= 0 or write_interval <= 0:
        raise ValueError("delta_t, end_time, and write_interval must be positive")

    case = Path(output).expanduser().resolve()
    if case.exists() and any(case.iterdir()):
        raise FileExistsError(
            f"Refusing to overwrite non-empty case directory: {case}. "
            "Choose a new output path to preserve existing simulation results."
        )

    (case / "0").mkdir(parents=True, exist_ok=True)
    (case / "constant").mkdir(parents=True, exist_ok=True)
    (case / "system").mkdir(parents=True, exist_ok=True)

    files = {
        case / "system" / "blockMeshDict": _block_mesh_dict(cells),
        case / "system" / "controlDict": _control_dict(delta_t, end_time, write_interval),
        case / "system" / "fvSchemes": _fv_schemes(),
        case / "system" / "fvSolution": _fv_solution(),
        case / "constant" / "physicalProperties": _physical_properties(alpha),
        case / "0" / "T": _initial_field(cells),
    }
    for path, content in files.items():
        path.write_text(content, encoding="utf-8")

    return case


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, help="Output case directory")
    parser.add_argument("--cells", type=int, default=200, help="Number of cells along x")
    parser.add_argument("--alpha", type=float, default=0.1, help="Diffusivity DT")
    parser.add_argument("--delta-t", type=float, default=0.001, help="Time step")
    parser.add_argument("--end-time", type=float, default=1.0, help="End time")
    parser.add_argument("--write-interval", type=float, default=0.01, help="Write interval")
    args = parser.parse_args()

    case = create_case(
        args.output,
        cells=args.cells,
        alpha=args.alpha,
        delta_t=args.delta_t,
        end_time=args.end_time,
        write_interval=args.write_interval,
    )
    print(f"Created OpenFOAM heat-diffusion case: {case}")
    print(f"Cells along x: {args.cells}")
    print(f"Diffusivity DT: {args.alpha:g}")
    print(f"deltaT: {args.delta_t:g}; endTime: {args.end_time:g}")
    print("\nNext steps (with OpenFOAM Foundation v14 sourced):")
    print(f"  cd {case}")
    print("  blockMesh")
    print("  checkMesh")
    print("  laplacianFoam")


if __name__ == "__main__":
    main()
