from __future__ import annotations

import importlib


def test_use_case_package_layout_and_imports() -> None:
    modules = (
        "use_cases.thermal_barrier.run",
        "use_cases.electronics_cooling.run",
        "use_cases.viscous_flow.run",
    )
    for module_name in modules:
        module = importlib.import_module(module_name)
        assert callable(module.main)

    viscous_flow = importlib.import_module("use_cases.viscous_flow.run")
    assert callable(viscous_flow.burgers_residual)
