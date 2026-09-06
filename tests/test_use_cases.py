from __future__ import annotations


def test_use_cases_are_importable() -> None:
    from use_cases import electronic_cooling, thermal_barrier, viscous_flow

    assert callable(thermal_barrier.main)
    assert callable(electronic_cooling.main)
    assert callable(viscous_flow.main)
    assert callable(viscous_flow.burgers_residual)
