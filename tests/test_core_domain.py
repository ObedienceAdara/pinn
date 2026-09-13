import pytest

from pinn.core import Domain


def test_steady_1d_domain_dimensions() -> None:
    d = Domain(spatial_bounds=[(-1.0, 1.0)])
    assert d.spatial_dim == 1
    assert d.input_dim == 1
    assert not d.is_time_dependent
    assert d.time_index is None
    assert list(d.faces()) == [(0, -1, -1.0), (0, 1, 1.0)]


def test_transient_2d_domain_dimensions() -> None:
    d = Domain(spatial_bounds=[(-1.0, 1.0), (0.0, 2.0)], time_bounds=(0.0, 1.0))
    assert d.spatial_dim == 2
    assert d.input_dim == 3
    assert d.is_time_dependent
    assert d.time_index == 2
    assert d.lower() == (-1.0, 0.0, 0.0)
    assert d.upper() == (1.0, 2.0, 1.0)
    assert len(list(d.faces())) == 4  # 2 spatial dims -> 4 edges


def test_3d_domain_has_six_faces() -> None:
    d = Domain(spatial_bounds=[(0.0, 1.0), (0.0, 1.0), (0.0, 1.0)])
    assert len(list(d.faces())) == 6


def test_rejects_inverted_or_degenerate_bounds() -> None:
    with pytest.raises(ValueError):
        Domain(spatial_bounds=[(1.0, -1.0)])
    with pytest.raises(ValueError):
        Domain(spatial_bounds=[(0.0, 0.0)])
    with pytest.raises(ValueError):
        Domain(spatial_bounds=[(-1.0, 1.0)], time_bounds=(1.0, 0.0))


def test_rejects_empty_spatial_bounds() -> None:
    with pytest.raises(ValueError):
        Domain(spatial_bounds=[])
