from __future__ import annotations

import h5py
import numpy as np
import pytest

from snowsar.utils.lidar_utils import (
    compute_pearson_correlation,
    cumulative_sum_through_date,
    extract_start_date_str,
)
from snowsar.utils.nisar_utils import _read_geogrid_coords, h5_get, resolve_h5_path


def test_resolve_h5_path_accepts_leading_slash_variants(tmp_path):
    h5_path = tmp_path / "test.h5"
    with h5py.File(h5_path, "w") as h5:
        h5.create_dataset("science/LSAR/data", data=[1, 2, 3])

    with h5py.File(h5_path, "r") as h5:
        assert resolve_h5_path(h5, "science/LSAR/data") == "/science/LSAR/data"
        assert resolve_h5_path(h5, "/science/LSAR/data") == "/science/LSAR/data"


def test_resolve_h5_path_uses_extra_candidates(tmp_path):
    h5_path = tmp_path / "test.h5"
    with h5py.File(h5_path, "w") as h5:
        h5.create_dataset("science/LSAR/data", data=[1, 2, 3])

    with h5py.File(h5_path, "r") as h5:
        resolved = resolve_h5_path(
            h5,
            "science/DOES_NOT_EXIST",
            extra_candidates=["science/LSAR/data"],
        )

    assert resolved == "science/LSAR/data"


def test_resolve_h5_path_includes_context_in_error(tmp_path):
    h5_path = tmp_path / "test.h5"
    with h5py.File(h5_path, "w") as h5:
        h5.create_group("science")

    with h5py.File(h5_path, "r") as h5:
        with pytest.raises(KeyError, match="Top-level keys: \\['science'\\]"):
            resolve_h5_path(h5, "missing/path")


def test_h5_get_returns_dataset_after_resolution(tmp_path):
    h5_path = tmp_path / "test.h5"
    with h5py.File(h5_path, "w") as h5:
        h5.create_dataset("science/LSAR/data", data=[4, 5, 6])

    with h5py.File(h5_path, "r") as h5:
        dataset = h5_get(h5, "science/LSAR/data")
        assert dataset[()].tolist() == [4, 5, 6]


def test_read_geogrid_coords_computes_spacing_and_handles_bad_projection(tmp_path):
    h5_path = tmp_path / "grid.h5"
    with h5py.File(h5_path, "w") as h5:
        grp = h5.create_group("grid")
        grp.create_dataset("xCoordinates", data=np.array([100.0, 130.0, 160.0]))
        grp.create_dataset("yCoordinates", data=np.array([50.0, 25.0, 0.0]))
        grp.create_dataset("projection", data=np.bytes_("EPSG:4326"))

    with h5py.File(h5_path, "r") as h5:
        x, y, dx, dy, epsg = _read_geogrid_coords(h5["grid"])

    assert x.tolist() == [100.0, 130.0, 160.0]
    assert y.tolist() == [50.0, 25.0, 0.0]
    assert dx == 30.0
    assert dy == -25.0
    assert epsg is None


def test_extract_start_date_str_supports_date_ranges():
    assert extract_start_date_str("snow_2023May11-12_data.tif") == "20230511"


def test_compute_pearson_correlation_filters_nonfinite_pairs():
    x = np.array([1.0, 2.0, np.nan, 4.0, np.inf])
    y = np.array([2.0, 4.0, 6.0, 8.0, 10.0])

    result = compute_pearson_correlation(x, y)

    assert result["count"] == 3
    assert result["valid"] is True
    assert result["reason"] == ""
    assert result["statistic"] == pytest.approx(1.0)


def test_compute_pearson_correlation_nan_mode_reports_invalid_constant_input():
    result = compute_pearson_correlation(
        np.array([5.0, 5.0, 5.0]),
        np.array([1.0, 2.0, 3.0]),
        on_invalid="nan",
    )

    assert result["count"] == 3
    assert result["valid"] is False
    assert np.isnan(result["statistic"])
    assert "constant input arrays" in result["reason"]


def test_cumulative_sum_through_date_is_inclusive():
    stack = np.array(
        [
            [[1.0, 2.0], [3.0, 4.0]],
            [[10.0, 20.0], [30.0, 40.0]],
            [[100.0, 200.0], [300.0, 400.0]],
        ]
    )
    dates = ["20240101", "20240110", "20240120"]

    result = cumulative_sum_through_date(stack, dates, "20240110")

    assert result.tolist() == [[11.0, 22.0], [33.0, 44.0]]


def test_cumulative_sum_through_date_rejects_bad_date_format():
    stack = np.zeros((1, 2, 2), dtype=float)

    with pytest.raises(ValueError, match="Expected YYYYMMDD format"):
        cumulative_sum_through_date(stack, ["2024-01-01"], "20240101")
