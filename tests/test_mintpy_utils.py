from __future__ import annotations

import sys
from types import SimpleNamespace

import h5py
import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import Point

from snowsar.utils.mintpy_utils import (
    align_insar_swe_with_snotel,
    build_station_windows,
    compare_station_windows,
    decode_h5_dates,
    displacement_to_swe_cm,
    mintpy_crs_from_attrs,
    mintpy_footprint_from_timeseries_h5,
    read_mintpy_displacement_series_at_latlon,
    sample_mintpy_incidence_angle,
    summarize_insar_swe_metrics,
)


def _install_fake_geometry_module(monkeypatch):
    fake_geometry = SimpleNamespace(
        get_valid_data_polygon_from_array=lambda *args, crs, **kwargs: gpd.GeoDataFrame(
            geometry=[Point(0, 0)],
            crs=crs,
        )
    )
    monkeypatch.setitem(sys.modules, "snowsar.utils.geometry", fake_geometry)


def test_mintpy_footprint_uses_epsg_from_attrs(monkeypatch):
    _install_fake_geometry_module(monkeypatch)

    attrs = {
        "X_FIRST": 479400.0,
        "Y_FIRST": 5021240.0,
        "X_STEP": 80.0,
        "Y_STEP": -80.0,
        "X_UNIT": "meters",
        "Y_UNIT": "meters",
        "EPSG": 32611,
    }
    data = np.ones((2, 3), dtype=np.float32)

    fake_readfile = SimpleNamespace(
        get_slice_list=lambda path: ["timeseries-20201213"],
        read=lambda path, datasetName=None: (data, attrs),
    )
    fake_mintpy_utils = SimpleNamespace(readfile=fake_readfile)
    fake_mintpy = SimpleNamespace(utils=fake_mintpy_utils)

    monkeypatch.setitem(sys.modules, "mintpy", fake_mintpy)
    monkeypatch.setitem(sys.modules, "mintpy.utils", fake_mintpy_utils)

    gdf = mintpy_footprint_from_timeseries_h5("fake_timeseries.h5")

    assert str(gdf.crs) == "EPSG:32611"


def test_mintpy_footprint_falls_back_to_degree_units(monkeypatch):
    _install_fake_geometry_module(monkeypatch)

    attrs = {
        "X_FIRST": -120.0,
        "Y_FIRST": 40.0,
        "X_STEP": 0.1,
        "Y_STEP": -0.1,
        "X_UNIT": "degrees",
        "Y_UNIT": "degrees",
    }
    data = np.ones((2, 2), dtype=np.float32)

    fake_readfile = SimpleNamespace(
        get_slice_list=lambda path: ["timeseries-20201213"],
        read=lambda path, datasetName=None: (data, attrs),
    )
    fake_mintpy_utils = SimpleNamespace(readfile=fake_readfile)
    fake_mintpy = SimpleNamespace(utils=fake_mintpy_utils)

    monkeypatch.setitem(sys.modules, "mintpy", fake_mintpy)
    monkeypatch.setitem(sys.modules, "mintpy.utils", fake_mintpy_utils)

    gdf = mintpy_footprint_from_timeseries_h5("fake_timeseries.h5")

    assert str(gdf.crs) == "EPSG:4326"


def _install_fake_mintpy_modules(monkeypatch, *, incidence_grid=None):
    class FakeCoord:
        def open(self):
            return None

        def geo2radar(self, lat, lon):
            return (1, 2, None, None)

    fake_readfile = SimpleNamespace(
        read=lambda path, datasetName=None: (
            np.asarray(
                incidence_grid
                if incidence_grid is not None
                else [[30.0, 31.0, 32.0], [33.0, 34.0, 35.0]]
            ),
            {},
        ),
        read_attribute=lambda path: {"DUMMY": True},
    )
    fake_ut = SimpleNamespace(
        read_timeseries_lalo=lambda **kwargs: (
            [pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-05")],
            np.array([0.0, -0.02]),
            np.array([0.1, 0.1]),
        ),
        readfile=fake_readfile,
        coordinate=lambda attrs, lookup_file=None: FakeCoord(),
    )
    fake_mintpy_utils = SimpleNamespace(readfile=fake_readfile, utils=fake_ut)
    fake_mintpy = SimpleNamespace(utils=fake_mintpy_utils)

    monkeypatch.setitem(sys.modules, "mintpy", fake_mintpy)
    monkeypatch.setitem(sys.modules, "mintpy.utils", fake_mintpy_utils)


def test_read_mintpy_displacement_series_at_latlon(monkeypatch):
    _install_fake_mintpy_modules(monkeypatch)

    dates, displacement = read_mintpy_displacement_series_at_latlon(
        39.0,
        -120.0,
        timeseries_file="timeseries.h5",
        lookup_file="lookup.tif",
        win_size=10,
    )

    assert dates.strftime("%Y-%m-%d").tolist() == ["2024-01-01", "2024-01-05"]
    assert displacement.tolist() == pytest.approx([0.0, -0.02])


def test_sample_mintpy_incidence_angle_returns_requested_unit(monkeypatch):
    _install_fake_mintpy_modules(monkeypatch)

    angle_rad = sample_mintpy_incidence_angle(
        39.0,
        -120.0,
        timeseries_file="timeseries.h5",
        lookup_file="lookup.tif",
        incidence_file="incidence.tif",
        unit="radians",
    )
    angle_deg = sample_mintpy_incidence_angle(
        39.0,
        -120.0,
        timeseries_file="timeseries.h5",
        lookup_file="lookup.tif",
        incidence_file="incidence.tif",
        unit="degrees",
    )

    assert angle_deg == pytest.approx(35.0)
    assert angle_rad == pytest.approx(np.deg2rad(35.0))


def test_align_insar_swe_with_snotel_and_summarize_metrics():
    station_df = pd.DataFrame(
        {
            "date_time_utc": pd.to_datetime(
                ["2024-01-01", "2024-01-05", "2024-01-10"]
            ),
            "value_cm": [10.0, 13.0, 15.0],
        }
    )
    insar_dates = pd.to_datetime(["2024-01-01", "2024-01-05", "2024-01-10"])
    insar_delta_swe_cm = np.array([0.0, 2.5, 1.5])

    aligned = align_insar_swe_with_snotel(
        station_df, insar_dates, insar_delta_swe_cm
    )
    metrics = summarize_insar_swe_metrics(aligned)

    assert aligned["snotel_delta_swe_cm"].tolist() == pytest.approx([0.0, 3.0, 2.0])
    assert aligned["insar_cumulative_swe_cm"].tolist() == pytest.approx([0.0, 2.5, 4.0])
    assert metrics.loc[0, "matched_dates"] == 3
    assert metrics.loc[0, "rmse_cm"] == pytest.approx(
        np.sqrt((0.0**2 + 0.5**2 + 0.5**2) / 3.0)
    )


def test_displacement_to_swe_cm_preserves_length():
    displacement = np.array([0.0, -0.01, -0.03])
    swe_cm = displacement_to_swe_cm(displacement, np.deg2rad(35.0))

    assert len(swe_cm) == len(displacement)
    assert swe_cm[0] == pytest.approx(0.0)


def test_decode_h5_dates_and_crs_helpers():
    assert decode_h5_dates([b"20240101", "20240105"]) == [
        "20240101",
        "20240105",
    ]

    crs = mintpy_crs_from_attrs({"EPSG": 32611})
    assert crs is not None
    assert crs.to_epsg() == 32611
    assert mintpy_crs_from_attrs({"X_UNIT": "meters", "Y_UNIT": "meters"}) is None


def test_build_station_windows_for_geographic_grid():
    attrs = {
        "X_FIRST": -120.0,
        "Y_FIRST": 40.0,
        "X_STEP": 0.5,
        "Y_STEP": -0.5,
        "WIDTH": 3,
        "LENGTH": 3,
        "X_UNIT": "degrees",
        "Y_UNIT": "degrees",
    }
    site_table = pd.DataFrame(
        {
            "site_name": ["Station A"],
            "latitude": [39.75],
            "longitude": [-119.75],
        }
    )

    windows = build_station_windows(site_table, attrs, win_size=1)

    assert len(windows) == 1
    assert windows[0]["row"] == 0
    assert windows[0]["col"] == 0
    assert windows[0]["y_max"] == 2
    assert windows[0]["x_max"] == 2


def test_compare_station_windows_skips_incomplete_insar(tmp_path):
    timeseries_file = tmp_path / "timeseries.h5"
    coherence_file = tmp_path / "ifgramStack.h5"
    incidence_file = tmp_path / "geometry.h5"

    factor = (
        (-0.6784 * np.deg2rad(35.0) ** 2)
        + (0.2899 * np.deg2rad(35.0))
        - 0.8473
    )
    ts_values = np.array(
        [
            np.zeros((3, 3), dtype=np.float32),
            np.full((3, 3), -(2.0 / 100.0) * -factor, dtype=np.float32),
            np.full((3, 3), -((2.0 + 3.0) / 100.0) * -factor, dtype=np.float32),
        ]
    )

    with h5py.File(timeseries_file, "w") as f:
        f.create_dataset("date", data=np.array([b"20240101", b"20240105", b"20240110"]))
        f.create_dataset("timeseries", data=ts_values)
        f.attrs["X_FIRST"] = -120.0
        f.attrs["Y_FIRST"] = 40.0
        f.attrs["X_STEP"] = 0.5
        f.attrs["Y_STEP"] = -0.5
        f.attrs["WIDTH"] = 3
        f.attrs["LENGTH"] = 3
        f.attrs["X_UNIT"] = "degrees"
        f.attrs["Y_UNIT"] = "degrees"

    with h5py.File(coherence_file, "w") as f:
        f.create_dataset(
            "date",
            data=np.array(
                [[b"20240101", b"20240105"], [b"20240105", b"20240110"]]
            ),
        )
        coherence = np.ones((2, 3, 3), dtype=np.float32)
        coherence[1, 0, 0] = 0.1
        f.create_dataset("coherence", data=coherence)

    with h5py.File(incidence_file, "w") as f:
        f.create_dataset(
            "incidenceAngle",
            data=np.full((3, 3), 35.0, dtype=np.float32),
        )

    site_table = pd.DataFrame(
        {
            "site_name": ["Station A"],
            "latitude": [39.75],
            "longitude": [-119.75],
        }
    )
    swe_data = {
        "Station A": pd.DataFrame(
            {
                "date_time_utc": pd.to_datetime(
                    ["2024-01-01", "2024-01-05", "2024-01-10"]
                ),
                "value_cm": [10.0, 12.0, 15.0],
            }
        )
    }

    all_results, failures = compare_station_windows(
        site_table,
        swe_data,
        timeseries_file=timeseries_file,
        coherence_file=coherence_file,
        incidence_file=incidence_file,
        incidence_dataset="incidenceAngle",
        coherence_dataset="coherence",
        coherence_date_dataset="date",
        coherence_threshold=0.35,
        apply_bias_correction=False,
        bias_station_names=[],
        win_size=0,
    )

    assert all_results == {}
    assert failures["site_name"].tolist() == ["Station A"]
    assert failures.loc[0, "error"] == (
        "InSAR incomplete: 1 interval(s) missing after masking/filtering."
    )
