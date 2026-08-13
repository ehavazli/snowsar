from __future__ import annotations

import sys
from types import SimpleNamespace

import pandas as pd
import pytest
from shapely.geometry import Point

from snowsar.utils.snotel_utils import (
    build_snotel_value_lookup,
    fetch_snotel_timeseries,
    snotel_site_table_from_results,
    summarize_snotel_results,
)


def test_fetch_snotel_timeseries_chunks_requests_and_filters_obs_hour(
    monkeypatch,
):
    calls = []
    responses = {
        ("WTEQ_H", "2024-01-01", "2024-01-02"): {
            "values": [
                {"date_time_utc": "2024-01-01T00:00:00Z", "value": "1.0"},
                {"date_time_utc": "2024-01-01T06:00:00Z", "value": "9.0"},
                {"date_time_utc": "2024-01-02T00:00:00Z", "value": "2.0"},
            ]
        },
        ("WTEQ_H", "2024-01-03", "2024-01-04"): {
            "values": [
                {"date_time_utc": "2024-01-03T00:00:00Z", "value": "3.0"},
                {"date_time_utc": "2024-01-04T00:00:00Z", "value": "4.0"},
            ]
        },
        ("WTEQ_H", "2024-01-05", "2024-01-05"): {
            "values": [
                {"date_time_utc": "2024-01-05T00:00:00Z", "value": "5.0"}
            ]
        },
    }

    def fake_get_values(
        wsdlurl, site_code, variable_code, start=None, end=None, timeout=None
    ):
        calls.append((site_code, variable_code, start, end, timeout))
        return responses[(variable_code, start, end)]

    fake_ulmo = SimpleNamespace(
        cuahsi=SimpleNamespace(
            wof=SimpleNamespace(get_values=fake_get_values)
        )
    )
    monkeypatch.setitem(sys.modules, "ulmo", fake_ulmo)

    sites = pd.DataFrame(
        {
            "code": ["site-1"],
            "name": ["Station 1"],
            "geometry": [Point(-105.0, 39.0)],
        }
    )

    result = fetch_snotel_timeseries(
        sites,
        "https://example.test/wsdl",
        start_date="2024-01-01",
        end_date="2024-01-05",
        obs_hour=0,
        include_temperature=False,
        request_chunk_days=2,
        request_timeout=123.0,
    )

    assert list(result) == ["Station 1"]
    df = result["Station 1"]
    assert df["date_time_utc"].dt.strftime("%Y-%m-%d %H:%M:%S").tolist() == [
        "2024-01-01 00:00:00",
        "2024-01-02 00:00:00",
        "2024-01-03 00:00:00",
        "2024-01-04 00:00:00",
        "2024-01-05 00:00:00",
    ]
    assert df["value_cm"].tolist() == pytest.approx(
        [2.54, 5.08, 7.62, 10.16, 12.7]
    )
    assert calls == [
        (
            "site-1",
            "WTEQ_H",
            "2024-01-01",
            "2024-01-02",
            123.0,
        ),
        (
            "site-1",
            "WTEQ_H",
            "2024-01-03",
            "2024-01-04",
            123.0,
        ),
        (
            "site-1",
            "WTEQ_H",
            "2024-01-05",
            "2024-01-05",
            123.0,
        ),
    ]


def test_fetch_snotel_timeseries_rejects_invalid_chunk_days():
    sites = pd.DataFrame(
        {"code": ["site-1"], "name": ["Station 1"], "geometry": [Point(0, 0)]}
    )

    with pytest.raises(ValueError, match="request_chunk_days must be >= 1"):
        fetch_snotel_timeseries(
            sites,
            "https://example.test/wsdl",
            start_date="2024-01-01",
            end_date="2024-01-01",
            request_chunk_days=0,
        )


def test_summarize_snotel_results_and_site_table():
    station_a = pd.DataFrame(
        {
            "date_time_utc": pd.to_datetime(["2024-01-01", "2024-01-02"]),
            "value_cm": [10.0, 12.0],
            "temp_c": [0.0, 1.0],
            "site_loc": [Point(-120.0, 39.0), Point(-120.0, 39.0)],
        }
    )
    station_b = pd.DataFrame(
        {
            "date_time_utc": pd.to_datetime(["2024-02-01"]),
            "value_cm": [5.0],
            "site_loc": [Point(-121.0, 40.0)],
        }
    )

    results = {"Station B": station_b, "Station A": station_a}

    summary = summarize_snotel_results(results)
    site_table = snotel_site_table_from_results(results)

    assert summary["station"].tolist() == ["Station A", "Station B"]
    assert summary["rows"].tolist() == [2, 1]
    assert summary["has_temperature"].tolist() == [True, False]

    assert site_table["site_name"].tolist() == ["Station A", "Station B"]
    assert site_table["latitude"].tolist() == [39.0, 40.0]
    assert site_table["longitude"].tolist() == [-120.0, -121.0]
    assert site_table["records"].tolist() == [2, 1]


def test_build_snotel_value_lookup_normalizes_dates():
    results = {
        "Station A": pd.DataFrame(
            {
                "date_time_utc": pd.to_datetime(
                    ["2024-01-01T06:00:00Z", "2024-01-02T06:00:00Z"]
                ),
                "value_cm": [10.0, 12.5],
            }
        )
    }

    lookup = build_snotel_value_lookup(results)

    assert list(lookup) == ["Station A"]
    assert lookup["Station A"].index.strftime("%Y-%m-%d").tolist() == [
        "2024-01-01",
        "2024-01-02",
    ]
    assert lookup["Station A"].tolist() == pytest.approx([10.0, 12.5])
