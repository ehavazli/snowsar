from __future__ import annotations

import re
from typing import Dict, List, Union

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_snotel_data(
    results: Dict[str, pd.DataFrame],
    reference_date: Union[str, pd.Timestamp],
    dates: List,
    *,
    x_axis: str = "days_since_reference",  # notebook default
    title_left: str = "Daily SWE at 12 AM (cm)",
    title_right: str = "Mean and Std Dev of Δ SWE on acquisition dates",
    show_legend: bool = False,
):
    """
    Notebook-style two-panel plot:

    Left:
      - SWE time series for each station
      - vertical dashed lines on acquisition dates

    Right:
      - mean and std dev of ΔSWE between consecutive acquisition dates
      - computed across stations, per acquisition date (the "later" date)

    `reference_date` accepts either:
      - "MM-DD" (anchored to acquisition year), or
      - any pandas-parseable date string / Timestamp.
    """
    if not results:
        raise ValueError("results is empty")

    dates = pd.to_datetime(dates).normalize()
    if len(dates) == 0:
        raise ValueError("dates is empty")

    reference_year = pd.to_datetime(dates.min()).year
    if isinstance(reference_date, str):
        ref_str = reference_date.strip()
        if re.fullmatch(r"\d{2}-\d{2}", ref_str):
            reference_date = pd.to_datetime(f"{reference_year}-{ref_str}")
        else:
            reference_date = pd.to_datetime(ref_str)
    else:
        reference_date = pd.to_datetime(reference_date)
    reference_date = reference_date.normalize()

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # ----- Left plot -----
    for site_name, df in results.items():
        if df.empty:
            continue

        if x_axis == "days_since_reference":
            x = df["days_since_reference"]
        elif x_axis == "date":
            x = pd.to_datetime(df["date_time_utc"]).dt.normalize()
        else:
            raise ValueError(
                "x_axis must be 'days_since_reference' or 'date'"
            )

        axes[0].plot(x, df["value_cm"], linestyle="-", label=site_name)

    # vertical markers
    if x_axis == "days_since_reference":
        days_since_ref = [(d - reference_date).days for d in dates]
        for day in days_since_ref:
            axes[0].axvline(
                day, color="k", linestyle="--", alpha=0.7, linewidth=1
            )
        axes[0].set_xlabel(f"Days Since {reference_date.date()}")
    else:
        for d in dates:
            axes[0].axvline(
                d, color="k", linestyle="--", alpha=0.4, linewidth=1
            )
        axes[0].set_xlabel("Date")
        axes[0].tick_params(axis="x", rotation=45)

    axes[0].set_ylabel("In Situ SWE (cm)")
    axes[0].set_title(title_left)
    axes[0].grid(True, alpha=0.25)
    if show_legend:
        axes[0].legend()

    # ----- Right plot: ΔSWE per acquisition date -----
    date_set = set(dates)

    all_dates = []
    all_deltas = []

    for df in results.values():
        if df.empty:
            continue
        tmp = df.copy()
        tmp["date"] = pd.to_datetime(tmp["date_time_utc"]).dt.normalize()
        tmp = tmp[tmp["date"].isin(date_set)].sort_values("date")

        if len(tmp) < 2:
            continue

        tmp["delta_swe_cm"] = tmp["value_cm"].diff()
        all_dates.extend(tmp["date"].iloc[1:].to_list())
        all_deltas.extend(tmp["delta_swe_cm"].iloc[1:].to_list())

    if len(all_dates) == 0:
        axes[1].set_title(title_right)
        axes[1].set_xlabel("Date")
        axes[1].set_ylabel("Δ SWE (cm)")
        axes[1].grid(True)
        plt.tight_layout()
        plt.show()
        return

    all_dates = np.array(
        pd.to_datetime(all_dates).normalize(), dtype="datetime64[D]"
    )
    all_deltas = np.array(all_deltas, dtype=float)

    unique_dates = np.unique(all_dates)
    mean_deltas = np.array(
        [np.nanmean(all_deltas[all_dates == d]) for d in unique_dates]
    )
    std_deltas = np.array(
        [np.nanstd(all_deltas[all_dates == d]) for d in unique_dates]
    )

    axes[1].errorbar(
        pd.to_datetime(unique_dates),
        mean_deltas,
        yerr=std_deltas,
        fmt="o-",
        capsize=3,
    )

    axes[1].set_xlabel("Date")
    axes[1].set_ylabel("Δ SWE (cm)")
    axes[1].set_title(title_right)
    axes[1].grid(True)
    axes[1].tick_params(axis="x", rotation=45)
    axes[1].set_xticks(dates)

    plt.tight_layout()
    plt.show()


def make_footprint_station_map(
    footprint_gdf: gpd.GeoDataFrame,
    snotel_sites: gpd.GeoDataFrame,
    *,
    zoom_start: int = 8,
    footprint_name: str = "Valid Data Area",
    marker_color: str = "blue",
    tiles: str = "OpenStreetMap",
):
    """
    Create an interactive Folium map with:
      - footprint polygon(s) as GeoJSON
      - SNOTEL station points as markers with popup "Name: ... - Code: ..."

    Parameters
    ----------
    footprint_gdf : GeoDataFrame
        GeoDataFrame containing the footprint geometry (polygon/multipolygon).
        Expected CRS: EPSG:4326 (lat/lon). If not, it will be reprojected.
    snotel_sites : GeoDataFrame
        GeoDataFrame with Point geometry and columns 'name' and 'code'.
        Expected CRS: EPSG:4326 (lat/lon). If not, it will be reprojected.
    zoom_start : int
        Initial zoom.
    footprint_name : str
        Layer name for footprint.
    marker_color : str
        Folium marker color.
    tiles : str
        Folium tileset.

    Returns
    -------
    folium.Map
        Map object displayable directly in a notebook cell.
    """
    import folium  # keep optional dependency local

    if footprint_gdf is None or footprint_gdf.empty:
        raise ValueError("footprint_gdf is empty")

    # Ensure lat/lon CRS for Folium
    fp = footprint_gdf
    if fp.crs is not None and str(fp.crs) != "EPSG:4326":
        fp = fp.to_crs("EPSG:4326")

    st = snotel_sites if snotel_sites is not None else gpd.GeoDataFrame()
    if not st.empty and st.crs is not None and str(st.crs) != "EPSG:4326":
        st = st.to_crs("EPSG:4326")

    # Center on footprint centroid (use unary_union to be safe)
    centroid = fp.geometry.unary_union.centroid
    m = folium.Map(
        location=[centroid.y, centroid.x], zoom_start=zoom_start, tiles=tiles
    )

    # Add footprint as GeoJSON
    folium.GeoJson(fp.to_json(), name=footprint_name).add_to(m)

    # Add station markers
    if not st.empty:
        for _, row in st.iterrows():
            name = row.get("name", "")
            code = row.get("code", "")
            geom = row.geometry
            if geom is None or geom.is_empty:
                continue

            folium.Marker(
                location=[geom.y, geom.x],  # lat, lon
                popup=f"Name: {name} - Code: {code}",
                icon=folium.Icon(color=marker_color, icon="info-sign"),
            ).add_to(m)

    folium.LayerControl().add_to(m)
    return m
