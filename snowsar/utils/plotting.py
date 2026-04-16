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
    x_axis: str = "days_since_reference",
    title_left: str = "Daily SWE at 12 AM (cm)",
    title_right: str = "Mean and Std Dev of Δ SWE on acquisition dates",
    show_legend: bool = False,
):
    """
    Create two-panel visualization of SNOTEL snow water equivalent (SWE) data.

    Generates a figure with:
    - Left panel: Time series of SWE for each station with vertical lines marking acquisition dates
    - Right panel: Mean and standard deviation of ΔSWE between consecutive acquisition dates

    Parameters
    ----------
    results : Dict[str, pd.DataFrame]
        Dictionary mapping station names to DataFrames. Each DataFrame must contain:
        - 'date_time_utc': datetime column
        - 'value_cm': SWE measurements in cm
        - 'days_since_reference': days relative to reference_date (if x_axis='days_since_reference')
    reference_date : str or pd.Timestamp
        Reference date for x-axis calculations. Accepts:
        - "MM-DD" format (e.g., "10-01") anchored to acquisition year
        - Any pandas-parseable date string (e.g., "2020-10-01")
        - pd.Timestamp object
    dates : List
        Acquisition dates to mark with vertical lines. Will be converted to pd.Timestamp.
    x_axis : str, default "days_since_reference"
        X-axis units for left panel. Must be one of:
        - "days_since_reference": days since reference_date
        - "date": calendar dates
    title_left : str, default "Daily SWE at 12 AM (cm)"
        Title for left panel
    title_right : str, default "Mean and Std Dev of Δ SWE on acquisition dates"
        Title for right panel
    show_legend : bool, default False
        Whether to show station names in left panel legend

    Raises
    ------
    ValueError
        If results is empty, dates is empty, or x_axis is invalid

    Examples
    --------
    >>> results = {
    ...     "Station_A": pd.DataFrame({
    ...         "date_time_utc": pd.date_range("2020-10-01", periods=100),
    ...         "value_cm": np.random.rand(100) * 50,
    ...         "days_since_reference": range(100)
    ...     }),
    ...     "Station_B": pd.DataFrame({
    ...         "date_time_utc": pd.date_range("2020-10-01", periods=100),
    ...         "value_cm": np.random.rand(100) * 45,
    ...         "days_since_reference": range(100)
    ...     })
    ... }
    >>> dates = ["2020-10-15", "2020-11-01", "2020-11-15"]
    >>> plot_snotel_data(results, reference_date="10-01", dates=dates)
    """
    if not results:
        raise ValueError("results is empty")

    dates = pd.to_datetime(dates).normalize()
    if len(dates) == 0:
        raise ValueError("dates is empty")

    if x_axis not in {"days_since_reference", "date"}:
        raise ValueError(
            f"x_axis must be 'days_since_reference' or 'date', got '{x_axis}'"
        )

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
        else:  # x_axis == "date"
            x = pd.to_datetime(df["date_time_utc"]).dt.normalize()

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
    Create an interactive Folium map with footprint polygon and SNOTEL stations.

    Generates a web-based interactive map showing:
    - InSAR footprint polygon(s) as a GeoJSON layer
    - SNOTEL station locations as markers with name and code popups
    - Layer control for toggling visibility

    Parameters
    ----------
    footprint_gdf : gpd.GeoDataFrame
        GeoDataFrame containing the footprint geometry (Polygon or MultiPolygon).
        Will be automatically reprojected to EPSG:4326 if in a different CRS.
    snotel_sites : gpd.GeoDataFrame
        GeoDataFrame with Point geometry and columns 'name' and 'code'.
        Will be automatically reprojected to EPSG:4326 if in a different CRS.
        Can be None or empty to show only footprint.
    zoom_start : int, default 8
        Initial zoom level (0=world, 18=building level). Typical range: 6-12.
    footprint_name : str, default "Valid Data Area"
        Display name for footprint layer in layer control
    marker_color : str, default "blue"
        Folium marker color. Options: "red", "blue", "green", "purple",
        "orange", "darkred", "lightred", "beige", "darkblue", "darkgreen",
        "cadetblue", "darkpurple", "white", "pink", "lightblue", "lightgreen",
        "gray", "black", "lightgray"
    tiles : str, default "OpenStreetMap"
        Basemap tileset. Common options: "OpenStreetMap", "Stamen Terrain",
        "Stamen Toner", "CartoDB positron", "CartoDB dark_matter"

    Returns
    -------
    folium.Map
        Interactive map object. Displays automatically in Jupyter notebooks.
        Can be saved to HTML with map.save("output.html").

    Raises
    ------
    ValueError
        If footprint_gdf is None or empty, or if zoom_start is out of valid range
    ImportError
        If folium package is not installed

    Notes
    -----
    Requires folium package (optional dependency): pip install folium

    Examples
    --------
    >>> footprint = gpd.GeoDataFrame(
    ...     geometry=[Polygon([(-120, 38), (-120, 39), (-119, 39), (-119, 38)])],
    ...     crs="EPSG:4326"
    ... )
    >>> stations = gpd.GeoDataFrame(
    ...     {"name": ["Station A"], "code": ["ABC:01"]},
    ...     geometry=[Point(-119.5, 38.5)],
    ...     crs="EPSG:4326"
    ... )
    >>> m = make_footprint_station_map(footprint, stations, zoom_start=10)
    >>> m.save("map.html")  # Save to file
    """
    try:
        import folium
    except ImportError as e:
        raise ImportError(
            "folium is required for make_footprint_station_map. "
            "Install with: pip install folium"
        ) from e

    if footprint_gdf is None or footprint_gdf.empty:
        raise ValueError("footprint_gdf is empty")

    if not (0 <= zoom_start <= 18):
        raise ValueError(
            f"zoom_start must be between 0 and 18, got {zoom_start}"
        )

    # Reproject to EPSG:4326 (lat/lon) for Folium
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
