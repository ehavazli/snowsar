from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Union

import geopandas as gpd
import numpy as np
import pandas as pd


def mintpy_dates_from_timeseries_h5(
    timeseries_h5: Union[str, Path],
) -> List[pd.Timestamp]:
    """
    Extract acquisition dates from MintPy geocoded timeseries HDF5 file.

    Reads slice names from MintPy timeseries file, extracts YYYYMMDD date tokens,
    and returns unique sorted dates as normalized timestamps.

    Parameters
    ----------
    timeseries_h5 : str or Path
        Path to MintPy geocoded timeseries file (e.g., geo_timeseries*.h5)

    Returns
    -------
    List[pd.Timestamp]
        Sorted list of unique acquisition dates, normalized to midnight (00:00:00)

    Raises
    ------
    ImportError
        If mintpy package is not installed
    ValueError
        If no slices found in file or no valid YYYYMMDD dates parsed

    Notes
    -----
    - Uses the last 8-digit token in each slice name as the date
    - Slice names typically follow format: "timeseries-YYYYMMDD" or "YYYYMMDD_YYYYMMDD"
    - Duplicate dates across slices are automatically deduplicated

    Examples
    --------
    >>> dates = mintpy_dates_from_timeseries_h5("geo_timeseries_ERA5_demErr.h5")
    >>> len(dates)
    45
    >>> dates[0]
    Timestamp('2020-10-01 00:00:00')
    """
    import re

    timeseries_h5 = Path(timeseries_h5)

    try:
        from mintpy.utils import readfile
    except ImportError as e:
        raise ImportError(
            "mintpy is required for mintpy_dates_from_timeseries_h5(). "
            "Install mintpy in your environment."
        ) from e

    slices = readfile.get_slice_list(str(timeseries_h5))
    if not slices:
        raise ValueError(f"No slices found in: {timeseries_h5}")

    dates: List[pd.Timestamp] = []

    for s in slices:
        tokens = re.findall(r"(\d{8})", str(s))
        if not tokens:
            continue
        # use last 8-digit token as the date
        dates.append(pd.to_datetime(tokens[-1], format="%Y%m%d").normalize())

    dates = sorted(set(dates))
    if not dates:
        raise ValueError(
            f"Could not parse any YYYYMMDD dates from MintPy slices in: {timeseries_h5}"
        )

    return dates


def mintpy_footprint_from_timeseries_h5(
    timeseries_h5: Union[str, Path],
    *,
    reference_slice: Optional[str] = None,
    footprint_mode: str = "single",
    crs: str = "EPSG:4326",
) -> gpd.GeoDataFrame:
    """
    Build valid-data footprint polygon from MintPy geocoded timeseries file.

    Reads one time slice from MintPy timeseries, extracts georeferencing metadata,
    and generates a polygon representing the valid (non-NaN) data extent.

    Parameters
    ----------
    timeseries_h5 : str or Path
        Path to MintPy geocoded timeseries file (e.g., geo_timeseries*.h5)
    reference_slice : str, optional
        Specific slice name to use for footprint generation.
        If None (default), uses the first available slice.
    footprint_mode : {"single", "union", "intersection"}, default "single"
        How to combine valid-data masks. "single" uses reference_slice (or the
        first slice), "union" uses pixels valid in any slice, and
        "intersection" uses pixels valid in every slice.
    crs : str, default "EPSG:4326"
        Target coordinate reference system for output geometry

    Returns
    -------
    gpd.GeoDataFrame
        Single-row GeoDataFrame with Polygon geometry representing valid data extent.
        Returns empty GeoDataFrame if no valid data found.

    Raises
    ------
    ImportError
        If mintpy package is not installed
    ValueError
        If no slices found in file or required geocoding attributes are missing
        (X_FIRST, Y_FIRST, X_STEP, Y_STEP)

    Notes
    -----
    - Footprint is derived from finite (non-NaN/non-inf) values in the selected slice
    - Uses first slice by default. Use footprint_mode="union" or
      "intersection" when masks vary temporally.
    - Geocoding bounds are computed from grid attributes and array shape

    Examples
    --------
    >>> footprint = mintpy_footprint_from_timeseries_h5("geo_timeseries_ERA5_demErr.h5")
    >>> footprint.geometry[0].area
    0.425
    >>> footprint.crs
    CRS.from_epsg(4326)
    """
    timeseries_h5 = Path(timeseries_h5)

    try:
        from mintpy.utils import readfile
    except ImportError as e:
        raise ImportError(
            "mintpy is required for mintpy_footprint_from_timeseries_h5(). "
            "Install mintpy in your environment."
        ) from e

    from .geometry import get_valid_data_polygon_from_array

    # Determine slice
    slices = readfile.get_slice_list(str(timeseries_h5))
    if not slices:
        raise ValueError(f"No slices found in: {timeseries_h5}")
    if footprint_mode not in {"single", "union", "intersection"}:
        raise ValueError(
            "footprint_mode must be one of: 'single', 'union', 'intersection'"
        )

    # Validate reference_slice if provided
    if reference_slice is not None:
        if reference_slice not in slices:
            raise ValueError(
                f"reference_slice '{reference_slice}' not found in file.\n"
                f"  File: {timeseries_h5}\n"
                f"  Available slices: {slices[:10]}" +
                (f" ... ({len(slices)} total)" if len(slices) > 10 else "")
            )
        slice_name = reference_slice
    else:
        slice_name = slices[0]

    data, atr = readfile.read(str(timeseries_h5), datasetName=slice_name)
    data = np.array(data)

    if footprint_mode != "single":
        valid = np.isfinite(data)
        for other_slice in slices:
            if other_slice == slice_name:
                continue
            other_data, _ = readfile.read(
                str(timeseries_h5), datasetName=other_slice
            )
            other_valid = np.isfinite(np.asarray(other_data))
            if other_valid.shape != valid.shape:
                raise ValueError(
                    f"Slice shape mismatch: {other_slice} has {other_valid.shape}, "
                    f"expected {valid.shape}."
                )
            if footprint_mode == "union":
                valid |= other_valid
            else:
                valid &= other_valid
        data = np.where(valid, 1.0, np.nan).astype(np.float32)

    # MintPy geocoding info
    try:
        x_first = float(atr["X_FIRST"])
        y_first = float(atr["Y_FIRST"])
        x_step = float(atr["X_STEP"])
        y_step = float(atr["Y_STEP"])
    except KeyError as e:
        raise ValueError(
            f"Missing required geocoding attribute {e} in attrs for {timeseries_h5.name}."
        ) from e

    nrows, ncols = data.shape

    west = x_first
    north = y_first
    east = west + (ncols * x_step)
    south = north + (nrows * y_step)

    # Normalize ordering
    west, east = (min(west, east), max(west, east))
    south, north = (min(south, north), max(south, north))

    gdf = get_valid_data_polygon_from_array(
        data,
        north=north,
        south=south,
        east=east,
        west=west,
        x_step=x_step,
        y_step=y_step,
        crs=crs,
    )

    return gdf
