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
    Extract acquisition dates from MintPy geo_timeseries*.h5.

    Uses mintpy.readfile.get_slice_list() and parses YYYYMMDD from slice names.
    Returns normalized pd.Timestamp list.
    """
    import re

    timeseries_h5 = Path(timeseries_h5)

    try:
        from mintpy.utils import readfile
    except Exception as e:
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
    crs: str = "EPSG:4326",
) -> gpd.GeoDataFrame:
    """
    Build a footprint polygon from a MintPy geo_timeseries*.h5 by reading one slice,
    and extracting the valid-data region using geocoding metadata.

    Uses geometry.get_valid_data_polygon_from_array() with the *lowercase* step args:
      x_step, y_step
    """
    timeseries_h5 = Path(timeseries_h5)

    try:
        from mintpy.utils import readfile
    except Exception as e:
        raise ImportError(
            "mintpy is required for mintpy_footprint_from_timeseries_h5(). "
            "Install mintpy in your environment."
        ) from e

    from .geometry import get_valid_data_polygon_from_array

    # Determine slice
    slices = readfile.get_slice_list(str(timeseries_h5))
    if not slices:
        raise ValueError(f"No slices found in: {timeseries_h5}")

    slice_name = reference_slice or slices[0]
    data, atr = readfile.read(str(timeseries_h5), datasetName=slice_name)
    data = np.array(data)

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
