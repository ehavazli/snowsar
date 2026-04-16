from __future__ import annotations

import re
from pathlib import Path
from typing import List, Sequence, Union

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import shapes
from shapely.geometry import Polygon, shape
from shapely.ops import unary_union

# Find any YYYYMMDD tokens (HyP3 names typically include 2 dates)
_DATE_RE = re.compile(r"(\d{8})")


def parse_unique_dates_from_hyp3_filenames(
    paths: Sequence[Union[str, Path]],
) -> List[pd.Timestamp]:
    """
    Parse sorted unique dates from HyP3 filenames by scanning YYYYMMDD tokens.

    Extracts all 8-digit date tokens (YYYYMMDD format) from filenames and
    returns unique dates as normalized (midnight) timestamps. HyP3 InSAR
    products typically include both reference and secondary dates in the filename.

    Parameters
    ----------
    paths : Sequence[str or Path]
        File paths or filenames to parse. Typically HyP3 GeoTIFF paths like
        "S1AA_20201215T123456_20201227T123456_VV_unwrapped_phase.tif"

    Returns
    -------
    List[pd.Timestamp]
        Sorted list of unique dates found across all filenames, normalized
        to midnight (00:00:00). Returns empty list if no valid dates found.

    Examples
    --------
    >>> paths = [
    ...     "S1_20201215_20201227_VV_unw_phase.tif",
    ...     "S1_20201227_20210108_VV_unw_phase.tif"
    ... ]
    >>> dates = parse_unique_dates_from_hyp3_filenames(paths)
    >>> len(dates)
    3
    >>> dates[0]
    Timestamp('2020-12-15 00:00:00')
    """
    if not paths:
        return []

    dates = set()
    for p in paths:
        name = Path(p).name
        for token in _DATE_RE.findall(name):
            try:
                dates.add(pd.to_datetime(token, format="%Y%m%d").normalize())
            except (ValueError, OverflowError):
                # Skip invalid date tokens (e.g., "99999999" or malformed dates)
                continue
    return sorted(dates)


def footprint_from_geotiffs(
    tif_paths: Sequence[Union[str, Path]],
    *,
    band: int = 1,
    hole_area_min: float = 0.0001,
    out_crs: str = "EPSG:4326",
) -> gpd.GeoDataFrame:
    """
    Build valid-data footprint polygon from GeoTIFF(s).

    Creates a unified polygon representing the valid data extent across one or
    more GeoTIFF files by:
    1. Reading raster validity masks (non-nodata areas)
    2. Polygonizing each mask with proper georeferencing
    3. Filtering small interior holes below area threshold
    4. Merging all polygons into a single geometry
    5. Reprojecting to target CRS

    Parameters
    ----------
    tif_paths : Sequence[str or Path]
        Paths to GeoTIFF files. Typically HyP3 unwrapped phase products.
    band : int, default 1
        Band number to read mask from (1-indexed)
    hole_area_min : float, default 0.0001
        Minimum area threshold for keeping interior holes in polygons.
        Holes smaller than this (in source CRS units squared) are filled.
        Default 0.0001 is ~0.01 deg² ≈ 1 km² at equator.
    out_crs : str, default "EPSG:4326"
        Target coordinate reference system for output geometry.
        None preserves source raster CRS.

    Returns
    -------
    gpd.GeoDataFrame
        Single-row GeoDataFrame with polygon geometry representing
        the unified valid data extent. Returns empty GeoDataFrame if
        no valid data found in any input file.

    Raises
    ------
    ValueError
        If band number is invalid (< 1)

    Examples
    --------
    >>> tif_files = [
    ...     "S1_20201215_20201227_VV_unw_phase.tif",
    ...     "S1_20201227_20210108_VV_unw_phase.tif"
    ... ]
    >>> footprint = footprint_from_geotiffs(tif_files)
    >>> footprint.crs
    CRS.from_epsg(4326)
    """
    if band < 1:
        raise ValueError(f"band must be >= 1, got {band}")
    tif_paths = [Path(p) for p in tif_paths]
    if len(tif_paths) == 0:
        return gpd.GeoDataFrame(geometry=[], crs=out_crs)

    list_parts = []
    src_crs = None

    for tif in tif_paths:
        with rasterio.open(tif) as src:
            src_crs = src.crs
            mask = src.read_masks(band)  # uint8 (0..255)

            shapes_generator = shapes(
                mask.astype("uint8"),
                mask=mask,  # same as your notebook (truthy where >0)
                transform=src.transform,
            )

            polygons = [shape(geom) for geom, _ in shapes_generator]

            for polygon in polygons:
                # Skip empties
                if polygon.is_empty:
                    continue

                list_interiors = []
                for interior in polygon.interiors:
                    # rasterio->shapely interiors are LinearRing; area computed via Polygon()
                    ring_poly = Polygon(interior)
                    if ring_poly.area > hole_area_min:
                        list_interiors.append(interior)

                temp_pol = Polygon(
                    polygon.exterior.coords, holes=list_interiors
                )
                if not temp_pol.is_empty:
                    list_parts.append(temp_pol)

    if not list_parts:
        # fall back to empty gdf with output crs
        return gpd.GeoDataFrame(geometry=[], crs=out_crs)

    valid_area = unary_union(list_parts)

    gdf = gpd.GeoDataFrame(geometry=[valid_area], crs=src_crs)

    # Reproject to lat/lon for downstream SNOTEL filtering
    if out_crs is not None:
        gdf = gdf.to_crs(out_crs)

    return gdf
