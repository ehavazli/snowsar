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
    Returns normalized pd.Timestamp (00:00).
    """
    dates = set()
    for p in paths:
        name = Path(p).name
        for token in _DATE_RE.findall(name):
            try:
                dates.add(pd.to_datetime(token, format="%Y%m%d").normalize())
            except Exception:
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
    Build valid-data footprint polygon from GeoTIFF(s), matching the notebook method:

    - uses src.read_masks(band) directly
    - polygonizes mask
    - filters polygon interior holes by area threshold
    - unions all polygons
    - sets CRS from raster, then reprojects to EPSG:4326 by default
    """
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
