from __future__ import annotations

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import shapes
from shapely.geometry import Polygon, shape
from shapely.ops import unary_union


def get_valid_data_polygon_from_array(
    array: np.ndarray,
    north: float,
    south: float,
    east: float,
    west: float,
    x_step: float,
    y_step: float,
    *,
    crs: str = "EPSG:4326",
    hole_area_min: float = 0.0,
    return_largest: bool = True,
) -> gpd.GeoDataFrame:
    """
    Extract valid-data polygon(s) from a 2D array with NaNs.
    """
    if array.ndim != 2:
        raise ValueError("array must be 2D")
    if not (west < east and south < north):
        raise ValueError("Invalid bounds: expected west<east and south<north")

    xsize = float(abs(x_step))
    ysize = float(abs(y_step))
    transform = rasterio.transform.from_origin(west, north, xsize, ysize)

    mask = np.isfinite(array).astype(np.uint8)

    polys = []
    for geom, value in shapes(
        mask, mask=mask.astype(bool), transform=transform
    ):
        if value != 1:
            continue

        shp = shape(geom)
        if shp.geom_type == "Polygon":
            candidates = [shp]
        elif shp.geom_type == "MultiPolygon":
            candidates = list(shp.geoms)
        else:
            continue

        for poly in candidates:
            if not poly.is_valid:
                poly = poly.buffer(0)

            if hole_area_min > 0 and poly.interiors:
                kept_holes = []
                for ring in poly.interiors:
                    hole_poly = Polygon(ring)
                    if hole_poly.area >= hole_area_min:
                        kept_holes.append(ring)
                poly = Polygon(poly.exterior.coords, holes=kept_holes)

            polys.append(poly)

    if not polys:
        return gpd.GeoDataFrame(geometry=[], crs=crs)

    merged = unary_union(polys)

    if return_largest:
        if merged.geom_type == "Polygon":
            out = merged
        elif merged.geom_type == "MultiPolygon":
            out = max(list(merged.geoms), key=lambda p: p.area)
        else:
            out = merged
    else:
        out = merged

    return gpd.GeoDataFrame(geometry=[out], crs=crs)
