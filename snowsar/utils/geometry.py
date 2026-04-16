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

    Builds a polygon representing the valid (finite) data extent by:
    1. Creating a binary mask of finite values
    2. Polygonizing the mask with proper georeferencing
    3. Filtering small holes within polygons
    4. Merging all polygons into a single geometry

    Parameters
    ----------
    array : np.ndarray
        2D array with data values. NaN/inf values are treated as invalid.
    north : float
        Northern boundary (maximum Y coordinate)
    south : float
        Southern boundary (minimum Y coordinate)
    east : float
        Eastern boundary (maximum X coordinate)
    west : float
        Western boundary (minimum X coordinate)
    x_step : float
        Pixel size in X direction (can be negative)
    y_step : float
        Pixel size in Y direction (can be negative)
    crs : str, default "EPSG:4326"
        Coordinate reference system for output geometry
    hole_area_min : float, default 0.0
        Minimum area threshold for keeping holes in polygons.
        Holes smaller than this are filled.
    return_largest : bool, default True
        If True and multiple disconnected polygons exist, return only
        the largest one. If False, return the MultiPolygon union.

    Returns
    -------
    gpd.GeoDataFrame
        Single-row GeoDataFrame with polygon geometry representing
        valid data extent. Returns empty GeoDataFrame if no valid data.

    Raises
    ------
    ValueError
        If array is not 2D, or if bounds are invalid (west >= east or south >= north)

    Examples
    --------
    >>> import numpy as np
    >>> data = np.random.rand(100, 100)
    >>> data[30:70, 30:70] = np.nan  # Create hole in center
    >>> gdf = get_valid_data_polygon_from_array(
    ...     data, north=50.0, south=40.0, east=-120.0, west=-121.0,
    ...     x_step=0.01, y_step=-0.01
    ... )
    >>> print(gdf.geometry[0].geom_type)
    Polygon
    """
    # Validate inputs
    if not isinstance(array, np.ndarray):
        raise TypeError(f"array must be numpy.ndarray, got {type(array)}")
    if array.ndim != 2:
        raise ValueError(f"array must be 2D, got shape {array.shape}")
    if array.size == 0:
        raise ValueError("array is empty")
    if not (west < east and south < north):
        raise ValueError(
            f"Invalid bounds: expected west < east and south < north, "
            f"got west={west}, east={east}, south={south}, north={north}"
        )
    if x_step == 0 or y_step == 0:
        raise ValueError(f"Step sizes must be non-zero, got x_step={x_step}, y_step={y_step}")
    if not np.isfinite([north, south, east, west, x_step, y_step]).all():
        raise ValueError("All coordinate parameters must be finite numbers")

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
