# utils/nisar_utils.py
from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Union

import geopandas as gpd
import numpy as np
import pandas as pd
from rasterio.features import shapes
from rasterio.transform import from_origin
from shapely.geometry import Polygon, shape
from shapely.ops import unary_union

logger = logging.getLogger(__name__)


# -----------------------------
# Paths / conventions (GUNW)
# -----------------------------
def gunw_unwrapped_phase_path(*, frequency: str = "A", pol: str = "HH") -> str:
    """
    Default raster used for footprint extraction.
    Matches your sample:
      science/LSAR/GUNW/grids/frequencyA/unwrappedInterferogram/HH/unwrappedPhase
    """
    frequency = frequency.upper()
    pol = pol.upper()
    return (
        f"science/LSAR/GUNW/grids/frequency{frequency}/"
        f"unwrappedInterferogram/{pol}/unwrappedPhase"
    )


def gunw_connected_components_path(*, frequency: str = "A", pol: str = "HH") -> str:
    """
    Optional alternative for footprint extraction if you prefer CC mask.
    """
    frequency = frequency.upper()
    pol = pol.upper()
    return (
        f"science/LSAR/GUNW/grids/frequency{frequency}/"
        f"unwrappedInterferogram/{pol}/connectedComponents"
    )


# -----------------------------
# Dates
# -----------------------------
def nisar_dates_from_gunw_h5(
    gunw_h5: Union[str, Path],
) -> List[pd.Timestamp]:
    """
    Return [reference_date, secondary_date] from identification times, normalized.

    Uses:
      science/LSAR/identification/referenceZeroDopplerStartTime
      science/LSAR/identification/secondaryZeroDopplerStartTime
    """
    gunw_h5 = Path(gunw_h5)

    try:
        import h5py
    except Exception as e:
        raise ImportError("h5py is required to read NISAR GUNW HDF5 files.") from e

    def _read_str(ds) -> str:
        val = ds[()]
        if isinstance(val, bytes):
            return val.decode("utf-8", errors="ignore")
        if isinstance(val, np.ndarray) and val.dtype.kind in {"S", "O"}:
            # scalar array-of-bytes / object
            v0 = val.item()
            return (
                v0.decode("utf-8", errors="ignore")
                if isinstance(v0, bytes)
                else str(v0)
            )
        return str(val)

    ref_path = "science/LSAR/identification/referenceZeroDopplerStartTime"
    sec_path = "science/LSAR/identification/secondaryZeroDopplerStartTime"

    with h5py.File(gunw_h5, "r") as f:
        if ref_path not in f or sec_path not in f:
            raise ValueError(
                f"Missing identification time datasets in {gunw_h5.name}. "
                f"Expected: {ref_path} and {sec_path}"
            )

        ref_str = _read_str(f[ref_path])
        sec_str = _read_str(f[sec_path])

    # Normalize to midnight to match your ctx convention
    ref_dt = pd.to_datetime(ref_str, errors="coerce")
    sec_dt = pd.to_datetime(sec_str, errors="coerce")

    if pd.isna(ref_dt) or pd.isna(sec_dt):
        raise ValueError(
            f"Could not parse reference/secondary times from file: {gunw_h5.name}\n"
            f"ref='{ref_str}' sec='{sec_str}'"
        )

    dates = sorted({ref_dt.normalize(), sec_dt.normalize()})
    return dates


# -----------------------------
# Footprint
# -----------------------------
def nisar_footprint_from_gunw_h5(
    gunw_h5: Union[str, Path],
    *,
    raster_path: Optional[str] = None,
    frequency: str = "A",
    pol: str = "HH",
    crs_out: str = "EPSG:4326",
    min_hole_area: float = 0.0,
) -> gpd.GeoDataFrame:
    """
    Build a valid-data footprint polygon from a GUNW HDF5 raster layer.

    - Reads xCoordinates/yCoordinates, spacing, and projection from the parent group.
    - Builds a finite-pixel mask (excluding fill values if present)
    - Polygonizes mask via rasterio.features.shapes
    - Unions polygons and returns GeoDataFrame reprojected to crs_out

    Parameters
    ----------
    gunw_h5 : path
        NISAR L2 GUNW HDF5 file.
    raster_path : str, optional
        Full HDF5 dataset path for the raster. If None, uses unwrappedPhase path
        derived from (frequency, pol).
    frequency : str
        "A" / "B" (used only if raster_path is None)
    pol : str
        "HH", "HV", etc (used only if raster_path is None)
    crs_out : str
        Output CRS (default EPSG:4326)
    min_hole_area : float
        If >0, remove holes smaller than this area (in source CRS units^2).

    Returns
    -------
    gpd.GeoDataFrame
        Single-row GeoDataFrame with footprint geometry.
    """
    gunw_h5 = Path(gunw_h5)
    raster_path = raster_path or gunw_unwrapped_phase_path(frequency=frequency, pol=pol)

    try:
        import h5py
    except Exception as e:
        raise ImportError("h5py is required to read NISAR GUNW HDF5 files.") from e

    with h5py.File(gunw_h5, "r") as f:
        if raster_path not in f:
            raise ValueError(
                f"Raster dataset not found in {gunw_h5.name}:\n  {raster_path}"
            )

        ds = f[raster_path]
        arr = ds[()]  # numpy array

        # Parent group holds x/y coords + projection
        grp = ds.parent

        # Required grids
        if "xCoordinates" not in grp or "yCoordinates" not in grp:
            raise ValueError(
                f"Missing xCoordinates/yCoordinates near raster path:\n  {raster_path}"
            )

        x = np.array(grp["xCoordinates"][()])
        y = np.array(grp["yCoordinates"][()])

        # Spacing (prefer explicit datasets, else infer)
        if "xCoordinateSpacing" in grp:
            dx = float(np.array(grp["xCoordinateSpacing"][()]).item())
        else:
            dx = float(x[1] - x[0])

        if "yCoordinateSpacing" in grp:
            dy = float(np.array(grp["yCoordinateSpacing"][()]).item())
        else:
            dy = float(y[1] - y[0])

        # Projection EPSG code (your sample stores integer like 32611)
        epsg = None
        if "projection" in grp:
            try:
                epsg = int(np.array(grp["projection"][()]).item())
            except Exception:
                epsg = None

        crs_src = f"EPSG:{epsg}" if epsg else None

        # Build mask: finite & not fill value
        mask = np.isfinite(arr)

        fill = ds.attrs.get("_FillValue", None)
        if fill is not None:
            try:
                fill_val = float(fill)
                mask &= arr != fill_val
            except Exception:
                pass

    # If x,y look like pixel centers, adjust half-pixel to get top-left corner
    # Use absolute resolution for rasterio transform
    res_x = abs(dx)
    res_y = abs(dy)

    west = float(np.min(x) - res_x / 2.0)
    east = float(np.max(x) + res_x / 2.0)
    south = float(np.min(y) - res_y / 2.0)
    north = float(np.max(y) + res_y / 2.0)

    transform = from_origin(west, north, res_x, res_y)

    # Polygonize valid area
    mask_u8 = mask.astype(np.uint8)

    polys = []
    for geom, val in shapes(mask_u8, mask=mask.astype(bool), transform=transform):
        if val != 1:
            continue
        p = shape(geom)

        # Optional hole filtering, similar to your HyP3 version
        if min_hole_area > 0 and hasattr(p, "interiors") and len(p.interiors) > 0:
            kept_holes = []
            for interior in p.interiors:
                ring_poly = Polygon(interior)
                if ring_poly.area >= min_hole_area:
                    kept_holes.append(interior)
            p = Polygon(p.exterior.coords, holes=kept_holes)

        polys.append(p)

    if not polys:
        # Return empty in the expected CRS
        return gpd.GeoDataFrame(geometry=[], crs=crs_out)

    merged = unary_union(polys)

    gdf = gpd.GeoDataFrame(geometry=[merged], crs=crs_src)

    # Reproject to EPSG:4326 (or whatever requested)
    if crs_out and gdf.crs is not None:
        gdf = gdf.to_crs(crs_out)
    elif crs_out and gdf.crs is None:
        # If projection missing, assume already lon/lat (rare for GUNW grids)
        gdf = gdf.set_crs(crs_out)

    return gdf


def nisar_union_footprints(
    gunw_files: Sequence[Union[str, Path]],
    *,
    raster_path: Optional[str] = None,
    frequency: str = "A",
    pol: str = "HH",
    crs_out: str = "EPSG:4326",
    min_hole_area: float = 0.0,
) -> gpd.GeoDataFrame:
    """
    Convenience: union footprints across multiple GUNW files.
    """
    geoms = []
    for f in gunw_files:
        gdf = nisar_footprint_from_gunw_h5(
            f,
            raster_path=raster_path,
            frequency=frequency,
            pol=pol,
            crs_out=crs_out,
            min_hole_area=min_hole_area,
        )
        if not gdf.empty:
            geoms.append(gdf.geometry.iloc[0])

    if not geoms:
        return gpd.GeoDataFrame(geometry=[], crs=crs_out)

    return gpd.GeoDataFrame(geometry=[unary_union(geoms)], crs=crs_out)
