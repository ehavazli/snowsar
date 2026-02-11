# utils/nisar_utils.py
from __future__ import annotations

import logging
import subprocess
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple, Union

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
def gunw_unwrapped_phase_path(
    *, frequency: str = "A", pol: str = "HH"
) -> str:
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


def gunw_connected_components_path(
    *, frequency: str = "A", pol: str = "HH"
) -> str:
    """
    Optional alternative for footprint extraction if you prefer CC mask.
    """
    frequency = frequency.upper()
    pol = pol.upper()
    return (
        f"science/LSAR/GUNW/grids/frequency{frequency}/"
        f"unwrappedInterferogram/{pol}/connectedComponents"
    )


def gunw_coherence_magnitude_path(
    *, frequency: str = "A", pol: str = "HH"
) -> str:
    frequency = frequency.upper()
    pol = pol.upper()
    return (
        f"science/LSAR/GUNW/grids/frequency{frequency}/"
        f"unwrappedInterferogram/{pol}/coherenceMagnitude"
    )


def gunw_ionosphere_phase_screen_path(
    *, frequency: str = "A", pol: str = "HH"
) -> str:
    frequency = frequency.upper()
    pol = pol.upper()
    return (
        f"science/LSAR/GUNW/grids/frequency{frequency}/"
        f"unwrappedInterferogram/{pol}/ionospherePhaseScreen"
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
        raise ImportError(
            "h5py is required to read NISAR GUNW HDF5 files."
        ) from e

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
    raster_path = raster_path or gunw_unwrapped_phase_path(
        frequency=frequency, pol=pol
    )
    try:
        import h5py
    except Exception as e:
        raise ImportError(
            "h5py is required to read NISAR GUNW HDF5 files."
        ) from e
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
    for geom, val in shapes(
        mask_u8, mask=mask.astype(bool), transform=transform
    ):
        if val != 1:
            continue
        p = shape(geom)
        # Optional hole filtering, similar to your HyP3 version
        if (
            min_hole_area > 0
            and hasattr(p, "interiors")
            and len(p.interiors) > 0
        ):
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


def _gunw_date_tokens_from_filename(
    gunw_h5: Union[str, Path],
    *,
    track_block: int = 5,
    frame_block: int = 7,
    ref_block: int = 12,
    sec_block: int = 13,
) -> Tuple[str, str, str, str]:
    """
    Parse reference/secondary date tokens from a NISAR GUNW filename.
    You specified:
      - split by "_"
      - dates are in blocks 11 and 13
    Example:
      NISAR_L2_PR_GUNW_..._SH_20081012T060911_20081012T060925_20081127T061000_...h5
                               ^ block 11                 ^ block 13
    Returns
    -------
    (ref_date, sec_date) strings exactly as in filename (e.g., "20081012T060911", "20081127T061000")
    """
    p = Path(gunw_h5)
    parts = p.stem.split("_")
    if len(parts) <= max(ref_block, sec_block):
        raise ValueError(
            f"Filename does not have enough '_' blocks for ref_block={ref_block}, sec_block={sec_block}:\n"
            f"  {p.name}\n"
            f"  n_blocks={len(parts)}"
        )
    ref_date = parts[ref_block].split("T")[
        0
    ]  # keep only date part (drop time)
    sec_date = parts[sec_block].split("T")[0]
    track = parts[track_block]
    frame = parts[frame_block]
    return ref_date, sec_date, track, frame


def _format_outname(
    gunw_h5: Union[str, Path],
    layer_name: str,
    *,
    track: int = 5,
    frame: int = 7,
) -> str:
    ref_date, sec_date, track, frame = _gunw_date_tokens_from_filename(
        gunw_h5
    )
    return f"{ref_date}_{sec_date}_{layer_name}_T{track}_F{frame}.tif"


def extract_gunw_layers_to_geotiff_batch(
    gunw_dir: Union[str, Path],
    pattern: str,
    out_dir: Union[str, Path],
    *,
    frequency: str = "A",
    pol: str = "HH",
    layers: Sequence[str] = (
        "unwrappedPhase",
        "coherenceMagnitude",
        "ionospherePhaseScreen",
        "connectedComponents",
    ),
    warp: bool = True,
    dst_epsg: Optional[int] = 4326,
    dst_res: Optional[Tuple[float, float]] = None,
    resampling: str = "nearest",
    overwrite: bool = False,
) -> Dict[Path, Dict[str, Path]]:
    """
    Batch extractor for NISAR GUNW GeoTIFF exports.

    Supported layer names
    ---------------------
    Geogrid (2D, non-interpolated):
      - unwrappedPhase
      - coherenceMagnitude
      - ionospherePhaseScreen
      - connectedComponents

    RadarGrid cubes (exported in BOTH forms):
      - incidenceAngle
      - totalTroposphere  (computed as hydrostatic + wet tropospheric phase screen in radarGrid space)

    For any requested RadarGrid cube, this function writes an interpolated cube→geogrid GeoTIFF using an on-the-fly (cached) DEM: <name>_interp

    Returns
    -------
    Dict[Path, Dict[str, Path]]
        Mapping: input GUNW file -> {output_label -> GeoTIFF path}
    """
    gunw_dir = Path(gunw_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    gunw_geogrid_group = (
        "unwrappedInterferogram"  # default geogrid group for x/y/projection
    )
    files = sorted(gunw_dir.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No files matched: {gunw_dir}/{pattern}")

    # Import here so module import doesn't hard-require dependencies unless used
    try:
        import h5py
    except Exception as e:
        raise ImportError(
            "h5py is required to read NISAR GUNW HDF5 files."
        ) from e

    try:
        import rasterio
        from rasterio.crs import CRS
        from rasterio.transform import from_origin
        from rasterio.warp import (
            Resampling,
            calculate_default_transform,
            reproject,
        )
    except Exception as e:
        raise ImportError(
            "rasterio is required to write/warp GeoTIFF outputs."
        ) from e

    if warp and dst_epsg is None:
        raise ValueError("dst_epsg must be provided when warp=True")

    # 2D geogrid layer path builders
    layer_to_path = {
        "unwrappedPhase": gunw_unwrapped_phase_path(
            frequency=frequency, pol=pol
        ),
        "coherenceMagnitude": gunw_coherence_magnitude_path(
            frequency=frequency, pol=pol
        ),
        "ionospherePhaseScreen": gunw_ionosphere_phase_screen_path(
            frequency=frequency, pol=pol
        ),
        "connectedComponents": gunw_connected_components_path(
            frequency=frequency, pol=pol
        ),
    }

    # RadarGrid "virtual" layers (cubes)
    cube_layers_supported = {
        "incidenceAngle",
        "localIncidenceAngle",
        "totalTroposphere",
    }

    # Split requested layers into geogrid vs cube
    geogrid_layers = [l for l in layers if l in layer_to_path]
    cube_layers = [l for l in layers if l in cube_layers_supported]

    unknown = [
        l
        for l in layers
        if (l not in layer_to_path and l not in cube_layers_supported)
    ]
    if unknown:
        raise ValueError(
            f"Unsupported layer(s): {unknown}. "
            f"Supported geogrid: {sorted(layer_to_path)}; supported cube: {sorted(cube_layers_supported)}"
        )

    resampling = resampling.strip().lower()
    if not hasattr(Resampling, resampling):
        raise ValueError(f"Invalid resampling='{resampling}'.")
    resamp_enum = getattr(Resampling, resampling)

    def _write_geotiff(
        *, out_path: Path, arr2d: np.ndarray, transform, crs, nodata
    ) -> None:
        profile = {
            "driver": "GTiff",
            "height": int(arr2d.shape[0]),
            "width": int(arr2d.shape[1]),
            "count": 1,
            "dtype": arr2d.dtype,
            "crs": crs,
            "transform": transform,
            "nodata": nodata,
        }
        with rasterio.open(out_path, "w", **profile) as dst_ds:
            dst_ds.write(arr2d, 1)

    all_outputs: Dict[Path, Dict[str, Path]] = {}

    for gunw_h5 in files:
        gunw_h5_path = Path(gunw_h5)
        per_file: Dict[str, Path] = {}
        # -------------------------------------------------
        # Destination (warp) grid template: compute ONCE per file
        # Use unwrappedPhase geogrid as the reference grid so ALL outputs
        # (including connectedComponents + incidence/local incidence) land on
        # exactly the same dst_transform/dst_width/dst_height.
        # -------------------------------------------------
        dst_crs = None
        dst_transform = None
        dst_width = None
        dst_height = None
        if warp:
            if dst_epsg is None:
                raise ValueError("dst_epsg must be provided when warp=True")
            dst_crs = CRS.from_epsg(int(dst_epsg))

            ref_path = gunw_unwrapped_phase_path(frequency=frequency, pol=pol)
            with h5py.File(gunw_h5_path, "r") as f_ref:
                if ref_path not in f_ref:
                    # Fall back: first requested geogrid layer that exists
                    for lnm in geogrid_layers:
                        p = layer_to_path[lnm]
                        if p in f_ref:
                            ref_path = p
                            break
                    else:
                        raise ValueError(
                            "warp=True requires a reference geogrid layer (unwrappedPhase or one of requested geogrid layers)."
                        )

                ds_ref = f_ref[ref_path]
                grp_ref = ds_ref.parent
                x_ref = np.array(grp_ref["xCoordinates"][()])
                y_ref = np.array(grp_ref["yCoordinates"][()])
                dx_ref = (
                    float(np.array(grp_ref["xCoordinateSpacing"][()]).item())
                    if "xCoordinateSpacing" in grp_ref
                    else float(x_ref[1] - x_ref[0])
                )
                dy_ref = (
                    float(np.array(grp_ref["yCoordinateSpacing"][()]).item())
                    if "yCoordinateSpacing" in grp_ref
                    else float(y_ref[1] - y_ref[0])
                )

                epsg_ref = int(np.array(grp_ref["projection"][()]).item())
                src_crs_ref = CRS.from_epsg(epsg_ref)

                bounds_ref, _, _ = _grid_bounds_from_xy(x_ref, y_ref)
                left, bottom, right, top = bounds_ref

                # Use the reference raster shape from the dataset itself
                h_ref, w_ref = ds_ref.shape
                dst_transform, dst_width, dst_height = (
                    calculate_default_transform(
                        src_crs_ref,
                        dst_crs,
                        w_ref,
                        h_ref,
                        left,
                        bottom,
                        right,
                        top,
                        resolution=dst_res,
                    )
                )

        # -------------------------------------------------
        # Build the "valid grid" mask from unwrappedPhase on the OUTPUT grid.
        # We use this mask for connectedComponents + incidence/local incidence so
        # values outside valid unwrappedPhase are filled with NaN.
        # -------------------------------------------------
        unw_valid = None
        unw_out_transform = None
        unw_out_crs = None
        unw_out_shape = None
        # Always compute mask from unwrappedPhase (even if not requested) because it defines the valid grid.
        with h5py.File(gunw_h5_path, "r") as f_unw:
            unw_path = gunw_unwrapped_phase_path(frequency=frequency, pol=pol)
            if unw_path not in f_unw:
                raise ValueError(
                    f"Missing unwrappedPhase dataset needed for validity mask:\n  {unw_path}"
                )
            ds_unw = f_unw[unw_path]
            grp_unw = ds_unw.parent
            unw_arr_native = ds_unw[()]
            if unw_arr_native.ndim != 2:
                raise ValueError(
                    f"Expected 2D unwrappedPhase, got shape={unw_arr_native.shape}"
                )

            x_unw = np.array(grp_unw["xCoordinates"][()])
            y_unw = np.array(grp_unw["yCoordinates"][()])
            dx_unw = (
                float(np.array(grp_unw["xCoordinateSpacing"][()]).item())
                if "xCoordinateSpacing" in grp_unw
                else float(x_unw[1] - x_unw[0])
            )
            dy_unw = (
                float(np.array(grp_unw["yCoordinateSpacing"][()]).item())
                if "yCoordinateSpacing" in grp_unw
                else float(y_unw[1] - y_unw[0])
            )

            epsg_unw = int(np.array(grp_unw["projection"][()]).item())
            unw_src_crs = CRS.from_epsg(epsg_unw)

            res_x_unw = abs(dx_unw)
            res_y_unw = abs(dy_unw)
            west_unw = float(np.min(x_unw) - res_x_unw / 2.0)
            north_unw = float(np.max(y_unw) + res_y_unw / 2.0)
            unw_src_transform = from_origin(
                west_unw, north_unw, res_x_unw, res_y_unw
            )

            unw_fill = ds_unw.attrs.get("_FillValue", None)
            try:
                unw_fill = float(unw_fill) if unw_fill is not None else None
            except Exception:
                unw_fill = None

            if warp:
                assert (
                    dst_crs is not None
                    and dst_transform is not None
                    and dst_width is not None
                    and dst_height is not None
                )
                unw_warp = np.full(
                    (dst_height, dst_width),
                    unw_fill if unw_fill is not None else np.nan,
                    dtype=np.float32,
                )

                reproject(
                    source=unw_arr_native.astype(np.float32, copy=False),
                    destination=unw_warp,
                    src_transform=unw_src_transform,
                    src_crs=unw_src_crs,
                    src_nodata=unw_fill,
                    dst_transform=dst_transform,
                    dst_crs=dst_crs,
                    dst_nodata=unw_fill if unw_fill is not None else np.nan,
                    resampling=Resampling.nearest,
                )

                unw_out_arr = unw_warp
                unw_out_transform = dst_transform
                unw_out_crs = dst_crs
            else:
                unw_out_arr = unw_arr_native.astype(np.float32, copy=False)
                unw_out_transform = unw_src_transform
                unw_out_crs = unw_src_crs

        # Valid = finite and not fill-value (if present)
        unw_valid = np.isfinite(unw_out_arr)
        if unw_fill is not None:
            # NOTE: if unw_fill==0.0, this can be over-strict for true 0-rad pixels, but matches the requested "valid grid of unwrappedPhase".
            unw_valid &= unw_out_arr != float(unw_fill)

        unw_out_shape = unw_out_arr.shape
        # -------------------------
        # 1) Non-interpolated 2D layers (geogrid)
        # -------------------------
        if geogrid_layers:
            with h5py.File(gunw_h5_path, "r") as f:
                for layer_name in geogrid_layers:
                    ds_path = layer_to_path[layer_name]
                    if ds_path not in f:
                        raise ValueError(
                            f"Dataset not found for {layer_name}:\n  {ds_path}"
                        )

                    ds = f[ds_path]
                    grp = ds.parent
                    arr = ds[()]

                    if arr.ndim != 2:
                        raise ValueError(
                            f"Expected 2D raster for {layer_name}, got shape={arr.shape}"
                        )

                    if "xCoordinates" not in grp or "yCoordinates" not in grp:
                        raise ValueError(
                            f"Missing xCoordinates/yCoordinates near:\n  {ds_path}"
                        )

                    x = np.array(grp["xCoordinates"][()])
                    y = np.array(grp["yCoordinates"][()])
                    dx = (
                        float(np.array(grp["xCoordinateSpacing"][()]).item())
                        if "xCoordinateSpacing" in grp
                        else float(x[1] - x[0])
                    )
                    dy = (
                        float(np.array(grp["yCoordinateSpacing"][()]).item())
                        if "yCoordinateSpacing" in grp
                        else float(y[1] - y[0])
                    )

                    epsg = None
                    if "projection" in grp:
                        try:
                            epsg = int(np.array(grp["projection"][()]).item())
                        except Exception:
                            epsg = None
                    if not epsg:
                        raise ValueError(
                            f"Could not read native EPSG from group 'projection' near:\n  {ds_path}"
                        )

                    src_crs = CRS.from_epsg(epsg)

                    nodata = ds.attrs.get("_FillValue", None)
                    if nodata is not None:
                        try:
                            nodata = float(nodata)
                        except Exception:
                            nodata = None

                    res_x = abs(dx)
                    res_y = abs(dy)
                    west = float(np.min(x) - res_x / 2.0)
                    north = float(np.max(y) + res_y / 2.0)
                    src_transform = from_origin(west, north, res_x, res_y)

                    height, width = arr.shape

                    if warp:
                        # Use per-file destination grid template
                        assert (
                            dst_crs is not None
                            and dst_transform is not None
                            and dst_width is not None
                            and dst_height is not None
                        )

                        dst = np.full(
                            (dst_height, dst_width),
                            nodata if nodata is not None else 0,
                            dtype=(
                                np.float32
                                if arr.dtype.kind == "f"
                                else arr.dtype
                            ),
                        )

                        reproject(
                            source=arr,
                            destination=dst,
                            src_transform=src_transform,
                            src_crs=src_crs,
                            src_nodata=nodata,
                            dst_transform=dst_transform,
                            dst_crs=dst_crs,
                            dst_nodata=nodata,
                            resampling=resamp_enum,
                        )

                        out_arr = dst
                        out_transform = dst_transform
                        out_crs = dst_crs
                    else:
                        out_arr = arr
                        out_transform = src_transform
                        out_crs = src_crs

                    # Enforce unwrappedPhase-valid grid masking for select layers.
                    # - connectedComponents should align to unwrappedPhase grid and be NaN outside valid.
                    # - unwrappedPhase itself should use the precomputed (warped) array used for the valid mask.
                    if layer_name == "unwrappedPhase":
                        out_arr = unw_out_arr
                        out_transform = unw_out_transform
                        out_crs = unw_out_crs
                        nodata = unw_fill
                    elif layer_name == "connectedComponents":
                        out_arr = out_arr.astype(np.float32, copy=False)
                        if unw_valid is not None:
                            out_arr = np.where(
                                unw_valid, out_arr, np.nan
                            ).astype(np.float32, copy=False)
                        nodata = np.nan

                    out_name = _format_outname(gunw_h5_path, layer_name)
                    out_path = out_dir / out_name

                    if out_path.exists() and not overwrite:
                        logger.info("Skipping existing output: %s", out_path)
                        per_file[layer_name] = out_path
                        continue

                    _write_geotiff(
                        out_path=out_path,
                        arr2d=out_arr,
                        transform=out_transform,
                        crs=out_crs,
                        nodata=nodata,
                    )
                    per_file[layer_name] = out_path
                    logger.info("Wrote %s -> %s", layer_name, out_path)

        # -------------------------
        # 2) RadarGrid cubes: for each requested cube layer, write BOTH
        #    - non-interpolated radarGrid slice
        #    - interpolated cube->geogrid (DEM downloaded on-the-fly)
        # -------------------------
        if cube_layers:
            # Prepare DEM once per file (needed for cube->geogrid interpolation)
            dem_out = dem_cache_path_for_gunw(
                gunw_h5_path,
                out_dir=out_dir,
                frequency=frequency,
                pol=pol,
                raster_path=None,
                buffer_deg=0.02,
                precision=3,
                data_source="COP",
                keep_egm=False,
            )
            if overwrite or (not dem_out.exists()):
                try:
                    download_dem_for_gunw_with_sardem(
                        gunw_h5_path,
                        dem_out,
                        frequency=frequency,
                        pol=pol,
                        raster_path=None,
                        overwrite=overwrite,
                        data_source="COP",
                        keep_egm=False,
                    )
                    logger.info("Prepared DEM -> %s", dem_out)
                except Exception as e:
                    raise RuntimeError(
                        f"Failed to download/prepare DEM for {gunw_h5_path}: {e}"
                    ) from e
            else:
                logger.info("Reusing cached DEM -> %s", dem_out)

            cube_base = "/science/LSAR/GUNW/metadata/radarGrid"

            # Read radarGrid coords/projection once (for slice export + combined cube)
            with h5py.File(gunw_h5_path, "r") as f:
                for p in (
                    f"{cube_base}/xCoordinates",
                    f"{cube_base}/yCoordinates",
                    f"{cube_base}/heightAboveEllipsoid",
                    f"{cube_base}/projection",
                ):
                    if p not in f:
                        raise ValueError(f"Missing radarGrid field:\n  {p}")

                xrg = np.array(f[f"{cube_base}/xCoordinates"][()])
                yrg = np.array(f[f"{cube_base}/yCoordinates"][()])
                zrg = np.array(f[f"{cube_base}/heightAboveEllipsoid"][()])
                epsg_rg = int(
                    np.array(f[f"{cube_base}/projection"][()]).item()
                )
                crs_rg = CRS.from_epsg(epsg_rg)

                # pixel spacing from coordinate diffs (robust to +/- direction)
                dx_rg = (
                    float(np.median(np.diff(xrg))) if xrg.size > 1 else 1.0
                )
                dy_rg = (
                    float(np.median(np.diff(yrg))) if yrg.size > 1 else 1.0
                )
                res_x_rg = abs(dx_rg)
                res_y_rg = abs(dy_rg)
                west_rg = float(np.min(xrg) - res_x_rg / 2.0)
                north_rg = float(np.max(yrg) + res_y_rg / 2.0)
                transform_rg = from_origin(
                    west_rg, north_rg, res_x_rg, res_y_rg
                )

                # choose a representative height slice (median of heights)
                k = int(np.argmin(np.abs(zrg - float(np.median(zrg)))))
                z_sel = float(zrg[k])

                # Load requested cubes (and compute totalTroposphere cube if requested)
                cubes: Dict[str, np.ndarray] = {}

                if ("incidenceAngle" in cube_layers) or (
                    "localIncidenceAngle" in cube_layers
                ):
                    inc_path = f"{cube_base}/incidenceAngle"
                    if inc_path in f:
                        cubes["incidenceAngle"] = np.asarray(f[inc_path][()])
                    else:
                        raise ValueError(
                            f"Missing radarGrid cube dataset:\n  {inc_path}"
                        )

                if "totalTroposphere" in cube_layers:
                    hydro_path = (
                        f"{cube_base}/hydrostaticTroposphericPhaseScreen"
                    )
                    wet_path = f"{cube_base}/wetTroposphericPhaseScreen"
                    if hydro_path not in f or wet_path not in f:
                        missing = []
                        if hydro_path not in f:
                            missing.append(hydro_path)
                        if wet_path not in f:
                            missing.append(wet_path)
                        raise ValueError(
                            "Missing required radarGrid cube dataset(s) to compute totalTroposphere:\n  "
                            + "\n  ".join(missing)
                        )
                    hydro = np.asarray(f[hydro_path][()])
                    wet = np.asarray(f[wet_path][()])
                    if hydro.shape != wet.shape:
                        raise ValueError(
                            f"Hydro and wet cubes have different shapes: hydro={hydro.shape}, wet={wet.shape}"
                        )
                    cubes["totalTroposphere"] = hydro + wet

            # --- (B) Interpolated cube->geogrid GeoTIFFs ---
            # Note: incidenceAngle + localIncidenceAngle are generated as a paired product
            # and (if warp=True) warped onto the exact same destination grid.
            need_inc = "incidenceAngle" in cube_layers
            need_local = "localIncidenceAngle" in cube_layers

            # (1) Incidence + Local incidence (paired)
            if need_inc or need_local:
                inc_label = "incidenceAngle_interp"
                local_label = "localIncidenceAngle_interp"
                inc_out = out_dir / _format_outname(gunw_h5_path, inc_label)
                local_out = out_dir / _format_outname(
                    gunw_h5_path, local_label
                )

                # Decide whether we need to (re)compute
                inc_done = inc_out.exists() and not overwrite
                local_done = local_out.exists() and not overwrite

                # If local is requested, we must compute both (local depends on LOS + incidence)
                must_compute = (need_local and not local_done) or (
                    need_inc and not inc_done
                )
                if must_compute:
                    if warp:
                        tmp_inc = out_dir / (inc_out.stem + "_native.tif")
                        tmp_local = out_dir / (local_out.stem + "_native.tif")
                    else:
                        tmp_inc = inc_out
                        tmp_local = local_out

                    interpolate_incidence_and_local_incidence_to_geotiff(
                        gunw_h5=gunw_h5_path,
                        dem_path=dem_out,
                        out_inc_tif=tmp_inc,
                        out_local_inc_tif=tmp_local,
                        frequency=frequency,
                        pol=pol,
                        gunw_geogrid_group=gunw_geogrid_group,
                        cube_interp_method="linear",
                        dem_resampling="bilinear",
                        overwrite=True,
                        dst_nodata=0.0,
                    )

                    if warp:
                        # Warp BOTH products onto the same destination grid computed from incidence
                        with rasterio.open(tmp_inc) as src_ds:
                            src_crs = src_ds.crs
                            src_transform = src_ds.transform
                            src_arr_inc = src_ds.read(1)
                            nodata = src_ds.nodata
                            h, w = src_arr_inc.shape
                            left, bottom, right, top = (
                                rasterio.transform.array_bounds(
                                    h, w, src_transform
                                )
                            )

                        # Use per-file destination grid template
                        assert (
                            dst_crs is not None
                            and dst_transform is not None
                            and dst_width is not None
                            and dst_height is not None
                        )

                        def _warp_arr(arr_in: np.ndarray) -> np.ndarray:
                            dst_arr = np.full(
                                (dst_height, dst_width),
                                nodata if nodata is not None else 0.0,
                                dtype=arr_in.dtype,
                            )
                            reproject(
                                source=arr_in,
                                destination=dst_arr,
                                src_transform=src_transform,
                                src_crs=src_crs,
                                src_nodata=nodata,
                                dst_transform=dst_transform,
                                dst_crs=dst_crs,
                                dst_nodata=(
                                    nodata if nodata is not None else 0.0
                                ),
                                resampling=resamp_enum,
                            )
                            return dst_arr

                        # incidence
                        if need_inc:
                            inc_w = _warp_arr(src_arr_inc).astype(
                                np.float32, copy=False
                            )
                            if unw_valid is not None:
                                inc_w = np.where(
                                    unw_valid, inc_w, np.nan
                                ).astype(np.float32, copy=False)
                            _write_geotiff(
                                out_path=inc_out,
                                arr2d=inc_w,
                                transform=unw_out_transform,
                                crs=unw_out_crs,
                                nodata=np.nan,
                            )
                            per_file[inc_label] = inc_out
                        else:
                            # keep in dict if it already existed (handled below)
                            pass

                        # local incidence (warp using same dst grid)
                        with rasterio.open(tmp_local) as src_local_ds:
                            src_arr_local = src_local_ds.read(1)
                        if need_local:
                            loc_w = _warp_arr(src_arr_local).astype(
                                np.float32, copy=False
                            )
                            if unw_valid is not None:
                                loc_w = np.where(
                                    unw_valid, loc_w, np.nan
                                ).astype(np.float32, copy=False)
                            _write_geotiff(
                                out_path=local_out,
                                arr2d=loc_w,
                                transform=unw_out_transform,
                                crs=unw_out_crs,
                                nodata=np.nan,
                            )
                            per_file[local_label] = local_out

                        # cleanup native temps
                        try:
                            tmp_inc.unlink()
                        except Exception:
                            pass
                        try:
                            tmp_local.unlink()
                        except Exception:
                            pass
                    else:
                        # no warp; outputs are already on geogrid. Still apply unwrappedPhase-valid mask.
                        if need_inc:
                            with rasterio.open(inc_out) as _ds:
                                _arr = _ds.read(1).astype(
                                    np.float32, copy=False
                                )
                            if unw_valid is not None:
                                _arr = np.where(
                                    unw_valid, _arr, np.nan
                                ).astype(np.float32, copy=False)
                            _write_geotiff(
                                out_path=inc_out,
                                arr2d=_arr,
                                transform=unw_out_transform,
                                crs=unw_out_crs,
                                nodata=np.nan,
                            )
                            per_file[inc_label] = inc_out
                        if need_local:
                            with rasterio.open(local_out) as _ds:
                                _arr = _ds.read(1).astype(
                                    np.float32, copy=False
                                )
                            if unw_valid is not None:
                                _arr = np.where(
                                    unw_valid, _arr, np.nan
                                ).astype(np.float32, copy=False)
                            _write_geotiff(
                                out_path=local_out,
                                arr2d=_arr,
                                transform=unw_out_transform,
                                crs=unw_out_crs,
                                nodata=np.nan,
                            )
                            per_file[local_label] = local_out
                else:
                    # already existed
                    if need_inc and inc_done:
                        per_file[inc_label] = inc_out
                    if need_local and local_done:
                        per_file[local_label] = local_out

            # (2) Other cube layers (e.g., totalTroposphere)
            for cube_name in [
                c
                for c in cube_layers
                if c not in ("incidenceAngle", "localIncidenceAngle")
            ]:
                label = f"{cube_name}_interp"
                out_name = _format_outname(gunw_h5_path, label)
                out_path = out_dir / out_name

                if out_path.exists() and not overwrite:
                    logger.info("Skipping existing output: %s", out_path)
                    per_file[label] = out_path
                    continue

                tmp_path = out_path
                if warp:
                    tmp_path = out_dir / (out_path.stem + "_native.tif")

                if cube_name == "totalTroposphere":
                    interpolate_gunw_radargrid_cube_to_geotiff(
                        gunw_h5_path,
                        dem_out,
                        cube_ds_name=None,
                        cube_data=cubes["totalTroposphere"],
                        out_tif=tmp_path,
                        frequency=frequency,
                        pol=pol,
                        cube_interp_method="linear",
                        dst_nodata=0.0,
                        overwrite=True,
                    )
                else:
                    raise ValueError(f"Unhandled cube layer: {cube_name}")

                if warp:
                    # Warp tmp_path -> out_path
                    with rasterio.open(tmp_path) as src_ds:
                        src_arr = src_ds.read(1)
                        src_transform = src_ds.transform
                        src_crs = src_ds.crs
                        nodata = src_ds.nodata
                        height, width = src_arr.shape
                        left, bottom, right, top = (
                            rasterio.transform.array_bounds(
                                height, width, src_transform
                            )
                        )

                        dst_crs = CRS.from_epsg(int(dst_epsg))
                        dst_transform, dst_width, dst_height = (
                            calculate_default_transform(
                                src_crs,
                                dst_crs,
                                width,
                                height,
                                left,
                                bottom,
                                right,
                                top,
                                resolution=dst_res,
                            )
                        )

                        dst = np.full(
                            (dst_height, dst_width),
                            nodata if nodata is not None else 0.0,
                            dtype=src_arr.dtype,
                        )

                        reproject(
                            source=src_arr,
                            destination=dst,
                            src_transform=src_transform,
                            src_crs=src_crs,
                            src_nodata=nodata,
                            dst_transform=dst_transform,
                            dst_crs=dst_crs,
                            dst_nodata=nodata if nodata is not None else 0.0,
                            resampling=resamp_enum,
                        )

                    _write_geotiff(
                        out_path=out_path,
                        arr2d=dst,
                        transform=dst_transform,
                        crs=dst_crs,
                        nodata=nodata if nodata is not None else 0.0,
                    )
                    per_file[label] = out_path
                    try:
                        tmp_path.unlink()
                    except Exception:
                        pass
                else:
                    per_file[label] = out_path
        all_outputs[gunw_h5_path] = per_file

    return all_outputs


def download_dem_for_gunw_with_sardem(
    gunw_h5: Union[str, Path],
    dem_out: Union[str, Path],
    *,
    frequency: str = "A",
    pol: str = "HH",
    raster_path: Optional[str] = None,
    buffer_deg: float = 0.02,
    data_source: str = "COP",
    output_format: str = "GTiff",
    output_type: str = "float32",
    xrate: Optional[int] = None,
    yrate: Optional[int] = None,
    keep_egm: bool = False,
    cache_dir: Optional[Union[str, Path]] = None,
    overwrite: bool = False,
) -> Path:
    """
    Download a DEM covering the NISAR GUNW raster footprint using `sardem`.
    This:
      1) builds a footprint polygon from a GUNW raster layer
      2) converts to EPSG:4326
      3) derives bbox (west, south, east, north) with optional buffer (degrees)
      4) runs `sardem --bbox ... --output ...`
    Parameters
    ----------
    gunw_h5 : path
        NISAR L2 GUNW HDF5 file
    dem_out : path
        Output DEM filepath (e.g. ".../dem.tif")
    frequency, pol, raster_path :
        Used to pick the raster driving the footprint (defaults to unwrappedPhase path builder).
    buffer_deg : float
        Buffer in degrees added to bbox in EPSG:4326.
    data_source : {"NASA","NASA_WATER","COP"}
        sardem data source (COP is default).
    output_format : {"GTiff","ENVI","ROI_PAC"}
        sardem output format.
    output_type : {"int16","float32","uint8"}
        sardem output type (float32 recommended).
    xrate, yrate : int, optional
        Upsampling factors passed to sardem (--xrate/--yrate).
    keep_egm : bool
        If True, pass --keep-egm (don’t convert geoid heights to ellipsoid heights).
    cache_dir : path, optional
        sardem cache directory.
    overwrite : bool
        If False and dem_out exists, returns existing path.
    Returns
    -------
    Path to DEM (dem_out).
    """
    gunw_h5 = Path(gunw_h5)
    dem_out = Path(dem_out)
    dem_out.parent.mkdir(parents=True, exist_ok=True)
    if dem_out.exists() and not overwrite:
        logger.info("DEM exists, skipping: %s", dem_out)
        return dem_out
    # 1) footprint -> EPSG:4326
    fp = nisar_footprint_from_gunw_h5(
        gunw_h5,
        raster_path=raster_path,
        frequency=frequency,
        pol=pol,
        crs_out="EPSG:4326",
    )
    if fp.empty:
        raise ValueError(f"Footprint is empty for: {gunw_h5}")
    geom = fp.geometry.iloc[0]
    west, south, east, north = geom.bounds
    # 2) buffer in degrees
    west -= buffer_deg
    south -= buffer_deg
    east += buffer_deg
    north += buffer_deg
    # 3) build sardem command
    cmd = [
        "sardem",
        "--bbox",
        f"{west}",
        f"{south}",
        f"{east}",
        f"{north}",
        "--data-source",
        str(data_source),
        "--output",
        str(dem_out),
        "--output-format",
        str(output_format),
        "--output-type",
        str(output_type),
    ]
    if xrate is not None:
        cmd += ["--xrate", str(int(xrate))]
    if yrate is not None:
        cmd += ["--yrate", str(int(yrate))]
    if keep_egm:
        cmd += ["--keep-egm"]
    if cache_dir is not None:
        cmd += ["--cache-dir", str(Path(cache_dir))]
    logger.info("Running: %s", " ".join(cmd))
    # 4) run
    try:
        subprocess.run(cmd, check=True)
    except FileNotFoundError as e:
        raise RuntimeError(
            "Could not find `sardem` executable. "
            "Make sure `sardem` is installed in this environment (conda-forge: sardem) "
            "and that your notebook kernel uses the same env."
        ) from e
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            f"sardem failed with exit code {e.returncode}"
        ) from e
    if not dem_out.exists():
        raise RuntimeError(
            f"sardem reported success but output not found: {dem_out}"
        )
    return dem_out


def dem_cache_path_for_gunw(
    gunw_h5: Union[str, Path],
    *,
    out_dir: Union[str, Path],
    frequency: str = "A",
    pol: str = "HH",
    raster_path: Optional[str] = None,
    buffer_deg: float = 0.02,
    precision: int = 3,
    data_source: str = "COP",
    keep_egm: bool = False,
) -> Path:
    """Return a deterministic DEM path for a scene footprint to enable reuse.

    The DEM filename is derived from the (buffered) EPSG:4326 footprint bounds,
    rounded to `precision` decimal places, plus key DEM generation options.
    """
    gunw_h5 = Path(gunw_h5)
    out_dir = Path(out_dir)
    dem_dir = out_dir / "dem_cache"
    dem_dir.mkdir(parents=True, exist_ok=True)

    fp = nisar_footprint_from_gunw_h5(
        gunw_h5,
        raster_path=raster_path,
        frequency=frequency,
        pol=pol,
        crs_out="EPSG:4326",
    )
    if fp.empty:
        raise ValueError(f"Footprint is empty for: {gunw_h5}")
    west, south, east, north = fp.geometry.iloc[0].bounds
    west -= buffer_deg
    south -= buffer_deg
    east += buffer_deg
    north += buffer_deg

    # Round bounds to reduce key explosion while still being area-specific.
    w = round(float(west), precision)
    s = round(float(south), precision)
    e = round(float(east), precision)
    n = round(float(north), precision)

    egm_tag = "egm" if keep_egm else "hae"
    fname = f"dem_{data_source}_{egm_tag}_w{w}_s{s}_e{e}_n{n}_buf{round(buffer_deg, precision)}.tif"
    return dem_dir / fname


# -----------------------------
# 3D (radarGrid cube) -> 2D (geogrid) interpolation helpers
#   Implemented to match prep_nisar.py:
#     - DEM warped to exact x/y grid with GDAL Warp MEM + targetAlignedPixels
#     - RegularGridInterpolator with axis flipping + NaN fill
#     - Interpolate only on valid pixels (finite unwrappedPhase + _FillValue)
# -----------------------------


def _grid_bounds_from_xy(
    xcoord: np.ndarray, ycoord: np.ndarray
) -> Tuple[Tuple[float, float, float, float], float, float]:
    """Compute pixel-edge bounds aligned to xcoord/ycoord (pixel centers)."""
    if xcoord.size < 2 or ycoord.size < 2:
        raise ValueError(
            "xcoord/ycoord must have at least 2 elements to infer spacing."
        )

    dx = float(xcoord[1] - xcoord[0])
    dy = float(ycoord[1] - ycoord[0])

    left = float(xcoord[0] - dx / 2.0)
    right = float(xcoord[-1] + dx / 2.0)

    top = float(ycoord[0] - dy / 2.0)
    bottom = float(ycoord[-1] + dy / 2.0)

    miny, maxy = (bottom, top) if bottom < top else (top, bottom)
    minx, maxx = (left, right) if left < right else (right, left)
    return (minx, miny, maxx, maxy), dx, dy


def _read_raster_epsg(path: Union[str, Path]) -> int:
    """Read EPSG code from a raster file using GDAL."""
    try:
        from osgeo import gdal, osr
    except Exception as e:
        raise ImportError("GDAL (osgeo) is required for DEM warping.") from e

    ds = gdal.Open(str(path), gdal.GA_ReadOnly)
    if ds is None:
        raise OSError(f"Cannot open raster: {path}")

    srs = osr.SpatialReference(wkt=ds.GetProjection())
    epsg = srs.GetAttrValue("AUTHORITY", 1)
    if epsg is None:
        raise ValueError(
            f"Could not determine EPSG from raster projection: {path}"
        )
    return int(epsg)


def _warp_to_grid_mem(
    *,
    src_path: Union[str, Path],
    src_epsg: int,
    dst_epsg: int,
    xcoord: np.ndarray,
    ycoord: np.ndarray,
    resample_alg: str,
) -> np.ndarray:
    """Warp raster to the exact xcoord/ycoord grid using GDAL MEM output."""
    try:
        from osgeo import gdal
    except Exception as e:
        raise ImportError("GDAL (osgeo) is required for DEM warping.") from e

    bounds, dx, dy = _grid_bounds_from_xy(xcoord, ycoord)

    warp_opts = gdal.WarpOptions(
        format="MEM",
        outputBounds=bounds,
        srcSRS=f"EPSG:{int(src_epsg)}",
        dstSRS=f"EPSG:{int(dst_epsg)}",
        xRes=abs(dx),
        yRes=abs(dy),
        targetAlignedPixels=True,
        resampleAlg=str(resample_alg),
    )
    dst = gdal.Warp("", str(src_path), options=warp_opts)
    if dst is None:
        raise RuntimeError(f"GDAL Warp failed for {src_path}")

    arr = dst.ReadAsArray()
    if arr is None:
        raise RuntimeError(f"Failed reading warped array for {src_path}")

    # Ensure 2D
    if arr.ndim == 3:
        arr = arr[0]
    return np.asarray(arr)


def _make_rgi(grid_axes, values, method: str = "linear"):
    """RegularGridInterpolator wrapper matching prep_nisar.py behavior."""
    try:
        from scipy.interpolate import RegularGridInterpolator
    except Exception as e:
        raise ImportError(
            "scipy is required for 3D interpolation (RegularGridInterpolator). "
            "Install with: conda install -c conda-forge scipy"
        ) from e

    axes = [np.asarray(a) for a in grid_axes]
    vals = values
    for dim, ax in enumerate(axes):
        if ax.size >= 2 and ax[0] > ax[-1]:
            axes[dim] = ax[::-1]
            vals = np.flip(vals, axis=dim)

    return RegularGridInterpolator(
        tuple(axes),
        vals,
        method=method,
        bounds_error=False,
        fill_value=np.nan,
    )


def _read_valid_unw_mask_full_geogrid(
    gunw_file: Union[str, Path],
    *,
    frequency: str,
    pol: str,
) -> np.ndarray:
    """Valid pixels defined as finite unwrappedPhase (+ _FillValue check), over the FULL geogrid."""
    try:
        import h5py
    except Exception as e:
        raise ImportError(
            "h5py is required to read NISAR GUNW HDF5 files."
        ) from e

    raster_path = gunw_unwrapped_phase_path(frequency=frequency, pol=pol)

    with h5py.File(Path(gunw_file), "r") as f:
        # Some files may be addressable with or without a leading '/'
        if raster_path in f:
            dset = f[raster_path]
        elif f"/{raster_path}" in f:
            dset = f[f"/{raster_path}"]
        else:
            raise ValueError(
                f"Missing unwrappedPhase dataset for validity mask: {raster_path}"
            )

        unw = dset[()]
        fill = dset.attrs.get("_FillValue", None)

    valid = np.isfinite(unw)
    if fill is not None:
        try:
            valid &= unw != float(fill)
        except Exception:
            pass
    return valid


def interpolate_gunw_radargrid_cube_to_geotiff(
    gunw_h5: Union[str, Path],
    dem_path: Union[str, Path],
    *,
    cube_ds_name: Optional[str] = None,
    cube_data: Optional[np.ndarray] = None,
    out_tif: Union[str, Path],
    frequency: str = "A",
    pol: str = "HH",
    gunw_geogrid_group: str = "unwrappedInterferogram",
    cube_interp_method: str = "linear",
    dem_resampling: str = "bilinear",
    overwrite: bool = False,
    dst_dtype: str = "float32",
    dst_nodata: float = 0.0,
) -> Path:
    """
    Interpolate a 3D NISAR L2 GUNW radarGrid metadata cube onto the GUNW geogrid using a DEM.

    Rewritten to match prep_nisar.py behavior:
      1) DEM is warped to EXACT geogrid x/y (pixel-aligned) using GDAL Warp MEM
      2) RegularGridInterpolator flips decreasing axes, returns NaN out-of-bounds
      3) interpolation happens ONLY on valid pixels (validity from unwrappedPhase)
    """
    gunw_h5 = Path(gunw_h5)
    dem_path = Path(dem_path)
    out_tif = Path(out_tif)
    out_tif.parent.mkdir(parents=True, exist_ok=True)
    if out_tif.exists() and not overwrite:
        return out_tif

    if cube_data is None and not cube_ds_name:
        raise ValueError(
            "Provide cube_ds_name (to read from file) or cube_data (in-memory cube)."
        )

    try:
        import h5py
    except Exception as e:
        raise ImportError(
            "h5py is required to read NISAR GUNW HDF5 files."
        ) from e

    try:
        import rasterio
        from rasterio.crs import CRS
        from rasterio.transform import from_origin
    except Exception as e:
        raise ImportError(
            "rasterio is required to write GeoTIFF outputs."
        ) from e

    # --------------------------
    # Read cube + coords (radarGrid) and output geogrid coords (x/y/epsg)
    # --------------------------
    product_type = "GUNW"
    cube_base = f"/science/LSAR/{product_type}/metadata/radarGrid"
    geo_base = f"/science/LSAR/{product_type}/grids/frequency{frequency}/{gunw_geogrid_group}/{pol}"
    xgeo_path = f"{geo_base}/xCoordinates"
    ygeo_path = f"{geo_base}/yCoordinates"
    proj_path = f"{geo_base}/projection"

    with h5py.File(gunw_h5, "r") as f:
        # radarGrid coords
        xcoords = np.array(f[f"{cube_base}/xCoordinates"][()])
        ycoords = np.array(f[f"{cube_base}/yCoordinates"][()])
        zcoords = np.array(f[f"{cube_base}/heightAboveEllipsoid"][()])

        # cube
        if cube_data is None:
            cube_path = f"{cube_base}/{cube_ds_name}"
            if cube_path not in f:
                raise ValueError(f"Cube dataset not found: {cube_path}")
            cube = np.asarray(f[cube_path][()])
        else:
            cube = np.asarray(cube_data)

        # output geogrid coords + EPSG
        for p in (xgeo_path, ygeo_path, proj_path):
            if p not in f:
                raise ValueError(f"Missing required geogrid dataset: {p}")
        x_out = np.array(f[xgeo_path][()])
        y_out = np.array(f[ygeo_path][()])
        out_epsg = int(np.array(f[proj_path][()]).item())

    if cube.ndim != 3:
        raise ValueError(f"Expected 3D cube, got shape={cube.shape}")

    # Baseline top/bottom special case
    if cube.shape[0] == 2 and zcoords.size >= 2:
        z_for_interp = np.array([zcoords[0], zcoords[-1]])
    else:
        z_for_interp = zcoords

    # --------------------------
    # 1) Warp DEM to exact output geogrid
    # --------------------------
    dem_src_epsg = _read_raster_epsg(dem_path)
    dem_on_grid = _warp_to_grid_mem(
        src_path=dem_path,
        src_epsg=dem_src_epsg,
        dst_epsg=out_epsg,
        xcoord=x_out,
        ycoord=y_out,
        resample_alg=dem_resampling,
    ).astype(np.float32, copy=False)

    # --------------------------
    # 2) Validity mask from unwrappedPhase (full grid)
    # --------------------------
    valid = _read_valid_unw_mask_full_geogrid(
        gunw_h5, frequency=frequency, pol=pol
    )

    if valid.shape != dem_on_grid.shape:
        raise ValueError(
            "Validity mask shape does not match DEM-on-grid shape. "
            f"mask={valid.shape}, dem={dem_on_grid.shape}. "
            "Check that unwrappedPhase geogrid matches x/y used for warping."
        )

    # --------------------------
    # 3) Interpolate cube at valid pixels only: pts=(z=DEM, y, x)
    # --------------------------
    Y_2d, X_2d = np.meshgrid(y_out, x_out, indexing="ij")

    out = np.full(dem_on_grid.shape, np.nan, dtype=np.float32)
    ii, jj = np.where(valid)
    if ii.size > 0:
        pts = np.column_stack(
            [
                dem_on_grid[ii, jj].astype(np.float64),
                Y_2d[ii, jj].astype(np.float64),
                X_2d[ii, jj].astype(np.float64),
            ]
        )
        itp = _make_rgi(
            (z_for_interp, ycoords, xcoords), cube, method=cube_interp_method
        )
        out[ii, jj] = itp(pts).astype(np.float32)

    # Replace NaN with dst_nodata
    out = np.where(np.isfinite(out), out, float(dst_nodata)).astype(
        np.float32
    )

    # --------------------------
    # 4) Write GeoTIFF (native geogrid CRS)
    # --------------------------
    bounds, dx, dy = _grid_bounds_from_xy(x_out, y_out)
    west, south, east, north = bounds
    transform = from_origin(west, north, abs(dx), abs(dy))
    crs = CRS.from_epsg(out_epsg)

    profile = {
        "driver": "GTiff",
        "height": int(out.shape[0]),
        "width": int(out.shape[1]),
        "count": 1,
        "dtype": dst_dtype,
        "crs": crs,
        "transform": transform,
        "nodata": float(dst_nodata),
        "tiled": True,
        "compress": "deflate",
    }

    with rasterio.open(out_tif, "w", **profile) as dst:
        dst.write(out.astype(dst_dtype), 1)

    return out_tif


def _approx_degree_spacing_meters(
    *, xcoord: np.ndarray, ycoord: np.ndarray
) -> Tuple[float, float]:
    """
    Approximate pixel spacing (dx, dy) in meters for a geographic grid (EPSG:4326-like).
    Uses mean latitude for the scene.
    """
    if xcoord.size < 2 or ycoord.size < 2:
        raise ValueError("xcoord/ycoord must have at least 2 elements.")
    dx_deg = float(xcoord[1] - xcoord[0])
    dy_deg = float(ycoord[1] - ycoord[0])
    lat0 = float(np.mean(ycoord))

    meters_per_deg_lat = 111132.92
    meters_per_deg_lon = 111319.49 * np.cos(np.deg2rad(lat0))

    dx_m = abs(dx_deg) * meters_per_deg_lon
    dy_m = abs(dy_deg) * meters_per_deg_lat
    return dx_m, dy_m


def _surface_normal_enu_from_dem(
    dem_m: np.ndarray,
    *,
    xcoord: np.ndarray,
    ycoord: np.ndarray,
    epsg: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute unit surface normal (E, N, U) from a DEM on the output grid.
    """
    if int(epsg) == 4326:
        dx_m, dy_m = _approx_degree_spacing_meters(
            xcoord=xcoord, ycoord=ycoord
        )
    else:
        _, dx, dy = _grid_bounds_from_xy(xcoord, ycoord)
        dx_m, dy_m = abs(float(dx)), abs(float(dy))

    dz_dy, dz_dx = np.gradient(dem_m.astype(np.float64), dy_m, dx_m)

    n_e = -dz_dx
    n_n = -dz_dy
    n_u = np.ones_like(dem_m, dtype=np.float64)

    norm = np.sqrt(n_e * n_e + n_n * n_n + n_u * n_u)
    norm = np.where(norm == 0.0, np.nan, norm)

    return (
        (n_e / norm).astype(np.float32),
        (n_n / norm).astype(np.float32),
        (n_u / norm).astype(np.float32),
    )


def interpolate_incidence_and_local_incidence_to_geotiff(
    gunw_h5: Union[str, Path],
    dem_path: Union[str, Path],
    *,
    out_inc_tif: Union[str, Path],
    out_local_inc_tif: Union[str, Path],
    frequency: str = "A",
    pol: str = "HH",
    gunw_geogrid_group: str = "unwrappedInterferogram",
    cube_interp_method: str = "linear",
    dem_resampling: str = "bilinear",
    overwrite: bool = False,
    dst_nodata: float = 0.0,
) -> Tuple[Path, Path]:
    """
    Produce BOTH:
      1) interpolated incidenceAngle
      2) interpolated local incidence angle using LOS unit vectors + DEM-derived surface normal

    Notes
    -----
    - Uses the same prep_nisar-style DEM warping + 3D interpolation.
    - Does NOT mask by unwrappedPhase (geometry fields should be defined everywhere).
    - Assumes losUnitVectorX/Y/Z are in local ENU (E, N, U).
    """
    gunw_h5 = Path(gunw_h5)
    dem_path = Path(dem_path)
    out_inc_tif = Path(out_inc_tif)
    out_local_inc_tif = Path(out_local_inc_tif)
    out_inc_tif.parent.mkdir(parents=True, exist_ok=True)
    out_local_inc_tif.parent.mkdir(parents=True, exist_ok=True)

    if (
        out_inc_tif.exists()
        and out_local_inc_tif.exists()
        and (not overwrite)
    ):
        return out_inc_tif, out_local_inc_tif

    try:
        import h5py
    except Exception as e:
        raise ImportError(
            "h5py is required to read NISAR GUNW HDF5 files."
        ) from e

    try:
        import rasterio
        from rasterio.crs import CRS
        from rasterio.transform import from_origin
    except Exception as e:
        raise ImportError(
            "rasterio is required to write GeoTIFF outputs."
        ) from e

    product_type = "GUNW"
    cube_base = f"/science/LSAR/{product_type}/metadata/radarGrid"

    geo_base = f"/science/LSAR/{product_type}/grids/frequency{frequency}/{gunw_geogrid_group}/{pol}"
    xgeo_path = f"{geo_base}/xCoordinates"
    ygeo_path = f"{geo_base}/yCoordinates"
    proj_path = f"{geo_base}/projection"

    with h5py.File(gunw_h5, "r") as f:
        xrg = np.array(f[f"{cube_base}/xCoordinates"][()])
        yrg = np.array(f[f"{cube_base}/yCoordinates"][()])
        zrg = np.array(f[f"{cube_base}/heightAboveEllipsoid"][()])

        inc = np.asarray(f[f"{cube_base}/incidenceAngle"][()])
        los_e = np.asarray(f[f"{cube_base}/losUnitVectorX"][()])
        los_n = np.asarray(f[f"{cube_base}/losUnitVectorY"][()])
        # Some NISAR files include only X/Y; derive Z from unit-length constraint.
        los_u = None
        los_u_path = f"{cube_base}/losUnitVectorZ"
        if los_u_path in f:
            los_u = np.asarray(f[los_u_path][()])

        x_out = np.array(f[xgeo_path][()])
        y_out = np.array(f[ygeo_path][()])
        out_epsg = int(np.array(f[proj_path][()]).item())

    cubes_to_check = [
        ("incidenceAngle", inc),
        ("losUnitVectorX", los_e),
        ("losUnitVectorY", los_n),
    ]
    if los_u is not None:
        cubes_to_check.append(("losUnitVectorZ", los_u))

    for name, cube in cubes_to_check:
        if cube.ndim != 3:
            raise ValueError(
                f"Expected 3D cube for {name}, got shape={cube.shape}"
            )
        if cube.shape != inc.shape:
            raise ValueError(
                f"Cube shape mismatch: {name}={cube.shape}, incidenceAngle={inc.shape}"
            )

    if inc.shape[0] == 2 and zrg.size >= 2:
        z_for_interp = np.array([zrg[0], zrg[-1]])
    else:
        z_for_interp = zrg

    dem_src_epsg = _read_raster_epsg(dem_path)
    dem_on_grid = _warp_to_grid_mem(
        src_path=dem_path,
        src_epsg=dem_src_epsg,
        dst_epsg=out_epsg,
        xcoord=x_out,
        ycoord=y_out,
        resample_alg=dem_resampling,
    ).astype(np.float32, copy=False)

    Y_2d, X_2d = np.meshgrid(y_out, x_out, indexing="ij")
    pts = np.column_stack(
        [
            dem_on_grid.ravel().astype(np.float64),
            Y_2d.ravel().astype(np.float64),
            X_2d.ravel().astype(np.float64),
        ]
    )

    itp_inc = _make_rgi(
        (z_for_interp, yrg, xrg), inc, method=cube_interp_method
    )
    itp_e = _make_rgi(
        (z_for_interp, yrg, xrg), los_e, method=cube_interp_method
    )
    itp_n = _make_rgi(
        (z_for_interp, yrg, xrg), los_n, method=cube_interp_method
    )
    itp_u = (
        _make_rgi((z_for_interp, yrg, xrg), los_u, method=cube_interp_method)
        if los_u is not None
        else None
    )

    inc_out = itp_inc(pts).reshape(dem_on_grid.shape).astype(np.float32)
    le = itp_e(pts).reshape(dem_on_grid.shape).astype(np.float32)
    ln = itp_n(pts).reshape(dem_on_grid.shape).astype(np.float32)
    if itp_u is not None:
        lu = itp_u(pts).reshape(dem_on_grid.shape).astype(np.float32)
    else:
        # Derive Up component. Assuming LOS is expressed in ENU and points from ground to sensor,
        # the Up component should be positive.
        lu_sq = 1.0 - (le * le + ln * ln)
        lu_sq = np.clip(lu_sq, 0.0, None)
        lu = np.sqrt(lu_sq).astype(np.float32)

    # Normalize LOS to unit vectors (guard against numerical drift)
    norm = np.sqrt(le * le + ln * ln + lu * lu)
    norm = np.where(norm > 0, norm, 1.0)
    le = (le / norm).astype(np.float32)
    ln = (ln / norm).astype(np.float32)
    lu = (lu / norm).astype(np.float32)

    n_e, n_n, n_u = _surface_normal_enu_from_dem(
        dem_on_grid, xcoord=x_out, ycoord=y_out, epsg=out_epsg
    )

    dot = le * n_e + ln * n_n + lu * n_u
    dot = np.clip(np.abs(dot), 0.0, 1.0)
    local_inc = np.degrees(np.arccos(dot)).astype(np.float32)

    inc_out = np.where(
        np.isfinite(inc_out), inc_out, float(dst_nodata)
    ).astype(np.float32)
    local_inc = np.where(
        np.isfinite(local_inc), local_inc, float(dst_nodata)
    ).astype(np.float32)

    bounds, dx, dy = _grid_bounds_from_xy(x_out, y_out)
    west, south, east, north = bounds
    transform = from_origin(west, north, abs(dx), abs(dy))
    crs = CRS.from_epsg(out_epsg)

    profile = {
        "driver": "GTiff",
        "height": int(inc_out.shape[0]),
        "width": int(inc_out.shape[1]),
        "count": 1,
        "dtype": "float32",
        "crs": crs,
        "transform": transform,
        "nodata": float(dst_nodata),
        "tiled": True,
        "compress": "deflate",
    }

    with rasterio.open(out_inc_tif, "w", **profile) as dst:
        dst.write(inc_out, 1)

    with rasterio.open(out_local_inc_tif, "w", **profile) as dst:
        dst.write(local_inc, 1)

    return out_inc_tif, out_local_inc_tif


def extract_incidence_angle_from_radargrid(
    gunw_h5: Union[str, Path],
    dem_path: Union[str, Path],
    out_dir: Union[str, Path],
    *,
    frequency: str = "A",
    pol: str = "HH",
    gunw_geogrid_group: str = "unwrappedInterferogram",
    cube_interp_method: str = "linear",
    dem_resampling: str = "bilinear",
    overwrite: bool = False,
) -> Path:
    """
    3D-interpolate radarGrid/incidenceAngle to the GUNW geogrid using DEM.
    Output name matches your other exports via _format_outname().
    """
    gunw_h5 = Path(gunw_h5)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_name = _format_outname(gunw_h5, "incidenceAngle")
    out_tif = out_dir / out_name
    return interpolate_gunw_radargrid_cube_to_geotiff(
        gunw_h5=gunw_h5,
        dem_path=dem_path,
        cube_ds_name="incidenceAngle",
        out_tif=out_tif,
        frequency=frequency,
        pol=pol,
        gunw_geogrid_group=gunw_geogrid_group,
        cube_interp_method=cube_interp_method,
        dem_resampling=dem_resampling,
        overwrite=overwrite,
        dst_dtype="float32",
        dst_nodata=0.0,
    )


def extract_incidence_and_local_incidence_angle_from_radargrid(
    gunw_h5: Union[str, Path],
    dem_path: Union[str, Path],
    out_dir: Union[str, Path],
    *,
    frequency: str = "A",
    pol: str = "HH",
    gunw_geogrid_group: str = "unwrappedInterferogram",
    cube_interp_method: str = "linear",
    dem_resampling: str = "bilinear",
    overwrite: bool = False,
) -> Dict[str, Path]:
    """
    Write BOTH interpolated incidenceAngle and localIncidenceAngle to GeoTIFF.
    Returns {"incidenceAngle": <path>, "localIncidenceAngle": <path>}.
    """
    gunw_h5 = Path(gunw_h5)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    inc_tif = out_dir / _format_outname(gunw_h5, "incidenceAngle_interp")
    local_tif = out_dir / _format_outname(
        gunw_h5, "localIncidenceAngle_interp"
    )

    interpolate_incidence_and_local_incidence_to_geotiff(
        gunw_h5=gunw_h5,
        dem_path=dem_path,
        out_inc_tif=inc_tif,
        out_local_inc_tif=local_tif,
        frequency=frequency,
        pol=pol,
        gunw_geogrid_group=gunw_geogrid_group,
        cube_interp_method=cube_interp_method,
        dem_resampling=dem_resampling,
        overwrite=overwrite,
        dst_nodata=0.0,
    )
    return {"incidenceAngle": inc_tif, "localIncidenceAngle": local_tif}


def extract_combined_tropo_phase_screen_from_radargrid(
    gunw_h5: Union[str, Path],
    dem_path: Union[str, Path],
    out_dir: Union[str, Path],
    *,
    frequency: str = "A",
    pol: str = "HH",
    gunw_geogrid_group: str = "unwrappedInterferogram",
    cube_interp_method: str = "linear",
    dem_resampling: str = "bilinear",
    overwrite: bool = False,
) -> Path:
    """
    combinedTroposphericPhaseScreen = hydrostaticTroposphericPhaseScreen + wetTroposphericPhaseScreen
    (sum in radarGrid space, then 3D interpolate to geogrid).
    """
    gunw_h5 = Path(gunw_h5)
    dem_path = Path(dem_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_name = _format_outname(gunw_h5, "combinedTroposphericPhaseScreen")
    out_tif = out_dir / out_name
    try:
        import h5py
    except Exception as e:
        raise ImportError(
            "h5py is required to read NISAR GUNW HDF5 files."
        ) from e
    # ---- radarGrid dataset paths (adjust if your sample differs) ----
    # These are the typical locations. If yours are different, we can wire them
    # through small path helper functions just like the geogrid ones.
    freq = frequency.upper()
    pol = pol.upper()
    hydro_path = (
        f"science/LSAR/GUNW/metadata/radarGrid/frequency{freq}/"
        f"hydrostaticTroposphericPhaseScreen"
    )
    wet_path = (
        f"science/LSAR/GUNW/metadata/radarGrid/frequency{freq}/"
        f"wetTroposphericPhaseScreen"
    )
    with h5py.File(gunw_h5, "r") as f:
        if hydro_path not in f:
            raise ValueError(f"Missing hydrostatic cube:\n  {hydro_path}")
        if wet_path not in f:
            raise ValueError(f"Missing wet cube:\n  {wet_path}")
        hydro_ds = f[hydro_path]
        wet_ds = f[wet_path]
        hydro = hydro_ds[()]
        wet = wet_ds[()]
        if hydro.shape != wet.shape:
            raise ValueError(
                f"Hydro and wet cube shapes differ: hydro={hydro.shape} wet={wet.shape}"
            )
        if hydro.ndim != 3:
            raise ValueError(
                f"Expected 3D cubes, got hydro.ndim={hydro.ndim}"
            )
        # Handle fill/nodata robustly
        hydro_fill = hydro_ds.attrs.get("_FillValue", None)
        wet_fill = wet_ds.attrs.get("_FillValue", None)
        hydro_mask = np.isfinite(hydro)
        wet_mask = np.isfinite(wet)
        if hydro_fill is not None:
            try:
                hydro_mask &= hydro != float(hydro_fill)
            except Exception:
                pass
        if wet_fill is not None:
            try:
                wet_mask &= wet != float(wet_fill)
            except Exception:
                pass
        valid = hydro_mask & wet_mask
        combined = np.full(hydro.shape, np.nan, dtype=np.float32)
        combined[valid] = hydro[valid].astype(np.float32) + wet[valid].astype(
            np.float32
        )
    # Now interpolate the *combined* cube
    return interpolate_gunw_radargrid_cube_to_geotiff(
        gunw_h5=gunw_h5,
        dem_path=dem_path,
        cube_data=combined,  # <-- key change
        out_tif=out_tif,
        frequency=frequency,
        pol=pol,
        gunw_geogrid_group=gunw_geogrid_group,
        cube_interp_method=cube_interp_method,
        dem_resampling=dem_resampling,
        overwrite=overwrite,
        dst_dtype="float32",
        dst_nodata=0.0,
    )


def extract_total_troposphere_from_radargrid(
    gunw_h5: Union[str, Path],
    dem_path: Union[str, Path],
    *,
    out_tif: Union[str, Path],
    frequency: str = "A",
    pol: str = "HH",
    overwrite: bool = False,
) -> Path:
    """Compute totalTroposphere = hydrostatic + wet (radarGrid cubes) and interpolate to geogrid.

    This follows the same cube->geogrid interpolation used elsewhere (prep_nisar-style).
    """
    gunw_h5 = Path(gunw_h5)
    out_tif = Path(out_tif)

    try:
        import h5py
        import numpy as np
    except Exception as e:
        raise ImportError(
            "h5py and numpy are required to read NISAR GUNW HDF5 files"
        ) from e

    cube_base = "/science/LSAR/GUNW/metadata/radarGrid"
    hydro_path = f"{cube_base}/hydrostaticTroposphericPhaseScreen"
    wet_path = f"{cube_base}/wetTroposphericPhaseScreen"

    with h5py.File(gunw_h5, "r") as f:
        if hydro_path not in f or wet_path not in f:
            missing = []
            if hydro_path not in f:
                missing.append(hydro_path)
            if wet_path not in f:
                missing.append(wet_path)
            raise ValueError(
                "Missing required radarGrid cube dataset(s) to compute totalTroposphere:\n  "
                + "\n  ".join(missing)
            )
        hydro = np.asarray(f[hydro_path][()])
        wet = np.asarray(f[wet_path][()])
    if hydro.shape != wet.shape:
        raise ValueError(
            f"Hydro and wet cubes have different shapes: hydro={hydro.shape}, wet={wet.shape}"
        )

    total = hydro + wet

    return interpolate_gunw_radargrid_cube_to_geotiff(
        gunw_h5=gunw_h5,
        dem_path=dem_path,
        cube_ds_name=None,
        cube_data=total,
        out_tif=out_tif,
        frequency=frequency,
        pol=pol,
        cube_interp_method="linear",
        overwrite=overwrite,
        dst_nodata=0.0,
    )
