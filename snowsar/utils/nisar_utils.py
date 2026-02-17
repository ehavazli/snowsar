# utils/nisar_utils.py
from __future__ import annotations

import logging
import posixpath
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import geopandas as gpd
import h5py
import numpy as np
import pandas as pd
from rasterio.features import shapes
from rasterio.transform import from_origin
from shapely.geometry import Polygon, shape
from shapely.ops import unary_union

logger = logging.getLogger(__name__)


# -----------------------------
# HDF5 path resolution helpers
# -----------------------------
def _h5_exists(f: Union[h5py.File, h5py.Group], path: str) -> bool:
    try:
        f[path]
        return True
    except KeyError:
        return False


def resolve_h5_path(
    f: Union[h5py.File, h5py.Group],
    path: str,
    *,
    extra_candidates: Optional[Sequence[str]] = None,
) -> str:
    """Resolve an HDF5 dataset/group path robustly.

    NISAR GUNW files are sometimes addressed with and without a leading '/'
    (e.g., '/science/LSAR/...' vs 'science/LSAR/...'). This helper tries both
    forms (plus any extra candidates) and returns the first that exists.

    Raises
    ------
    KeyError
        If none of the candidate paths exist.
    """
    p = posixpath.normpath(path.strip())

    candidates: List[str] = []
    if p.startswith("/"):
        candidates.extend([p, p.lstrip("/")])
    else:
        candidates.extend(["/" + p, p])

    if extra_candidates:
        candidates.extend(
            [posixpath.normpath(c.strip()) for c in extra_candidates]
        )

    # Deduplicate while preserving order
    seen = set()
    uniq: List[str] = []
    for c in candidates:
        if c not in seen:
            uniq.append(c)
            seen.add(c)

    for c in uniq:
        if _h5_exists(f, c):
            return c

    top_keys = list(f.keys())[:30]
    raise KeyError(
        f"HDF5 path not found. Tried: {uniq}. Top-level keys: {top_keys}"
    )


def h5_get(
    f: Union[h5py.File, h5py.Group], path: str
) -> Union[h5py.Dataset, h5py.Group]:
    """Convenience wrapper returning the dataset/group at a resolved path."""
    return f[resolve_h5_path(f, path)]


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

    def _read_str(ds) -> str:
        val = ds[()]
        if isinstance(val, bytes):
            return val.decode("utf-8", errors="ignore")
        if isinstance(val, np.ndarray) and val.dtype.kind in {"S", "O"}:
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
        try:
            ref_p = resolve_h5_path(f, ref_path)
            sec_p = resolve_h5_path(f, sec_path)
        except KeyError as e:
            raise ValueError(
                f"Missing identification time datasets in {gunw_h5.name}. "
                f"Expected: {ref_path} and {sec_path}"
            ) from e
        ref_str = _read_str(f[ref_p])
        sec_str = _read_str(f[sec_p])

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
    """
    gunw_h5 = Path(gunw_h5)
    raster_path = raster_path or gunw_unwrapped_phase_path(
        frequency=frequency, pol=pol
    )

    with h5py.File(gunw_h5, "r") as f:
        try:
            rp = resolve_h5_path(f, raster_path)
        except KeyError as e:
            raise ValueError(
                f"Raster dataset not found in {gunw_h5.name}:\n  {raster_path}"
            ) from e
        ds = f[rp]
        arr = ds[()]  # numpy array
        grp = ds.parent
        if "xCoordinates" not in grp or "yCoordinates" not in grp:
            raise ValueError(
                f"Missing xCoordinates/yCoordinates near raster path:\n  {raster_path}"
            )
        x = np.array(grp["xCoordinates"][()])
        y = np.array(grp["yCoordinates"][()])
        if "xCoordinateSpacing" in grp:
            dx = float(np.array(grp["xCoordinateSpacing"][()]).item())
        else:
            dx = float(x[1] - x[0])
        if "yCoordinateSpacing" in grp:
            dy = float(np.array(grp["yCoordinateSpacing"][()]).item())
        else:
            dy = float(y[1] - y[0])
        epsg = None
        if "projection" in grp:
            try:
                epsg = int(np.array(grp["projection"][()]).item())
            except Exception:
                epsg = None
        crs_src = f"EPSG:{epsg}" if epsg else None
        mask = np.isfinite(arr)
        fill = ds.attrs.get("_FillValue", None)
        if fill is not None:
            try:
                fill_val = float(fill)
                mask &= arr != fill_val
            except Exception:
                pass

    res_x = abs(dx)
    res_y = abs(dy)
    west = float(np.min(x) - res_x / 2.0)
    north = float(np.max(y) + res_y / 2.0)
    transform = from_origin(west, north, res_x, res_y)

    mask_u8 = mask.astype(np.uint8)
    polys = []
    for geom, val in shapes(
        mask_u8, mask=mask.astype(bool), transform=transform
    ):
        if val != 1:
            continue
        p = shape(geom)
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
        return gpd.GeoDataFrame(geometry=[], crs=crs_out)

    merged = unary_union(polys)
    gdf = gpd.GeoDataFrame(geometry=[merged], crs=crs_src)
    if crs_out and gdf.crs is not None:
        gdf = gdf.to_crs(crs_out)
    elif crs_out and gdf.crs is None:
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
    for f_ in gunw_files:
        gdf = nisar_footprint_from_gunw_h5(
            f_,
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
    Returns (ref_date, sec_date, track, frame) as strings.
    """
    p = Path(gunw_h5)
    parts = p.stem.split("_")
    if len(parts) <= max(ref_block, sec_block):
        raise ValueError(
            f"Filename does not have enough '_' blocks for ref_block={ref_block}, sec_block={sec_block}:\n"
            f"  {p.name}\n"
            f"  n_blocks={len(parts)}"
        )
    ref_date = parts[ref_block].split("T")[0]
    sec_date = parts[sec_block].split("T")[0]
    track = parts[track_block]
    frame = parts[frame_block]
    return ref_date, sec_date, track, frame


def _format_outname(
    gunw_h5: Union[str, Path],
    layer_name: str,
) -> str:
    ref_date, sec_date, track, frame = _gunw_date_tokens_from_filename(
        gunw_h5
    )
    return f"{ref_date}_{sec_date}_{layer_name}_T{track}_F{frame}.tif"


# -----------------------------
# Dataset discovery / routing
# -----------------------------
@dataclass(frozen=True)
class DatasetInfo:
    """Lightweight index entry for an HDF5 dataset."""

    path: str
    name: str
    ndim: int
    shape: Tuple[int, ...]
    parent_path: str


def build_dataset_index(f: h5py.File) -> Dict[str, List[DatasetInfo]]:
    """Build an index: dataset basename -> list of candidates across the file."""
    out: Dict[str, List[DatasetInfo]] = {}

    def _visitor(name: str, obj) -> None:
        if not isinstance(obj, h5py.Dataset):
            return
        base = name.split("/")[-1]
        parent = "/".join(name.split("/")[:-1])
        full_path = ("/" + name) if not name.startswith("/") else name
        parent_path = (
            ("/" + parent)
            if (parent and not parent.startswith("/"))
            else ("/" if not parent else parent)
        )
        info = DatasetInfo(
            path=full_path,
            name=base,
            ndim=obj.ndim,
            shape=tuple(obj.shape),
            parent_path=parent_path,
        )
        out.setdefault(base, []).append(info)

    f.visititems(_visitor)
    return out


def pick_best_candidate(
    candidates: List[DatasetInfo],
    *,
    frequency: str,
    pol: str,
    prefer_geogrid: bool = True,
) -> DatasetInfo:
    """Pick the best dataset match among multiple candidates.

    Preference order (roughly):
    1) 2D geogrid datasets under /grids/ (if prefer_geogrid=True)
    2) matching frequency{A/B}
    3) matching polarization folder
    4) radarGrid cubes under /metadata/radarGrid/ (as fallback)
    """
    freq = f"frequency{frequency.upper()}"
    pol_u = pol.upper()

    def score(c: DatasetInfo) -> int:
        p = c.path
        s = 0
        if prefer_geogrid and "/grids/" in p and c.ndim == 2:
            s += 100
        if f"/{freq}/" in p:
            s += 30
        if f"/{pol_u}/" in p:
            s += 25
        if "/metadata/radarGrid/" in p and c.ndim == 3:
            s += 20
        if "/unwrappedInterferogram/" in p:
            s += 5
        return s

    return sorted(candidates, key=score, reverse=True)[0]


def is_geogrid_2d(f: h5py.File, info: DatasetInfo) -> bool:
    if info.ndim != 2:
        return False
    try:
        parent = h5_get(f, info.parent_path)
    except KeyError:
        return False
    return (
        isinstance(parent, h5py.Group)
        and ("xCoordinates" in parent)
        and ("yCoordinates" in parent)
    )


def is_radargrid_cube(f: h5py.File, info: DatasetInfo) -> bool:
    if info.ndim != 3:
        return False
    if "/metadata/radarGrid/" in info.path:
        return True
    try:
        parent = h5_get(f, info.parent_path)
    except KeyError:
        return False
    return isinstance(parent, h5py.Group) and (
        ("xCoordinates" in parent or "yCoordinates" in parent)
        and ("heightAboveEllipsoid" in parent or "zCoordinates" in parent)
    )


def resolve_layer_requests_for_file(
    f: h5py.File,
    *,
    requested: Union[str, Sequence[str]],
    frequency: str,
    pol: str,
    prefer_geogrid: bool = True,
) -> List[DatasetInfo]:
    """Resolve requested layer names to concrete dataset paths within this file.

    If requested == 'all' (case-insensitive), returns all 2D geogrid datasets that match
    the frequency/pol preference ranking (i.e., picks one best candidate per basename).
    """
    idx = build_dataset_index(f)

    if isinstance(requested, str) and requested.lower() == "all":
        picked: List[DatasetInfo] = []
        for name, cands in idx.items():
            best = pick_best_candidate(
                cands,
                frequency=frequency,
                pol=pol,
                prefer_geogrid=prefer_geogrid,
            )
            if is_geogrid_2d(f, best):
                picked.append(best)
        return sorted(picked, key=lambda d: d.name)

    if isinstance(requested, str):
        requested_names = [requested]
    else:
        requested_names = list(requested)

    out: List[DatasetInfo] = []
    missing: List[str] = []
    for name in requested_names:
        if name in idx:
            best = pick_best_candidate(
                idx[name],
                frequency=frequency,
                pol=pol,
                prefer_geogrid=prefer_geogrid,
            )
            out.append(best)
        else:
            missing.append(name)

    if missing:
        raise ValueError(f"Layer(s) not found in file: {missing}")
    return out


# -----------------------------
# Batch extraction
# -----------------------------
def extract_gunw_layers_to_geotiff_batch(
    gunw_dir: Union[str, Path],
    pattern: str,
    out_dir: Union[str, Path],
    *,
    frequency: str = "A",
    pol: str = "HH",
    layers: Union[str, Sequence[str]] = (
        "unwrappedPhase",
        "coherenceMagnitude",
        "ionospherePhaseScreen",
        "connectedComponents",
    ),
    warp: bool = True,
    dst_epsg: Optional[int] = None,
    dst_res: Optional[float] = None,
    resampling: str = "nearest",
    overwrite: bool = False,
    dem_dir: Optional[Union[str, Path]] = None,
    dem_precision: int = 2,
) -> Dict[Path, Dict[str, Path]]:
    """Extract arbitrary layer names from NISAR GUNW HDF5 to GeoTIFFs.

    - Prefers 2D geogrid datasets when both geogrid (2D) and radarGrid (3D cube) exist.
    - If a requested layer is only available as a radarGrid cube, it is interpolated to geogrid
      using DEM and exported (suffix *_interp).
    - Computed aliases:
        * totalTroposphere = hydrostaticTroposphericPhaseScreen + wetTroposphericPhaseScreen (radarGrid cubes)
        * localIncidenceAngle derived from DEM surface normal and LOS unit vectors (radarGrid cubes)
    - When warp=True, all outputs are reprojected onto a single per-file template grid derived from
      unwrappedPhase (or first available geogrid 2D dataset).
    """
    gunw_dir = Path(gunw_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(gunw_dir.glob(pattern))
    if not files:
        raise ValueError(f"No files found: {gunw_dir}/{pattern}")

    try:
        import rasterio
        from rasterio.crs import CRS
        from rasterio.enums import Resampling
        from rasterio.warp import calculate_default_transform, reproject
    except Exception as e:
        raise ImportError(
            "rasterio is required for GeoTIFF export/warp."
        ) from e

    resamp_map = {
        "nearest": Resampling.nearest,
        "bilinear": Resampling.bilinear,
        "cubic": Resampling.cubic,
        "average": Resampling.average,
    }
    if resampling not in resamp_map:
        raise ValueError(
            f"Unsupported resampling: {resampling}. Choose from {list(resamp_map)}"
        )
    user_resamp = resamp_map[resampling]

    def _write_geotiff(
        arr2d: np.ndarray,
        out_path: Path,
        *,
        crs,
        transform,
        nodata=None,
    ) -> None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        profile = {
            "driver": "GTiff",
            "height": arr2d.shape[0],
            "width": arr2d.shape[1],
            "count": 1,
            "dtype": arr2d.dtype,
            "crs": crs,
            "transform": transform,
            "nodata": nodata,
            "tiled": True,
            "compress": "deflate",
        }
        with rasterio.open(out_path, "w", **profile) as dst_ds:
            dst_ds.write(arr2d, 1)

    def _warp_to_template(
        src_arr: np.ndarray,
        *,
        src_transform,
        src_crs,
        dst_transform,
        dst_crs,
        dst_shape: Tuple[int, int],
        resamp,
        src_nodata=None,
        dst_nodata=np.nan,
    ) -> np.ndarray:
        dst = np.full(dst_shape, dst_nodata, dtype=np.float32)
        reproject(
            source=src_arr.astype(np.float32, copy=False),
            destination=dst,
            src_transform=src_transform,
            src_crs=src_crs,
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            resampling=resamp,
            src_nodata=src_nodata,
            dst_nodata=dst_nodata,
        )
        return dst

    # Computed alias registry
    computed_aliases = {"totalTroposphere", "localIncidenceAngle"}

    all_outputs: Dict[Path, Dict[str, Path]] = {}

    for gunw_h5 in files:
        gunw_h5_path = Path(gunw_h5)
        per_file: Dict[str, Path] = {}

        with h5py.File(gunw_h5_path, "r") as f:
            idx = build_dataset_index(f)

            # Determine requested names and dataset infos
            template_info: Optional[DatasetInfo] = (
                None  # ensure defined for warp=False too
            )

            if isinstance(layers, str) and layers.lower() == "all":
                requested_infos = resolve_layer_requests_for_file(
                    f,
                    requested="all",
                    frequency=frequency,
                    pol=pol,
                    prefer_geogrid=True,
                )
                requested_names = [d.name for d in requested_infos]
            else:
                requested_names = (
                    [layers] if isinstance(layers, str) else list(layers)
                )
                requested_infos = []
                missing = []
                for nm in requested_names:
                    if nm in idx:
                        best = pick_best_candidate(
                            idx[nm],
                            frequency=frequency,
                            pol=pol,
                            prefer_geogrid=True,
                        )
                        requested_infos.append(best)
                    elif nm in computed_aliases:
                        continue
                    else:
                        missing.append(nm)
                if missing:
                    raise ValueError(
                        f"Layer(s) not found in {gunw_h5_path.name}: {missing}"
                    )

            # -----------------------------
            # Build warp template grid ONCE
            # -----------------------------
            dst_crs = None
            dst_transform = None
            dst_width = None
            dst_height = None

            if warp:
                if dst_epsg is None:
                    raise ValueError(
                        "dst_epsg must be provided when warp=True"
                    )
                dst_crs = CRS.from_epsg(int(dst_epsg))

                if "unwrappedPhase" in idx:
                    cand = pick_best_candidate(
                        idx["unwrappedPhase"],
                        frequency=frequency,
                        pol=pol,
                        prefer_geogrid=True,
                    )
                    if is_geogrid_2d(f, cand):
                        template_info = cand
                if template_info is None:
                    for info in requested_infos:
                        if is_geogrid_2d(f, info):
                            template_info = info
                            break
                if template_info is None:
                    raise ValueError(
                        "warp=True requires at least one 2D geogrid dataset to define the template grid."
                    )

                ds_ref = h5_get(f, template_info.path)
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

                res_x_ref = abs(dx_ref)
                res_y_ref = abs(dy_ref)
                west_ref = float(np.min(x_ref) - res_x_ref / 2.0)
                north_ref = float(np.max(y_ref) + res_y_ref / 2.0)
                h_ref, w_ref = ds_ref.shape
                src_transform_ref = from_origin(
                    west_ref, north_ref, res_x_ref, res_y_ref
                )

                left = west_ref
                right = left + w_ref * res_x_ref
                top = north_ref
                bottom = top - h_ref * res_y_ref

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

            # -----------------------------
            # Build valid-mask from unwrappedPhase (preferred) else template layer
            # -----------------------------
            valid_info: Optional[DatasetInfo] = None
            if "unwrappedPhase" in idx:
                cand = pick_best_candidate(
                    idx["unwrappedPhase"],
                    frequency=frequency,
                    pol=pol,
                    prefer_geogrid=True,
                )
                if is_geogrid_2d(f, cand):
                    valid_info = cand
            if valid_info is None:
                # If warp, template_info will exist; otherwise fall back to first geogrid 2D requested
                if template_info is not None:
                    valid_info = template_info
                else:
                    for info in requested_infos:
                        if is_geogrid_2d(f, info):
                            valid_info = info
                            break
            if valid_info is None:
                raise ValueError(
                    "Could not determine a geogrid 2D dataset to build a validity mask."
                )

            ds_valid = h5_get(f, valid_info.path)
            grp_valid = ds_valid.parent
            arr_valid_native = ds_valid[()]

            x_v = np.array(grp_valid["xCoordinates"][()])
            y_v = np.array(grp_valid["yCoordinates"][()])
            dx_v = (
                float(np.array(grp_valid["xCoordinateSpacing"][()]).item())
                if "xCoordinateSpacing" in grp_valid
                else float(x_v[1] - x_v[0])
            )
            dy_v = (
                float(np.array(grp_valid["yCoordinateSpacing"][()]).item())
                if "yCoordinateSpacing" in grp_valid
                else float(y_v[1] - y_v[0])
            )
            epsg_v = int(np.array(grp_valid["projection"][()]).item())
            src_crs_v = CRS.from_epsg(epsg_v)

            res_x_v = abs(dx_v)
            res_y_v = abs(dy_v)
            west_v = float(np.min(x_v) - res_x_v / 2.0)
            north_v = float(np.max(y_v) + res_y_v / 2.0)
            src_transform_v = from_origin(west_v, north_v, res_x_v, res_y_v)

            fill_v = ds_valid.attrs.get("_FillValue", None)

            if warp:
                assert (
                    dst_transform is not None
                    and dst_crs is not None
                    and dst_width is not None
                    and dst_height is not None
                )
                out_valid_arr = _warp_to_template(
                    arr_valid_native,
                    src_transform=src_transform_v,
                    src_crs=src_crs_v,
                    dst_transform=dst_transform,
                    dst_crs=dst_crs,
                    dst_shape=(dst_height, dst_width),
                    resamp=Resampling.nearest,
                    src_nodata=float(fill_v) if fill_v is not None else None,
                    dst_nodata=np.nan,
                )
                unw_valid = np.isfinite(out_valid_arr)
                if fill_v is not None:
                    unw_valid &= out_valid_arr != float(fill_v)
            else:
                out_valid_arr = arr_valid_native.astype(
                    np.float32, copy=False
                )
                unw_valid = np.isfinite(out_valid_arr)
                if fill_v is not None:
                    unw_valid &= out_valid_arr != float(fill_v)

            # -----------------------------
            # DEM preparation (only if any cube extraction or derived LIA/tropo is needed)
            # -----------------------------
            need_dem = False
            if any(
                n in requested_names
                for n in ("incidenceAngle", "localIncidenceAngle")
            ):
                need_dem = True
            for nm in requested_names:
                if nm in computed_aliases:
                    need_dem = True
                elif nm in idx:
                    best = pick_best_candidate(
                        idx[nm],
                        frequency=frequency,
                        pol=pol,
                        prefer_geogrid=True,
                    )
                    if is_radargrid_cube(f, best) and not is_geogrid_2d(
                        f, best
                    ):
                        need_dem = True

            dem_out: Optional[Path] = None
            if need_dem:
                dem_dir_path = (
                    Path(dem_dir) if dem_dir else (out_dir / "dem_cache")
                )
                dem_dir_path.mkdir(parents=True, exist_ok=True)

                gdf = nisar_footprint_from_gunw_h5(
                    gunw_h5_path,
                    raster_path=gunw_unwrapped_phase_path(
                        frequency=frequency, pol=pol
                    ),
                    frequency=frequency,
                    pol=pol,
                    crs_out="EPSG:4326",
                )
                minx, miny, maxx, maxy = gdf.total_bounds
                key = (
                    round(float(minx), dem_precision),
                    round(float(miny), dem_precision),
                    round(float(maxx), dem_precision),
                    round(float(maxy), dem_precision),
                )
                dem_out = (
                    dem_dir_path
                    / f"dem_{key[0]}_{key[1]}_{key[2]}_{key[3]}.tif"
                )

                if overwrite or (not dem_out.exists()):
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
                else:
                    logger.info("Reusing cached DEM -> %s", dem_out)

                # Keep per-file copy
                dem_copy = out_dir / f"{gunw_h5_path.stem}_DEM.tif"
                if overwrite or (not dem_copy.exists()):
                    shutil.copy2(dem_out, dem_copy)
                per_file["DEM"] = dem_copy

            # -----------------------------
            # Extract 2D geogrid datasets
            # -----------------------------
            for info in requested_infos:
                if not is_geogrid_2d(f, info):
                    continue

                ds = h5_get(f, info.path)
                grp = ds.parent
                arr = ds[()]
                if arr.ndim != 2:
                    continue

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
                epsg = int(np.array(grp["projection"][()]).item())
                src_crs = CRS.from_epsg(epsg)
                res_x = abs(dx)
                res_y = abs(dy)
                west = float(np.min(x) - res_x / 2.0)
                north = float(np.max(y) + res_y / 2.0)
                src_transform = from_origin(west, north, res_x, res_y)

                out_name = _format_outname(gunw_h5_path, info.name)
                out_path = out_dir / out_name
                if out_path.exists() and not overwrite:
                    per_file[info.name] = out_path
                    continue

                resamp_here = (
                    Resampling.nearest
                    if info.name == "connectedComponents"
                    else user_resamp
                )
                src_fill = ds.attrs.get("_FillValue", None)

                if warp:
                    dst_arr = _warp_to_template(
                        arr,
                        src_transform=src_transform,
                        src_crs=src_crs,
                        dst_transform=dst_transform,
                        dst_crs=dst_crs,
                        dst_shape=(dst_height, dst_width),
                        resamp=resamp_here,
                        src_nodata=(
                            float(src_fill) if src_fill is not None else None
                        ),
                        dst_nodata=np.nan,
                    )
                    if info.name == "connectedComponents":
                        dst_arr[~unw_valid] = np.nan
                    _write_geotiff(
                        dst_arr.astype(np.float32),
                        out_path,
                        crs=dst_crs,
                        transform=dst_transform,
                        nodata=np.nan,
                    )
                else:
                    arr_out = arr.astype(np.float32, copy=False)
                    if info.name == "connectedComponents":
                        arr_out[~unw_valid] = np.nan
                    _write_geotiff(
                        arr_out,
                        out_path,
                        crs=src_crs,
                        transform=src_transform,
                        nodata=np.nan,
                    )

                per_file[info.name] = out_path

            # -----------------------------
            # Derived angles: localIncidenceAngle (and optionally incidenceAngle from cube)
            # -----------------------------
            need_local = ("localIncidenceAngle" in requested_names) or (
                "incidenceAngle" in requested_names
            )
            if need_local:
                if dem_out is None:
                    raise ValueError(
                        "DEM is required to derive localIncidenceAngle but was not prepared."
                    )

                loc_label = "localIncidenceAngle_interp"
                loc_out = out_dir / _format_outname(gunw_h5_path, loc_label)

                want_inc_interp = ("incidenceAngle" in requested_names) and (
                    "incidenceAngle" not in per_file
                )
                inc_label = "incidenceAngle_interp"
                inc_out = out_dir / _format_outname(gunw_h5_path, inc_label)

                have_loc = loc_out.exists() and not overwrite
                have_inc = inc_out.exists() and not overwrite

                if (not have_loc) or (want_inc_interp and not have_inc):
                    inc_arr, local_arr, src_t, src_c = (
                        interpolate_incidence_and_local_incidence(
                            gunw_h5_path,
                            dem_out,
                            frequency=frequency,
                            pol=pol,
                        )
                    )

                    if warp:
                        dst_shape = (dst_height, dst_width)
                        loc_w = _warp_to_template(
                            local_arr,
                            src_transform=src_t,
                            src_crs=src_c,
                            dst_transform=dst_transform,
                            dst_crs=dst_crs,
                            dst_shape=dst_shape,
                            resamp=Resampling.nearest,
                            src_nodata=np.nan,
                            dst_nodata=np.nan,
                        )
                        loc_w[~unw_valid] = np.nan
                        _write_geotiff(
                            loc_w.astype(np.float32),
                            loc_out,
                            crs=dst_crs,
                            transform=dst_transform,
                            nodata=np.nan,
                        )

                        if want_inc_interp:
                            inc_w = _warp_to_template(
                                inc_arr,
                                src_transform=src_t,
                                src_crs=src_c,
                                dst_transform=dst_transform,
                                dst_crs=dst_crs,
                                dst_shape=dst_shape,
                                resamp=Resampling.nearest,
                                src_nodata=np.nan,
                                dst_nodata=np.nan,
                            )
                            inc_w[~unw_valid] = np.nan
                            _write_geotiff(
                                inc_w.astype(np.float32),
                                inc_out,
                                crs=dst_crs,
                                transform=dst_transform,
                                nodata=np.nan,
                            )
                    else:
                        local_arr = local_arr.astype(np.float32, copy=False)
                        local_arr[~unw_valid] = np.nan
                        _write_geotiff(
                            local_arr,
                            loc_out,
                            crs=src_c,
                            transform=src_t,
                            nodata=np.nan,
                        )

                        if want_inc_interp:
                            inc_arr = inc_arr.astype(np.float32, copy=False)
                            inc_arr[~unw_valid] = np.nan
                            _write_geotiff(
                                inc_arr,
                                inc_out,
                                crs=src_c,
                                transform=src_t,
                                nodata=np.nan,
                            )

                per_file[loc_label] = loc_out
                if want_inc_interp:
                    per_file[inc_label] = inc_out

            # -----------------------------
            # Extract cube datasets (interpolate -> geogrid -> warp template)
            # -----------------------------
            for nm in requested_names:
                if nm in ("incidenceAngle", "localIncidenceAngle"):
                    continue
                if nm in per_file:
                    continue

                # computed alias: totalTroposphere
                if nm == "totalTroposphere":
                    if dem_out is None:
                        raise ValueError(
                            "DEM is required for cube interpolation but was not prepared."
                        )
                    label = f"{nm}_interp"
                    out_path = out_dir / _format_outname(gunw_h5_path, label)
                    if out_path.exists() and not overwrite:
                        per_file[label] = out_path
                        continue

                    cube_base = "/science/LSAR/GUNW/metadata/radarGrid"
                    hydro_path = (
                        f"{cube_base}/hydrostaticTroposphericPhaseScreen"
                    )
                    wet_path = f"{cube_base}/wetTroposphericPhaseScreen"
                    try:
                        hydro = np.asarray(h5_get(f, hydro_path)[()])
                        wet = np.asarray(h5_get(f, wet_path)[()])
                    except KeyError as e:
                        raise ValueError(
                            "Cannot compute totalTroposphere; missing hydrostatic/wet troposphere datasets "
                            f"under {cube_base}"
                        ) from e
                    cube = hydro + wet

                    arr_i, src_t, src_c = (
                        interpolate_radargrid_cube_to_geogrid(
                            gunw_h5_path,
                            dem_out,
                            cube_ds_name=None,
                            cube_data=cube,
                            frequency=frequency,
                            pol=pol,
                            gunw_geogrid_group="unwrappedInterferogram",
                        )
                    )
                    if warp:
                        dst_arr = _warp_to_template(
                            arr_i,
                            src_transform=src_t,
                            src_crs=src_c,
                            dst_transform=dst_transform,
                            dst_crs=dst_crs,
                            dst_shape=(dst_height, dst_width),
                            resamp=Resampling.nearest,
                            src_nodata=np.nan,
                            dst_nodata=np.nan,
                        )
                        dst_arr[~unw_valid] = np.nan
                        _write_geotiff(
                            dst_arr.astype(np.float32),
                            out_path,
                            crs=dst_crs,
                            transform=dst_transform,
                            nodata=np.nan,
                        )
                    else:
                        arr_i = arr_i.astype(np.float32, copy=False)
                        arr_i[~unw_valid] = np.nan
                        _write_geotiff(
                            arr_i,
                            out_path,
                            crs=src_c,
                            transform=src_t,
                            nodata=np.nan,
                        )

                    per_file[label] = out_path
                    continue

                if nm not in idx:
                    continue

                best = pick_best_candidate(
                    idx[nm], frequency=frequency, pol=pol, prefer_geogrid=True
                )
                if not is_radargrid_cube(f, best):
                    continue
                if dem_out is None:
                    raise ValueError(
                        "DEM is required for cube interpolation but was not prepared."
                    )

                label = f"{nm}_interp"
                out_path = out_dir / _format_outname(gunw_h5_path, label)
                if out_path.exists() and not overwrite:
                    per_file[label] = out_path
                    continue

                arr_i, src_t, src_c = interpolate_radargrid_cube_to_geogrid(
                    gunw_h5_path,
                    dem_out,
                    cube_ds_name=nm,
                    cube_data=None,
                    frequency=frequency,
                    pol=pol,
                    gunw_geogrid_group="unwrappedInterferogram",
                )

                if warp:
                    dst_arr = _warp_to_template(
                        arr_i,
                        src_transform=src_t,
                        src_crs=src_c,
                        dst_transform=dst_transform,
                        dst_crs=dst_crs,
                        dst_shape=(dst_height, dst_width),
                        resamp=Resampling.nearest,
                        src_nodata=np.nan,
                        dst_nodata=np.nan,
                    )
                    dst_arr[~unw_valid] = np.nan
                    _write_geotiff(
                        dst_arr.astype(np.float32),
                        out_path,
                        crs=dst_crs,
                        transform=dst_transform,
                        nodata=np.nan,
                    )
                else:
                    arr_i = arr_i.astype(np.float32, copy=False)
                    arr_i[~unw_valid] = np.nan
                    _write_geotiff(
                        arr_i,
                        out_path,
                        crs=src_c,
                        transform=src_t,
                        nodata=np.nan,
                    )

                per_file[label] = out_path

        all_outputs[gunw_h5_path] = per_file

    return all_outputs


# -----------------------------
# DEM download (sardem)
# -----------------------------
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
    """
    gunw_h5 = Path(gunw_h5)
    dem_out = Path(dem_out)
    dem_out.parent.mkdir(parents=True, exist_ok=True)
    if dem_out.exists() and not overwrite:
        logger.info("DEM exists, skipping: %s", dem_out)
        return dem_out

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

    west -= buffer_deg
    south -= buffer_deg
    east += buffer_deg
    north += buffer_deg

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
    """Return a deterministic DEM path for a scene footprint to enable reuse."""
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

    w = round(float(west), precision)
    s = round(float(south), precision)
    e = round(float(east), precision)
    n = round(float(north), precision)

    egm_tag = "egm" if keep_egm else "hae"
    fname = f"dem_{data_source}_{egm_tag}_w{w}_s{s}_e{e}_n{n}_buf{round(buffer_deg, precision)}.tif"
    return dem_dir / fname


# -----------------------------
# 3D (radarGrid cube) -> 2D (geogrid) interpolation helpers
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
    raster_path = gunw_unwrapped_phase_path(frequency=frequency, pol=pol)

    with h5py.File(Path(gunw_file), "r") as f:
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
        import rasterio
        from rasterio.crs import CRS
        from rasterio.transform import from_origin as rio_from_origin
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
        xcoords = np.array(h5_get(f, f"{cube_base}/xCoordinates")[()])
        ycoords = np.array(h5_get(f, f"{cube_base}/yCoordinates")[()])
        zcoords = np.array(h5_get(f, f"{cube_base}/heightAboveEllipsoid")[()])

        if cube_data is None:
            cube_path = f"{cube_base}/{cube_ds_name}"
            cube_path_res = resolve_h5_path(f, cube_path)
            cube = np.asarray(f[cube_path_res][()])
        else:
            cube = np.asarray(cube_data)

        x_out = np.array(h5_get(f, xgeo_path)[()])
        y_out = np.array(h5_get(f, ygeo_path)[()])
        out_epsg = int(np.array(h5_get(f, proj_path)[()]).item())

    if cube.ndim != 3:
        raise ValueError(f"Expected 3D cube, got shape={cube.shape}")

    if cube.shape[0] == 2 and zcoords.size >= 2:
        z_for_interp = np.array([zcoords[0], zcoords[-1]])
    else:
        z_for_interp = zcoords

    dem_src_epsg = _read_raster_epsg(dem_path)
    dem_on_grid = _warp_to_grid_mem(
        src_path=dem_path,
        src_epsg=dem_src_epsg,
        dst_epsg=out_epsg,
        xcoord=x_out,
        ycoord=y_out,
        resample_alg=dem_resampling,
    ).astype(np.float32, copy=False)

    valid = _read_valid_unw_mask_full_geogrid(
        gunw_h5, frequency=frequency, pol=pol
    )
    if valid.shape != dem_on_grid.shape:
        raise ValueError(
            "Validity mask shape does not match DEM-on-grid shape. "
            f"mask={valid.shape}, dem={dem_on_grid.shape}."
        )

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

    out = np.where(np.isfinite(out), out, float(dst_nodata)).astype(
        np.float32
    )

    bounds, dx, dy = _grid_bounds_from_xy(x_out, y_out)
    west, south, east, north = bounds
    transform = rio_from_origin(west, north, abs(dx), abs(dy))
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
    """Approximate pixel spacing (dx, dy) in meters for a geographic grid."""
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
    """Compute unit surface normal (E, N, U) from a DEM on the output grid."""
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
    Produce BOTH interpolated incidenceAngle and local incidence angle using LOS unit vectors + DEM normal.
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
        import rasterio
        from rasterio.crs import CRS
        from rasterio.transform import from_origin as rio_from_origin
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
        xrg = np.array(h5_get(f, f"{cube_base}/xCoordinates")[()])
        yrg = np.array(h5_get(f, f"{cube_base}/yCoordinates")[()])
        zrg = np.array(h5_get(f, f"{cube_base}/heightAboveEllipsoid")[()])

        inc = np.asarray(h5_get(f, f"{cube_base}/incidenceAngle")[()])
        los_e = np.asarray(h5_get(f, f"{cube_base}/losUnitVectorX")[()])
        los_n = np.asarray(h5_get(f, f"{cube_base}/losUnitVectorY")[()])

        los_u = None
        los_u_path = f"{cube_base}/losUnitVectorZ"
        if _h5_exists(f, los_u_path) or _h5_exists(f, los_u_path.lstrip("/")):
            los_u = np.asarray(h5_get(f, los_u_path)[()])

        x_out = np.array(h5_get(f, xgeo_path)[()])
        y_out = np.array(h5_get(f, ygeo_path)[()])
        out_epsg = int(np.array(h5_get(f, proj_path)[()]).item())

    for name, cube in [
        ("incidenceAngle", inc),
        ("losUnitVectorX", los_e),
        ("losUnitVectorY", los_n),
    ]:
        if cube.ndim != 3:
            raise ValueError(
                f"Expected 3D cube for {name}, got shape={cube.shape}"
            )
        if cube.shape != inc.shape:
            raise ValueError(
                f"Cube shape mismatch: {name}={cube.shape}, incidenceAngle={inc.shape}"
            )

    if los_u is not None:
        if los_u.ndim != 3 or los_u.shape != inc.shape:
            raise ValueError(
                "losUnitVectorZ cube shape mismatch vs incidenceAngle"
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
        lu_sq = 1.0 - (le * le + ln * ln)
        lu_sq = np.clip(lu_sq, 0.0, None)
        lu = np.sqrt(lu_sq).astype(np.float32)

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
    transform = rio_from_origin(west, north, abs(dx), abs(dy))
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


# -----------------------------
# Wrapper functions expected by the batch extractor
# -----------------------------
def interpolate_radargrid_cube_to_geogrid(
    gunw_h5: Union[str, Path],
    dem_path: Union[str, Path],
    *,
    cube_ds_name: Optional[str],
    cube_data: Optional[np.ndarray],
    frequency: str,
    pol: str,
    gunw_geogrid_group: str = "unwrappedInterferogram",
) -> Tuple[np.ndarray, "rasterio.Affine", "rasterio.crs.CRS"]:
    """
    Return (array, transform, crs) on the native GUNW geogrid for a radarGrid cube.
    Implemented by writing a temp GeoTIFF using interpolate_gunw_radargrid_cube_to_geotiff()
    and reading it back (simple + robust).
    """
    import tempfile

    import rasterio

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td) / "tmp_interp.tif"
        interpolate_gunw_radargrid_cube_to_geotiff(
            gunw_h5=gunw_h5,
            dem_path=dem_path,
            cube_ds_name=cube_ds_name,
            cube_data=cube_data,
            out_tif=tmp,
            frequency=frequency,
            pol=pol,
            gunw_geogrid_group=gunw_geogrid_group,
            overwrite=True,
            dst_nodata=np.nan,
        )
        with rasterio.open(tmp) as ds:
            arr = ds.read(1).astype(np.float32)
            return arr, ds.transform, ds.crs


def interpolate_incidence_and_local_incidence(
    gunw_h5: Union[str, Path],
    dem_path: Union[str, Path],
    *,
    frequency: str,
    pol: str,
    gunw_geogrid_group: str = "unwrappedInterferogram",
) -> Tuple[np.ndarray, np.ndarray, "rasterio.Affine", "rasterio.crs.CRS"]:
    """
    Return (incidenceAngle_2d, localIncidenceAngle_2d, transform, crs) on the native GUNW geogrid.
    """
    import tempfile

    import rasterio

    with tempfile.TemporaryDirectory() as td:
        inc_tif = Path(td) / "inc.tif"
        lia_tif = Path(td) / "lia.tif"
        interpolate_incidence_and_local_incidence_to_geotiff(
            gunw_h5=gunw_h5,
            dem_path=dem_path,
            out_inc_tif=inc_tif,
            out_local_inc_tif=lia_tif,
            frequency=frequency,
            pol=pol,
            gunw_geogrid_group=gunw_geogrid_group,
            overwrite=True,
            dst_nodata=np.nan,
        )
        with rasterio.open(inc_tif) as ds_inc:
            inc = ds_inc.read(1).astype(np.float32)
            transform = ds_inc.transform
            crs = ds_inc.crs
        with rasterio.open(lia_tif) as ds_lia:
            lia = ds_lia.read(1).astype(np.float32)
        return inc, lia, transform, crs
