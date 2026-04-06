from __future__ import annotations

import glob
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import h5py
import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.windows import Window, from_bounds
from rasterio.transform import Affine
from rasterio.warp import reproject


_DATE_PATTERN = re.compile(r"(?P<year>\d{4})(?P<month>[A-Za-z]{3})(?P<day>\d{1,2})(?:-(?P<end_day>\d{1,2}))?")


@dataclass(frozen=True)
class MintPyGrid:
    length: int
    width: int
    x_first: float
    x_step: float
    y_first: float
    y_step: float
    transform: Affine
    crs: CRS
    attrs: Dict[str, Any]


def list_hdf5_root_datasets(h5_path: str | Path) -> List[str]:
    """List root-level datasets in an HDF5 file."""
    h5_path = Path(h5_path)
    with h5py.File(h5_path, "r") as h5:
        return sorted([key for key in h5.keys() if isinstance(h5[key], h5py.Dataset)])


def read_hdf5_root_attributes(h5_path: str | Path) -> Dict[str, Any]:
    """Read root-level HDF5 attributes with bytes decoded to strings."""
    h5_path = Path(h5_path)
    with h5py.File(h5_path, "r") as h5:
        return {key: _decode_attr(value) for key, value in h5.attrs.items()}


def _decode_attr(value: Any) -> Any:
    if isinstance(value, (bytes, bytearray)):
        return value.decode()
    return value


def get_mintpy_grid(mintpy_timeseries_path: str | Path) -> MintPyGrid:
    """Read a regular MintPy geocoded grid definition from attributes or lat/lon datasets."""
    mintpy_timeseries_path = Path(mintpy_timeseries_path)

    with h5py.File(mintpy_timeseries_path, "r") as h5:
        attrs = {key: _decode_attr(value) for key, value in h5.attrs.items()}

        if all(
            key in attrs for key in ("X_FIRST", "X_STEP", "Y_FIRST", "Y_STEP", "LENGTH", "WIDTH")
        ):
            x_first = float(attrs["X_FIRST"])
            x_step = float(attrs["X_STEP"])
            y_first = float(attrs["Y_FIRST"])
            y_step = float(attrs["Y_STEP"])
            length = int(attrs["LENGTH"])
            width = int(attrs["WIDTH"])
        elif "latitude" in h5 and "longitude" in h5:
            lat = h5["latitude"][()]
            lon = h5["longitude"][()]

            if lat.ndim == 1 and lon.ndim == 1:
                lat = np.repeat(lat[:, None], lon.size, axis=1)
                lon = np.repeat(lon[None, :], lat.shape[0], axis=0)

            if lat.ndim != 2 or lon.ndim != 2 or lat.shape != lon.shape:
                raise ValueError("MintPy latitude/longitude datasets must define a shared 2D grid.")

            length, width = lat.shape
            x_first = float(lon[0, 0])
            y_first = float(lat[0, 0])
            x_step = float(np.nanmedian(np.diff(lon[0, :])))
            y_step = float(np.nanmedian(np.diff(lat[:, 0])))
            tol = 1e-9
            lon_regular = np.allclose(np.diff(lon, axis=1), x_step, atol=tol, equal_nan=True)
            lat_regular = np.allclose(np.diff(lat, axis=0), y_step, atol=tol, equal_nan=True)
            if not (lon_regular and lat_regular):
                raise ValueError("MintPy latitude/longitude grid is not regular enough for affine output.")
        else:
            raise ValueError(
                "MintPy file must provide X/Y grid attributes or latitude/longitude datasets."
            )

    transform = Affine(x_step, 0.0, x_first, 0.0, y_step, y_first)
    return MintPyGrid(
        length=length,
        width=width,
        x_first=x_first,
        x_step=x_step,
        y_first=y_first,
        y_step=y_step,
        transform=transform,
        crs=CRS.from_epsg(4326),
        attrs=attrs,
    )


def get_geocoded_hdf5_grid(h5_path: str | Path) -> MintPyGrid:
    """Read a regular geocoded HDF5 grid definition from root attributes."""
    h5_path = Path(h5_path)
    with h5py.File(h5_path, "r") as h5:
        attrs = {key: _decode_attr(value) for key, value in h5.attrs.items()}
    required = {"X_FIRST", "X_STEP", "Y_FIRST", "Y_STEP", "LENGTH", "WIDTH"}
    if not required.issubset(attrs):
        missing = sorted(required - set(attrs))
        raise ValueError(f"{h5_path} is missing geocoded grid attributes: {missing}")
    x_first = float(attrs["X_FIRST"])
    x_step = float(attrs["X_STEP"])
    y_first = float(attrs["Y_FIRST"])
    y_step = float(attrs["Y_STEP"])
    length = int(attrs["LENGTH"])
    width = int(attrs["WIDTH"])
    transform = Affine(x_step, 0.0, x_first, 0.0, y_step, y_first)
    return MintPyGrid(
        length=length,
        width=width,
        x_first=x_first,
        x_step=x_step,
        y_first=y_first,
        y_step=y_step,
        transform=transform,
        crs=CRS.from_epsg(4326),
        attrs=attrs,
    )


def write_mintpy_array_as_geotiff(
    input_array: np.ndarray,
    mintpy_timeseries_path: str | Path,
    out_tif_path: str | Path,
    *,
    nodata: float = np.nan,
    compress: str = "deflate",
    predictor: int = 3,
    bigtiff: str = "IF_SAFER",
    creation_overviews: bool = False,
    overview_levels: Sequence[int] = (2, 4, 8, 16),
    description: str = "Layer on MintPy geocoded grid",
    extra_tags: Dict[str, Any] | None = None,
) -> Path:
    """Write an array aligned to a MintPy grid as a geocoded GeoTIFF."""
    if input_array.ndim != 2:
        raise ValueError("input_array must be 2D (LENGTH x WIDTH).")

    grid = get_mintpy_grid(mintpy_timeseries_path)
    if input_array.shape != (grid.length, grid.width):
        raise ValueError(
            f"input_array shape {input_array.shape} does not match MintPy grid {(grid.length, grid.width)}."
        )

    out_tif_path = Path(out_tif_path)
    dtype = np.float32 if np.issubdtype(input_array.dtype, np.floating) else input_array.dtype
    profile = {
        "driver": "GTiff",
        "height": grid.length,
        "width": grid.width,
        "count": 1,
        "dtype": dtype,
        "crs": grid.crs,
        "transform": grid.transform,
        "tiled": True,
        "blockxsize": 256,
        "blockysize": 256,
        "compress": compress,
        "predictor": predictor,
        "bigtiff": bigtiff,
        "nodata": nodata,
    }

    with rasterio.open(out_tif_path, "w", **profile) as dst:
        dst.write(input_array.astype(dtype, copy=False), 1)
        dst.set_band_description(1, description)

        tags = {
            "LENGTH": str(grid.length),
            "WIDTH": str(grid.width),
            "X_FIRST": str(grid.x_first),
            "X_STEP": str(grid.x_step),
            "Y_FIRST": str(grid.y_first),
            "Y_STEP": str(grid.y_step),
            "Description": description,
            "Source_MintPy_File": str(mintpy_timeseries_path),
        }
        if extra_tags:
            tags.update({str(key): str(value) for key, value in extra_tags.items()})
        for key in ("UTM_ZONE", "REF_YEAR", "REF_DATE", "CENTER_LAT", "CENTER_LON"):
            if key in grid.attrs:
                tags[f"MintPy_{key}"] = str(grid.attrs[key])
        dst.update_tags(**tags)

        if creation_overviews:
            dst.build_overviews(list(overview_levels), Resampling.average)
            dst.update_tags(ns="rio_overview", resampling="average")

    return out_tif_path


def resample_geotiff_to_mintpy_grid(
    geotiff_path: str | Path,
    mintpy_timeseries_path: str | Path,
    *,
    output_path: str | Path | None = None,
    write_output: bool = False,
    resampling: Resampling = Resampling.bilinear,
    output_description: str = "LIDAR resampled to MintPy geocoded grid",
) -> np.ndarray | Path:
    """Resample a raster onto a MintPy geocoded grid with nodata-aware reprojection."""
    geotiff_path = Path(geotiff_path)
    grid = get_mintpy_grid(mintpy_timeseries_path)
    destination = np.full((grid.length, grid.width), np.nan, dtype=np.float32)

    with rasterio.open(geotiff_path) as src:
        if src.crs is None:
            raise ValueError(f"{geotiff_path} has no CRS defined.")

        src_array = src.read(1).astype(np.float32, copy=False)
        src_nodata = src.nodata
        if src_nodata is not None:
            src_array = np.where(src_array == src_nodata, np.nan, src_array)

        reproject(
            source=src_array,
            destination=destination,
            src_transform=src.transform,
            src_crs=src.crs,
            src_nodata=np.nan,
            dst_transform=grid.transform,
            dst_crs=grid.crs,
            dst_nodata=np.nan,
            resampling=resampling,
        )

    if not write_output:
        return destination

    if output_path is None:
        output_path = geotiff_path.with_name(f"resampled_{geotiff_path.name}")

    return write_mintpy_array_as_geotiff(
        destination,
        mintpy_timeseries_path,
        output_path,
        creation_overviews=True,
        description=output_description,
    )


def extract_start_date_str(path: str | Path) -> str:
    """Extract a YYYYMMDD start date from names like 2023Apr09 or 2023May11-12."""
    match = _DATE_PATTERN.search(str(path))
    if not match:
        raise ValueError(f"No date token found in: {path}")

    start_token = (
        f"{match.group('year')}{match.group('month').title()}{int(match.group('day')):02d}"
    )
    return datetime.strptime(start_token, "%Y%b%d").strftime("%Y%m%d")


def read_geotiff_stack_sorted_by_date(
    pattern: str,
    *,
    dtype: str = "float32",
    strict_same_grid: bool = True,
) -> Tuple[List[str], np.ndarray, Dict[str, Any]]:
    """Read matching GeoTIFFs into a date-sorted stack."""
    paths = sorted(glob.glob(pattern, recursive=True))
    if not paths:
        raise FileNotFoundError(f"No files matched pattern: {pattern}")

    items = sorted((extract_start_date_str(path), path) for path in paths)
    dates = [date for date, _ in items]
    sorted_paths = [path for _, path in items]

    with rasterio.open(sorted_paths[0]) as first_ds:
        height, width = first_ds.height, first_ds.width
        transform = first_ds.transform
        crs = first_ds.crs
        profile = first_ds.profile.copy()

    data = np.full((len(sorted_paths), height, width), np.nan, dtype=np.dtype(dtype))
    for idx, path in enumerate(sorted_paths):
        with rasterio.open(path) as ds:
            if ds.height != height or ds.width != width:
                raise ValueError(f"Shape mismatch: {path} has {(ds.height, ds.width)} != {(height, width)}")
            if strict_same_grid:
                if ds.crs != crs:
                    raise ValueError(f"CRS mismatch: {path} has {ds.crs} != {crs}")
                if not np.allclose(tuple(ds.transform), tuple(transform)):
                    raise ValueError(f"Transform mismatch for {path}")

            band = ds.read(1, masked=True)
            data[idx] = np.where(np.ma.getmaskarray(band), np.nan, band.astype(dtype, copy=False))

    metadata = {
        "profile": profile,
        "transform": transform,
        "crs": crs,
        "paths": [str(path) for path in sorted_paths],
    }
    return dates, data, metadata


def build_lidar_timeseries_h5(
    resampled_pattern: str,
    mintpy_timeseries_path: str | Path,
    output_path: str | Path,
) -> Path:
    """Create a MintPy-like HDF5 timeseries from resampled LIDAR GeoTIFFs."""
    from mintpy.utils import utils as ut

    dates, stack, _ = read_geotiff_stack_sorted_by_date(resampled_pattern)
    metadata = ut.readfile.read_attribute(str(mintpy_timeseries_path))
    for key in ("REF_DATE", "REF_LAT", "REF_LON", "REF_X", "REF_Y"):
        metadata.pop(key, None)

    num_dates = len(dates)
    ds_name_dict = {
        "date": [np.dtype("S8"), (num_dates,), np.array(dates, np.bytes_)],
        "bperp": [np.float32, (num_dates,), np.zeros(num_dates, dtype=np.float32)],
        "timeseries": [np.float32, stack.shape, stack.astype(np.float32, copy=False)],
    }
    output_path = Path(output_path)
    ut.writefile.layout_hdf5(str(output_path), ds_name_dict, metadata=metadata)
    return output_path


def filter_finite_pairs(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return flattened finite-value pairs from two same-shaped arrays."""
    if x.shape != y.shape:
        raise ValueError(f"Array shapes must match, got {x.shape} and {y.shape}.")
    mask = np.isfinite(x) & np.isfinite(y)
    return np.asarray(x[mask]).ravel(), np.asarray(y[mask]).ravel()


def read_geometry_datasets(
    geometry_h5_path: str | Path,
    dataset_names: Sequence[str],
) -> Dict[str, np.ndarray]:
    """Read selected root-level datasets from a MintPy geometry HDF5 file."""
    geometry_h5_path = Path(geometry_h5_path)
    arrays: Dict[str, np.ndarray] = {}
    with h5py.File(geometry_h5_path, "r") as h5:
        for name in dataset_names:
            if name not in h5:
                raise KeyError(f"Dataset '{name}' not found in {geometry_h5_path}.")
            arrays[name] = np.asarray(h5[name][()])
    return arrays


def subset_radar_geometry_h5(
    geometry_h5_path: str | Path,
    output_path: str | Path,
    *,
    lat_range: Tuple[float, float],
    lon_range: Tuple[float, float],
) -> Path:
    """
    Subset a radar-coordinate MintPy geometry file using its latitude/longitude datasets.

    The output remains radar-coordinate but is cropped to the bounding rows/columns
    enclosing the requested geographic box.
    """
    geometry_h5_path = Path(geometry_h5_path)
    output_path = Path(output_path)
    arrays = read_geometry_datasets(geometry_h5_path, ["latitude", "longitude"])
    lat = arrays["latitude"]
    lon = arrays["longitude"]

    lat_min, lat_max = sorted((float(lat_range[0]), float(lat_range[1])))
    lon_min, lon_max = sorted((float(lon_range[0]), float(lon_range[1])))
    inside = (
        np.isfinite(lat)
        & np.isfinite(lon)
        & (lat >= lat_min)
        & (lat <= lat_max)
        & (lon >= lon_min)
        & (lon <= lon_max)
    )
    if not np.any(inside):
        raise ValueError(
            f"No geometryRadar.h5 pixels found inside lat={lat_range}, lon={lon_range}."
        )

    rows, cols = np.where(inside)
    row_min, row_max = int(rows.min()), int(rows.max())
    col_min, col_max = int(cols.min()), int(cols.max())

    attrs = read_hdf5_root_attributes(geometry_h5_path)
    datasets = list_hdf5_root_datasets(geometry_h5_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(geometry_h5_path, "r") as src, h5py.File(output_path, "w") as dst:
        for key, value in attrs.items():
            dst.attrs[key] = value
        subset_length = row_max - row_min + 1
        subset_width = col_max - col_min + 1
        for key in ("LENGTH", "length"):
            if key in dst.attrs:
                dst.attrs[key] = subset_length
        for key in ("WIDTH", "width", "XMAX", "xmax"):
            if key in dst.attrs:
                dst.attrs[key] = subset_width
        for key in ("YMAX", "ymax"):
            if key in dst.attrs:
                dst.attrs[key] = subset_length
        for key in ("SUBSET_ROW_MIN", "SUBSET_ROW_MAX", "SUBSET_COL_MIN", "SUBSET_COL_MAX"):
            if key in dst.attrs:
                del dst.attrs[key]
        dst.attrs["SUBSET_ROW_MIN"] = row_min
        dst.attrs["SUBSET_ROW_MAX"] = row_max
        dst.attrs["SUBSET_COL_MIN"] = col_min
        dst.attrs["SUBSET_COL_MAX"] = col_max

        for name in datasets:
            data = src[name][()]
            if data.ndim >= 2 and data.shape[-2:] == lat.shape:
                subset = data[..., row_min : row_max + 1, col_min : col_max + 1]
            else:
                subset = data
            kwargs: Dict[str, Any] = {}
            if isinstance(src[name], h5py.Dataset) and src[name].compression is not None:
                kwargs["compression"] = src[name].compression
            dst.create_dataset(name, data=subset, **kwargs)

    return output_path


def los_unit_vector_from_inc_azimuth(
    incidence_deg: np.ndarray,
    azimuth_deg: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build the MintPy LOS unit vector in local ENU.

    Assumes MintPy geometry definition:
    - incidenceAngle is measured from vertical at the target.
    - azimuthAngle is measured from north with anti-clockwise positive.
    """
    theta = np.deg2rad(incidence_deg)
    psi = np.deg2rad(azimuth_deg)
    s_east = -np.sin(theta) * np.sin(psi)
    s_north = np.sin(theta) * np.cos(psi)
    s_up = -np.cos(theta)
    norm = np.sqrt(s_east**2 + s_north**2 + s_up**2)
    norm = np.where(norm == 0, 1.0, norm)
    return s_east / norm, s_north / norm, s_up / norm


def llh_to_ecef(
    lon_deg: np.ndarray,
    lat_deg: np.ndarray,
    height_m: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert lon/lat/height to ECEF on WGS84."""
    wgs84_a = 6378137.0
    wgs84_e2 = 6.69437999014e-3
    lon = np.deg2rad(lon_deg)
    lat = np.deg2rad(lat_deg)
    sin_lat, cos_lat = np.sin(lat), np.cos(lat)
    sin_lon, cos_lon = np.sin(lon), np.cos(lon)
    radius = wgs84_a / np.sqrt(1.0 - wgs84_e2 * sin_lat * sin_lat)
    x = (radius + height_m) * cos_lat * cos_lon
    y = (radius + height_m) * cos_lat * sin_lon
    z = (radius * (1.0 - wgs84_e2) + height_m) * sin_lat
    return x, y, z


def ecef_to_enu(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    lon0_deg: np.ndarray,
    lat0_deg: np.ndarray,
    h0_m: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert ECEF coordinates to local ENU relative to per-pixel reference points."""
    x0, y0, z0 = llh_to_ecef(lon0_deg, lat0_deg, h0_m)
    dx = x - x0
    dy = y - y0
    dz = z - z0

    lon0 = np.deg2rad(lon0_deg)
    lat0 = np.deg2rad(lat0_deg)
    sin_lat, cos_lat = np.sin(lat0), np.cos(lat0)
    sin_lon, cos_lon = np.sin(lon0), np.cos(lon0)

    east = -sin_lon * dx + cos_lon * dy
    north = -sin_lat * cos_lon * dx - sin_lat * sin_lon * dy + cos_lat * dz
    up = cos_lat * cos_lon * dx + cos_lat * sin_lon * dy + sin_lat * dz
    return east, north, up


def surface_normal_from_geometry(
    longitude_deg: np.ndarray,
    latitude_deg: np.ndarray,
    height_m: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Estimate unit surface normals from central differences on a geocoded MintPy grid."""
    lon_xp = np.roll(longitude_deg, -1, axis=1)
    lat_xp = np.roll(latitude_deg, -1, axis=1)
    h_xp = np.roll(height_m, -1, axis=1)
    lon_xm = np.roll(longitude_deg, 1, axis=1)
    lat_xm = np.roll(latitude_deg, 1, axis=1)
    h_xm = np.roll(height_m, 1, axis=1)
    lon_yp = np.roll(longitude_deg, -1, axis=0)
    lat_yp = np.roll(latitude_deg, -1, axis=0)
    h_yp = np.roll(height_m, -1, axis=0)
    lon_ym = np.roll(longitude_deg, 1, axis=0)
    lat_ym = np.roll(latitude_deg, 1, axis=0)
    h_ym = np.roll(height_m, 1, axis=0)

    x_xp, y_xp, z_xp = llh_to_ecef(lon_xp, lat_xp, h_xp)
    x_xm, y_xm, z_xm = llh_to_ecef(lon_xm, lat_xm, h_xm)
    x_yp, y_yp, z_yp = llh_to_ecef(lon_yp, lat_yp, h_yp)
    x_ym, y_ym, z_ym = llh_to_ecef(lon_ym, lat_ym, h_ym)

    ex_p, nx_p, ux_p = ecef_to_enu(x_xp, y_xp, z_xp, longitude_deg, latitude_deg, height_m)
    ex_m, nx_m, ux_m = ecef_to_enu(x_xm, y_xm, z_xm, longitude_deg, latitude_deg, height_m)
    ey_p, ny_p, uy_p = ecef_to_enu(x_yp, y_yp, z_yp, longitude_deg, latitude_deg, height_m)
    ey_m, ny_m, uy_m = ecef_to_enu(x_ym, y_ym, z_ym, longitude_deg, latitude_deg, height_m)

    tx_e = ex_p - ex_m
    tx_n = nx_p - nx_m
    tx_u = ux_p - ux_m
    ty_e = ey_p - ey_m
    ty_n = ny_p - ny_m
    ty_u = uy_p - uy_m

    normal_e = tx_n * ty_u - tx_u * ty_n
    normal_n = tx_u * ty_e - tx_e * ty_u
    normal_u = tx_e * ty_n - tx_n * ty_e

    norm = np.sqrt(normal_e**2 + normal_n**2 + normal_u**2)
    norm = np.where(norm == 0, np.nan, norm)
    normal_e = normal_e / norm
    normal_n = normal_n / norm
    normal_u = normal_u / norm

    flip = normal_u < 0
    normal_e[flip] *= -1
    normal_n[flip] *= -1
    normal_u[flip] *= -1

    for array in (normal_e, normal_n, normal_u):
        array[:, 0] = np.nan
        array[:, -1] = np.nan
        array[0, :] = np.nan
        array[-1, :] = np.nan

    return normal_e, normal_n, normal_u


def local_incidence_from_geometry(
    geometry_h5_path: str | Path,
) -> Dict[str, np.ndarray]:
    """Compute terrain-aware local incidence from MintPy geometry datasets."""
    arrays = read_geometry_datasets(
        geometry_h5_path,
        ["longitude", "latitude", "height", "incidenceAngle", "azimuthAngle"],
    )
    los_e, los_n, los_u = los_unit_vector_from_inc_azimuth(
        arrays["incidenceAngle"],
        arrays["azimuthAngle"],
    )
    normal_e, normal_n, normal_u = surface_normal_from_geometry(
        arrays["longitude"],
        arrays["latitude"],
        arrays["height"],
    )
    cos_local_incidence = -(normal_e * los_e + normal_n * los_n + normal_u * los_u)
    cos_local_incidence = np.clip(cos_local_incidence, -1.0, 1.0)
    local_incidence_deg = np.degrees(np.arccos(cos_local_incidence)).astype(np.float32)
    return {
        "local_incidence_deg": local_incidence_deg,
        "cos_local_incidence": cos_local_incidence.astype(np.float32),
        "surface_normal_east": normal_e.astype(np.float32),
        "surface_normal_north": normal_n.astype(np.float32),
        "surface_normal_up": normal_u.astype(np.float32),
        "los_east": los_e.astype(np.float32),
        "los_north": los_n.astype(np.float32),
        "los_up": los_u.astype(np.float32),
    }


def project_scalar_field_to_los(
    scalar_field: np.ndarray,
    geometry_h5_path: str | Path,
    *,
    quantity_direction: str = "surface_normal",
    mask_shadow: bool = True,
    allow_flat_fallback: bool = True,
    target_raster_path: str | Path | None = None,
) -> Dict[str, Any]:
    """
    Project a scalar field on the MintPy geogrid into LOS using geometryGeo.h5.

    `quantity_direction` controls the assumed direction of the scalar quantity:
    - `surface_normal`: scalar acts along the local terrain normal.
    - `vertical`: scalar acts along local up.
    """
    geometry_h5_path = Path(geometry_h5_path)
    scalar_field = np.asarray(scalar_field, dtype=np.float32)
    available = set(list_hdf5_root_datasets(geometry_h5_path))
    required_for_terrain = {"longitude", "latitude", "height", "incidenceAngle", "azimuthAngle"}

    if quantity_direction not in {"surface_normal", "vertical"}:
        raise ValueError("quantity_direction must be 'surface_normal' or 'vertical'.")

    projection_mode = "flat"
    reason = ""
    if quantity_direction == "surface_normal" and required_for_terrain.issubset(available):
        geometry = local_incidence_from_geometry(geometry_h5_path)
        factor = geometry["cos_local_incidence"]
        projection_mode = "terrain"
    else:
        if "incidenceAngle" not in available:
            raise KeyError(
                f"{geometry_h5_path} does not contain incidenceAngle required for LOS projection."
            )
        if quantity_direction == "surface_normal" and not allow_flat_fallback:
            missing = sorted(required_for_terrain - available)
            raise KeyError(
                f"{geometry_h5_path} is missing datasets for terrain-aware projection: {missing}"
            )
        arrays = read_geometry_datasets(geometry_h5_path, ["incidenceAngle"])
        factor = np.cos(np.deg2rad(arrays["incidenceAngle"])).astype(np.float32)
        if quantity_direction == "surface_normal":
            reason = "Terrain-aware projection unavailable; fell back to flat incidence."
        else:
            reason = "Used flat incidence for vertical-to-LOS projection."

    if scalar_field.shape != factor.shape:
        if target_raster_path is None:
            raise ValueError(
                f"scalar_field shape {scalar_field.shape} does not match geometry shape {factor.shape}."
            )
        geometry_grid = get_geocoded_hdf5_grid(geometry_h5_path)
        with rasterio.open(target_raster_path) as target_ds:
            destination_factor = np.full((target_ds.height, target_ds.width), np.nan, dtype=np.float32)
            reproject(
                source=factor.astype(np.float32, copy=False),
                destination=destination_factor,
                src_transform=geometry_grid.transform,
                src_crs=geometry_grid.crs,
                src_nodata=np.nan,
                dst_transform=target_ds.transform,
                dst_crs=target_ds.crs,
                dst_nodata=np.nan,
                resampling=Resampling.bilinear,
            )
            factor = destination_factor

            if mask_shadow and "shadowMask" in available:
                shadow_mask = read_geometry_datasets(geometry_h5_path, ["shadowMask"])["shadowMask"].astype(np.float32)
                destination_shadow = np.zeros((target_ds.height, target_ds.width), dtype=np.float32)
                reproject(
                    source=shadow_mask,
                    destination=destination_shadow,
                    src_transform=geometry_grid.transform,
                    src_crs=geometry_grid.crs,
                    src_nodata=np.nan,
                    dst_transform=target_ds.transform,
                    dst_crs=target_ds.crs,
                    dst_nodata=np.nan,
                    resampling=Resampling.nearest,
                )
                shadow_mask = destination_shadow >= 0.5
            else:
                shadow_mask = None
        if scalar_field.shape != factor.shape:
            raise ValueError(
                f"scalar_field shape {scalar_field.shape} does not match resampled geometry shape {factor.shape}."
            )
    else:
        shadow_mask = None

    projected = scalar_field * factor
    if mask_shadow and "shadowMask" in available:
        if shadow_mask is None:
            shadow_mask = read_geometry_datasets(geometry_h5_path, ["shadowMask"])["shadowMask"].astype(bool)
        projected = np.where(shadow_mask, np.nan, projected)

    return {
        "projected_los": projected.astype(np.float32),
        "projection_factor": factor.astype(np.float32),
        "projection_mode": projection_mode,
        "quantity_direction": quantity_direction,
        "reason": reason,
    }


def resample_geometry_dataset_to_raster(
    geometry_h5_path: str | Path,
    dataset_name: str,
    target_raster_path: str | Path,
    *,
    resampling: Resampling = Resampling.bilinear,
) -> np.ndarray:
    """Resample a geocoded geometry dataset onto a target raster grid."""
    geometry_h5_path = Path(geometry_h5_path)
    target_raster_path = Path(target_raster_path)
    source = read_geometry_datasets(geometry_h5_path, [dataset_name])[dataset_name].astype(np.float32)
    geometry_grid = get_geocoded_hdf5_grid(geometry_h5_path)

    with rasterio.open(target_raster_path) as target_ds:
        destination = np.full((target_ds.height, target_ds.width), np.nan, dtype=np.float32)
        reproject(
            source=source,
            destination=destination,
            src_transform=geometry_grid.transform,
            src_crs=geometry_grid.crs,
            src_nodata=np.nan,
            dst_transform=target_ds.transform,
            dst_crs=target_ds.crs,
            dst_nodata=np.nan,
            resampling=resampling,
        )
    return destination


def compute_pearson_correlation(
    x: np.ndarray,
    y: np.ndarray,
    *,
    min_points: int = 3,
    on_invalid: str = "raise",
) -> Dict[str, Any]:
    """Compute a guarded Pearson correlation on finite paired samples."""
    from scipy.stats import pearsonr

    x_valid, y_valid = filter_finite_pairs(x, y)
    if x_valid.size < min_points:
        message = f"Need at least {min_points} valid paired samples, found {x_valid.size}."
        if on_invalid == "nan":
            return {
                "count": int(x_valid.size),
                "statistic": np.nan,
                "pvalue": np.nan,
                "valid": False,
                "reason": message,
            }
        raise ValueError(message)
    if np.allclose(x_valid, x_valid[0]) or np.allclose(y_valid, y_valid[0]):
        message = "Pearson correlation is undefined for constant input arrays."
        if on_invalid == "nan":
            return {
                "count": int(x_valid.size),
                "statistic": np.nan,
                "pvalue": np.nan,
                "valid": False,
                "reason": message,
            }
        raise ValueError(message)
    if on_invalid not in {"raise", "nan"}:
        raise ValueError("on_invalid must be either 'raise' or 'nan'.")

    result = pearsonr(x_valid, y_valid)
    return {
        "count": int(x_valid.size),
        "statistic": float(result.statistic),
        "pvalue": float(result.pvalue),
        "valid": True,
        "reason": "",
    }


def cumulative_sum_through_date(
    stack: np.ndarray,
    dates: Sequence[str],
    inclusive_end_date: str,
) -> np.ndarray:
    """Sum a stack through an inclusive YYYYMMDD cutoff."""
    if stack.ndim != 3:
        raise ValueError("stack must have shape (time, length, width).")
    if stack.shape[0] != len(dates):
        raise ValueError("dates length must match the first dimension of stack.")

    selected_indices = [idx for idx, date in enumerate(dates) if date <= inclusive_end_date]
    if not selected_indices:
        raise ValueError(f"No slices found on or before {inclusive_end_date}.")
    return np.nansum(stack[selected_indices, :, :], axis=0)


def mintpy_slice_names(timeseries_path: str | Path) -> List[str]:
    """Return MintPy dataset slice names."""
    from mintpy.utils import readfile

    return list(readfile.get_slice_list(str(timeseries_path)))


def mintpy_date_slice_names(timeseries_path: str | Path) -> List[str]:
    """Return MintPy slice names that contain a YYYYMMDD token."""
    slice_names = mintpy_slice_names(timeseries_path)
    return [name for name in slice_names if re.search(r"\d{8}", name)]


def mintpy_date_from_slice_name(slice_name: str) -> str:
    """Extract the final YYYYMMDD token from a MintPy slice name."""
    matches = re.findall(r"(\d{8})", slice_name)
    if not matches:
        raise ValueError(f"No YYYYMMDD token found in slice name: {slice_name}")
    return matches[-1]


def resample_many_geotiffs(
    geotiff_paths: Iterable[str | Path],
    mintpy_timeseries_path: str | Path,
    *,
    output_dir: str | Path | None = None,
    overwrite: bool = False,
) -> List[Path]:
    """Resample multiple rasters and return the output paths."""
    output_dir_path = Path(output_dir) if output_dir is not None else None
    written_paths: List[Path] = []
    for geotiff_path in geotiff_paths:
        geotiff_path = Path(geotiff_path)
        output_path = (
            output_dir_path / f"resampled_{geotiff_path.name}"
            if output_dir_path is not None
            else geotiff_path.with_name(f"resampled_{geotiff_path.name}")
        )
        if output_path.exists() and not overwrite:
            written_paths.append(output_path)
            continue
        output_path.parent.mkdir(parents=True, exist_ok=True)
        written_paths.append(
            resample_geotiff_to_mintpy_grid(
                geotiff_path,
                mintpy_timeseries_path,
                output_path=output_path,
                write_output=True,
            )
        )
    return written_paths


def subset_geotiff_by_bbox(
    geotiff_path: str | Path,
    output_path: str | Path,
    *,
    lat_range: Tuple[float, float],
    lon_range: Tuple[float, float],
) -> Path:
    """Crop a geocoded GeoTIFF to a lon/lat bounding box and write a valid GeoTIFF."""
    geotiff_path = Path(geotiff_path)
    output_path = Path(output_path)
    south, north = sorted((float(lat_range[0]), float(lat_range[1])))
    west, east = sorted((float(lon_range[0]), float(lon_range[1])))

    with rasterio.open(geotiff_path) as src:
        window = from_bounds(west, south, east, north, src.transform)
        # Match MintPy subset indexing by flooring both window bounds instead of
        # rounding lengths, which can drop edge pixels on one side.
        col_off = max(0, int(np.floor(window.col_off)))
        row_off = max(0, int(np.floor(window.row_off)))
        col_end = min(src.width, int(np.floor(window.col_off + window.width)))
        row_end = min(src.height, int(np.floor(window.row_off + window.height)))
        window = Window(
            col_off=col_off,
            row_off=row_off,
            width=col_end - col_off,
            height=row_end - row_off,
        )
        if window.width <= 0 or window.height <= 0:
            raise ValueError(
                f"Bounding box lon={lon_range}, lat={lat_range} does not overlap {geotiff_path}."
            )

        data = src.read(window=window)
        transform = src.window_transform(window)
        profile = src.profile.copy()
        profile.update(
            height=data.shape[1],
            width=data.shape[2],
            transform=transform,
        )

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(output_path, "w", **profile) as dst:
            dst.write(data)

    return output_path
