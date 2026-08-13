"""
Streaming and local-cache utilities for NISAR cloud access.

The validated streaming path uses ``earthaccess`` to open NISAR HDF5 products
directly from NASA Earthdata Cloud. For repeated use, granules can also be
cached locally and reused by the repo's path-based processing utilities.
"""
from __future__ import annotations

import getpass
import hashlib
import logging
import netrc
import os
import subprocess
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Optional, Sequence, Union
from urllib.parse import urlparse

import numpy as np

if TYPE_CHECKING:
    import h5py

logger = logging.getLogger(__name__)

NISAR_S3_CREDENTIALS_ENDPOINT = (
    "https://nisar.asf.earthdatacloud.nasa.gov/s3credentials"
)
DEFAULT_NISAR_CACHE_DIR = Path.home() / ".cache" / "snowsar" / "nisar"
DEFAULT_NISAR_SUBSET_CACHE_DIR = DEFAULT_NISAR_CACHE_DIR / "subsets"
NISAR_PRODUCT_SHORT_NAMES = {
    "RSLC": "NISAR_L1_RSLC_BETA_V1",
    "RIFG": "NISAR_L1_RIFG_BETA_V1",
    "ROFF": "NISAR_L1_ROFF_BETA_V1",
    "RUNW": "NISAR_L1_RUNW_BETA_V1",
    "GSLC": "NISAR_L2_GSLC_BETA_V1",
    "GOFF": "NISAR_L2_GOFF_BETA_V1",
    "GCOV": "NISAR_L2_GCOV_BETA_V1",
    "GUNW": "NISAR_L2_GUNW_BETA_V1",
    "SME2": "NISAR_L3_SME2_BETA_V1",
}
ASF_FLIGHT_DIRECTIONS = {"ASCENDING", "DESCENDING"}
EARTHDATA_LOGIN_HOST = "urs.earthdata.nasa.gov"
NISAR_GUNW_ASF_COLLECTIONS = [
    "C2850261892-ASF",  # public NISAR GUNW
    "C4052499921-ASF",  # private ephemeral archive
]
_ASF_SEARCH_SESSION = None


def _netrc_path() -> Path:
    """Return the configured netrc path."""
    configured = os.getenv("NETRC")
    if configured:
        return Path(configured).expanduser()
    return Path.home() / ".netrc"


def _earthdata_credentials_from_netrc() -> Optional[tuple[str, str]]:
    """Return Earthdata username/password from netrc when present."""
    path = _netrc_path()
    if not path.exists():
        return None

    try:
        auth = netrc.netrc(str(path)).authenticators(EARTHDATA_LOGIN_HOST)
    except (netrc.NetrcParseError, OSError) as e:
        raise RuntimeError(f"Failed to read Earthdata credentials from {path}") from e

    if not auth:
        return None
    username, _, password = auth
    if not username or not password:
        return None
    return username, password


def _prompt_earthdata_credentials() -> tuple[str, str]:
    """Prompt interactively for Earthdata username/password."""
    username = input("Earthdata username: ")
    password = getpass.getpass("Earthdata password: ")
    return username, password


def _local_path_from_granule(granule: Any) -> Optional[Path]:
    """Return a local filesystem path when the input is path-like."""
    if isinstance(granule, Path):
        return granule.expanduser()
    if isinstance(granule, str):
        parsed = urlparse(granule)
        if parsed.scheme == "file":
            return Path(parsed.path).expanduser()
        if parsed.scheme == "":
            return Path(granule).expanduser()
    return None


def _earthaccess_authenticated(earthaccess_module) -> bool:
    """Return True when earthaccess already has an authenticated session."""
    try:
        earthaccess_module.auth_environ()
        return True
    except RuntimeError:
        return False


def setup_earthaccess_auth(
    *,
    persist: bool = True,
    force_reauth: bool = False,
) -> None:
    """
    Authenticate with NASA Earthdata Login for NISAR cloud access.

    Parameters
    ----------
    persist : bool, default True
        Save credentials for future sessions when earthaccess supports it.
    force_reauth : bool, default False
        Force re-authentication even if a prior session exists.
    """
    try:
        import earthaccess
    except ImportError as e:
        raise ImportError(
            "earthaccess is required for NISAR streaming.\n"
            "Install with: conda install -c conda-forge earthaccess"
        ) from e

    if force_reauth or not _earthaccess_authenticated(earthaccess):
        strategy = "netrc" if _earthdata_credentials_from_netrc() else "interactive"
        auth = earthaccess.login(strategy=strategy, persist=persist)
        if not getattr(auth, "authenticated", False):
            raise RuntimeError(
                "Earthdata authentication failed. "
                "Please check your credentials at https://urs.earthdata.nasa.gov/"
            )
        logger.info("Earthdata authentication successful")
    else:
        logger.info("Using existing Earthdata credentials")


def setup_asf_search_auth(*, force_reauth: bool = False):
    """
    Return an authenticated ASF Search session for downloads.

    Credentials are read from ``~/.netrc`` or the ``NETRC`` environment path
    when available. If no Earthdata Login entry exists, prompt interactively.
    """
    try:
        import asf_search as asf
    except ImportError as e:
        raise ImportError(
            "asf_search required. Install with: pip install asf-search"
        ) from e

    global _ASF_SEARCH_SESSION
    if _ASF_SEARCH_SESSION is not None and not force_reauth:
        return _ASF_SEARCH_SESSION

    credentials = _earthdata_credentials_from_netrc()
    if credentials is None:
        credentials = _prompt_earthdata_credentials()

    username, password = credentials
    _ASF_SEARCH_SESSION = asf.ASFSession().auth_with_creds(username, password)
    return _ASF_SEARCH_SESSION


def _is_earthaccess_granule(granule: Any) -> bool:
    """Return True for earthaccess-style DataGranule objects."""
    return hasattr(granule, "data_links") and callable(granule.data_links)


def _is_asf_result(granule: Any) -> bool:
    """Return True for ASF Search result objects."""
    return hasattr(granule, "get_urls") and callable(granule.get_urls)


def get_nisar_granule_urls(granule: Any) -> list[str]:
    """Extract candidate data URLs from a supported granule object."""
    if isinstance(granule, Path):
        return [str(granule)]
    if isinstance(granule, str):
        return [granule]
    if _is_earthaccess_granule(granule):
        return [str(url) for url in granule.data_links()]
    if _is_asf_result(granule):
        return [str(url) for url in granule.get_urls()]
    return []


def get_nisar_granule_name(granule: Any) -> str:
    """Return a stable display name for a NISAR granule or URL."""
    for url in get_nisar_granule_urls(granule):
        parsed = urlparse(url)
        candidate = Path(parsed.path).name if parsed.scheme else Path(url).name
        if candidate:
            return candidate

    if _is_earthaccess_granule(granule) or isinstance(granule, dict):
        umm = getattr(granule, "umm", None)
        if umm is None and isinstance(granule, dict):
            umm = granule.get("umm")
        if isinstance(umm, dict):
            granule_ur = umm.get("GranuleUR")
            if granule_ur:
                return str(granule_ur)

    if _is_asf_result(granule):
        properties = getattr(granule, "properties", None)
        if isinstance(properties, dict):
            filename = properties.get("fileName")
            if filename:
                return str(filename)
            scene_name = properties.get("sceneName")
            if scene_name:
                return str(scene_name)

    text = str(granule).strip()
    if not text:
        raise ValueError("Could not determine a filename for the requested granule.")
    return Path(text).name or text.replace("/", "_")


def _granule_urls(granule: Any) -> list[str]:
    """Extract candidate data URLs from a supported granule object."""
    return get_nisar_granule_urls(granule)


def _granule_basename(granule: Any) -> str:
    """Return a stable local filename for a granule or URL."""
    return get_nisar_granule_name(granule)


def _bbox_to_wkt(bbox: tuple[float, float, float, float]) -> str:
    """Convert a west/south/east/north bbox to a WKT polygon."""
    west, south, east, north = bbox
    return (
        f"POLYGON(({west} {south}, {east} {south}, {east} {north}, "
        f"{west} {north}, {west} {south}))"
    )


def _coordinate_slice(coords, lower: float, upper: float) -> slice:
    """Return a slice selecting sorted ascending or descending coordinates."""
    coords = np.asarray(coords)
    if coords.ndim != 1 or coords.size == 0:
        raise ValueError("Coordinate array must be one-dimensional and non-empty.")

    if coords[0] <= coords[-1]:
        start = int(np.searchsorted(coords, lower, side="left"))
        stop = int(np.searchsorted(coords, upper, side="right"))
    else:
        ascending = coords[::-1]
        start_asc = int(np.searchsorted(ascending, lower, side="left"))
        stop_asc = int(np.searchsorted(ascending, upper, side="right"))
        start = coords.size - stop_asc
        stop = coords.size - start_asc

    start = max(0, start)
    stop = min(coords.size, stop)
    if stop <= start:
        raise ValueError("Requested bbox does not overlap the dataset coordinates.")
    return slice(start, stop)


def _coordinate_step(coords) -> float:
    """Return the signed pixel spacing for a coordinate vector."""
    coords = np.asarray(coords)
    if coords.size > 1:
        return float(np.median(np.diff(coords)))
    return 1.0


def _coordinates_to_affine(xcoords, ycoords):
    """Return a rasterio affine transform from center coordinate vectors."""
    try:
        from affine import Affine
    except ImportError as e:
        raise ImportError(
            "affine is required to write cached NISAR subset GeoTIFFs.\n"
            "Install with: conda install -c conda-forge affine"
        ) from e

    xcoords = np.asarray(xcoords)
    ycoords = np.asarray(ycoords)
    if xcoords.size == 0 or ycoords.size == 0:
        raise ValueError("Cannot build GeoTIFF transform from empty coordinates.")

    x_step = _coordinate_step(xcoords)
    y_step = _coordinate_step(ycoords)
    return Affine.translation(
        float(xcoords[0]) - x_step / 2,
        float(ycoords[0]) - y_step / 2,
    ) * Affine.scale(x_step, y_step)


def _write_subset_geotiff(
    path: Path,
    data,
    xcoords,
    ycoords,
    epsg: int,
) -> None:
    """Write one geogridded subset as a single-band GeoTIFF."""
    try:
        import rasterio
        from rasterio.crs import CRS
    except ImportError as e:
        raise ImportError(
            "rasterio is required to cache NISAR subsets as GeoTIFFs.\n"
            "Install with: conda install -c conda-forge rasterio"
        ) from e

    data = np.asarray(data)
    if data.ndim != 2:
        raise ValueError("Only 2D NISAR subset arrays can be cached as GeoTIFFs.")

    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=data.shape[0],
        width=data.shape[1],
        count=1,
        dtype=data.dtype,
        crs=CRS.from_epsg(epsg),
        transform=_coordinates_to_affine(xcoords, ycoords),
        compress="deflate",
    ) as dst:
        dst.write(data, 1)


def _read_subset_geotiff(path: Path):
    """Read a cached single-band GeoTIFF subset."""
    try:
        import rasterio
        from rasterio.transform import xy
    except ImportError as e:
        raise ImportError(
            "rasterio is required to read cached NISAR subset GeoTIFFs.\n"
            "Install with: conda install -c conda-forge rasterio"
        ) from e

    with rasterio.open(path) as src:
        data = src.read(1)
        x, _ = xy(
            src.transform,
            np.zeros(src.width, dtype=int),
            np.arange(src.width),
            offset="center",
        )
        _, y = xy(
            src.transform,
            np.arange(src.height),
            np.zeros(src.height, dtype=int),
            offset="center",
        )
        epsg = src.crs.to_epsg() if src.crs else None

    if epsg is None:
        raise ValueError(f"Cached GeoTIFF is missing an EPSG CRS: {path}")
    return data, np.asarray(x), np.asarray(y), int(epsg)


def read_nisar_h5_bbox(
    h5: h5py.File,
    dataset_path: str,
    bbox: tuple[float, float, float, float],
):
    """
    Read a geogridded NISAR HDF5 dataset subset for a lon/lat bbox.

    Parameters
    ----------
    h5 : h5py.File
        Open NISAR HDF5 file handle.
    dataset_path : str
        Dataset path whose parent group contains ``xCoordinates``,
        ``yCoordinates``, and ``projection``.
    bbox : tuple
        ``(west, south, east, north)`` in EPSG:4326 lon/lat.

    Returns
    -------
    data : numpy.ndarray
        Dataset subset.
    x : numpy.ndarray
        Subset x coordinates in the product projection.
    y : numpy.ndarray
        Subset y coordinates in the product projection.
    epsg : int
        Product projection EPSG code.
    """
    ds = h5[dataset_path]
    grp = ds.parent
    if "xCoordinates" not in grp or "yCoordinates" not in grp:
        raise ValueError(f"Missing xCoordinates/yCoordinates near {dataset_path}")
    if "projection" not in grp:
        raise ValueError(f"Missing projection near {dataset_path}")

    xcoords = grp["xCoordinates"][()]
    ycoords = grp["yCoordinates"][()]
    epsg = int(grp["projection"][()].item())

    if epsg == 4326:
        west, south, east, north = bbox
    else:
        try:
            from rasterio.crs import CRS
            from rasterio.warp import transform_bounds
        except ImportError as e:
            raise ImportError(
                "rasterio is required to subset projected NISAR data by bbox.\n"
                "Install with: conda install -c conda-forge rasterio"
            ) from e
        west, south, east, north = transform_bounds(
            CRS.from_epsg(4326),
            CRS.from_epsg(epsg),
            *bbox,
            densify_pts=21,
        )
    x_slice = _coordinate_slice(xcoords, west, east)
    y_slice = _coordinate_slice(ycoords, south, north)

    data = ds[y_slice, x_slice]
    return data, xcoords[x_slice], ycoords[y_slice], epsg


def _normalize_earthaccess_target(granule: Any) -> Union[str, Any]:
    """
    Convert supported inputs into a value accepted by earthaccess.open/download.

    earthaccess natively accepts DataGranule objects, while ASF results are
    normalized to their first data URL.
    """
    if _is_earthaccess_granule(granule):
        return granule
    if _is_asf_result(granule):
        urls = granule.get_urls()
        if not urls:
            raise ValueError("ASF Search result has no URLs to open or download.")
        return str(urls[0])
    if isinstance(granule, (str, Path)):
        return str(granule)
    raise TypeError(
        "Unsupported granule input. Expected an earthaccess granule, "
        "ASF Search result, URL, or local path."
    )


def get_nisar_cache_path(
    granule: Any,
    *,
    cache_dir: Optional[Union[str, Path]] = None,
) -> Path:
    """
    Return the deterministic local cache path for a granule.

    Parameters
    ----------
    granule : object
        earthaccess granule, ASF Search result, URL, or local path
    cache_dir : str or Path, optional
        Cache root directory. Defaults to ``~/.cache/snowsar/nisar``.
    """
    root = Path(cache_dir) if cache_dir is not None else DEFAULT_NISAR_CACHE_DIR
    return root / _granule_basename(granule)


def get_nisar_subset_cache_path(
    granule: Any,
    dataset_path: str,
    bbox: tuple[float, float, float, float],
    *,
    cache_dir: Optional[Union[str, Path]] = None,
) -> Path:
    """
    Return the deterministic cache path for one bbox subset.

    The cached artifact is a GeoTIFF containing only the requested data window,
    not the source HDF5 granule.
    """
    root = (
        Path(cache_dir)
        if cache_dir is not None
        else DEFAULT_NISAR_SUBSET_CACHE_DIR
    )
    basename = Path(_granule_basename(granule)).stem
    layer_name = Path(dataset_path).name or "dataset"
    urls = "|".join(_granule_urls(granule))
    bbox_key = ",".join(f"{value:.10g}" for value in bbox)
    key = "\n".join([basename, urls, dataset_path, bbox_key])
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]
    return root / f"{basename}_{layer_name}_{digest}.tif"


def write_nisar_subset_geotiff(
    path: Union[str, Path],
    data,
    xcoords,
    ycoords,
    epsg: int,
    *,
    overwrite: bool = False,
) -> Path:
    """
    Write an already-read NISAR bbox subset as a GeoTIFF.
    """
    output_path = Path(path)
    if output_path.exists() and not overwrite:
        return output_path
    _write_subset_geotiff(output_path, data, xcoords, ycoords, epsg)
    return output_path


def read_nisar_h5_bbox_cached(
    granule: Any,
    dataset_path: str,
    bbox: tuple[float, float, float, float],
    *,
    cache_dir: Optional[Union[str, Path]] = None,
    provider: str = "earthaccess",
    use_cache: bool = True,
    overwrite: bool = False,
):
    """
    Read a bbox subset from a streamed NISAR HDF5 file.

    When ``use_cache`` is True, cache only the requested data window as a
    GeoTIFF. Returns ``(data, x, y, epsg, cache_path)``; ``cache_path`` is
    ``None`` when caching is disabled.
    """
    cache_path = get_nisar_subset_cache_path(
        granule,
        dataset_path,
        bbox,
        cache_dir=cache_dir,
    )
    if use_cache and cache_path.exists() and not overwrite:
        data, x, y, epsg = _read_subset_geotiff(cache_path)
        return data, x, y, epsg, cache_path

    with open_nisar_h5_stream(granule, provider=provider) as h5:
        data, x, y, epsg = read_nisar_h5_bbox(h5, dataset_path, bbox)

    if not use_cache:
        return data, x, y, epsg, None

    _write_subset_geotiff(cache_path, data, x, y, epsg)
    logger.info("Cached NISAR bbox subset GeoTIFF -> %s", cache_path)
    return data, x, y, epsg, cache_path


def _streamed_bbox_output_path(
    granule: Any,
    layer_label: str,
    bbox: tuple[float, float, float, float],
    *,
    out_dir: Union[str, Path],
) -> Path:
    """Return a deterministic output path for one streamed bbox layer."""
    out_root = Path(out_dir)
    basename = Path(get_nisar_granule_name(granule)).stem or "nisar_granule"
    bbox_key = ",".join(f"{value:.10g}" for value in bbox)
    digest = hashlib.sha256(
        "\n".join([basename, layer_label, bbox_key]).encode("utf-8")
    ).hexdigest()[:16]
    return out_root / f"{basename}_{layer_label}_{digest}.tif"


def _coordinate_slice_padded(
    coords, lower: float, upper: float, *, pad: int = 1
) -> slice:
    """Return a slice over sorted coordinates with a small interpolation pad."""
    coords = np.asarray(coords)
    if coords.ndim != 1 or coords.size == 0:
        raise ValueError("Coordinate array must be one-dimensional and non-empty.")

    if coords[0] <= coords[-1]:
        start = int(np.searchsorted(coords, lower, side="left"))
        stop = int(np.searchsorted(coords, upper, side="right"))
    else:
        ascending = coords[::-1]
        start_asc = int(np.searchsorted(ascending, lower, side="left"))
        stop_asc = int(np.searchsorted(ascending, upper, side="right"))
        start = coords.size - stop_asc
        stop = coords.size - start_asc

    start = max(0, start - pad)
    stop = min(coords.size, stop + pad)
    if stop <= start:
        raise ValueError("Requested bbox does not overlap the dataset coordinates.")
    return slice(start, stop)


def _bbox_dem_cache_path(
    bbox: tuple[float, float, float, float],
    *,
    cache_dir: Union[str, Path],
    buffer_deg: float,
    data_source: str,
) -> Path:
    """Return a deterministic DEM cache path for a lon/lat bbox."""
    root = Path(cache_dir)
    bbox_key = ",".join(f"{value:.10g}" for value in bbox)
    digest = hashlib.sha256(
        "\n".join([bbox_key, str(buffer_deg), data_source]).encode("utf-8")
    ).hexdigest()[:16]
    return root / f"dem_bbox_{digest}.tif"


def _download_dem_for_bbox_with_sardem(
    bbox: tuple[float, float, float, float],
    dem_out: Union[str, Path],
    *,
    buffer_deg: float,
    data_source: str = "COP",
    output_format: str = "GTiff",
    output_type: str = "float32",
    keep_egm: bool = False,
    overwrite: bool = False,
) -> Path:
    """Download a DEM covering one lon/lat bbox using sardem."""
    dem_out = Path(dem_out)
    dem_out.parent.mkdir(parents=True, exist_ok=True)
    if dem_out.exists() and not overwrite:
        return dem_out

    west, south, east, north = bbox
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
    if keep_egm:
        cmd += ["--keep-egm"]

    try:
        subprocess.run(cmd, check=True)
    except FileNotFoundError as e:
        raise RuntimeError(
            "Could not find `sardem` executable. Install `sardem` in this environment."
        ) from e
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"sardem failed with exit code {e.returncode}") from e

    if not dem_out.exists():
        raise RuntimeError(
            f"sardem reported success but output not found: {dem_out}"
        )
    return dem_out


def _read_valid_unwrapped_bbox(
    h5,
    bbox: tuple[float, float, float, float],
    *,
    frequency: str,
    pol: str,
):
    """Return the unwrapped subset plus validity mask for the requested bbox."""
    from snowsar.utils.nisar_utils import gunw_unwrapped_phase_path, resolve_h5_path

    raster_path = gunw_unwrapped_phase_path(frequency=frequency, pol=pol)
    resolved = resolve_h5_path(h5, raster_path)
    subset, x_out, y_out, epsg = read_nisar_h5_bbox(h5, resolved, bbox)
    fill = h5[resolved].attrs.get("_FillValue", None)

    valid = np.isfinite(subset)
    if fill is not None:
        valid &= subset != fill

    return subset, valid, x_out, y_out, int(epsg)


def _prepare_bbox_dem_on_grid(
    dem_path: Union[str, Path],
    *,
    x_out,
    y_out,
    out_epsg: int,
):
    """Warp a DEM onto the bbox output grid."""
    from snowsar.utils.nisar_utils import _read_raster_epsg, _warp_to_grid_mem

    dem_src_epsg = _read_raster_epsg(dem_path)
    return _warp_to_grid_mem(
        src_path=dem_path,
        src_epsg=dem_src_epsg,
        dst_epsg=out_epsg,
        xcoord=np.asarray(x_out),
        ycoord=np.asarray(y_out),
        resample_alg="bilinear",
    ).astype(np.float32, copy=False)


def _read_radargrid_axes_and_slices_for_bbox(h5, x_out, y_out):
    """Return radarGrid axis subsets covering the target bbox output grid."""
    from snowsar.utils.nisar_utils import h5_get

    cube_base = "/science/LSAR/GUNW/metadata/radarGrid"
    xrg = np.asarray(h5_get(h5, f"{cube_base}/xCoordinates")[()])
    yrg = np.asarray(h5_get(h5, f"{cube_base}/yCoordinates")[()])
    zrg = np.asarray(h5_get(h5, f"{cube_base}/heightAboveEllipsoid")[()])

    x_min = float(np.min(x_out))
    x_max = float(np.max(x_out))
    y_min = float(np.min(y_out))
    y_max = float(np.max(y_out))

    x_slice = _coordinate_slice_padded(xrg, x_min, x_max, pad=1)
    y_slice = _coordinate_slice_padded(yrg, y_min, y_max, pad=1)
    return xrg[x_slice], yrg[y_slice], zrg, x_slice, y_slice


def _interpolate_cube_array_to_bbox_grid(
    cube: np.ndarray,
    *,
    xrg,
    yrg,
    zrg,
    dem_on_grid,
    x_out,
    y_out,
    valid_mask,
    method: str = "linear",
) -> np.ndarray:
    """Interpolate one radarGrid cube onto the bbox output grid."""
    from snowsar.utils.nisar_utils import _make_rgi

    if cube.ndim != 3:
        raise ValueError(f"Expected 3D cube, got shape={cube.shape}")

    if cube.shape[0] == 2 and zrg.size >= 2:
        z_for_interp = np.array([zrg[0], zrg[-1]])
    else:
        z_for_interp = np.asarray(zrg)

    y2d, x2d = np.meshgrid(y_out, x_out, indexing="ij")
    out = np.full(dem_on_grid.shape, np.nan, dtype=np.float32)
    ii, jj = np.where(valid_mask)
    if ii.size == 0:
        return out

    pts = np.column_stack(
        [
            dem_on_grid[ii, jj].astype(np.float64),
            y2d[ii, jj].astype(np.float64),
            x2d[ii, jj].astype(np.float64),
        ]
    )
    interpolator = _make_rgi((z_for_interp, yrg, xrg), cube, method=method)
    out[ii, jj] = interpolator(pts).astype(np.float32)
    out[~valid_mask] = np.nan
    return out


def _interpolate_cube_layer_to_bbox_grid(
    h5,
    *,
    cube_ds_name: Optional[str],
    cube_data: Optional[np.ndarray],
    dem_on_grid,
    x_out,
    y_out,
    valid_mask,
    method: str = "linear",
) -> np.ndarray:
    """Interpolate one radarGrid layer or cube data onto the bbox output grid."""
    from snowsar.utils.nisar_utils import h5_get

    xrg, yrg, zrg, x_slice, y_slice = _read_radargrid_axes_and_slices_for_bbox(
        h5, x_out, y_out
    )

    if cube_data is None:
        if not cube_ds_name:
            raise ValueError("cube_ds_name or cube_data is required.")
        cube = np.asarray(
            h5_get(h5, f"/science/LSAR/GUNW/metadata/radarGrid/{cube_ds_name}")[
                :, y_slice, x_slice
            ]
        )
    else:
        cube = np.asarray(cube_data)

    return _interpolate_cube_array_to_bbox_grid(
        cube,
        xrg=xrg,
        yrg=yrg,
        zrg=zrg,
        dem_on_grid=dem_on_grid,
        x_out=x_out,
        y_out=y_out,
        valid_mask=valid_mask,
        method=method,
    )


def _interpolate_incidence_and_local_to_bbox_grid(
    h5,
    *,
    dem_on_grid,
    x_out,
    y_out,
    valid_mask,
    out_epsg: int,
    method: str = "linear",
):
    """Return interpolated incidence and local incidence on the bbox output grid."""
    from snowsar.utils.nisar_utils import _surface_normal_enu_from_dem, h5_get

    xrg, yrg, zrg, x_slice, y_slice = _read_radargrid_axes_and_slices_for_bbox(
        h5, x_out, y_out
    )
    cube_base = "/science/LSAR/GUNW/metadata/radarGrid"

    inc = np.asarray(h5_get(h5, f"{cube_base}/incidenceAngle")[:, y_slice, x_slice])
    los_e = np.asarray(h5_get(h5, f"{cube_base}/losUnitVectorX")[:, y_slice, x_slice])
    los_n = np.asarray(h5_get(h5, f"{cube_base}/losUnitVectorY")[:, y_slice, x_slice])

    try:
        los_u = np.asarray(
            h5_get(h5, f"{cube_base}/losUnitVectorZ")[:, y_slice, x_slice]
        )
    except KeyError:
        los_u = None

    inc_out = _interpolate_cube_array_to_bbox_grid(
        inc,
        xrg=xrg,
        yrg=yrg,
        zrg=zrg,
        dem_on_grid=dem_on_grid,
        x_out=x_out,
        y_out=y_out,
        valid_mask=valid_mask,
        method=method,
    )
    le = _interpolate_cube_array_to_bbox_grid(
        los_e,
        xrg=xrg,
        yrg=yrg,
        zrg=zrg,
        dem_on_grid=dem_on_grid,
        x_out=x_out,
        y_out=y_out,
        valid_mask=valid_mask,
        method=method,
    )
    ln = _interpolate_cube_array_to_bbox_grid(
        los_n,
        xrg=xrg,
        yrg=yrg,
        zrg=zrg,
        dem_on_grid=dem_on_grid,
        x_out=x_out,
        y_out=y_out,
        valid_mask=valid_mask,
        method=method,
    )
    if los_u is not None:
        lu = _interpolate_cube_array_to_bbox_grid(
            los_u,
            xrg=xrg,
            yrg=yrg,
            zrg=zrg,
            dem_on_grid=dem_on_grid,
            x_out=x_out,
            y_out=y_out,
            valid_mask=valid_mask,
            method=method,
        )
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
        dem_on_grid, xcoord=np.asarray(x_out), ycoord=np.asarray(y_out), epsg=out_epsg
    )
    dot = le * n_e + ln * n_n + lu * n_u
    dot = np.clip(np.abs(dot), 0.0, 1.0)
    local_inc = np.degrees(np.arccos(dot)).astype(np.float32)
    local_inc[~valid_mask] = np.nan
    inc_out[~valid_mask] = np.nan
    return inc_out.astype(np.float32), local_inc.astype(np.float32)


def _is_transient_stream_error(exc: Exception) -> bool:
    """Return True for retriable remote streaming failures."""
    status = getattr(exc, "status", None)
    if isinstance(status, int) and status in {500, 502, 503, 504}:
        return True

    text = str(exc).lower()
    markers = (
        "bad gateway",
        "gateway timeout",
        "service unavailable",
        "clientresponseerror",
        "connection reset",
        "temporarily unavailable",
        "timeout",
        "remote disconnected",
    )
    return any(marker in text for marker in markers)


def extract_gunw_layers_to_geotiff_bbox_streamed(
    granule: Any,
    bbox: tuple[float, float, float, float],
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
    provider: str = "earthaccess",
    overwrite: bool = False,
    dem_cache_dir: Optional[Union[str, Path]] = None,
    dem_buffer_deg: float = 0.02,
    cube_interp_method: str = "linear",
    dem_data_source: str = "COP",
    max_retries: int = 2,
    retry_delay: float = 1.0,
) -> Dict[str, Path]:
    """
    Stream and extract selected NISAR GUNW layers for one bbox without caching the full granule.

    Direct 2D geogrid layers are read only over the requested bbox. RadarGrid cube
    layers and computed aliases are interpolated onto the bbox geogrid using a DEM.
    """
    from snowsar.utils.nisar_utils import (
        build_dataset_index,
        is_geogrid_2d,
        is_radargrid_cube,
        pick_best_candidate,
        resolve_h5_path,
    )

    requested_names = [layers] if isinstance(layers, str) else list(layers)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    computed_aliases = {"totalTroposphere", "localIncidenceAngle"}
    outputs: Dict[str, Path] = {}

    if max_retries < 0:
        raise ValueError("max_retries must be >= 0")
    if retry_delay < 0:
        raise ValueError("retry_delay must be >= 0")

    for attempt in range(max_retries + 1):
        outputs = {}
        try:
            with open_nisar_h5_stream(granule, provider=provider) as h5:
                index = build_dataset_index(h5)
                _, valid_mask, x_out, y_out, out_epsg = _read_valid_unwrapped_bbox(
                    h5, bbox, frequency=frequency, pol=pol
                )
                if x_out.size == 0 or y_out.size == 0:
                    raise ValueError("Requested bbox produced an empty output grid.")

                selected_infos = {}
                missing = []
                for name in requested_names:
                    if name in index:
                        selected_infos[name] = pick_best_candidate(
                            index[name],
                            frequency=frequency,
                            pol=pol,
                            prefer_geogrid=True,
                        )
                    elif name not in computed_aliases:
                        missing.append(name)
                if missing:
                    raise ValueError(
                        f"Layer(s) not found in streamed granule: {missing}"
                    )

                need_dem = any(
                    name
                    in {"incidenceAngle", "localIncidenceAngle", "totalTroposphere"}
                    or (
                        name in selected_infos
                        and is_radargrid_cube(h5, selected_infos[name])
                        and not is_geogrid_2d(h5, selected_infos[name])
                    )
                    for name in requested_names
                )

                dem_on_grid = None
                if need_dem:
                    dem_root = (
                        Path(dem_cache_dir)
                        if dem_cache_dir is not None
                        else out_dir / "dem_cache"
                    )
                    dem_path = _bbox_dem_cache_path(
                        bbox,
                        cache_dir=dem_root,
                        buffer_deg=dem_buffer_deg,
                        data_source=dem_data_source,
                    )
                    _download_dem_for_bbox_with_sardem(
                        bbox,
                        dem_path,
                        buffer_deg=dem_buffer_deg,
                        data_source=dem_data_source,
                        overwrite=overwrite,
                    )
                    dem_on_grid = _prepare_bbox_dem_on_grid(
                        dem_path, x_out=x_out, y_out=y_out, out_epsg=out_epsg
                    )

                # Direct 2D geogrid layers.
                for name, info in selected_infos.items():
                    if not is_geogrid_2d(h5, info):
                        continue
                    resolved = resolve_h5_path(h5, info.path)
                    data, x_layer, y_layer, epsg = read_nisar_h5_bbox(
                        h5, resolved, bbox
                    )
                    fill = h5[resolved].attrs.get("_FillValue", None)
                    data = data.astype(np.float32, copy=False)
                    if fill is not None:
                        data = np.where(data == float(fill), np.nan, data)
                    if name == "connectedComponents":
                        data = data.astype(np.float32, copy=False)
                        data[~valid_mask] = np.nan
                    out_path = _streamed_bbox_output_path(
                        granule, name, bbox, out_dir=out_dir
                    )
                    write_nisar_subset_geotiff(
                        out_path,
                        data,
                        x_layer,
                        y_layer,
                        int(epsg),
                        overwrite=overwrite,
                    )
                    outputs[name] = out_path

                # Derived incidence/local incidence.
                need_local = "localIncidenceAngle" in requested_names
                need_inc_interp = (
                    "incidenceAngle" in requested_names
                    and "incidenceAngle" not in outputs
                )
                if need_local or need_inc_interp:
                    if dem_on_grid is None:
                        raise ValueError(
                            "DEM preparation failed for incidence interpolation."
                        )
                    inc_arr, local_arr = _interpolate_incidence_and_local_to_bbox_grid(
                        h5,
                        dem_on_grid=dem_on_grid,
                        x_out=x_out,
                        y_out=y_out,
                        valid_mask=valid_mask,
                        out_epsg=out_epsg,
                        method=cube_interp_method,
                    )
                    if need_local:
                        label = "localIncidenceAngle_interp"
                        out_path = _streamed_bbox_output_path(
                            granule, label, bbox, out_dir=out_dir
                        )
                        write_nisar_subset_geotiff(
                            out_path,
                            local_arr,
                            x_out,
                            y_out,
                            out_epsg,
                            overwrite=overwrite,
                        )
                        outputs[label] = out_path
                    if need_inc_interp:
                        label = "incidenceAngle_interp"
                        out_path = _streamed_bbox_output_path(
                            granule, label, bbox, out_dir=out_dir
                        )
                        write_nisar_subset_geotiff(
                            out_path,
                            inc_arr,
                            x_out,
                            y_out,
                            out_epsg,
                            overwrite=overwrite,
                        )
                        outputs[label] = out_path

                # Cube-derived layers.
                for name in requested_names:
                    if (
                        name in outputs
                        or name in {"incidenceAngle", "localIncidenceAngle"}
                    ):
                        continue
                    if name == "totalTroposphere":
                        if dem_on_grid is None:
                            raise ValueError(
                                "DEM preparation failed for cube interpolation."
                            )
                        from snowsar.utils.nisar_utils import h5_get

                        xrg, yrg, zrg, x_slice, y_slice = (
                            _read_radargrid_axes_and_slices_for_bbox(
                                h5, x_out, y_out
                            )
                        )
                        cube_base = "/science/LSAR/GUNW/metadata/radarGrid"
                        hydro = np.asarray(
                            h5_get(
                                h5,
                                f"{cube_base}/hydrostaticTroposphericPhaseScreen",
                            )[:, y_slice, x_slice]
                        )
                        wet = np.asarray(
                            h5_get(h5, f"{cube_base}/wetTroposphericPhaseScreen")[
                                :, y_slice, x_slice
                            ]
                        )
                        arr = _interpolate_cube_array_to_bbox_grid(
                            hydro + wet,
                            xrg=xrg,
                            yrg=yrg,
                            zrg=zrg,
                            dem_on_grid=dem_on_grid,
                            x_out=x_out,
                            y_out=y_out,
                            valid_mask=valid_mask,
                            method=cube_interp_method,
                        )
                        label = "totalTroposphere_interp"
                        out_path = _streamed_bbox_output_path(
                            granule, label, bbox, out_dir=out_dir
                        )
                        write_nisar_subset_geotiff(
                            out_path,
                            arr,
                            x_out,
                            y_out,
                            out_epsg,
                            overwrite=overwrite,
                        )
                        outputs[label] = out_path
                        continue

                    info = selected_infos.get(name)
                    if info is None or not is_radargrid_cube(h5, info):
                        continue
                    if dem_on_grid is None:
                        raise ValueError(
                            "DEM preparation failed for cube interpolation."
                        )

                    arr = _interpolate_cube_layer_to_bbox_grid(
                        h5,
                        cube_ds_name=name,
                        cube_data=None,
                        dem_on_grid=dem_on_grid,
                        x_out=x_out,
                        y_out=y_out,
                        valid_mask=valid_mask,
                        method=cube_interp_method,
                    )
                    label = f"{name}_interp"
                    out_path = _streamed_bbox_output_path(
                        granule, label, bbox, out_dir=out_dir
                    )
                    write_nisar_subset_geotiff(
                        out_path,
                        arr,
                        x_out,
                        y_out,
                        out_epsg,
                        overwrite=overwrite,
                    )
                    outputs[label] = out_path

            return outputs
        except Exception as exc:
            if attempt >= max_retries or not _is_transient_stream_error(exc):
                raise
            delay = retry_delay * (2**attempt)
            logger.warning(
                "Transient streamed read failure for %s (attempt %s/%s): %s. Retrying in %.1fs",
                get_nisar_granule_name(granule),
                attempt + 1,
                max_retries + 1,
                exc,
                delay,
            )
            if delay > 0:
                time.sleep(delay)

    return outputs


def open_nisar_h5_stream(
    granule: Any,
    *,
    provider: str = "earthaccess",
    mode: str = "r",
) -> h5py.File:
    """
    Stream a NISAR HDF5 file directly from Earthdata Cloud.

    Parameters
    ----------
    granule : object
        earthaccess DataGranule, ASF Search result, HTTPS URL, or local path
    provider : str, default "earthaccess"
        Streaming provider. Only ``earthaccess`` is currently validated.
    mode : str, default "r"
        File open mode
    """
    local_path = _local_path_from_granule(granule)
    if local_path is not None:
        return _open_local_h5(local_path, mode=mode)

    if provider != "earthaccess":
        raise NotImplementedError(
            "Only provider='earthaccess' is currently supported for validated "
            "NISAR streaming."
        )
    return _open_with_earthaccess(granule, mode=mode)


def _open_local_h5(
    path: Union[str, Path],
    *,
    mode: str = "r",
) -> h5py.File:
    """Open a cached local HDF5 file without using Earthdata auth."""
    try:
        import h5py
    except ImportError as e:
        raise ImportError(
            "h5py is required to open local NISAR HDF5 files.\n"
            "Install with: conda install -c conda-forge h5py"
        ) from e

    local_path = Path(path).expanduser()
    if not local_path.exists():
        raise FileNotFoundError(f"Local NISAR file not found: {local_path}")

    return h5py.File(local_path, mode=mode)


def _open_with_earthaccess(
    granule: Any,
    *,
    mode: str = "r",
) -> h5py.File:
    """Open NISAR HDF5 via earthaccess."""
    try:
        import earthaccess
        import h5py
    except ImportError as e:
        raise ImportError(
            "earthaccess and h5py are required for NISAR streaming.\n"
            "Install with: conda install -c conda-forge earthaccess h5py"
        ) from e

    class _StreamBackedH5File(h5py.File):
        """h5py.File that also closes its backing earthaccess stream on close()."""

        def __init__(self, stream: Any, **kwargs: Any) -> None:
            super().__init__(stream, **kwargs)
            self._nisar_stream = stream

        def close(self) -> None:
            try:
                super().close()
            finally:
                self._nisar_stream.close()

    if not _earthaccess_authenticated(earthaccess):
        setup_earthaccess_auth()

    target = _normalize_earthaccess_target(granule)
    try:
        opened = earthaccess.open(
            [target],
            credentials_endpoint=NISAR_S3_CREDENTIALS_ENDPOINT,
            show_progress=False,
        )
        if not opened:
            raise ValueError("earthaccess.open() returned no file handles.")
        stream = opened[0]
        try:
            return _StreamBackedH5File(stream, mode=mode)
        except Exception:
            stream.close()
            raise
    except Exception as e:
        raise RuntimeError(
            "Failed to stream the requested NISAR granule via earthaccess. "
            "Verify authentication and that the granule belongs to the NISAR "
            "Earthdata Cloud archive."
        ) from e


def cache_nisar_granule(
    granule: Any,
    *,
    cache_dir: Optional[Union[str, Path]] = None,
    provider: str = "earthaccess",
    overwrite: bool = False,
) -> Path:
    """
    Download one granule into a deterministic local cache and return its path.

    This is useful when a streamed file becomes part of a repeated workflow and
    should be reused by path-based utilities in this repo.
    """
    if provider not in {"earthaccess", "asf_search"}:
        raise ValueError(
            f"Unknown provider: {provider}. Choose 'earthaccess' or 'asf_search'."
        )

    # Handle local .h5 files that are already on disk
    source_path = _local_path_from_granule(granule)
    if source_path is not None and source_path.exists() and source_path.suffix == ".h5":
        target_path = get_nisar_cache_path(granule, cache_dir=cache_dir)
        target_path.parent.mkdir(parents=True, exist_ok=True)

        # If source and target are the same, just return it
        if source_path.resolve() == target_path.resolve():
            logger.info("Reusing cached NISAR granule -> %s", target_path)
            return target_path

        # If target exists and overwrite is False, reuse it
        if target_path.exists() and not overwrite:
            logger.info("Reusing cached NISAR granule -> %s", target_path)
            return target_path

        # Copy or move the local file into cache
        import shutil

        shutil.copy2(source_path, target_path)
        logger.info("Cached local NISAR granule -> %s", target_path)
        return target_path

    target_path = get_nisar_cache_path(granule, cache_dir=cache_dir)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    if target_path.exists() and not overwrite:
        logger.info("Reusing cached NISAR granule -> %s", target_path)
        return target_path

    downloaded = download_with_progress(
        [granule],
        output_dir=target_path.parent,
        provider=provider,
        overwrite=overwrite,
    )
    if not downloaded:
        raise RuntimeError("Download completed without returning a local file path.")

    downloaded_path = Path(downloaded[0])
    logger.info("Cached NISAR granule -> %s", downloaded_path)
    return downloaded_path


def search_nisar_data(
    *,
    bbox: Optional[tuple[float, float, float, float]] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    processing_level: str = "GUNW",
    provider: str = "earthaccess",
    flight_direction: Optional[str] = None,
    max_results: int = 100,
):
    """
    Search for NISAR products using earthaccess or ASF Search.
    """
    if provider == "earthaccess":
        if flight_direction is not None:
            raise ValueError(
                "flight_direction is not supported when provider='earthaccess'. "
                "Use provider='asf_search' to filter by flight direction."
            )
        return _search_with_earthaccess(
            bbox=bbox,
            start_date=start_date,
            end_date=end_date,
            processing_level=processing_level,
            max_results=max_results,
        )
    if provider == "asf_search":
        return _search_with_asf_search(
            bbox=bbox,
            start_date=start_date,
            end_date=end_date,
            processing_level=processing_level,
            flight_direction=flight_direction,
            max_results=max_results,
        )
    raise ValueError(
        f"Unknown provider: {provider}. Choose 'earthaccess' or 'asf_search'."
    )


def _search_with_earthaccess(
    *,
    bbox,
    start_date,
    end_date,
    processing_level,
    max_results,
):
    """Search NISAR data via earthaccess."""
    try:
        import earthaccess
    except ImportError as e:
        raise ImportError(
            "earthaccess required. Install with: conda install -c conda-forge earthaccess"
        ) from e

    short_name = NISAR_PRODUCT_SHORT_NAMES.get(processing_level.upper())
    if not short_name:
        logger.warning(
            "Unknown processing_level '%s'; using the supplied value directly.",
            processing_level,
        )
        short_name = processing_level

    results = earthaccess.search_data(
        short_name=short_name,
        bounding_box=bbox,
        temporal=(start_date, end_date),
        count=max_results,
    )

    logger.info("Found %s NISAR granules via earthaccess", len(results))
    return results


def _search_with_asf_search(
    *,
    bbox,
    start_date,
    end_date,
    processing_level,
    flight_direction,
    max_results,
):
    """Search NISAR data via asf_search."""
    try:
        import asf_search as asf
    except ImportError as e:
        raise ImportError(
            "asf_search required. Install with: pip install asf-search"
        ) from e

    search_params = {
        "dataset": "NISAR",
        "processingLevel": processing_level.upper(),
        "maxResults": max_results,
    }
    if processing_level.upper() == "GUNW":
        search_params["collections"] = NISAR_GUNW_ASF_COLLECTIONS
    if start_date:
        search_params["start"] = start_date
    if end_date:
        search_params["end"] = end_date
    if bbox:
        search_params["intersectsWith"] = _bbox_to_wkt(bbox)
    if flight_direction:
        flight_direction = flight_direction.upper()
        if flight_direction not in ASF_FLIGHT_DIRECTIONS:
            raise ValueError(
                "flight_direction must be 'ASCENDING' or 'DESCENDING'."
            )
        search_params["flightDirection"] = flight_direction

    search_params["session"] = setup_asf_search_auth()
    results = asf.search(opts=asf.ASFSearchOptions(**search_params))
    logger.info("Found %s NISAR granules via asf_search", len(results))
    return results


def download_with_progress(
    granule_results,
    *,
    output_dir: Union[str, Path],
    provider: str = "earthaccess",
    overwrite: bool = False,
) -> list[Path]:
    """
    Download NISAR granules for offline or repeated local processing.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if provider == "earthaccess":
        try:
            import earthaccess
        except ImportError as e:
            raise ImportError(
                "earthaccess required. Install with: conda install -c conda-forge earthaccess"
            ) from e

        if not _earthaccess_authenticated(earthaccess):
            setup_earthaccess_auth()

        normalized = [
            _normalize_earthaccess_target(granule) for granule in granule_results
        ]
        downloaded = earthaccess.download(
            normalized,
            str(output_dir),
            credentials_endpoint=NISAR_S3_CREDENTIALS_ENDPOINT,
            show_progress=False,
            force=overwrite,
        )
        return [Path(p) for p in downloaded]

    if provider == "asf_search":
        try:
            import asf_search as asf
        except ImportError as e:
            raise ImportError(
                "asf_search required. Install with: pip install asf-search"
            ) from e

        session = setup_asf_search_auth()
        downloaded: list[Path] = []
        for result in granule_results:
            filename = _granule_basename(result)
            target_file = output_dir / filename
            if target_file.exists() and not overwrite:
                downloaded.append(target_file)
                continue
            result.download(path=str(output_dir), session=session)
            downloaded.append(target_file)
        return downloaded

    raise ValueError(
        f"Unknown provider: {provider}. Choose 'earthaccess' or 'asf_search'."
    )
