from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING, List, Sequence, Tuple, Union

if TYPE_CHECKING:
    import pandas as pd

# Find any YYYYMMDD tokens (HyP3 names typically include 2 dates)
_DATE_RE = re.compile(r"(\d{8})")

# HyP3 MintPy-relevant GeoTIFF suffixes
HYP3_MINTPY_SUFFIXES = (
    "_water_mask.tif",
    "_corr.tif",
    "_unw_phase.tif",
    "_dem.tif",
    "_lv_theta.tif",
    "_lv_phi.tif",
    "_inc_map.tif",
)


def _parse_yyyymmdd(token: str):
    """Return a normalized pandas timestamp for a YYYYMMDD token."""
    try:
        import pandas as pd
    except Exception as e:
        raise ImportError(
            "pandas is required to parse HyP3 acquisition dates."
        ) from e

    return pd.to_datetime(token, format="%Y%m%d").normalize()


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
                dates.add(_parse_yyyymmdd(token))
            except (ValueError, OverflowError):
                # Skip invalid date tokens (e.g., "99999999" or malformed dates)
                continue
    return sorted(dates)


def parse_date_pairs_from_hyp3_filenames(
    paths: Sequence[Union[str, Path]],
) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
    """
    Parse sorted unique reference/secondary date pairs from HyP3 filenames.

    The first two valid YYYYMMDD tokens in each filename are treated as the
    interferogram pair. Filenames without at least two valid date tokens are
    skipped.
    """
    pairs = set()
    for p in paths:
        parsed = []
        for token in _DATE_RE.findall(Path(p).name):
            try:
                parsed.append(_parse_yyyymmdd(token))
            except (ValueError, OverflowError):
                continue
            if len(parsed) == 2:
                break
        if len(parsed) == 2:
            pairs.add(tuple(parsed))
    return sorted(pairs)


def common_geotiff_overlap(
    tif_paths: Sequence[Union[str, Path]],
) -> Tuple[float, float, float, float]:
    """
    Return common overlap bounds across multiple GeoTIFFs as (left, top, right, bottom).

    Computes the intersection of all raster bounds. All rasters must share the same CRS.
    Returns bounds in GDAL projWin order: (ulx, uly, lrx, lry).

    Parameters
    ----------
    tif_paths : Sequence[str or Path]
        Paths to GeoTIFF files

    Returns
    -------
    Tuple[float, float, float, float]
        (left, top, right, bottom) bounds of common overlap in source CRS

    Raises
    ------
    ValueError
        If paths is empty, rasters have no CRS, CRS mismatch, or no overlap exists

    Examples
    --------
    >>> tif_files = [
    ...     "S1_20201215_20201227_VV_unw_phase.tif",
    ...     "S1_20201227_20210108_VV_unw_phase.tif"
    ... ]
    >>> bounds = common_geotiff_overlap(tif_files)
    >>> left, top, right, bottom = bounds
    """
    try:
        import rasterio
    except Exception as e:
        raise ImportError(
            "rasterio is required to compute GeoTIFF overlap."
        ) from e

    tif_paths = [Path(p) for p in tif_paths]
    if not tif_paths:
        raise ValueError("tif_paths cannot be empty")

    first_crs = None
    first_file = None
    left_vals, right_vals, bottom_vals, top_vals = [], [], [], []

    for idx, tif in enumerate(tif_paths):
        with rasterio.open(tif) as src:
            if src.crs is None:
                raise ValueError(f"Raster has no CRS: {tif}")

            if idx == 0:
                first_crs = src.crs
                first_file = tif
            elif src.crs != first_crs:
                raise ValueError(
                    f"CRS mismatch:\n"
                    f"  {first_file.name}: {first_crs}\n"
                    f"  {tif.name}: {src.crs}"
                )

            bounds = src.bounds
            left_vals.append(bounds.left)
            right_vals.append(bounds.right)
            bottom_vals.append(bounds.bottom)
            top_vals.append(bounds.top)

    left = max(left_vals)
    right = min(right_vals)
    bottom = max(bottom_vals)
    top = min(top_vals)

    if left >= right or bottom >= top:
        raise ValueError(
            f"No overlap exists across rasters:\n"
            f"  computed bounds: left={left}, top={top}, right={right}, bottom={bottom}"
        )

    return (left, top, right, bottom)


def clip_geotiff_to_bounds(
    src_path: Union[str, Path],
    dst_path: Union[str, Path],
    bounds: Tuple[float, float, float, float],
    *,
    overwrite: bool = False,
) -> Path:
    """
    Clip one GeoTIFF to (left, top, right, bottom) bounds in its native CRS.

    Parameters
    ----------
    src_path : str or Path
        Path to input GeoTIFF
    dst_path : str or Path
        Path for output clipped GeoTIFF
    bounds : Tuple[float, float, float, float]
        (left, top, right, bottom) in source raster CRS
    overwrite : bool, default False
        If True, overwrite existing output file. If False, skip if output exists.

    Returns
    -------
    Path
        Path to created clipped GeoTIFF

    Raises
    ------
    ValueError
        If bounds don't overlap raster or output exists and overwrite=False

    Examples
    --------
    >>> bounds = (123.4, 45.6, 123.5, 45.5)  # left, top, right, bottom
    >>> clip_geotiff_to_bounds("input.tif", "output.tif", bounds)
    PosixPath('output.tif')
    """
    try:
        import rasterio
        from rasterio.windows import from_bounds
    except Exception as e:
        raise ImportError(
            "rasterio is required to clip GeoTIFF files."
        ) from e

    src_path = Path(src_path)
    dst_path = Path(dst_path)

    if dst_path.exists() and not overwrite:
        raise ValueError(f"Output file exists and overwrite=False: {dst_path}")

    left, top, right, bottom = bounds

    with rasterio.open(src_path) as src:
        window = from_bounds(left, bottom, right, top, src.transform)

        # Floor window bounds to match MintPy indexing
        import numpy as np
        col_off = max(0, int(np.floor(window.col_off)))
        row_off = max(0, int(np.floor(window.row_off)))
        col_end = min(src.width, int(np.floor(window.col_off + window.width)))
        row_end = min(src.height, int(np.floor(window.row_off + window.height)))

        from rasterio.windows import Window
        window = Window(
            col_off=col_off,
            row_off=row_off,
            width=col_end - col_off,
            height=row_end - row_off,
        )

        if window.width <= 0 or window.height <= 0:
            raise ValueError(
                f"Bounds {bounds} don't overlap {src_path}"
            )

        data = src.read(window=window)
        transform = src.window_transform(window)
        profile = src.profile.copy()
        profile.update(
            height=window.height,
            width=window.width,
            transform=transform,
        )

        dst_path.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(dst_path, "w", **profile) as dst:
            dst.write(data)

    return dst_path


def clip_hyp3_products_to_common_overlap(
    data_dir: Union[str, Path],
    *,
    dem_pattern: str = "*/*_dem.tif",
    suffixes: Sequence[str] = HYP3_MINTPY_SUFFIXES,
    overwrite: bool = False,
) -> List[Path]:
    """
    Clip HyP3 MintPy input products to their common DEM overlap.

    Finds all DEM files matching dem_pattern, computes common overlap bounds,
    then clips all files matching suffixes to that extent. Output files are
    written as siblings with '_clipped' suffix added to stem.

    Parameters
    ----------
    data_dir : str or Path
        Directory containing HyP3 products (typically organized as subdirs per pair)
    dem_pattern : str, default "*/*_dem.tif"
        Glob pattern to find DEM files for overlap computation
    suffixes : Sequence[str], default HYP3_MINTPY_SUFFIXES
        File suffixes to clip. Each matching file will be clipped to common bounds.
    overwrite : bool, default False
        If True, overwrite existing clipped files. If False, skip existing outputs.

    Returns
    -------
    List[Path]
        Paths to all clipped output files (existing skipped files are included)

    Raises
    ------
    ValueError
        If no DEM files found or no overlap exists

    Examples
    --------
    >>> clipped = clip_hyp3_products_to_common_overlap(
    ...     "hyp3_data",
    ...     dem_pattern="*/*_dem.tif"
    ... )
    >>> len(clipped)
    42
    """
    data_dir = Path(data_dir)

    # Find DEM files and compute common overlap
    dem_files = sorted(data_dir.glob(dem_pattern))
    if not dem_files:
        raise ValueError(f"No DEM files found with pattern '{dem_pattern}' in {data_dir}")

    overlap = common_geotiff_overlap(dem_files)

    # Clip all matching suffixes
    output_paths = []
    for suffix in suffixes:
        for file in data_dir.rglob(f"*{suffix}"):
            dst_file = file.parent / f"{file.stem}_clipped{file.suffix}"

            if dst_file.exists() and not overwrite:
                output_paths.append(dst_file)
                continue

            try:
                clipped = clip_geotiff_to_bounds(file, dst_file, overlap, overwrite=overwrite)
                output_paths.append(clipped)
            except ValueError as e:
                # Skip files that don't overlap (e.g., different subswaths)
                import warnings
                warnings.warn(f"Skipping {file.name}: {e}", UserWarning)
                continue

    return output_paths


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
    try:
        import geopandas as gpd
        import rasterio
        from rasterio.features import shapes
        from shapely.geometry import Polygon, shape
        from shapely.ops import unary_union
    except Exception as e:
        raise ImportError(
            "geopandas, rasterio, and shapely are required to build HyP3 footprints."
        ) from e

    if band < 1:
        raise ValueError(f"band must be >= 1, got {band}")
    tif_paths = [Path(p) for p in tif_paths]
    if len(tif_paths) == 0:
        return gpd.GeoDataFrame(geometry=[], crs=out_crs)

    list_parts = []
    src_crs = None
    first_crs_file = None

    for idx, tif in enumerate(tif_paths):
        with rasterio.open(tif) as src:
            # Validate CRS exists
            if src.crs is None:
                if out_crs is not None:
                    raise ValueError(
                        f"Raster has no CRS but out_crs={out_crs} was requested.\n"
                        f"  File: {tif}\n"
                        f"  Cannot reproject from undefined CRS."
                    )
                # If out_crs is None, we can proceed but warn
                import warnings
                warnings.warn(
                    f"Raster has no CRS defined: {tif}\n"
                    f"  Geometries will have undefined CRS.",
                    UserWarning
                )

            # Check CRS consistency across files
            if idx == 0:
                src_crs = src.crs
                first_crs_file = tif
            elif src.crs != src_crs:
                raise ValueError(
                    f"Mixed CRS detected in input GeoTIFFs:\n"
                    f"  First file CRS: {src_crs} ({first_crs_file.name})\n"
                    f"  Current file CRS: {src.crs} ({tif.name})\n"
                    f"  All input rasters must have the same CRS before union."
                )

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
