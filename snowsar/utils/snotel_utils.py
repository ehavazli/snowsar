from __future__ import annotations

import logging
from typing import Dict

import geopandas as gpd
import pandas as pd
from shapely.geometry import Point
from shapely.ops import transform as shp_transform

logger = logging.getLogger(__name__)

# Constants
_INCHES_TO_CM = 2.54


def f_to_c(temp_f: float) -> float:
    """
    Convert temperature from Fahrenheit to Celsius.

    Parameters
    ----------
    temp_f : float
        Temperature in degrees Fahrenheit

    Returns
    -------
    float
        Temperature in degrees Celsius
    """
    return (temp_f - 32.0) * 5.0 / 9.0


def fetch_snotel_sites(wsdlurl: str) -> gpd.GeoDataFrame:
    """
    Fetch SNOTEL station metadata from CUAHSI WaterOneFlow web service.

    Retrieves all available SNOTEL sites from the specified WSDL URL and
    returns them as a GeoDataFrame with standardized columns. Handles various
    metadata schemas and ensures consistent 'code', 'name', and 'geometry' columns.

    Parameters
    ----------
    wsdlurl : str
        CUAHSI WaterOneFlow WSDL endpoint URL. Example:
        "https://hydroportal.cuahsi.org/Snotel/cuahsi_1_1.asmx?WSDL"

    Returns
    -------
    gpd.GeoDataFrame
        GeoDataFrame with SNOTEL sites in EPSG:4326. Columns:
        - code: station identifier (str)
        - name: station name (str)
        - geometry: Point location

    Raises
    ------
    ValueError
        If response lacks 'location' column or required metadata
    ConnectionError
        If unable to connect to WSDL service

    Examples
    --------
    >>> wsdl = "https://hydroportal.cuahsi.org/Snotel/cuahsi_1_1.asmx?WSDL"
    >>> sites = fetch_snotel_sites(wsdl)
    >>> len(sites)
    854
    >>> sites.crs
    CRS.from_epsg(4326)
    """
    import ulmo

    sites = ulmo.cuahsi.wof.get_sites(wsdlurl)
    df = pd.DataFrame.from_dict(sites, orient="index")

    if "location" not in df.columns:
        raise ValueError(
            f"Expected 'location' column; columns={list(df.columns)}"
        )

    def _get_lon_lat(loc):
        if not isinstance(loc, dict):
            return None, None
        if "longitude" in loc and "latitude" in loc:
            return loc.get("longitude"), loc.get("latitude")
        for k in ("geoLocation", "geolocation", "GeoLocation"):
            if k in loc and isinstance(loc[k], dict):
                gl = loc[k]
                if "longitude" in gl and "latitude" in gl:
                    return gl.get("longitude"), gl.get("latitude")
        return None, None

    lon_lat = df["location"].apply(_get_lon_lat)
    df["longitude"] = lon_lat.apply(lambda x: x[0])
    df["latitude"] = lon_lat.apply(lambda x: x[1])

    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
    df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
    df = df.dropna(subset=["longitude", "latitude"]).copy()

    # Preserve the dict key explicitly
    df = df.reset_index().rename(columns={"index": "site_key"})

    # Name field normalization (only if needed)
    if "name" not in df.columns and "SiteName" in df.columns:
        df = df.rename(columns={"SiteName": "name"})

    # Pick a code that get_values() can actually use
    # Prefer metadata 'code' if present; otherwise try common alternatives; else fallback to site_key
    for cand in ("code", "site_code", "siteid"):
        if cand in df.columns:
            df["code"] = df[cand].astype(str)
            break
    else:
        df["code"] = df["site_key"].astype(str)

    # Ensure a stable station name column exists for downstream consumers.
    if "name" not in df.columns:
        df["name"] = df["site_key"].astype(str)
    else:
        fallback_name = df["site_key"].astype(str)
        df["name"] = (
            df["name"].astype(str).where(df["name"].notna(), fallback_name)
        )

    # Build geometry
    df["geometry"] = [
        Point(lon, lat) for lon, lat in zip(df["longitude"], df["latitude"])
    ]

    gdf = gpd.GeoDataFrame(df, geometry="geometry", crs="EPSG:4326")

    # Keep the minimal set your downstream expects
    keep = [c for c in ["code", "name", "geometry"] if c in gdf.columns]
    return gdf[keep].copy()


def _reproject_geometry(geom, src_crs, dst_crs):
    """
    Reproject a single Shapely geometry between coordinate reference systems.

    Parameters
    ----------
    geom : shapely.geometry
        Geometry to reproject (Point, Polygon, etc.)
    src_crs : str or CRS
        Source coordinate reference system
    dst_crs : str or CRS
        Destination coordinate reference system

    Returns
    -------
    shapely.geometry
        Reprojected geometry. Returns original geometry if src_crs == dst_crs
        or if either CRS is None. Returns None if input geometry is None.
    """
    if geom is None:
        return None
    if src_crs is None or dst_crs is None:
        return geom
    if str(src_crs) == str(dst_crs):
        return geom

    import pyproj  # dependency of geopandas

    tfm = pyproj.Transformer.from_crs(
        src_crs, dst_crs, always_xy=True
    ).transform
    return shp_transform(tfm, geom)


def filter_sites_by_polygon(
    sites_gdf: gpd.GeoDataFrame,
    footprint_geom,
    *,
    footprint_crs=None,
) -> gpd.GeoDataFrame:
    """
    Filter SNOTEL sites to those intersecting a footprint polygon.

    Spatially filters sites by intersection with a polygon boundary. Handles
    CRS mismatches by reprojecting sites to match the footprint CRS when needed.

    Parameters
    ----------
    sites_gdf : gpd.GeoDataFrame
        SNOTEL sites with Point geometry (typically from fetch_snotel_sites)
    footprint_geom : shapely.geometry
        Polygon or MultiPolygon geometry defining the area of interest
    footprint_crs : str or CRS, optional
        Coordinate reference system of footprint_geom. If provided and different
        from sites_gdf.crs, sites will be reprojected to footprint_crs before
        filtering. If None, assumes footprint_geom is already in sites_gdf.crs.

    Returns
    -------
    gpd.GeoDataFrame
        Filtered sites that intersect the footprint. Returns empty GeoDataFrame
        if no sites intersect. CRS matches footprint_crs if provided, otherwise
        matches input sites_gdf.crs.

    Examples
    --------
    >>> from shapely.geometry import Polygon
    >>> sites = fetch_snotel_sites(wsdl_url)
    >>> footprint = Polygon([(-120, 38), (-120, 39), (-119, 39), (-119, 38)])
    >>> filtered = filter_sites_by_polygon(sites, footprint, footprint_crs="EPSG:4326")
    >>> len(filtered) < len(sites)
    True
    """
    if sites_gdf.empty:
        return sites_gdf.copy()

    sites = sites_gdf

    # If user provides footprint CRS and it's different, reproject sites to footprint CRS.
    if (
        footprint_crs is not None
        and sites.crs is not None
        and str(sites.crs) != str(footprint_crs)
    ):
        sites = sites.to_crs(footprint_crs)

    mask = sites.intersects(footprint_geom)
    return sites.loc[mask].copy()


def _iter_date_chunks(
    start_date: str, end_date: str, chunk_days: int
) -> list[tuple[str, str]]:
    """Split an inclusive date range into smaller inclusive chunks."""
    start = pd.to_datetime(start_date).normalize()
    end = pd.to_datetime(end_date).normalize()

    if end < start:
        raise ValueError("end_date must be on or after start_date")
    if chunk_days < 1:
        raise ValueError("request_chunk_days must be >= 1")

    chunks: list[tuple[str, str]] = []
    chunk_span = pd.Timedelta(days=chunk_days - 1)
    current_start = start

    while current_start <= end:
        current_end = min(current_start + chunk_span, end)
        chunks.append(
            (
                current_start.strftime("%Y-%m-%d"),
                current_end.strftime("%Y-%m-%d"),
            )
        )
        current_start = current_end + pd.Timedelta(days=1)

    return chunks


def _fetch_values_chunked(
    ulmo,
    wsdlurl: str,
    site_code: str,
    variable_code: str,
    start_date: str,
    end_date: str,
    *,
    request_chunk_days: int,
    request_timeout: float | None,
) -> pd.DataFrame:
    """Fetch one site/variable across smaller date chunks and combine rows."""
    frames: list[pd.DataFrame] = []

    for chunk_start, chunk_end in _iter_date_chunks(
        start_date, end_date, request_chunk_days
    ):
        resp = ulmo.cuahsi.wof.get_values(
            wsdlurl,
            site_code,
            variable_code,
            start=chunk_start,
            end=chunk_end,
            timeout=request_timeout,
        )
        values = resp.get("values", None)
        if not values:
            continue

        frame = pd.DataFrame.from_dict(values)
        if frame.empty:
            continue
        frames.append(frame)

    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)
    if "date_time_utc" in combined.columns:
        combined = combined.drop_duplicates(subset=["date_time_utc"]).copy()
    else:
        combined = combined.drop_duplicates().copy()
    return combined


def fetch_snotel_timeseries(
    snotel_sites: pd.DataFrame,
    wsdlurl: str,
    start_date: str,
    end_date: str,
    *,
    reference_date: str = "12-01",
    obs_hour: int = 0,
    include_temperature: bool = True,
    request_chunk_days: int = 31,
    request_timeout: float | None = None,
    errors: str = "warn",
) -> Dict[str, pd.DataFrame]:
    """
    Fetch SNOTEL snow water equivalent and temperature time series for multiple sites.

    Retrieves hourly SWE (WTEQ_H) and optionally temperature (TOBS_H) data from
    CUAHSI WaterOneFlow service for all provided sites. Filters to a specific
    hour of day, converts units, and computes days since reference date.

    Parameters
    ----------
    snotel_sites : pd.DataFrame
        DataFrame with SNOTEL sites. Must contain columns:
        - code: site identifier for API queries
        - name: site name for result keys
        - geometry: Point geometry (shapely)
    wsdlurl : str
        CUAHSI WaterOneFlow WSDL endpoint URL
    start_date : str
        Start date in pandas-parseable format (e.g., "2020-10-01")
    end_date : str
        End date in pandas-parseable format (e.g., "2021-05-31")
    reference_date : str, default "12-01"
        Reference date in "MM-DD" format for computing days_since_reference.
        Year is taken from start_date.
    obs_hour : int, default 0
        Hour of day (0-23) to extract from hourly data. Default 0 = midnight.
    include_temperature : bool, default True
        Whether to fetch and include temperature (TOBS_H) data. If False or
        if temperature data unavailable, temp_c column will contain NaN.
    request_chunk_days : int, default 31
        Maximum number of days to request per WaterOneFlow call. Smaller chunks
        reduce service-side query failures for long hourly ranges.
    request_timeout : float or None, default None
        Optional SOAP request timeout in seconds for each WaterOneFlow call.
        If None, ulmo uses its default timeout.
    errors : {"warn", "raise"}, default "warn"
        If "warn", log site-level fetch failures and continue. If "raise",
        re-raise fetch failures for reproducible batch runs.

    Returns
    -------
    Dict[str, pd.DataFrame]
        Dictionary mapping site names to DataFrames. Each DataFrame contains:
        - date_time_utc: datetime (pd.Timestamp)
        - days_since_reference: days since reference_date (int)
        - value_cm: snow water equivalent in cm (float32)
        - temp_c: temperature in Celsius (float32, NaN if unavailable)
        - site_loc: station Point geometry (shapely)

        Sites with no data or errors are omitted from results.

    Raises
    ------
    ValueError
        If snotel_sites is missing required columns ('code', 'name', 'geometry')
        or if obs_hour is not in range [0, 23], or if errors is invalid.

    Notes
    -----
    - Filters to quality_control_level_code == "1" when available
    - Converts SWE from inches to cm using 1 inch = 2.54 cm
    - Converts temperature from Fahrenheit to Celsius
    - Logs warnings for sites with errors or missing data unless errors="raise"
    - Handles duplicate site names by appending " (code)" or " #N"

    Examples
    --------
    >>> sites = fetch_snotel_sites(wsdl_url)
    >>> filtered = filter_sites_by_polygon(sites, footprint, footprint_crs="EPSG:4326")
    >>> results = fetch_snotel_timeseries(
    ...     filtered, wsdl_url,
    ...     start_date="2020-10-01",
    ...     end_date="2021-05-31",
    ...     reference_date="10-01",
    ...     obs_hour=0
    ... )
    >>> for site_name, df in results.items():
    ...     print(f"{site_name}: {len(df)} records")
    """
    required = {"code", "name", "geometry"}
    missing = required - set(snotel_sites.columns)
    if missing:
        raise ValueError(f"snotel_sites missing {sorted(missing)}")

    if not (0 <= obs_hour <= 23):
        raise ValueError("obs_hour must be in [0, 23]")
    if request_chunk_days < 1:
        raise ValueError("request_chunk_days must be >= 1")
    if errors not in {"warn", "raise"}:
        raise ValueError("errors must be either 'warn' or 'raise'")

    import ulmo

    results: Dict[str, pd.DataFrame] = {}

    # Preserve your notebook behavior: reference year taken from start_date
    reference_year = pd.to_datetime(start_date).year
    reference_datetime = pd.to_datetime(f"{reference_year}-{reference_date}")

    for site_code, site_name, site_loc in zip(
        snotel_sites["code"], snotel_sites["name"], snotel_sites["geometry"]
    ):
        try:
            # --- SWE ---
            swe_df = _fetch_values_chunked(
                ulmo,
                wsdlurl,
                site_code,
                "WTEQ_H",
                start_date,
                end_date,
                request_chunk_days=request_chunk_days,
                request_timeout=request_timeout,
            )
            if swe_df.empty:
                continue

            # QC filter only if present (matches your intent)
            if "quality_control_level_code" in swe_df.columns:
                swe_df = swe_df[swe_df["quality_control_level_code"] == "1"]

            # Drop only what exists
            drop_cols = [
                "qualifiers",
                "censor_code",
                "method_id",
                "method_code",
                "source_code",
                "quality_control_level_code",
                "datetime",
            ]
            swe_df = swe_df.drop(
                columns=[c for c in drop_cols if c in swe_df.columns],
                errors="ignore",
            )

            # If schema isn't what we expect, skip site (minimal)
            if (
                "date_time_utc" not in swe_df.columns
                or "value" not in swe_df.columns
            ):
                logger.warning(
                    "Skipping %s (%s): unexpected SWE schema cols=%s",
                    site_code,
                    site_name,
                    list(swe_df.columns),
                )
                continue

            swe_df["date_time_utc"] = (
                pd.to_datetime(
                    swe_df["date_time_utc"], errors="coerce", utc=True
                )
                .dt.tz_convert(None)
            )
            swe_df = swe_df.dropna(subset=["date_time_utc"]).copy()

            swe_df["value"] = pd.to_numeric(
                swe_df["value"], errors="coerce"
            ).astype("float32")
            swe_df["value_cm"] = swe_df["value"] * _INCHES_TO_CM

            swe_at_hour = swe_df[
                swe_df["date_time_utc"].dt.hour == obs_hour
            ].copy()

            # --- Temperature ---
            tmp_at_hour = pd.DataFrame(columns=["date_time_utc", "temp_c"])

            if include_temperature:
                try:
                    tmp_df = _fetch_values_chunked(
                        ulmo,
                        wsdlurl,
                        site_code,
                        "TOBS_H",
                        start_date,
                        end_date,
                        request_chunk_days=request_chunk_days,
                        request_timeout=request_timeout,
                    )
                    if not tmp_df.empty:
                        if "quality_control_level_code" in tmp_df.columns:
                            tmp_df = tmp_df[
                                tmp_df["quality_control_level_code"] == "1"
                            ]

                        tmp_df = tmp_df.drop(
                            columns=[c for c in drop_cols if c in tmp_df.columns],
                            errors="ignore",
                        )

                        if (
                            "date_time_utc" in tmp_df.columns
                            and "value" in tmp_df.columns
                        ):
                            tmp_df["date_time_utc"] = (
                                pd.to_datetime(
                                    tmp_df["date_time_utc"],
                                    errors="coerce",
                                    utc=True,
                                )
                                .dt.tz_convert(None)
                            )
                            tmp_df = tmp_df.dropna(
                                subset=["date_time_utc"]
                            ).copy()

                            tmp_df["value"] = pd.to_numeric(
                                tmp_df["value"], errors="coerce"
                            ).astype("float32")

                            tmp_df["temp_c"] = tmp_df["value"].apply(f_to_c)

                            tmp_at_hour = tmp_df[
                                tmp_df["date_time_utc"].dt.hour == obs_hour
                            ][["date_time_utc", "temp_c"]].copy()

                except Exception as e_temp:
                    if errors == "raise":
                        raise
                    logger.warning(
                        "TOBS_H failed for %s (%s): %s",
                        site_code,
                        site_name,
                        e_temp,
                    )

            # --- Merge ---
            merged = swe_at_hour.merge(
                tmp_at_hour, on="date_time_utc", how="left"
            )

            merged["days_since_reference"] = (
                merged["date_time_utc"] - reference_datetime
            ).dt.days
            merged["site_loc"] = site_loc

            site_name_str = str(site_name).strip()
            if not site_name_str:
                site_name_str = str(site_code)
            result_key = site_name_str
            if result_key in results:
                result_key = f"{site_name_str} ({site_code})"
            if result_key in results:
                i = 2
                while f"{result_key} #{i}" in results:
                    i += 1
                result_key = f"{result_key} #{i}"

            results[result_key] = merged[
                [
                    "date_time_utc",
                    "days_since_reference",
                    "value_cm",
                    "temp_c",
                    "site_loc",
                ]
            ].copy()

        except Exception as e:
            if errors == "raise":
                raise
            logger.warning(
                "Skipping site %s (%s): %s", site_code, site_name, e
            )

    return results
