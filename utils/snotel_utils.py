from __future__ import annotations

import logging
from typing import Dict

import geopandas as gpd
import pandas as pd
import ulmo
from shapely.geometry import Point
from shapely.ops import transform as shp_transform

logger = logging.getLogger(__name__)


def f_to_c(temp_f: float) -> float:
    return (temp_f - 32.0) * 5.0 / 9.0


def fetch_snotel_sites(wsdlurl: str) -> gpd.GeoDataFrame:
    sites = ulmo.cuahsi.wof.get_sites(wsdlurl)
    df = pd.DataFrame.from_dict(sites, orient="index")

    if "location" not in df.columns:
        raise ValueError(f"Expected 'location' column; columns={list(df.columns)}")

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
    Reproject a single shapely geometry from src_crs -> dst_crs.
    """
    if geom is None:
        return None
    if src_crs is None or dst_crs is None:
        return geom
    if str(src_crs) == str(dst_crs):
        return geom

    # geopandas/pyproj transformer
    transformer = (
        gpd.GeoSeries([Point(0, 0)], crs=src_crs).to_crs(dst_crs).crs
    )  # ensures CRS parse
    # Use pyproj directly via GeoSeries.transform pattern:
    # We can do it more cleanly using pyproj.Transformer if available through geopandas:
    import pyproj  # dependency of geopandas

    tfm = pyproj.Transformer.from_crs(src_crs, dst_crs, always_xy=True).transform
    return shp_transform(tfm, geom)


def filter_sites_by_polygon(
    sites_gdf: gpd.GeoDataFrame,
    footprint_geom,
    *,
    footprint_crs=None,
) -> gpd.GeoDataFrame:
    """
    Filter sites that intersect the footprint polygon.

    Key fix: handles CRS mismatch safely.
    - If footprint_crs is provided and differs from sites_gdf.crs, we reproject sites.
    - If footprint_crs is None, we assume footprint_geom is already in sites_gdf.crs.
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


def fetch_snotel_timeseries(
    snotel_sites: pd.DataFrame,
    wsdlurl: str,
    start_date: str,
    end_date: str,
    *,
    reference_date: str = "12-01",
    obs_hour: int = 0,
    include_temperature: bool = True,
) -> Dict[str, pd.DataFrame]:
    """
    Fetch SNOTEL SWE (WTEQ_H) and optionally hourly TOBS_H for multiple sites.

    Returns dict[site_name] -> DataFrame with:
      - date_time_utc
      - days_since_reference
      - value_cm
      - temp_c  (if include_temperature, else NaN)
      - site_loc
    """
    required = {"code", "name", "geometry"}
    missing = required - set(snotel_sites.columns)
    if missing:
        raise ValueError(f"snotel_sites missing {sorted(missing)}")

    if not (0 <= obs_hour <= 23):
        raise ValueError("obs_hour must be in [0, 23]")

    results: Dict[str, pd.DataFrame] = {}

    # Preserve your notebook behavior: reference year taken from start_date
    reference_year = pd.to_datetime(start_date).year
    reference_datetime = pd.to_datetime(f"{reference_year}-{reference_date}")

    for site_code, site_name, site_loc in zip(
        snotel_sites["code"], snotel_sites["name"], snotel_sites["geometry"]
    ):
        try:
            # --- SWE ---
            swe_resp = ulmo.cuahsi.wof.get_values(
                wsdlurl, site_code, "WTEQ_H", start=start_date, end=end_date
            )
            swe_vals = swe_resp.get("values", None)
            if not swe_vals:
                continue

            swe_df = pd.DataFrame.from_dict(swe_vals)
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
                columns=[c for c in drop_cols if c in swe_df.columns], errors="ignore"
            )

            # If schema isn't what we expect, skip site (minimal)
            if "date_time_utc" not in swe_df.columns or "value" not in swe_df.columns:
                logger.warning(
                    "Skipping %s (%s): unexpected SWE schema cols=%s",
                    site_code,
                    site_name,
                    list(swe_df.columns),
                )
                continue

            swe_df["date_time_utc"] = pd.to_datetime(
                swe_df["date_time_utc"], errors="coerce"
            )
            swe_df = swe_df.dropna(subset=["date_time_utc"]).copy()

            swe_df["value"] = pd.to_numeric(swe_df["value"], errors="coerce").astype(
                "float32"
            )
            swe_df["value_cm"] = swe_df["value"] * 2.54

            swe_at_hour = swe_df[swe_df["date_time_utc"].dt.hour == obs_hour].copy()

            # --- Temperature ---
            tmp_at_hour = pd.DataFrame(columns=["date_time_utc", "temp_c"])

            if include_temperature:
                try:
                    tmp_resp = ulmo.cuahsi.wof.get_values(
                        wsdlurl, site_code, "TOBS_H", start=start_date, end=end_date
                    )
                    tmp_vals = tmp_resp.get("values", None)
                    if tmp_vals:
                        tmp_df = pd.DataFrame.from_dict(tmp_vals)

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
                                tmp_df["date_time_utc"] = pd.to_datetime(
                                    tmp_df["date_time_utc"], errors="coerce"
                                )
                                tmp_df = tmp_df.dropna(subset=["date_time_utc"]).copy()

                                tmp_df["value"] = pd.to_numeric(
                                    tmp_df["value"], errors="coerce"
                                ).astype("float32")

                                tmp_df["temp_c"] = tmp_df["value"].apply(f_to_c)

                                tmp_at_hour = tmp_df[
                                    tmp_df["date_time_utc"].dt.hour == obs_hour
                                ][["date_time_utc", "temp_c"]].copy()

                except Exception as e_temp:
                    logger.warning(
                        "TOBS_H failed for %s (%s): %s", site_code, site_name, e_temp
                    )

            # --- Merge ---
            merged = swe_at_hour.merge(tmp_at_hour, on="date_time_utc", how="left")

            merged["days_since_reference"] = (
                merged["date_time_utc"] - reference_datetime
            ).dt.days
            merged["site_loc"] = site_loc

            results[site_name] = merged[
                [
                    "date_time_utc",
                    "days_since_reference",
                    "value_cm",
                    "temp_c",
                    "site_loc",
                ]
            ].copy()

        except Exception as e:
            logger.warning("Skipping site %s (%s): %s", site_code, site_name, e)

    return results
