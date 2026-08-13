from __future__ import annotations

from pathlib import Path
from typing import Any, List, Optional, Union

import h5py
import geopandas as gpd
import numpy as np
import pandas as pd
from pyproj import CRS, Transformer

from .snotel_utils import build_snotel_value_lookup


def _normalize_mintpy_attr(value: Any) -> Any:
    """Normalize HDF5 attribute values into Python scalars when possible."""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="ignore")
    return value


def _infer_mintpy_crs(attrs: dict, fallback_crs: Optional[str] = None) -> str:
    """
    Infer CRS from MintPy geocoding attributes.

    Prefers the explicit EPSG attribute when present. Falls back to EPSG:4326
    only for degree-based coordinate units.
    """
    epsg = _normalize_mintpy_attr(attrs.get("EPSG"))
    if epsg not in (None, ""):
        epsg_text = str(epsg).strip()
        if epsg_text.upper().startswith("EPSG:"):
            return epsg_text.upper()
        try:
            return f"EPSG:{int(float(epsg_text))}"
        except ValueError:
            pass

    x_unit = str(_normalize_mintpy_attr(attrs.get("X_UNIT", ""))).lower()
    y_unit = str(_normalize_mintpy_attr(attrs.get("Y_UNIT", ""))).lower()
    if "degree" in x_unit and "degree" in y_unit:
        return "EPSG:4326"

    if fallback_crs is not None:
        return fallback_crs

    raise ValueError(
        "Could not infer MintPy CRS from attributes. "
        "Expected an EPSG attribute or degree-based X/Y units."
    )


def decode_h5_dates(values) -> List[str]:
    """Decode byte or scalar HDF5 date values into strings."""
    return [
        value.decode("utf-8") if isinstance(value, bytes) else str(value)
        for value in values
    ]


def mintpy_crs_from_attrs(
    attrs: dict,
    fallback_crs: Optional[str] = None,
) -> Optional[CRS]:
    """
    Infer a MintPy grid CRS from HDF5 attributes.

    Returns None when the attributes do not provide enough information and no
    fallback CRS is supplied.
    """
    try:
        return CRS.from_user_input(_infer_mintpy_crs(attrs, fallback_crs))
    except ValueError:
        return None


def mintpy_grid_centers(attrs: dict) -> tuple[np.ndarray, np.ndarray]:
    """Return 1D x/y center-coordinate arrays for a MintPy grid."""
    x_step = float(attrs["X_STEP"])
    y_step = float(attrs["Y_STEP"])
    width = int(float(attrs["WIDTH"]))
    length = int(float(attrs["LENGTH"]))
    x0 = float(attrs["X_FIRST"]) + x_step / 2.0
    y0 = float(attrs["Y_FIRST"]) + y_step / 2.0
    x_coords = x0 + x_step * np.arange(width)
    y_coords = y0 + y_step * np.arange(length)
    return x_coords, y_coords


def station_xy_for_grid(
    site_row,
    grid_crs: Optional[CRS],
) -> tuple[float, float]:
    """Return station coordinates in the MintPy grid CRS."""
    x_in = float(site_row.longitude)
    y_in = float(site_row.latitude)

    if grid_crs is None:
        return x_in, y_in

    looks_geographic = abs(x_in) <= 180.0 and abs(y_in) <= 90.0
    if grid_crs.is_projected and looks_geographic:
        transformer = Transformer.from_crs("EPSG:4326", grid_crs, always_xy=True)
        return transformer.transform(x_in, y_in)

    if (not grid_crs.is_projected) and (not looks_geographic):
        raise ValueError(
            f"Station coordinates ({x_in}, {y_in}) look projected, "
            f"but MintPy grid CRS is geographic ({grid_crs})."
        )

    return x_in, y_in


def build_station_windows(
    site_table: pd.DataFrame,
    attrs: dict,
    win_size: int,
) -> List[dict]:
    """Build pixel windows around each station on a MintPy grid."""
    grid_crs = mintpy_crs_from_attrs(attrs)
    x_coords, y_coords = mintpy_grid_centers(attrs)
    stations = []

    for _, site_row in site_table.iterrows():
        x_coord, y_coord = station_xy_for_grid(site_row, grid_crs)
        col = int(np.argmin(np.abs(x_coords - x_coord)))
        row = int(np.argmin(np.abs(y_coords - y_coord)))
        stations.append(
            {
                "site_name": site_row.site_name,
                "latitude": float(site_row.latitude),
                "longitude": float(site_row.longitude),
                "x_coord": float(x_coord),
                "y_coord": float(y_coord),
                "row": row,
                "col": col,
                "y_min": max(row - win_size, 0),
                "y_max": min(row + win_size + 1, len(y_coords)),
                "x_min": max(col - win_size, 0),
                "x_max": min(col + win_size + 1, len(x_coords)),
            }
        )
    return stations


def mintpy_dates_from_timeseries_h5(
    timeseries_h5: Union[str, Path],
) -> List[pd.Timestamp]:
    """
    Extract acquisition dates from MintPy geocoded timeseries HDF5 file.

    Reads slice names from MintPy timeseries file, extracts YYYYMMDD date tokens,
    and returns unique sorted dates as normalized timestamps.

    Parameters
    ----------
    timeseries_h5 : str or Path
        Path to MintPy geocoded timeseries file (e.g., geo_timeseries*.h5)

    Returns
    -------
    List[pd.Timestamp]
        Sorted list of unique acquisition dates, normalized to midnight (00:00:00)

    Raises
    ------
    ImportError
        If mintpy package is not installed
    ValueError
        If no slices found in file or no valid YYYYMMDD dates parsed

    Notes
    -----
    - Uses the last 8-digit token in each slice name as the date
    - Slice names typically follow format: "timeseries-YYYYMMDD" or "YYYYMMDD_YYYYMMDD"
    - Duplicate dates across slices are automatically deduplicated

    Examples
    --------
    >>> dates = mintpy_dates_from_timeseries_h5("geo_timeseries_ERA5_demErr.h5")
    >>> len(dates)
    45
    >>> dates[0]
    Timestamp('2020-10-01 00:00:00')
    """
    import re

    timeseries_h5 = Path(timeseries_h5)

    try:
        from mintpy.utils import readfile
    except ImportError as e:
        raise ImportError(
            "mintpy is required for mintpy_dates_from_timeseries_h5(). "
            "Install mintpy in your environment."
        ) from e

    slices = readfile.get_slice_list(str(timeseries_h5))
    if not slices:
        raise ValueError(f"No slices found in: {timeseries_h5}")

    dates: List[pd.Timestamp] = []

    for s in slices:
        tokens = re.findall(r"(\d{8})", str(s))
        if not tokens:
            continue
        # use last 8-digit token as the date
        dates.append(pd.to_datetime(tokens[-1], format="%Y%m%d").normalize())

    dates = sorted(set(dates))
    if not dates:
        raise ValueError(
            f"Could not parse any YYYYMMDD dates from MintPy slices in: {timeseries_h5}"
        )

    return dates


def mintpy_footprint_from_timeseries_h5(
    timeseries_h5: Union[str, Path],
    *,
    reference_slice: Optional[str] = None,
    footprint_mode: str = "single",
    crs: Optional[str] = None,
) -> gpd.GeoDataFrame:
    """
    Build valid-data footprint polygon from MintPy geocoded timeseries file.

    Reads one time slice from MintPy timeseries, extracts georeferencing metadata,
    and generates a polygon representing the valid (non-NaN) data extent.

    Parameters
    ----------
    timeseries_h5 : str or Path
        Path to MintPy geocoded timeseries file (e.g., geo_timeseries*.h5)
    reference_slice : str, optional
        Specific slice name to use for footprint generation.
        If None (default), uses the first available slice.
    footprint_mode : {"single", "union", "intersection"}, default "single"
        How to combine valid-data masks. "single" uses reference_slice (or the
        first slice), "union" uses pixels valid in any slice, and
        "intersection" uses pixels valid in every slice.
    crs : str, optional
        Override coordinate reference system for output geometry. If omitted,
        infer CRS from MintPy attributes, preferring the EPSG attribute.

    Returns
    -------
    gpd.GeoDataFrame
        Single-row GeoDataFrame with Polygon geometry representing valid data extent.
        Returns empty GeoDataFrame if no valid data found.

    Raises
    ------
    ImportError
        If mintpy package is not installed
    ValueError
        If no slices found in file or required geocoding attributes are missing
        (X_FIRST, Y_FIRST, X_STEP, Y_STEP)

    Notes
    -----
    - Footprint is derived from finite (non-NaN/non-inf) values in the selected slice
    - Uses first slice by default. Use footprint_mode="union" or
      "intersection" when masks vary temporally.
    - Geocoding bounds are computed from grid attributes and array shape

    Examples
    --------
    >>> footprint = mintpy_footprint_from_timeseries_h5("geo_timeseries_ERA5_demErr.h5")
    >>> footprint.geometry[0].area
    0.425
    >>> footprint.crs
    CRS.from_epsg(4326)
    """
    timeseries_h5 = Path(timeseries_h5)

    try:
        from mintpy.utils import readfile
    except ImportError as e:
        raise ImportError(
            "mintpy is required for mintpy_footprint_from_timeseries_h5(). "
            "Install mintpy in your environment."
        ) from e

    from .geometry import get_valid_data_polygon_from_array

    # Determine slice
    slices = readfile.get_slice_list(str(timeseries_h5))
    if not slices:
        raise ValueError(f"No slices found in: {timeseries_h5}")
    if footprint_mode not in {"single", "union", "intersection"}:
        raise ValueError(
            "footprint_mode must be one of: 'single', 'union', 'intersection'"
        )

    # Validate reference_slice if provided
    if reference_slice is not None:
        if reference_slice not in slices:
            raise ValueError(
                f"reference_slice '{reference_slice}' not found in file.\n"
                f"  File: {timeseries_h5}\n"
                f"  Available slices: {slices[:10]}" +
                (f" ... ({len(slices)} total)" if len(slices) > 10 else "")
            )
        slice_name = reference_slice
    else:
        slice_name = slices[0]

    data, atr = readfile.read(str(timeseries_h5), datasetName=slice_name)
    data = np.array(data)

    if footprint_mode != "single":
        valid = np.isfinite(data)
        for other_slice in slices:
            if other_slice == slice_name:
                continue
            other_data, _ = readfile.read(
                str(timeseries_h5), datasetName=other_slice
            )
            other_valid = np.isfinite(np.asarray(other_data))
            if other_valid.shape != valid.shape:
                raise ValueError(
                    f"Slice shape mismatch: {other_slice} has {other_valid.shape}, "
                    f"expected {valid.shape}."
                )
            if footprint_mode == "union":
                valid |= other_valid
            else:
                valid &= other_valid
        data = np.where(valid, 1.0, np.nan).astype(np.float32)

    # MintPy geocoding info
    try:
        x_first = float(atr["X_FIRST"])
        y_first = float(atr["Y_FIRST"])
        x_step = float(atr["X_STEP"])
        y_step = float(atr["Y_STEP"])
    except KeyError as e:
        raise ValueError(
            f"Missing required geocoding attribute {e} in attrs for {timeseries_h5.name}."
        ) from e

    nrows, ncols = data.shape

    west = x_first
    north = y_first
    east = west + (ncols * x_step)
    south = north + (nrows * y_step)

    # Normalize ordering
    west, east = (min(west, east), max(west, east))
    south, north = (min(south, north), max(south, north))

    inferred_crs = _infer_mintpy_crs(atr, fallback_crs=crs)

    gdf = get_valid_data_polygon_from_array(
        data,
        north=north,
        south=south,
        east=east,
        west=west,
        x_step=x_step,
        y_step=y_step,
        crs=inferred_crs,
    )

    return gdf


def _read_mintpy_incidence_grid(
    incidence_file: Union[str, Path],
    dataset_name: Optional[str] = "incidenceAngle",
) -> np.ndarray:
    """Read an incidence-angle grid from a MintPy-readable raster or HDF5 file."""
    incidence_file = Path(incidence_file)

    try:
        from mintpy.utils import readfile
    except ImportError as e:
        raise ImportError(
            "mintpy is required for _read_mintpy_incidence_grid(). "
            "Install mintpy in your environment."
        ) from e

    if dataset_name:
        try:
            data, _ = readfile.read(str(incidence_file), datasetName=dataset_name)
            return np.asarray(data)
        except Exception:
            pass

    data, _ = readfile.read(str(incidence_file))
    return np.asarray(data)


def read_mintpy_displacement_series_at_latlon(
    lat: float,
    lon: float,
    *,
    timeseries_file: Union[str, Path],
    lookup_file: Union[str, Path],
    win_size: int = 1,
) -> tuple[pd.DatetimeIndex, np.ndarray]:
    """
    Extract a MintPy displacement time series at a geographic point.

    Returns normalized acquisition dates plus the displacement vector.
    """
    timeseries_file = Path(timeseries_file)
    lookup_file = Path(lookup_file)

    try:
        from mintpy.utils import utils as ut
    except ImportError as e:
        raise ImportError(
            "mintpy is required for read_mintpy_displacement_series_at_latlon(). "
            "Install mintpy in your environment."
        ) from e

    dates, displacement, _ = ut.read_timeseries_lalo(
        lat=lat,
        lon=lon,
        win_size=win_size,
        ts_file=str(timeseries_file),
        lookup_file=str(lookup_file),
    )
    date_index = pd.to_datetime([d.date() for d in dates]).normalize()
    return date_index, np.asarray(displacement, dtype=float)


def sample_mintpy_incidence_angle(
    lat: float,
    lon: float,
    *,
    timeseries_file: Union[str, Path],
    lookup_file: Union[str, Path],
    incidence_file: Union[str, Path],
    dataset_name: Optional[str] = "incidenceAngle",
    unit: str = "radians",
) -> float:
    """
    Sample incidence angle at a geographic point from a MintPy grid.

    Parameters
    ----------
    lat, lon : float
        Geographic coordinates in degrees.
    timeseries_file : str or Path
        MintPy time-series file used for coordinate metadata.
    lookup_file : str or Path
        Lookup or geometry file used by MintPy to map lat/lon to grid indices.
    incidence_file : str or Path
        Raster or HDF5 file containing incidence angle values in degrees.
    dataset_name : str, optional
        Dataset to read from HDF5-style incidence inputs. Ignored when the file
        can be read directly without a dataset name.
    unit : {"radians", "degrees"}, default "radians"
        Unit for the returned incidence angle.
    """
    timeseries_file = Path(timeseries_file)
    lookup_file = Path(lookup_file)
    incidence_file = Path(incidence_file)

    if unit not in {"radians", "degrees"}:
        raise ValueError("unit must be either 'radians' or 'degrees'")

    try:
        from mintpy.utils import utils as ut
    except ImportError as e:
        raise ImportError(
            "mintpy is required for sample_mintpy_incidence_angle(). "
            "Install mintpy in your environment."
        ) from e

    incidence = np.squeeze(
        _read_mintpy_incidence_grid(incidence_file, dataset_name=dataset_name)
    )
    if incidence.ndim != 2:
        raise ValueError(
            f"Incidence grid must be 2D after squeeze(), got shape {incidence.shape}."
        )

    attrs = ut.readfile.read_attribute(str(timeseries_file))
    coord = ut.coordinate(attrs, lookup_file=str(lookup_file))
    coord.open()
    y, x = coord.geo2radar(lat, lon)[0:2]
    if y is None or x is None:
        raise ValueError(f"Could not map lat/lon ({lat}, {lon}) into the MintPy grid.")

    y = int(np.rint(y))
    x = int(np.rint(x))
    if y < 0 or x < 0 or y >= incidence.shape[0] or x >= incidence.shape[1]:
        raise IndexError(
            f"Mapped incidence index {(y, x)} is outside the incidence grid shape {incidence.shape}."
        )

    angle_deg = float(incidence[y, x])
    if unit == "degrees":
        return angle_deg
    return float(np.deg2rad(angle_deg))


def compare_station_windows(
    site_table: pd.DataFrame,
    swe_data: dict[str, pd.DataFrame],
    *,
    timeseries_file: Union[str, Path],
    coherence_file: Union[str, Path],
    incidence_file: Union[str, Path],
    incidence_dataset: str,
    coherence_dataset: str,
    coherence_date_dataset: str,
    coherence_threshold: float,
    apply_bias_correction: bool,
    bias_station_names: list[str],
    win_size: int,
) -> tuple[dict[str, dict], pd.DataFrame]:
    """
    Compare MintPy interval SWE against SNOTEL across station windows.

    Returns successful station results plus a failure table for skipped stations.
    Stations with incomplete InSAR coverage after masking/filtering are excluded
    from the results.
    """
    timeseries_file = Path(timeseries_file)
    coherence_file = Path(coherence_file)
    incidence_file = Path(incidence_file)

    interval_rows: dict[str, list[dict]] = {
        site_name: [] for site_name in site_table.site_name
    }
    snotel_lookup = build_snotel_value_lookup(swe_data)

    with (
        h5py.File(timeseries_file, "r") as ts_h5,
        h5py.File(coherence_file, "r") as coh_h5,
        h5py.File(incidence_file, "r") as inc_h5,
    ):
        attrs = {key: value for key, value in ts_h5.attrs.items()}
        stations = build_station_windows(site_table, attrs, win_size)

        incidence_deg = np.asarray(inc_h5[incidence_dataset][...], dtype=float)
        incidence_rad = np.deg2rad(incidence_deg)
        factor_grid = (
            (-0.6784 * incidence_rad**2)
            + (0.2899 * incidence_rad)
            - 0.8473
        )

        ts_dates = decode_h5_dates(ts_h5["date"][...])
        pair_index = {
            tuple(decode_h5_dates(row)): idx
            for idx, row in enumerate(coh_h5[coherence_date_dataset][...])
        }
        expected_interval_count = sum(
            1
            for i_day in range(len(ts_dates) - 1)
            if (ts_dates[i_day], ts_dates[i_day + 1]) in pair_index
        )

        for i_day in range(len(ts_dates) - 1):
            date1 = ts_dates[i_day]
            date2 = ts_dates[i_day + 1]
            pair = (date1, date2)
            if pair not in pair_index:
                print(f"[skip] missing coherence pair for {date1} -> {date2}")
                continue

            displacement_delta = -(
                np.asarray(ts_h5["timeseries"][i_day + 1], dtype=float)
                - np.asarray(ts_h5["timeseries"][i_day], dtype=float)
            )
            swe_map = displacement_delta / -factor_grid * 100.0
            coherence_map = np.asarray(
                coh_h5[coherence_dataset][pair_index[pair]],
                dtype=float,
            )
            swe_map = np.where(coherence_map == 0, np.nan, swe_map)

            daily_rows = []
            for station in stations:
                y_min = station["y_min"]
                y_max = station["y_max"]
                x_min = station["x_min"]
                x_max = station["x_max"]

                swe_window = swe_map[y_min:y_max, x_min:x_max]
                coh_window = coherence_map[y_min:y_max, x_min:x_max]
                insar_raw = (
                    float(np.nanmean(swe_window))
                    if np.isfinite(swe_window).any()
                    else np.nan
                )
                coherence_mean = (
                    float(np.nanmean(coh_window))
                    if np.isfinite(coh_window).any()
                    else np.nan
                )

                series = snotel_lookup[station["site_name"]]
                start = pd.Timestamp(date1)
                end = pd.Timestamp(date2)
                if start in series.index and end in series.index:
                    snotel_delta = float(series.loc[end] - series.loc[start])
                else:
                    snotel_delta = np.nan

                daily_rows.append(
                    {
                        **station,
                        "date_start": start,
                        "date_end": end,
                        "date": end,
                        "coherence": coherence_mean,
                        "insar_delta_swe_cm_raw": insar_raw,
                        "snotel_delta_swe_cm": snotel_delta,
                        "incidence_angle_deg": float(
                            incidence_deg[station["row"], station["col"]]
                        ),
                    }
                )

            bias_cm = np.nan
            if apply_bias_correction and daily_rows:
                calibration = pd.DataFrame(daily_rows)
                calibration = calibration[
                    np.isfinite(calibration["insar_delta_swe_cm_raw"])
                    & np.isfinite(calibration["snotel_delta_swe_cm"])
                    & (calibration["coherence"] >= coherence_threshold)
                ]
                if bias_station_names:
                    calibration = calibration[
                        calibration["site_name"].isin(bias_station_names)
                    ]
                if not calibration.empty:
                    bias_cm = float(
                        np.nanmean(
                            calibration["insar_delta_swe_cm_raw"]
                            - calibration["snotel_delta_swe_cm"]
                        )
                    )

            for row in daily_rows:
                corrected = row["insar_delta_swe_cm_raw"]
                if np.isfinite(corrected) and np.isfinite(bias_cm):
                    corrected = corrected - bias_cm
                if not (
                    np.isfinite(row["coherence"])
                    and row["coherence"] >= coherence_threshold
                ):
                    corrected = np.nan
                row["bias_cm"] = bias_cm
                row["insar_delta_swe_cm"] = corrected
                interval_rows[row["site_name"]].append(row)

    all_results = {}
    failures = []
    for station in build_station_windows(site_table, attrs, win_size):
        site_name = station["site_name"]
        aligned = pd.DataFrame(interval_rows[site_name]).sort_values(
            "date"
        ).reset_index(drop=True)
        if aligned.empty:
            failures.append(
                {"site_name": site_name, "error": "No valid intervals found."}
            )
            continue

        if len(aligned) != expected_interval_count:
            failures.append(
                {
                    "site_name": site_name,
                    "error": (
                        "InSAR incomplete: "
                        f"expected {expected_interval_count} intervals, "
                        f"found {len(aligned)}."
                    ),
                }
            )
            continue

        missing_insar_count = int(aligned["insar_delta_swe_cm"].isna().sum())
        if missing_insar_count:
            failures.append(
                {
                    "site_name": site_name,
                    "error": (
                        "InSAR incomplete: "
                        f"{missing_insar_count} interval(s) missing after "
                        "masking/filtering."
                    ),
                }
            )
            continue

        aligned["insar_cumulative_swe_cm"] = aligned["insar_delta_swe_cm"].cumsum()
        aligned["snotel_cumulative_swe_cm"] = aligned[
            "snotel_delta_swe_cm"
        ].fillna(0.0).cumsum()
        metrics = summarize_insar_swe_metrics(aligned)
        all_results[site_name] = {
            "site_name": site_name,
            "latitude": station["latitude"],
            "longitude": station["longitude"],
            "x_coord": station["x_coord"],
            "y_coord": station["y_coord"],
            "incidence_angle_deg": float(aligned["incidence_angle_deg"].iloc[0]),
            "aligned": aligned,
            "metrics": metrics,
        }

    return all_results, pd.DataFrame(failures)


def displacement_to_swe_cm(
    displacement: np.ndarray,
    incidence_angle_rad: float,
) -> np.ndarray:
    """
    Convert incremental LOS displacement into incremental SWE in centimeters.
    """
    displacement = np.asarray(displacement, dtype=float)
    factor = (
        (-0.6784 * incidence_angle_rad**2)
        + (0.2899 * incidence_angle_rad)
        - 0.8473
    )
    delta_displacement = np.diff(-displacement)
    insar_swe = delta_displacement / -factor
    return np.insert(insar_swe, 0, 0.0) * 100.0


def align_insar_swe_with_snotel(
    station_df: pd.DataFrame,
    insar_dates: Union[pd.DatetimeIndex, list, np.ndarray],
    insar_delta_swe_cm: np.ndarray,
) -> pd.DataFrame:
    """
    Align incremental InSAR SWE with SNOTEL observations on shared dates.
    """
    required = {"date_time_utc", "value_cm"}
    missing = required - set(station_df.columns)
    if missing:
        raise ValueError(f"station_df missing required columns: {sorted(missing)}")

    snotel = station_df.copy()
    snotel["date"] = pd.to_datetime(snotel["date_time_utc"]).dt.normalize()
    insar_dates = pd.to_datetime(insar_dates).normalize()

    snotel = snotel[snotel["date"].isin(insar_dates)].sort_values("date")
    snotel["snotel_delta_swe_cm"] = snotel["value_cm"].diff().fillna(0.0)

    insar = pd.DataFrame(
        {
            "date": insar_dates,
            "insar_delta_swe_cm": np.asarray(insar_delta_swe_cm, dtype=float),
        }
    )

    aligned = insar.merge(
        snotel[["date", "value_cm", "snotel_delta_swe_cm"]],
        on="date",
        how="inner",
    ).sort_values("date").reset_index(drop=True)

    aligned["insar_cumulative_swe_cm"] = aligned["insar_delta_swe_cm"].cumsum()
    aligned["snotel_cumulative_swe_cm"] = aligned["snotel_delta_swe_cm"].cumsum()
    return aligned


def summarize_insar_swe_metrics(aligned: pd.DataFrame) -> pd.DataFrame:
    """
    Summarize agreement metrics for aligned InSAR and SNOTEL SWE increments.
    """
    required = {
        "insar_delta_swe_cm",
        "snotel_delta_swe_cm",
        "insar_cumulative_swe_cm",
        "snotel_cumulative_swe_cm",
    }
    missing = required - set(aligned.columns)
    if missing:
        raise ValueError(f"aligned missing required columns: {sorted(missing)}")

    from .lidar_utils import compute_pearson_correlation

    y_true = aligned["snotel_delta_swe_cm"].to_numpy(dtype=float)
    y_pred = aligned["insar_delta_swe_cm"].to_numpy(dtype=float)
    resid = y_pred - y_true

    rmse = np.sqrt(np.mean(resid**2)) if len(aligned) else np.nan
    denom = np.sum((y_true - y_true.mean()) ** 2)
    r2 = (
        1.0 - (np.sum(resid**2) / denom)
        if len(aligned) and denom > 0
        else np.nan
    )

    delta_corr = compute_pearson_correlation(y_pred, y_true, on_invalid="nan")
    cumulative_corr = compute_pearson_correlation(
        aligned["insar_cumulative_swe_cm"].to_numpy(dtype=float),
        aligned["snotel_cumulative_swe_cm"].to_numpy(dtype=float),
        on_invalid="nan",
    )

    return pd.DataFrame(
        [
            {
                "matched_dates": len(aligned),
                "rmse_cm": rmse,
                "r2": r2,
                "delta_corr": delta_corr["statistic"],
                "delta_pvalue": delta_corr["pvalue"],
                "cumulative_corr": cumulative_corr["statistic"],
                "cumulative_pvalue": cumulative_corr["pvalue"],
            }
        ]
    )
