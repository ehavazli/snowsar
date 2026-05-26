from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union

import geopandas as gpd
import pandas as pd


def _has_nonempty_geometry(gdf: gpd.GeoDataFrame) -> bool:
    """
    Check if GeoDataFrame has non-empty geometry.

    Parameters
    ----------
    gdf : gpd.GeoDataFrame
        GeoDataFrame to check

    Returns
    -------
    bool
        True if GeoDataFrame is non-empty, has geometry column, and at least
        one non-empty geometry
    """
    return (
        (not gdf.empty)
        and ("geometry" in gdf.columns)
        and (not gdf.geometry.is_empty.all())
    )


@dataclass(frozen=True)
class InsarContext:
    """
    Standardized InSAR workflow context for downstream analysis.

    Provides a unified interface for HyP3, MintPy, and NISAR workflows, extracting
    acquisition dates and valid-data footprint from processed InSAR products.

    Attributes
    ----------
    source : str
        Processing workflow: "hyp3", "mintpy", or "nisar"
    dates : List[pd.Timestamp]
        Acquisition dates, normalized to midnight (00:00:00)
    footprint : gpd.GeoDataFrame
        Single-row GeoDataFrame with Polygon geometry representing valid data extent.
        CRS is EPSG:4326 (WGS84 lat/lon).
    source_files : Tuple[str, ...]
        Source files used to build the context.

    Notes
    -----
    This dataclass is immutable (frozen=True) to ensure consistency across
    analysis pipelines that consume InSAR context.

    Examples
    --------
    >>> # HyP3 workflow
    >>> ctx = build_insar_context(
    ...     source="hyp3",
    ...     hyp3_tifs=["S1_20201215_20201227_VV_unw.tif", "S1_20201227_20210108_VV_unw.tif"]
    ... )
    >>> len(ctx.dates)
    3
    >>> ctx.source
    'hyp3'

    >>> # MintPy workflow
    >>> ctx = build_insar_context(
    ...     source="mintpy",
    ...     mintpy_timeseries_h5="geo_timeseries_ERA5_demErr.h5"
    ... )
    >>> ctx.footprint.crs
    CRS.from_epsg(4326)
    """

    source: str  # "hyp3", "mintpy", or "nisar"
    dates: List[pd.Timestamp]  # normalized (midnight) timestamps
    footprint: (
        gpd.GeoDataFrame
    )  # single-row GeoDataFrame with polygon geometry
    source_files: Tuple[str, ...] = ()


def build_insar_context(
    *,
    source: str,
    hyp3_tifs: Optional[Sequence[Union[str, Path]]] = None,
    mintpy_timeseries_h5: Optional[Union[str, Path]] = None,
    mintpy_reference_slice: Optional[str] = None,
    nisar_gunw_h5s: Optional[Sequence[Union[str, Path]]] = None,
    nisar_frequency: str = "A",
    nisar_pol: str = "HH",
) -> InsarContext:
    """
    Build InSAR workflow context (dates, footprint) from HyP3, MintPy, or NISAR products.

    Factory function that routes to appropriate workflow-specific extractors based
    on source parameter. Enforces mutual exclusivity between HyP3, MintPy, and
    NISAR inputs.

    Parameters
    ----------
    source : str
        Processing workflow: "hyp3", "mintpy", or "nisar"
    hyp3_tifs : Sequence[str or Path], optional
        HyP3 GeoTIFF paths (e.g., *_unw_phase_clipped.tif).
        Required if source="hyp3", must be None if source="mintpy".
    mintpy_timeseries_h5 : str or Path, optional
        MintPy geocoded timeseries HDF5 file (e.g., geo_timeseries*.h5).
        Required if source="mintpy", must be None if source="hyp3".
    mintpy_reference_slice : str, optional
        Slice name to use for MintPy footprint generation.
        If None, uses first slice. Only used when source="mintpy".
    nisar_gunw_h5s : Sequence[str or Path], optional
        NISAR GUNW HDF5 files. Required if source="nisar".
    nisar_frequency : str, default "A"
        NISAR frequency for footprint extraction.
    nisar_pol : str, default "HH"
        NISAR polarization for footprint extraction.

    Returns
    -------
    InsarContext
        Standardized context with source, dates, and footprint

    Raises
    ------
    ValueError
        If source is not "hyp3", "mintpy", or "nisar"; if inputs for multiple
        workflows are provided; if required inputs are missing; or if footprint
        extraction returns empty geometry.

    Notes
    -----
    HyP3, MintPy, and NISAR are mutually exclusive workflows:
    - HyP3: dates extracted from filenames, footprint from raster masks
    - MintPy: dates extracted from HDF5 slices, footprint from one time slice
    - NISAR: dates and footprint extracted from GUNW HDF5 products

    Examples
    --------
    >>> # HyP3 workflow
    >>> tifs = [
    ...     "S1_20201215_20201227_VV_unw_phase_clipped.tif",
    ...     "S1_20201227_20210108_VV_unw_phase_clipped.tif"
    ... ]
    >>> ctx = build_insar_context(source="hyp3", hyp3_tifs=tifs)
    >>> len(ctx.dates)
    3
    >>> ctx.footprint.crs
    CRS.from_epsg(4326)

    >>> # MintPy workflow
    >>> ctx = build_insar_context(
    ...     source="mintpy",
    ...     mintpy_timeseries_h5="geo_timeseries_ERA5_demErr.h5"
    ... )
    >>> ctx.source
    'mintpy'
    """
    source = source.lower().strip()
    if source not in {"hyp3", "mintpy", "nisar"}:
        raise ValueError("source must be one of: 'hyp3', 'mintpy', 'nisar'")

    # Enforce mutual exclusivity
    provided = [
        hyp3_tifs is not None,
        mintpy_timeseries_h5 is not None,
        nisar_gunw_h5s is not None,
    ]
    if sum(provided) > 1:
        raise ValueError(
            "Provide inputs for only one workflow: hyp3_tifs, "
            "mintpy_timeseries_h5, or nisar_gunw_h5s."
        )

    if source == "hyp3":
        if not hyp3_tifs:
            raise ValueError("source='hyp3' requires hyp3_tifs")

        from .hyp3_utils import (
            footprint_from_geotiffs,
            parse_unique_dates_from_hyp3_filenames,
        )

        dates = parse_unique_dates_from_hyp3_filenames(hyp3_tifs)

        # Validate dates are non-empty
        if not dates:
            raise ValueError(
                "No valid dates parsed from HyP3 filenames.\n"
                f"  Files: {[str(Path(p).name) for p in hyp3_tifs[:5]]}"
                + (f" ... ({len(hyp3_tifs)} total)" if len(hyp3_tifs) > 5 else "")
            )

        footprint = footprint_from_geotiffs(hyp3_tifs)

        if not _has_nonempty_geometry(footprint):
            raise ValueError(
                "HyP3 footprint_from_geotiffs() returned empty geometry."
            )

        return InsarContext(
            source="hyp3",
            dates=dates,
            footprint=footprint,
            source_files=tuple(str(Path(p)) for p in hyp3_tifs),
        )

    if source == "nisar":
        if not nisar_gunw_h5s:
            raise ValueError("source='nisar' requires nisar_gunw_h5s")

        from .nisar_utils import nisar_dates_from_gunw_h5, nisar_union_footprints

        dates = sorted(
            {
                date
                for gunw_h5 in nisar_gunw_h5s
                for date in nisar_dates_from_gunw_h5(gunw_h5)
            }
        )
        if not dates:
            raise ValueError("No valid dates parsed from NISAR GUNW files.")

        footprint = nisar_union_footprints(
            nisar_gunw_h5s,
            frequency=nisar_frequency,
            pol=nisar_pol,
            crs_out="EPSG:4326",
        )
        if not _has_nonempty_geometry(footprint):
            raise ValueError(
                "NISAR nisar_union_footprints() returned empty geometry."
            )

        return InsarContext(
            source="nisar",
            dates=dates,
            footprint=footprint,
            source_files=tuple(str(Path(p)) for p in nisar_gunw_h5s),
        )

    # source == "mintpy"
    if mintpy_timeseries_h5 is None:
        raise ValueError("source='mintpy' requires mintpy_timeseries_h5")

    from .mintpy_utils import (
        mintpy_dates_from_timeseries_h5,
        mintpy_footprint_from_timeseries_h5,
    )

    dates = mintpy_dates_from_timeseries_h5(mintpy_timeseries_h5)
    footprint = mintpy_footprint_from_timeseries_h5(
        mintpy_timeseries_h5, reference_slice=mintpy_reference_slice
    )

    if not _has_nonempty_geometry(footprint):
        raise ValueError(
            "MintPy footprint_from_timeseries_h5() returned empty geometry."
        )

    return InsarContext(
        source="mintpy",
        dates=dates,
        footprint=footprint,
        source_files=(str(Path(mintpy_timeseries_h5)),),
    )
