from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Union

import geopandas as gpd
import pandas as pd


@dataclass(frozen=True)
class InsarContext:
    """
    Standardized output for downstream workflows (SNOTEL, plotting, etc.)
    """

    source: str  # "hyp3" or "mintpy"
    dates: List[pd.Timestamp]  # normalized (midnight) timestamps
    footprint: gpd.GeoDataFrame  # single-row GeoDataFrame with polygon geometry


def build_insar_context(
    *,
    source: str,
    hyp3_tifs: Optional[Sequence[Union[str, Path]]] = None,
    mintpy_timeseries_h5: Optional[Union[str, Path]] = None,
    mintpy_reference_slice: Optional[str] = None,
) -> InsarContext:
    """
    Build (dates, footprint) using the correct workflow based on `source`.

    Rules:
      - source must be "hyp3" or "mintpy"
      - hyp3 workflow uses ONLY hyp3_tifs
      - mintpy workflow uses ONLY mintpy_timeseries_h5 (+ optional slice name)
      - providing both hyp3 and mintpy inputs is an error

    Parameters
    ----------
    source : str
        "hyp3" or "mintpy"
    hyp3_tifs : list[path]
        HyP3 clipped GeoTIFFs (e.g., *unw_phase_clipped.tif)
    mintpy_timeseries_h5 : path
        MintPy geo_timeseries*.h5 file
    mintpy_reference_slice : str, optional
        Slice name to use for footprint derivation; if None, uses first slice.

    Returns
    -------
    InsarContext
    """
    source = source.lower().strip()
    if source not in {"hyp3", "mintpy"}:
        raise ValueError("source must be one of: 'hyp3', 'mintpy'")

    # Enforce mutual exclusivity
    if hyp3_tifs is not None and mintpy_timeseries_h5 is not None:
        raise ValueError(
            "Do not provide both hyp3_tifs and mintpy_timeseries_h5. "
            "HyP3 and MintPy are mutually exclusive workflows."
        )

    if source == "hyp3":
        if not hyp3_tifs:
            raise ValueError("source='hyp3' requires hyp3_tifs")

        from .hyp3_utils import (
            footprint_from_geotiffs,
            parse_unique_dates_from_hyp3_filenames,
        )

        dates = parse_unique_dates_from_hyp3_filenames(hyp3_tifs)
        footprint = footprint_from_geotiffs(hyp3_tifs)

        if footprint.empty:
            raise ValueError("HyP3 footprint_from_geotiffs() returned empty geometry.")

        return InsarContext(source="hyp3", dates=dates, footprint=footprint)

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

    if footprint.empty:
        raise ValueError(
            "MintPy footprint_from_timeseries_h5() returned empty geometry."
        )

    return InsarContext(source="mintpy", dates=dates, footprint=footprint)
