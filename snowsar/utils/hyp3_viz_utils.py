"""Visualization utilities for HyP3 InSAR products."""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Optional, Sequence, Tuple, Union

if TYPE_CHECKING:
    import matplotlib.axes
    import matplotlib.figure
    import numpy as np


def read_geotiff_as_array(
    path: Union[str, Path],
    mask_nodata: bool = True
) -> Tuple["np.ndarray", dict]:
    """
    Read GeoTIFF and return data array with nodata masked as NaN.

    Parameters
    ----------
    path : str or Path
        Path to GeoTIFF file
    mask_nodata : bool, default True
        Replace nodata values with NaN

    Returns
    -------
    data : np.ndarray
        2D array with nodata masked as NaN (if mask_nodata=True)
    profile : dict
        Rasterio profile metadata

    Examples
    --------
    >>> data, profile = read_geotiff_as_array("unwrapped_phase.tif")
    >>> data.shape
    (1000, 1000)
    """
    try:
        import numpy as np
        import rasterio
    except Exception as e:
        raise ImportError(
            "numpy and rasterio are required for GeoTIFF reading."
        ) from e

    with rasterio.open(path) as src:
        data = src.read(1).astype(np.float32)
        profile = src.profile

        if mask_nodata and src.nodata is not None:
            data = np.where(data == src.nodata, np.nan, data)

    return data, profile




def find_matching_products(
    unw_path: Path,
    wrapped_files: Sequence[Path],
    corr_files: Sequence[Path]
) -> Tuple[Optional[Path], Optional[Path]]:
    """
    Find wrapped phase and correlation files matching an unwrapped phase file.

    Matches based on parent directory (assumes products from same pair are in same folder).

    Parameters
    ----------
    unw_path : Path
        Path to unwrapped phase file
    wrapped_files : Sequence[Path]
        List of all wrapped phase files
    corr_files : Sequence[Path]
        List of all correlation files

    Returns
    -------
    wrapped_path : Path or None
        Matching wrapped phase file
    corr_path : Path or None
        Matching correlation file

    Examples
    --------
    >>> unw = Path("pair1/S1_20201215_20201227_unw_phase.tif")
    >>> wrapped = [Path("pair1/S1_20201215_20201227_wrapped_phase.tif")]
    >>> corr = [Path("pair1/S1_20201215_20201227_corr.tif")]
    >>> w, c = find_matching_products(unw, wrapped, corr)
    >>> w.name
    'S1_20201215_20201227_wrapped_phase.tif'
    """
    parent_dir = unw_path.parent

    wrapped_match = None
    for wf in wrapped_files:
        if wf.parent == parent_dir:
            wrapped_match = wf
            break

    corr_match = None
    for cf in corr_files:
        if cf.parent == parent_dir:
            corr_match = cf
            break

    return wrapped_match, corr_match


def plot_hyp3_trio(
    unw_path: Union[str, Path],
    wrapped_path: Optional[Union[str, Path]],
    corr_path: Optional[Union[str, Path]],
    *,
    unw_vlim_rad: Optional[Tuple[float, float]] = None,
    corr_vlim: Tuple[float, float] = (0, 1),
    figsize: Tuple[float, float] = (18, 6),
    save_path: Optional[Union[str, Path]] = None,
    dpi: int = 150,
) -> Tuple["matplotlib.figure.Figure", "np.ndarray"]:
    """
    Create 3-panel figure showing unwrapped phase, wrapped phase, and correlation.

    Parameters
    ----------
    unw_path : str or Path
        Path to unwrapped phase GeoTIFF
    wrapped_path : str or Path or None
        Path to wrapped phase GeoTIFF
    corr_path : str or Path or None
        Path to correlation GeoTIFF
    unw_vlim_rad : tuple or None, default None
        Color limits for unwrapped phase in radians (vmin, vmax)
    corr_vlim : tuple, default (0, 1)
        Color limits for correlation
    figsize : tuple, default (18, 6)
        Figure size (width, height) in inches
    save_path : str or Path or None, default None
        If provided, save figure to this path
    dpi : int, default 150
        DPI for saved figure

    Returns
    -------
    fig : matplotlib.figure.Figure
        The created figure
    axes : np.ndarray
        Array of axes objects

    Examples
    --------
    >>> fig, axes = plot_hyp3_trio(
    ...     "pair1/unw_phase.tif",
    ...     "pair1/wrapped_phase.tif",
    ...     "pair1/corr.tif"
    ... )
    """
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except Exception as e:
        raise ImportError(
            "matplotlib and numpy are required for visualization."
        ) from e

    from snowsar.utils.hyp3_utils import parse_date_pairs_from_hyp3_filenames

    # Read data
    unw_data, _ = read_geotiff_as_array(unw_path)
    wrapped_data, _ = read_geotiff_as_array(wrapped_path) if wrapped_path else (None, None)
    corr_data, _ = read_geotiff_as_array(corr_path) if corr_path else (None, None)

    # Parse dates from filename
    date_pairs = parse_date_pairs_from_hyp3_filenames([unw_path])
    if date_pairs:
        ref_date, sec_date = date_pairs[0]
        date_str = f"{ref_date.strftime('%Y%m%d')} - {sec_date.strftime('%Y%m%d')}"
    else:
        date_str = Path(unw_path).stem

    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=figsize)

    # --- Panel 1: Unwrapped Phase ---
    im1 = axes[0].imshow(
        unw_data,
        cmap="RdBu_r",
        vmin=unw_vlim_rad[0] if unw_vlim_rad else None,
        vmax=unw_vlim_rad[1] if unw_vlim_rad else None,
    )
    axes[0].set_title(f"Unwrapped Phase\n{date_str}", fontsize=12)
    axes[0].axis("off")
    cbar1 = plt.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)
    cbar1.set_label("Phase (radians)", rotation=270, labelpad=15)

    # --- Panel 2: Wrapped Phase ---
    if wrapped_data is not None:
        im2 = axes[1].imshow(
            wrapped_data,
            cmap="twilight",
            vmin=-np.pi,
            vmax=np.pi,
        )
        axes[1].set_title(f"Wrapped Phase (Interferogram)\n{date_str}", fontsize=12)
        axes[1].axis("off")
        cbar2 = plt.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)
        cbar2.set_label("Phase (radians)", rotation=270, labelpad=15)
        cbar2.set_ticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
        cbar2.set_ticklabels(['-π', '-π/2', '0', 'π/2', 'π'])
    else:
        axes[1].text(0.5, 0.5, "Wrapped phase\nfile not found",
                     ha="center", va="center", transform=axes[1].transAxes)
        axes[1].set_title(f"Wrapped Phase\n{date_str}", fontsize=12)
        axes[1].axis("off")

    # --- Panel 3: Correlation ---
    if corr_data is not None:
        im3 = axes[2].imshow(
            corr_data,
            cmap="gray",
            vmin=corr_vlim[0],
            vmax=corr_vlim[1],
        )
        axes[2].set_title(f"Correlation (Coherence)\n{date_str}", fontsize=12)
        axes[2].axis("off")
        cbar3 = plt.colorbar(im3, ax=axes[2], fraction=0.046, pad=0.04)
        cbar3.set_label("Correlation", rotation=270, labelpad=15)
    else:
        axes[2].text(0.5, 0.5, "Correlation\nfile not found",
                     ha="center", va="center", transform=axes[2].transAxes)
        axes[2].set_title(f"Correlation\n{date_str}", fontsize=12)
        axes[2].axis("off")

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        print(f"Saved figure to {save_path}")

    return fig, axes


def plot_correlation_histogram(
    corr_files: Sequence[Union[str, Path]],
    *,
    bins: int = 50,
    figsize: Tuple[float, float] = (10, 6)
) -> Tuple["matplotlib.figure.Figure", "matplotlib.axes.Axes"]:
    """
    Plot histogram of correlation values across all interferograms.

    Helps assess overall data quality and temporal decorrelation patterns.

    Parameters
    ----------
    corr_files : Sequence[str or Path]
        Paths to correlation GeoTIFF files
    bins : int, default 50
        Number of histogram bins
    figsize : tuple, default (10, 6)
        Figure size (width, height) in inches

    Returns
    -------
    fig : matplotlib.figure.Figure
    ax : matplotlib.axes.Axes

    Examples
    --------
    >>> corr_paths = ["pair1/corr.tif", "pair2/corr.tif"]
    >>> fig, ax = plot_correlation_histogram(corr_paths)
    """
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except Exception as e:
        raise ImportError(
            "matplotlib and numpy are required for histogram plotting."
        ) from e

    all_corr_values = []

    for cf in corr_files:
        data, _ = read_geotiff_as_array(cf)
        valid_data = data[~np.isnan(data)]
        all_corr_values.extend(valid_data.flatten())

    all_corr_values = np.array(all_corr_values)

    fig, ax = plt.subplots(figsize=figsize)

    ax.hist(all_corr_values, bins=bins, color="steelblue", alpha=0.7, edgecolor="black")

    # Add quality threshold lines
    ax.axvline(0.7, color="green", linestyle="--", label="Good (>0.7)")
    ax.axvline(0.4, color="orange", linestyle="--", label="Moderate (0.4-0.7)")

    # Statistics
    mean_corr = np.nanmean(all_corr_values)
    median_corr = np.nanmedian(all_corr_values)

    ax.set_xlabel("Correlation", fontsize=12)
    ax.set_ylabel("Frequency", fontsize=12)
    ax.set_title(
        f"Correlation Distribution Across {len(corr_files)} Interferograms\n"
        f"Mean: {mean_corr:.3f} | Median: {median_corr:.3f}",
        fontsize=13
    )
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig, ax
