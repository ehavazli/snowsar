"""
Streaming and local-cache utilities for NISAR cloud access.

The validated streaming path uses ``earthaccess`` to open NISAR HDF5 products
directly from NASA Earthdata Cloud. For repeated use, granules can also be
cached locally and reused by the repo's path-based processing utilities.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional, Union
from urllib.parse import urlparse

if TYPE_CHECKING:
    import h5py

logger = logging.getLogger(__name__)

NISAR_S3_CREDENTIALS_ENDPOINT = (
    "https://nisar.asf.earthdatacloud.nasa.gov/s3credentials"
)
DEFAULT_NISAR_CACHE_DIR = Path.home() / ".cache" / "snowsar" / "nisar"


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
        auth = earthaccess.login(persist=persist)
        if not getattr(auth, "authenticated", False):
            raise RuntimeError(
                "Earthdata authentication failed. "
                "Please check your credentials at https://urs.earthdata.nasa.gov/"
            )
        logger.info("Earthdata authentication successful")
    else:
        logger.info("Using existing Earthdata credentials")


def _is_earthaccess_granule(granule: Any) -> bool:
    """Return True for earthaccess-style DataGranule objects."""
    return hasattr(granule, "data_links") and callable(granule.data_links)


def _is_asf_result(granule: Any) -> bool:
    """Return True for ASF Search result objects."""
    return hasattr(granule, "get_urls") and callable(granule.get_urls)


def _granule_urls(granule: Any) -> list[str]:
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


def _granule_basename(granule: Any) -> str:
    """Return a stable local filename for a granule or URL."""
    for url in _granule_urls(granule):
        parsed = urlparse(url)
        candidate = Path(parsed.path).name if parsed.scheme else Path(url).name
        if candidate:
            return candidate

    if _is_earthaccess_granule(granule):
        umm = getattr(granule, "umm", None)
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

    text = str(granule).strip()
    if not text:
        raise ValueError("Could not determine a filename for the requested granule.")
    return Path(text).name or text.replace("/", "_")


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
        return h5py.File(opened[0], mode=mode)
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
    if isinstance(granule, (str, Path)):
        source_path = Path(granule)
        if source_path.exists() and source_path.suffix == ".h5":
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
    max_results: int = 100,
):
    """
    Search for NISAR products using earthaccess or ASF Search.
    """
    if provider == "earthaccess":
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

    collection_map = {
        "GUNW": "NISAR_L2_GUNW_BETA_V1",
        "GSLC": "NISAR_L2_GSLC_BETA_V1",
        "GCOV": "NISAR_L2_GCOV_BETA_V1",
        "RSLC": "NISAR_L1_RSLC_BETA_V1",
    }
    short_name = collection_map.get(processing_level.upper())
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
    if start_date:
        search_params["start"] = start_date
    if end_date:
        search_params["end"] = end_date
    if bbox:
        search_params["bbox"] = ",".join(str(x) for x in bbox)

    results = asf.search(**search_params)
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

        session = asf.ASFSession()
        downloaded: list[Path] = []
        for result in granule_results:
            filename = _granule_basename(result)
            target_file = output_dir / filename
            if target_file.exists() and not overwrite:
                continue
            result.download(path=str(output_dir), session=session)
            downloaded.append(target_file)
        return downloaded

    raise ValueError(
        f"Unknown provider: {provider}. Choose 'earthaccess' or 'asf_search'."
    )
