# NISAR Streaming Guide

This guide covers the validated NISAR cloud-access workflow in `snowsar`:

- Stream a granule directly with `earthaccess`
- Cache bbox subsets locally when you plan to reuse them
- Extract bbox-scoped layers directly from streamed granules
- Cache full `.h5` granules only for path-based utilities that require files

## Install

Create or update the environment:

```bash
conda env update -f environment.yml
conda activate snowsar
pip install -e .
```

The streaming workflow relies on:

- `earthaccess` for search, auth, streaming, and download
- `asf-search` for optional ASF-side search/download helpers

## Authenticate

```python
from snowsar.utils import setup_asf_search_auth, setup_earthaccess_auth

setup_earthaccess_auth(persist=True)
asf_session = setup_asf_search_auth()
```

These helpers read Earthdata credentials from `~/.netrc` or the path in
`NETRC` when available. If no Earthdata Login entry exists, they prompt for
credentials interactively. `setup_earthaccess_auth(persist=True)` stores
Earthdata credentials for future sessions when supported by `earthaccess`.

For `provider="asf_search"`, searches use an authenticated ASF session and
include both public NISAR GUNW and the NISAR private ephemeral archive
collection when your Earthdata account has access.

## Quick Start

For a runnable notebook version of this workflow, see
`notebooks/NISAR_streaming_example.ipynb`.

```python
from snowsar.utils import (
    read_nisar_h5_bbox_cached,
    search_nisar_data,
)

use_subset_cache = True
subset_cache_dir = "./data/nisar_subset_cache"

results = search_nisar_data(
    bbox=(-120, 37, -119, 38),
    start_date="2026-01-01",
    end_date="2026-03-31",
    processing_level="GUNW",
    provider="earthaccess",
    max_results=1,
)

if not results:
    print("No NISAR granules found for the requested search.")
else:
    unwrapped, x, y, epsg, cache_path = read_nisar_h5_bbox_cached(
        results[0],
        "science/LSAR/GUNW/grids/frequencyA/unwrappedInterferogram/HH/unwrappedPhase",
        bbox=(-120, 37, -119, 38),
        cache_dir=subset_cache_dir,
        use_cache=use_subset_cache,
    )
    print(unwrapped.shape)
    print(cache_path)
```

## When To Stream Vs Cache

Stream when you are:

- inspecting metadata
- plotting one or two layers
- experimenting in a notebook

Cache bbox subsets when you are:

- revisiting the same granule multiple times
- running repeated notebook analyses

Cache full granules only when you are:

- using existing path-based utilities such as
  `extract_gunw_layers_to_geotiff_batch()`

Use streamed bbox extraction when you are:

- happy with a chosen search bbox and want GeoTIFF outputs without caching the full `.h5`
- extracting a mix of direct geogrid layers and interpolated/derived layers for one area of interest
- running the same bbox extraction across many returned granules

## Search

Use `earthaccess` by default:

```python
from snowsar.utils import search_nisar_data

results = search_nisar_data(
    bbox=(-120.5, 37.0, -118.5, 38.5),
    start_date="2026-01-01",
    end_date="2026-03-31",
    processing_level="GUNW",
    provider="earthaccess",
    max_results=10,
)
```

If you prefer ASF search metadata:

```python
results = search_nisar_data(
    bbox=(-120.5, 37.0, -118.5, 38.5),
    start_date="2026-01-01",
    end_date="2026-03-31",
    processing_level="GUNW",
    provider="asf_search",
    max_results=10,
)
```

## Streaming

`open_nisar_h5_stream()` accepts:

- an `earthaccess` `DataGranule`
- an ASF Search result
- a direct HTTPS URL
- a cached local path

Example:

```python
from snowsar.utils import open_nisar_h5_stream, read_nisar_h5_bbox

with open_nisar_h5_stream(results[0]) as f:
    coherence, x, y, epsg = read_nisar_h5_bbox(
        f,
        "science/LSAR/GUNW/grids/frequencyA/unwrappedInterferogram/HH/coherenceMagnitude",
        bbox=(-120, 37, -119, 38),
    )
```

The validated streaming backend is `earthaccess`. Earlier experimental
`fsspec` examples were removed because they were not validated against NISAR's
credential flow.

## Subset Cache

`read_nisar_h5_bbox_cached()` streams from the source HDF5, reads the requested
bbox window, and optionally caches only that subset as a GeoTIFF.

```python
from snowsar.utils import read_nisar_h5_bbox_cached

use_subset_cache = True
subset_cache_dir = "./data/nisar_subset_cache"

if not results:
    print("No NISAR granules found for the requested search.")
else:
    data, x, y, epsg, cache_path = read_nisar_h5_bbox_cached(
        results[0],
        "science/LSAR/GUNW/grids/frequencyA/unwrappedInterferogram/HH/coherenceMagnitude",
        bbox=(-120, 37, -119, 38),
        cache_dir=subset_cache_dir,
        use_cache=use_subset_cache,
    )
    print(cache_path)
```

## Streamed Bbox Extraction

`extract_gunw_layers_to_geotiff_bbox_streamed()` streams one granule, reads
direct 2D geogrid layers only over the requested bbox, and interpolates
radarGrid cube layers or computed aliases onto that bbox grid. This avoids
caching the full source granule locally.

For interpolated and derived layers, the environment still needs:

- `sardem`
- `GDAL`
- `scipy`
- `rasterio`

Example for one granule:

```python
from snowsar.utils import extract_gunw_layers_to_geotiff_bbox_streamed

layers = [
    "unwrappedPhase",
    "coherenceMagnitude",
    "ionospherePhaseScreen",
    "connectedComponents",
    "incidenceAngle",
    "localIncidenceAngle",
    "totalTroposphere",
    "losUnitVectorX",
    "losUnitVectorY",
    "elevationAngle",
    "slantRangeSolidEarthTidesPhase",
]

if not results:
    print("No NISAR granules found for the requested search.")
else:
    outputs = extract_gunw_layers_to_geotiff_bbox_streamed(
        results[0],
        bbox=(-120, 37, -119, 38),
        out_dir="./outputs/nisar_bbox_layers",
        frequency="A",
        pol="HH",
        layers=layers,
        provider="earthaccess",
        overwrite=False,
        max_retries=2,
        retry_delay=1.0,
    )
    print(outputs)
```

Interpolated or computed outputs follow the same naming convention used by the
local batch extractor, so some keys include `_interp`, for example:

- `incidenceAngle_interp`
- `localIncidenceAngle_interp`
- `totalTroposphere_interp`

Example across all returned results:

```python
from snowsar.utils import (
    extract_gunw_layers_to_geotiff_bbox_streamed,
    get_nisar_granule_name,
)

batch_outputs = {}
batch_failures = {}

for granule in results:
    granule_name = get_nisar_granule_name(granule)
    try:
        batch_outputs[granule_name] = extract_gunw_layers_to_geotiff_bbox_streamed(
            granule,
            bbox=(-120, 37, -119, 38),
            out_dir="./outputs/nisar_bbox_layers_batch",
            frequency="A",
            pol="HH",
            layers=layers,
            provider="earthaccess",
            max_retries=2,
            retry_delay=1.0,
        )
    except Exception as exc:
        batch_failures[granule_name] = str(exc)
```

The retry knobs handle transient 5xx-style remote read failures. In batch mode,
it is still worth catching exceptions per granule so one failed stream does not
abort the rest of the loop.

## Full Granule Cache

`cache_nisar_granule()` downloads a single granule into a deterministic cache
location and reuses the local file on later calls.

```python
from snowsar.utils import cache_nisar_granule, get_nisar_cache_path

if not results:
    print("No NISAR granules found for the requested search.")
else:
    cache_path = get_nisar_cache_path(results[0])
    local_h5 = cache_nisar_granule(results[0])
    assert local_h5 == cache_path
```

You can also choose a project-local cache:

```python
if not results:
    print("No NISAR granules found for the requested search.")
else:
    local_h5 = cache_nisar_granule(results[0], cache_dir="./data/nisar_cache")
```

## Batch Processing

Once cached, pass the local directory into existing batch utilities:

```python
from pathlib import Path

from snowsar.utils import cache_nisar_granule
from snowsar.utils.nisar_utils import extract_gunw_layers_to_geotiff_batch

local_h5 = cache_nisar_granule(results[0], cache_dir="./data/nisar_cache")

outputs = extract_gunw_layers_to_geotiff_batch(
    gunw_dir=Path(local_h5).parent,
    pattern=Path(local_h5).name,
    out_dir=Path("./outputs"),
    frequency="A",
    pol="HH",
    layers=["unwrappedPhase", "coherenceMagnitude"],
    warp=True,
    dst_epsg=32610,
    dst_res=90.0,
)
```

## Validation Status

The current implementation is validated for:

- local unit tests of granule normalization and cache reuse
- Earthdata and ASF auth/search wiring
- bbox subset reads and subset GeoTIFF cache behavior
- streamed bbox extraction for direct geogrid layers
- streamed bbox extraction routing for interpolated and derived layers
- transient retry handling for streamed bbox extraction

Full remote integration still depends on having valid Earthdata credentials and
network access in the current environment.
