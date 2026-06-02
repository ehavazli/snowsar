# NISAR Streaming Guide

This guide covers the validated NISAR cloud-access workflow in `snowsar`:

- Stream a granule directly with `earthaccess`
- Cache a granule locally when you plan to reuse it
- Hand cached `.h5` files to the repo's existing path-based utilities

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
from snowsar.utils import setup_earthaccess_auth

setup_earthaccess_auth(persist=True)
```

This stores Earthdata credentials for future sessions when supported by
`earthaccess`.

## Quick Start

```python
from snowsar.utils import (
    cache_nisar_granule,
    open_nisar_h5_stream,
    search_nisar_data,
)

results = search_nisar_data(
    bbox=(-120, 37, -119, 38),
    start_date="2026-01-01",
    end_date="2026-03-31",
    processing_level="GUNW",
    provider="earthaccess",
    max_results=1,
)

with open_nisar_h5_stream(results[0]) as f:
    unwrapped = f[
        "science/LSAR/GUNW/grids/frequencyA/unwrappedInterferogram/HH/unwrappedPhase"
    ][()]

local_h5 = cache_nisar_granule(results[0])
print(local_h5)
```

## When To Stream Vs Cache

Stream when you are:

- inspecting metadata
- plotting one or two layers
- experimenting in a notebook

Cache locally when you are:

- revisiting the same granule multiple times
- running repeated notebook analyses
- using existing path-based utilities such as
  `extract_gunw_layers_to_geotiff_batch()`

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
from snowsar.utils import open_nisar_h5_stream

with open_nisar_h5_stream(results[0]) as f:
    coherence = f[
        "science/LSAR/GUNW/grids/frequencyA/unwrappedInterferogram/HH/coherenceMagnitude"
    ][()]
```

The validated streaming backend is `earthaccess`. Earlier experimental
`fsspec` examples were removed because they were not validated against NISAR's
credential flow.

## Local Cache

`cache_nisar_granule()` downloads a single granule into a deterministic cache
location and reuses the local file on later calls.

```python
from snowsar.utils import cache_nisar_granule, get_nisar_cache_path

cache_path = get_nisar_cache_path(results[0])
local_h5 = cache_nisar_granule(results[0])
assert local_h5 == cache_path
```

You can also choose a project-local cache:

```python
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
- `earthaccess` object handling in the streaming helper

Full remote integration still depends on having valid Earthdata credentials and
network access in the current environment.
