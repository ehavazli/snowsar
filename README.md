# snowsar

**snowsar** is a software package for estimating Snow Water Equivalent (SWE) from InSAR time series data, supporting the characterization of seasonal snowpack dynamics and cryospheric processes.

## Features

- **NISAR Data Support**: Process NISAR GUNW Level-2 interferograms
- **Cloud Streaming**: Stream NISAR data directly from NASA Earthdata Cloud without downloading (see [STREAMING.md](STREAMING.md))
- **HyP3 Integration**: Work with ASF HyP3 processed Sentinel-1 InSAR products
- **MintPy Compatible**: Leverage MintPy time series analysis capabilities
- **SNOTEL Integration**: Access in-situ snow measurements for validation

---

## Snow Environment Setup

This repository is easiest to use from the provided Conda environment because several workflows rely on geospatial libraries that are more reliable from conda-forge than from a bare `pip` install.

### Create a New Environment

To create a new Conda environment named `snowsar` and install the required packages:

```bash
git clone git@github.com:ehavazli/snowsar.git;
conda env create -f environment.yml;
conda activate snowsar;
```

Alternatively, to create it manually:

```bash
git clone git@github.com:ehavazli/snowsar.git
conda create -n snowsar -c conda-forge rasterio sardem shapely geopandas contextily leafmap mintpy libgdal-hdf5 h5py earthaccess scipy pip
conda activate snowsar
```

After activating the environment, install the package into that environment:

```bash
pip install -e .
```

## Cloud Streaming for NISAR Data

This package supports **streaming NISAR data directly from NASA Earthdata Cloud** without downloading entire files. This is ideal for:

- Quick exploration of NISAR products
- Extracting specific layers (unwrapped phase, coherence, etc.)
- Cloud-native processing workflows
- Storage-constrained environments

**Quick start:**

```python
from snowsar.utils import (
    cache_nisar_granule,
    open_nisar_h5_stream,
    search_nisar_data,
)

# Search for NISAR data
results = search_nisar_data(
    bbox=(-120, 37, -119, 38),
    start_date="2026-01-01",
    processing_level="GUNW"
)

if not results:
    print("No granules found for the given search parameters")
else:
    # Stream data without downloading
    with open_nisar_h5_stream(results[0]) as f:
        unwrapped = f['science/LSAR/GUNW/grids/frequencyA/unwrappedInterferogram/HH/unwrappedPhase'][()]

    # Cache the same granule locally when you plan to reuse it
    local_h5 = cache_nisar_granule(results[0])

    # Re-open the cached file locally without going back through Earthdata
    with open_nisar_h5_stream(local_h5) as f:
        print(f["science"].keys())
```

See [STREAMING.md](STREAMING.md) for complete documentation.

### If the Environment Already Exists

If you already have the `snowsar` environment, you can install missing packages manually:

```bash
conda install -c conda-forge rasterio sardem shapely geopandas contextily leafmap libgdal-hdf5 h5py earthaccess scipy mintpy
pip install -e .
```

---

## License

This project is licensed under the terms of the [Apache License 2.0](LICENSE).

## Contact

For questions, please contact the repository maintainer.
