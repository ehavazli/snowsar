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
git clone git@github.com:ehavazli/snowsar.git
cd snowsar
conda env create -f environment.yml
conda activate snowsar
```

Alternatively, if you want to create the Conda environment manually, install the core compiled/geospatial stack first:

```bash
git clone git@github.com:ehavazli/snowsar.git
cd snowsar
conda create -n snowsar -c conda-forge python=3.10 rasterio gdal sardem shapely geopandas contextily leafmap folium matplotlib mintpy libgdal-hdf5 h5py earthaccess scipy jupyterlab nb_conda_kernels pytest pip
conda activate snowsar
```

After activating the environment, install the package into that environment:

```bash
pip install -e .
```

That editable install pulls in the remaining Python package dependencies declared in `pyproject.toml`, including `ulmo`, `asf-search`, and `suds-jurko`.

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

If you already have the `snowsar` environment, you can install missing Conda packages manually and then refresh the Python package install:

```bash
conda install -c conda-forge rasterio gdal sardem shapely geopandas contextily leafmap folium matplotlib libgdal-hdf5 h5py earthaccess scipy mintpy jupyterlab nb_conda_kernels pytest
pip install -e .
```

---

## License

This project is licensed under the terms of the [Apache License 2.0](LICENSE).

## Contact

For questions, please contact the repository maintainer.
