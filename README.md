# snowsar

**snowsar** is a software package for estimating Snow Water Equivalent (SWE) from InSAR time series data, supporting the characterization of seasonal snowpack dynamics and cryospheric processes.

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
