# snowsar

**snowsar** is a software package for estimating Snow Water Equivalent (SWE) from InSAR time series data, supporting the characterization of seasonal snowpack dynamics and cryospheric processes.

---

## Snow Environment Setup

This repository requires a Conda environment with specific geospatial libraries and additional packages for accessing SNOTEL data.

### Create a New Environment

To create a new Conda environment named `snowsar` and install the required packages:

```bash
git clone git@github.com:ehavazli/snowsar.git;
conda env create -f environment.yml;
conda activate snowsar;
pip install -e .
```

Alternatively, to create it manually:

```bash
git clone git@github.com:ehavazli/snowsar.git
conda create -n snowsar rasterio sardem shapely geopandas contextily leafmap mintpy libgdal-hdf5 pip
conda activate snowsar
```

After activating the environment, install the additional Python packages needed for SNOTEL data access:

```bash
pip install ulmo "suds-jurko @ https://github.com/drivendataorg/suds-jurko-wheel/releases/download/v0.6/suds_jurko-0.6-py3-none-any.whl"
```

### If the Environment Already Exists

If you already have the `snowsar` environment, you can install missing packages manually:

```bash
conda install rasterio sardem shapely geopandas contextily leafmap libgdal-hdf5
pip install ulmo "suds-jurko @ https://github.com/drivendataorg/suds-jurko-wheel/releases/download/v0.6/suds_jurko-0.6-py3-none-any.whl"
```

---

## License

This project is licensed under the terms of the [Apache License 2.0](LICENSE).

## Contact

For questions, please contact the repository maintainer.
