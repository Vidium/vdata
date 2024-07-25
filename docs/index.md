---
hide:
  - navigation
---
# VData - Temporally annotated data manipulation and storage

**VData** is used for storing and manipulating multivariate observations of timestamped data.

![The VData structure](images/vdata_overview_light.png#only-light)
![The VData structure](images/vdata_overview_dark.png#only-dark)

VData is a Python package based on the well known [AnnData](https://anndata.readthedocs.io/en/latest/) project for 
manipulating and storing annotated datasets. VData adds an additional `time` dimension for easily handling timestamped 
datasets (e.g. simulated data). It is a complete reimplementation with important optimizations for high speed and low
memory footprint.

## Features

- explicit handling of timestamped data, especially suited for simulated single-cell datesets
- complete Python reimplementation based on [ h5py ](https://docs.h5py.org/en/latest)
- complete compatibility with the [ scverse ](https://scverse.org/) ecosystem 
- very fast loading of any dataset
- memory-efficient data manipulation (<<1GB) even for datasets of hundreds of GB.

## Installation

VData requires Python 3.9+

### Installing with pip
Stable releases are published on pip. Install the latest version with :

```shell
pip install vdata
```

Visit the Python Package Index at : [https://pypi.org/project/vdata](https://pypi.org/project/vdata)


### Installing from source
You can clone the development version from GitHub :

```shell
git clone git@github.com:Vidium/vdata.git
```

??? info "For developpers"
    VData and its dependencies are managed with [Poetry](https://python-poetry.org/).

    - To install the required dependencies, you can run `poetry install`.
    - To also install the development dependencies, you can run `poetry install --with dev,docs`

## Citation

You can cite the **VData** pre-print as :

> VData: Temporally annotated data manipulation and storage
> 
> Matteo Bouvier, Arnaud Bonnaffoux
> 
> bioRxiv 2023.08.29.555297; doi: https://doi.org/10.1101/2023.08.29.555297 
