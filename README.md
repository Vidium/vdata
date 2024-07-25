# 🗂 VData

**VData** is used for storing and manipulating multivariate observations of timestamped data.

![The VData structure](docs/images/vdata_diagram.png)

It extends the [AnnData](https://anndata.readthedocs.io/en/latest/) object by adding the **time** dimension.

**Example** : The VData object allows to efficiently store information about cells (**observations**), whose gene 
expression (**variables**) is measured over multiple **time points**. It is build around layers (.layers). Each layer 
is a 3D matrix of : `obs` x `var` x `time points`. Around those layers, DataFrames allow to describe variables and 
time points, while custom TemporalDataFrames describe observations.

The **uns** dictionnary is used to store additional unstructure data.

More generally, VData objects can be used to store any timestamped datasets where annotation of observations and
variables is required.

## 🌟 Features

- complete Python reimplementation based on [ h5py ](https://docs.h5py.org/en/latest)
- very fast loading of any dataset
- memory-efficient data manipulation (<1GB) even for datasets of hundreds of GB.
- explicit handling of timestamped data, especially suited for simulated single-cell datesets
- complete compatibility with the [ scverse ](https://scverse.org/) ecosystem 

## 👁 Overview

### General

The `vdata` library exposes the actual **VData** object alongside with the **TemporalDataFrame** object which extends
the common `pandas.DataFrame` to a third `time` axis.

**VData** objects can be created from in-RAM objects such as `AnnData`, `TemporalDataFrame`, `pandas.DataFrame` or 
mappings of `<layer name>`:`DataFrame`. 

It is also possible to load data from a `VData` or an `AnnData` saved as a 
[ hdf5 website ](https://www.hdfgroup.org/solutions/hdf5/) file or in `csv` format.

> 🔵 **Note**
> An important distinction with `AnnData` is that when a **VData** is backed on (read from) an hdf5 file, the *whole* 
> object is only loaded on-demand and by small chunks of data. As a result, VData objects will always consume small 
amounts of RAM and will be very fast to read.

### Layers and data annotation

The bulk of the data is stored in `TemporalDataFrames`, themselves stacked up in the **layers** dictionnary. Data is
thus represented as `observations` x `variables` x `time points` dataframes. Observation indices can either be unique 
at each time point or strictly the same (e.g. to store simulated data where a single cell can be recorded multiple 
times).

![TemporalDataFrames, one with unique observations and one with identical observations at all timepoints](docs/images/TDF_diagram.png)

Three additional dataframes are used for annotating the observations (**obs**), variables (**var**) and timepoints
(**timepoints**).

### Multi-dimension annotation

There are two additional mappings for storing multi-dimensional annotations (i.e. that require more than one column to
be stored). These are the `obsm` and `varm` mappings, which respectively contain TemporalDataFrames and pandas 
DataFrames.

> 🟢 **Example**
> You can store PCA or UMAP coordinates in obsm.

### Pairwise annotation

The last two mappings (`obsp` and `varp`) contain pariwise annotations : data in square matrices of `obs` x `obs` 
or `var` x `var`.

> 🟢 **Example** 
> You can store distance values between observations in obsp.

## 📀 Installation

VData requires Python 3.9+

### pip installation (stable)

```shell
pip install vdata
```

### using git (latest)

```shell
git clone git@github.com:Vidium/vdata.git
```

## 📑 Documentation

See the complete documentation at [INCOMING].

Read the VData article at https://www.biorxiv.org/content/10.1101/2023.08.29.555297

## 🖋 Citation

You can cite the **VData** pre-print as :

> VData: Temporally annotated data manipulation and storage
> 
> Matteo Bouvier, Arnaud Bonnaffoux
> 
> bioRxiv 2023.08.29.555297; doi: https://doi.org/10.1101/2023.08.29.555297 
