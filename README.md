# VData - Annotated multivariate observation of timestamped data.

VData provides a container for simulated data across time points, heavily inspired by the AnnData project 
(https://anndata.readthedocs.io/en/latest/).

The VData object allows to store information about cells (observations), whose gene expression (variables) is 
measured over a time series (time points).

A VData object is build around layers (.layers). Each layer is a 3D matrix of : obs (cells) x var (genes) x time 
points (2D matrices at each time point can have different numbers of obs). All layers must have the same shape.

Around the layers, pandas DataFrames allow to describe variables (.var) and time points (.time_points), while custom 
TemporalDataFrames (dataframes extended by one dimension) describe observations (.obs). 

![doc/img.png](doc/VData.png)

## Multi-dimension annotation

The arrays .obsm and .varm allow to store annotations that need more than one dimension (e.g. PCA coordinates).

## Pairwise annotation

The arrays .obsp and .varp allow to store annotations on pairs of observations or variables (e.g. distance matrices).

## Unstructured annotations

The array .uns allows to store information not directly related to observations, variables and time points, which 
can take many different shapes (e.g. colors for plottings groups of observations). 
