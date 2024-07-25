# TemporalDataFrame

`TemporalDataFrame`s are equivalent to `pandas.DataFrame`s with an added `time` axis. 

!!! note
    An important distinction between
    TemporalDataFrames and regular pandas.DataFrames is that numerical and string data are stored in separate dedicated 
    arrays in TemporalDataFrames. The order of the columns will thus be primarily defined by the data type.

## Creation


## Data storage

-> in RAM
-> in hdf5 file for memory efficiency (.data attribute, is_backed, is_closed)

## Getting and setting values

-> subsetting rules
-> shortcut for single column
-> loc, iloc, at, iat
-> views


## Axes (get and set)

-> timepoints
-> index
-> columns


## Operations

-> in-place : memory efficient
-> creating new TDFs

## Locking

## Compatibility with other data formats

-> to_dict
-> to_pandas
