# TemporalDataFrame

`TemporalDataFrame`s are equivalent to `pandas.DataFrame`s with an added `time` axis. 

!!! note
    An important distinction between
    TemporalDataFrames and regular pandas.DataFrames is that numerical and string data are stored in separate dedicated 
    arrays in TemporalDataFrames. The order of the columns will thus be primarily defined by the data type.

## Creation

### From a dictionary

Much like `pandas.DataFrame`s, TemporalDataFrames can be created from a dictionary of `column name` to `column values`.
Column values can be either integer, floating or string data.

``` py
>>> import vdata
>>> vdata.TemporalDataFrame({'col1': [1, 2, 3], 'col2': [4, 5, 6], 'col3': ['a', 'b', 'c']})
TemporalDataFrame No_Name
Time point : 0.0 hours
‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾
  Time-point     col1 col2     col3
0       0.0h  ｜  1.0  4.0  ｜    a
1       0.0h  ｜  2.0  5.0  ｜    b
2       0.0h  ｜  3.0  6.0  ｜    c
[3 rows x 3 columns]
```
### From a pandas DataFrame

TemporalDataFrames can also be created from a `pandas.DataFrame` :

``` py
>>> import pandas as pd
>>> df = pd.DataFrame({'col1': [1, 2, 3], 'col2': [4, 5, 6]})
>>> df
   col1  col2
0     1     4
1     2     5
2     3     6
>>>
>>> vdata.TemporalDataFrame(df)
TemporalDataFrame No_Name
Time point : 0.0 hours
‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾
  Time-point    col1 col2
0       0.0h  ｜  1.0  4.0
1       0.0h  ｜  2.0  5.0
2       0.0h  ｜  3.0  6.0
[3 rows x 2 columns]
```

### From a 2D array of values

Finally, TemporalDataFrames can be created from 2D arrays of values such as numpy arrays :

``` py
>>> import numpy as np
>>> data = np.arange(1, 7).reshape((2, 3)).T
>>> data
array([[1, 4],
       [2, 5],
       [3, 6]])
>>>
>>> vdata.TemporalDataFrame(data, columns=['col1', 'col2'])
TemporalDataFrame No_Name
Time point : 0.0 hours
‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾
  Time-point    col1 col2
0       0.0h  ｜  1.0  4.0
1       0.0h  ｜  2.0  5.0
2       0.0h  ｜  3.0  6.0
[3 rows x 2 columns]
```

### Defining time points

By default, all values are set to time point `0h`. You can however define a particular time point for each data row
with the `timepoints` parameter. It can be:

- a unique TimePoint (or object castable to a TimePoint) that applies to all data rows
- a sequence of TimePoints of the same length as the number of data rows

``` py
>>> data = np.arange(1, 13).reshape((2, 6)).T
>>> vdata.TemporalDataFrame(data, columns=['col1', 'col2'], timepoints=[1, 1, 2, 2, 2, 2])
TemporalDataFrame No_Name
Time point : 1.0 hour
‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾
  Time-point    col1 col2
0       1.0h  ｜  1.0  7.0
1       1.0h  ｜  2.0  8.0
[2 rows x 2 columns]

Time point : 2.0 hours
‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾
  Time-point    col1  col2
2       2.0h  ｜  3.0   9.0
3       2.0h  ｜  4.0  10.0
4       2.0h  ｜  5.0  11.0
5       2.0h  ｜  6.0  12.0
[4 rows x 2 columns]
```

### RepeatingIndex

The index of a TemporalDataFrame is usually a sequence of unique values, either numerical or string. In this case, there
can be different number of data rows at each time point in the TemporalDataFrame.

To accommodate for simulated data, where data for a single object (e.g. a cell) can be obtained at multiple time points,
VData provides a special [`RepeatingIndex`](../api/repeating_index.md) class for defining row indices that repeat exactly at each time points in a 
TemporalDataFrame. In this case, there must thus be the same number of data rows at each time point.

TODO: example

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
