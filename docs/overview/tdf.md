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

``` py
>>> vdata.RepeatingIndex([1, 2, 3], repeats=2)
RepeatingIndex([1, 2, 3, 1, 2, 3], dtype='int64', repeating=True)
```

## Data storage

When TemporalDataFrames are created as described in the section above, all the data is hosted in RAM. Manipulating very
large datasets can however become too much of a burden and they need to be stored elsewhere. TemporalDataFrames thus 
rely on the hdf5 file format for saving and their data. Once saved to hdf5 format, datasets can be read one small chunk 
at a time to keep memory usage low. In that case, we say the TemporalDataFrame is `backed` on the hdf5 file.

``` py title="saving a TemporalDataFrame to hdf5 file"
>>> tdf
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

>>> tdf.write('tdf.h5')
>>> tdf
Backed TemporalDataFrame No_Name
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

>>> tdf.is_backed
True
```

Once the `TemporalDataFrame.write()` method has been called, the underlying data is dropped from the RAM and replacced
with a [`Ch5mpy`](https://gitlab.vidium.fr/vidium/ch5mpy) `H5Dict` wrapper around the hdf5 data. It can be accessed with
the `TemporalDataFrame.data` attribute.

``` py title="access the underlying data structure"
>>> tdf.data
H5Dict{
        columns_numerical: H5Array(['col1', 'col2'], shape=(2,), dtype='<U4'),
        columns_string: H5Array([], shape=(0,), dtype='<U1'),
        index: H5Array([0, 1, 2, 3, 4, 5], shape=(6,), dtype=int64),
        numerical_array: H5Array([[1.0, 7.0],
                 [2.0, 8.0],
                 [3.0, 9.0],
                 [4.0, 10.0],
                 [5.0, 11.0],
                 [6.0, 12.0]], shape=(6, 2), dtype=float64),
        string_array: H5Array([], shape=(6, 0), dtype='<U1'),
        timepoints_index: TimePointIndex[0 --1.0h--> 2 --2.0h--> 6]
}
```

See the [hdf5 topic](../topics/hdf5.md) for more examples on reading from and writing to hdf5 files.

## Getting and setting values

Data in TemporalDataFrames is stored in two separate arrays, one dedicated to numerical data and one to string data.
Both arrays can be accessed with the `TemporalDataFrame.values_num` and `TemporalDataFrame.values_str` attributes.
These attributes are best for interacting directly with the data, even when the TemporalDataFrame is backed on a hdf5
file.

For convenience, a concatenated array can be obtained from the `TemporalDataFrame.values` attribute and should be used 
in most cases.

### Subsetting rules

Because TemporalDataFrames are 3D objects, the subsetting rules differ slightly from AnnData's. Subsetting is performed
with the square brackets operator and with one to three selection objects. The first selects `time points`, the second 
`indices` and the third `columns`.

``` py
>>> tdf
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

>>> tdf['1h']
View of TemporalDataFrame No_Name
Time point : 1.0 hour
‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾
  Time-point    col1 col2
0       1.0h  ｜  1.0  7.0
1       1.0h  ｜  2.0  8.0
[2 rows x 2 columns]


>>> tdf[:, [0, 2, 4]]
View of TemporalDataFrame No_Name
Time point : 1.0 hour
‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾
  Time-point    col1 col2
0       1.0h  ｜  1.0  7.0
[1 rows x 2 columns]

Time point : 2.0 hours
‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾
  Time-point    col1  col2
2       2.0h  ｜  3.0   9.0
4       2.0h  ｜  5.0  11.0
[2 rows x 2 columns]


>>> tdf[:, :, 'col1']
View of TemporalDataFrame No_Name
Time point : 1.0 hour
‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾
  Time-point    col1
0       1.0h  ｜  1.0
1       1.0h  ｜  2.0
[2 rows x 1 columns]

Time point : 2.0 hours
‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾
  Time-point    col1
2       2.0h  ｜  3.0
3       2.0h  ｜  4.0
4       2.0h  ｜  5.0
5       2.0h  ｜  6.0
[4 rows x 1 columns]

```

For selecting unique columns, an equivalent of the syntax `tdf[:, :, <col_name>]` is `tdf.col_name`.

!!! note
    This only works for string column names which do not contain space characters.

### loc, iloc, at and iat methods

As it is done in `pandas.DataFrame`s, methods `loc`, `iloc`, `at` and `iat` allow to select indices and columns. 
No time points selection is possible with those methods.

## Axes (get and set)

TemporalDataFrame are 3D object with axes being : time points, indices and columns. Together, all 3 axis defined the
TemporalDataFrame's shape `(<nb time points>, <nb indices>, <nb columns>)`.

### Timepoints

``` py
tdf.timepoints                      # unique time points as a TimePointArray
tdf.timepoints_index                # time point column as a TimePointIndex
tdf.timepoints_column               # time point column as a TimePointArray
tdf.timepoints_column_str           # time point column casted to strings
tdf.timepoints_column_numerical     # time point column casted to floats
tdf.n_timepoints                    # number of unique time points
```

### Index

``` py
tdf.index                           # index getter (as RepeatingIndex) and setter
tdf.index_at(timepoint)             # index as ReapeatingIndex at a given time point
tdf.n_index                         # number of index rows
tdf.n_index_at(timepoint)           # number of index rows at a given time point
```

### Columns

``` py
tdf.columns                         # column getter and setter
tdf.columns_num                     # columns getter and setter for the numerical data
tdf.columns_str                     # columns getter and setter for the string data
tdf.n_columns                       # total number of columns
tdf.n_columns_str                   # number of numerical columns
tdf.n_columns_num                   # number of string columns
```

## Operations

Usual operations are possible on `TemporalDataFrame`s : addition, subtractio, multiplication and division by a
numerical value or by another TemporalDataFrame (in which case the operation will be applied element by element and 
time points, indices and columns must match).

!!! warning
    When dealing with large datasets, one must keep in mind that most operations will create a brand new 
    in-RAM TemporalDataFrame instance.
    Only in-place operations `+=`, `-=`, `*=` and `/=` will modify the values directly.

## Locking

Indices and columns axes can be locked to prevent modifications (e.g. renaming indices or columns).

``` py
tdf.has_locked_indices                  # boolean
tdf.has_locked_columns                  # boolean
tdf.lock_indices()
tdf.lock_columns()
tdf.unlock_indices()
tdf.unlock_columns()
```

## Compatibility with other data formats

For compatibility with `pandas.DataFrame`, `TemporalDataFrame.to_dict()` and `TemporalDataFrame.to_pandas()` methods 
exist for converting a TemporalDataFrame to dict or pandas format respectively.
