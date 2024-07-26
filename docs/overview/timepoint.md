# Time management

VData aims at easily dealing with time-dependent data. The package comes with a variety of classes for efficiently
representing and operating on individual or collections of time points.

# [TimePoint](../api/timepoint.md#TimePoint)

This class is used to represent a single time point. It expects a value and an optional unit.

The **unit** must be one of `s` (seconds), `m` (minutes), `h` (hours), `D` (days), `M` (months) or `Y` (years).

The **value** can be :

- an integer or floating point value
- a string or bytes representation of a timepoint
- another TimePoint instance

In the case where the value is a string representation, it must follow the following rules :

- it can be casted to a float
- or it is of the format `<value><unit>` where `<value>` can be casted to a float and `<unit>` is one of the valid
units defined above.

``` py
>>> import vdata
>>> vdata.TimePoint(1)
1.0 hour
>>> vdata.TimePoint('0.1m')
0.1 minutes
>>> vdata.TimePoint(vd.TimePoint(0.5))
0.5 hours
```


`TimePoint` instances can be compared using regular comparison operators (==, <, >, <=, >=), added, subtracted, 
multiplied and divided together.

A `TimePoint` can be converted to another unit using the `TimePoint.value_as()` method.


# [TimePointArray](../api/timepoint.md#TimePointArray)

`TimePointArray`s are subclasses of numpy `ndarray`s specialized for storing time points of the same unit. They
implement regular operations on numpy.ndarrays and support [read/write operations to hdf5 files](../topics/hdf5.md).
TimePointArrays are limited to 1-dimensional arrays.

``` py
>>> vdata.timepoint.TimePointArray([0, 1, 2, 3], unit='s')
TimePointArray([0.0s, 1.0s, 2.0s, 3.0s])
```

TimePointArrays come with 2 utility functions :

- `atleast_1d` for converting the input `TimePoint` or collection of `TimePoint`s to a `TimePointArray`.
- `as_timepointarray` for casting the input time_list to a `TimePointArray`. The time_list argument can be a variety of
things : a TimePointArray, a TimePointRange or a collection of objects that could be casted to TimePoints.


# [TimePointRange](../api/timepoint.md#TimePointRange)

`TimePointRange`s are equivalent to regular ranges but for `TimePoints`. You can define a start and stop time point and 
a step time interval. You can then iterate through the TimePointRange as a regular range.


# [TimePointIndex](../api/timepoint.md#TimePointIndex)

`TimePointIndices` resemble `TimePointArray`s but are specialized for efficiently dealing with collections of ordered
time points. 

You are not likely to create an instance yourself but are used internally by [TemporalDataFrames](./tdf.md) to store the
`time` dimension which can be accessed with the `TemporalDataFrame.timepoints_index` attribute.

``` py
>>> timepoints = vdata.timepoint.TimePointArray([0, 1, 2])
>>> index = vdata.timepoints.TimePointIndex(timepoints, [3, 7, 10])
>>> index
TimePointIndex[0 --0.0h--> 3 --1.0h--> 7 --2.0h--> 10]
```

You can subset and iterate through a TimePointIndex. It support [read/write operations to hdf5 files](../topics/hdf5.md).

TimePointIndices are most usefull for creating masks as arrays of indices in the index where it matches a particular 
time points value.

``` py
>>> index.at(vdata.TimePoint('1h'))
array([3, 4, 5, 6])
```
