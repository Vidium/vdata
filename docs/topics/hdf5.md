# Writing TemporalDataFrames and VData objects

``` py title="writing objects to hdf5 files"
>>> tdf = vdata.TemporalDataFrame(...)
>>>
>>> tdf.write('/path/to/file.h5')  # with a file path
>>>
>>> d = ch.H5Dict(...)
>>> tdf.write(d)  # with a Ch5mpy Group or H5Dict
```

# Reading saved TemporalDataFrames and VData objects

``` py title="reading objects from hdf5 files"
>>> vdata.TemporalDataFrame.read('/path/to/file.h5')  # with a file path
>>>
>>> d = ch.H5Dict(...)
>>> vdata.TemporalDataFrame.read(d)  # with a Ch5mpy Group of H5Dict
```

By default, TemporalDataFrame and VData objects are read in `READ` (`r`) mode, a.k.a read-only mode and thus
data modification will not be possible.
To modify objects, you must open them in `READ_WRITE` (`r+`) mode :

``` py
>>> vdata.TemporalDataFrame.read('/path/to/file.h5', mode=vdata.H5Mode.READ_WRITE)
```

