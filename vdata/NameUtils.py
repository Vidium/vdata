# coding: utf-8
# Created on 11/4/20 2:45 PM
# Author : matteo

# ====================================================
# imports
import h5py
import numpy as np
import pandas as pd
from typing import Union, Sequence
from typing_extensions import Literal

import vdata
from . import utils

# ====================================================
# types
DataFrame = Union[pd.DataFrame, 'vdata.TemporalDataFrame']

DTypes = {int: int,
          "int": np.int32,
          "int8": np.int8,
          "int16": np.int16,
          "int32": np.int32,
          "int64": np.int64,
          float: float,
          "float": np.float32,
          "float16": np.float16,
          "float32": np.float32,
          "float64": np.float64,
          "float128": np.float128,
          np.int: np.int64,
          np.int_: np.int64,
          np.int8: np.int8,
          np.int16: np.int16,
          np.int32: np.int32,
          np.int64: np.int64,
          np.float: np.float64,
          np.float_: np.float64,
          np.float16: np.float16,
          np.float32: np.float32,
          np.float64: np.float64,
          np.float128: np.float128,
          np.object: np.object}

DType = Union[int, float, np.int_, np.float_, np.object]
str_DType = Union[Literal["int", "int8", "int16", "int32", "int64", "float", "float16", "float32", "float64",
                          "float128"]]

LoggingLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
LoggingLevels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]

H5Group = Union[h5py.File, h5py.Group, h5py.Dataset]

Slicer = Union[Sequence[Union[int, float, str, bool, utils.TimePoint]], range, slice]
PreSlicer = Union[int, float, str, utils.TimePoint, Slicer, 'ellipsis']
