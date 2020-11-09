# coding: utf-8
# Created on 11/4/20 2:45 PM
# Author : matteo

# ====================================================
# imports
import numpy as np
import pandas as pd
from scipy import sparse
from typing import Union
from typing_extensions import Literal

# ====================================================
# types
ArrayLike_2D = Union[np.ndarray, sparse.spmatrix, pd.DataFrame]
ArrayLike_3D = Union[np.ndarray, sparse.spmatrix]

DTypes = {int: int,
          "int": np.int32,
          "int8": np.int8,
          "int16": np.int16,
          "int32": np.int32,
          "int64": np.int64,
          float: float,
          "float": np.float32,
          "float16": np.float16,
          "float34": np.float32,
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
          np.float128: np.float128}

DType = Union[str, int, float, np.dtype]
# DType = Union[str, int, float, np.int, np.int8, np.int16, np.int32, np.int64, np.float, np.float16, np.float32, np.float64, np.float128, np.int_, np.float_]

LoggingLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
