# coding: utf-8
# Created on 11/4/20 2:45 PM
# Author : matteo

"""
Global types for typing in the vdata module.
"""

# ====================================================
# imports
from typing import Union, Literal

import numpy as np

# ====================================================
# types
_HAS_FLOAT128 = getattr(np, "float128", None) is not None

DTypes = {
    int: int,
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
    np.integer: np.int64,
    np.int8: np.int8,
    np.int16: np.int16,
    np.int32: np.int32,
    np.int64: np.int64,
    np.floating: np.float64,
    np.float16: np.float16,
    np.float32: np.float32,
    np.float64: np.float64,
    object: object,
}

if _HAS_FLOAT128:
    DTypes |= {
        "float128": np.float128,
        np.float128: np.float128,
    }

DType = Union[int, float, np.integer, np.floating, object]
StrDType = Literal[
    "int",
    "int8",
    "int16",
    "int32",
    "int64",
    "float",
    "float16",
    "float32",
    "float64",
    "float128",
]

LoggingLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
LoggingLevels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
