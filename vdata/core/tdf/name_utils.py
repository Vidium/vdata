# coding: utf-8
# Created on 29/03/2022 11:44
# Author : matteo

# ====================================================
# imports
import numpy as np
from numbers import Number
from vdata.h5pickle import File
from vdata.h5pickle import Group

from typing import Union, Collection

from vdata.time_point import TimePoint


# ====================================================
# code
SLICER = Union[Number, np.number, str, TimePoint,
               Collection[Union[Number, np.number, str, TimePoint]],
               range, slice, 'ellipsis']

H5Data = Union[File, Group]

DEFAULT_TIME_POINTS_COL_NAME = 'Time-point'
