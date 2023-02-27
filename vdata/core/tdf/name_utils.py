# coding: utf-8
# Created on 29/03/2022 11:44
# Author : matteo

# ====================================================
# imports
import numpy as np
from ch5mpy import File
from ch5mpy import Group
from numbers import Number

from typing import Union
from typing import Collection

from vdata.time_point import TimePoint


# ====================================================
# code
SLICER = Union[Number, np.number, str, TimePoint,
               Collection[Union[Number, np.number, str, TimePoint]],
               range, slice, 'ellipsis']

H5Data = Union[File, Group]

DEFAULT_TIME_POINTS_COL_NAME = 'Time-point'
