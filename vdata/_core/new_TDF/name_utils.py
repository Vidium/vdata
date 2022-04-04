# coding: utf-8
# Created on 29/03/2022 11:44
# Author : matteo

# ====================================================
# imports
import numpy as np
from numbers import Number
from h5py import File, Group

from typing import Union, Collection

from vdata.new_time_point import TimePoint


# ====================================================
# code
SLICER = Union[Number, np.number, str, TimePoint,
               Collection[Union[Number, np.number, str, TimePoint]],
               range, slice, 'ellipsis']

H5Data = Union[File, Group]


class H5Mode(str):
    READ = 'r'
    READ_WRITE = 'r+'
    WRITE_TRUNCATE = 'w'
    WRITE = 'w-'
    READ_WRITE_CREATE = 'a'
