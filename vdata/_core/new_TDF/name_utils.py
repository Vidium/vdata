# coding: utf-8
# Created on 29/03/2022 11:44
# Author : matteo

# ====================================================
# imports
from h5py import File, Group

from typing import Union, Collection

from vdata import TimePoint


# ====================================================
# code
H5Data = Union[File, Group]
TIMEPOINTS_SLICER = Union[None,
                          Union[int, float, str, TimePoint],
                          Collection[Union[int, float, str, TimePoint]],
                          range,
                          slice]


class H5Mode(str):
    READ = 'r'
    READ_WRITE = 'r+'
    WRITE_TRUNCATE = 'w'
    WRITE = 'w-'
    READ_WRITE_CREATE = 'a'
