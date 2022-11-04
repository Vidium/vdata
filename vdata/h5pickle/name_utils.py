# coding: utf-8
# Created on 21/10/2021 15:40
# Author : matteo

# ====================================================
# imports
from typing import Union

from .h5pickle import File, Group, Dataset

# ====================================================
# code
H5Group = Union[File, Group, Dataset]


class H5Mode(str):
    READ = 'r'
    READ_WRITE = 'r+'
    WRITE_TRUNCATE = 'w'
    WRITE = 'w-'
    READ_WRITE_CREATE = 'a'
