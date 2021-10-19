# coding: utf-8
# Created on 25/02/2021 11:26
# Author : matteo

# ====================================================
# imports
import h5pickle as h5py
from typing import Union

# ====================================================
# code
H5Group = Union[h5py.File, h5py.Group, h5py.Dataset]
