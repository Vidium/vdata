# coding: utf-8
# Created on 11/20/20 4:30 PM
# Author : matteo

# ====================================================
# imports


# ====================================================
# code
# ====================================================
# imports
import h5py
import pandas as pd
from functools import singledispatch
from typing import Dict

from .logger import generalLogger


# ====================================================
# code
@singledispatch
def write_data(data, save_file: h5py.File, key: str) -> None:
    """
    TODO
    """
    generalLogger.warning(f"H5 writing not yet implemented for data of type '{type(data)}'.")


@write_data.register
def write_DataFrame(data: pd.DataFrame, save_file: h5py.File, key: str) -> None:
    """
    obs, var, time_points
    """
    print('save DF', key)


@write_data.register(dict)
def write_Dict(data: Dict, save_file: h5py.File, key: str) -> None:
    """
    TODO
    """
    print('save dict', key)


@write_data.register
def write_Dict(data: None, save_file: h5py.File, key: str) -> None:
    """
    TODO
    """
    print('save None', key)
