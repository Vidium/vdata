# coding: utf-8
# Created on 31/03/2022 11:25
# Author : matteo

# ====================================================
# imports
from h5py import File
from pathlib import Path

from typing import Union

from .name_utils import H5Data, H5Mode
from .dataframe import TemporalDataFrame


# ====================================================
# code
def read_TemporalDataFrame(file: Union[str, Path, H5Data],
                           mode: H5Mode = H5Mode.READ) -> TemporalDataFrame:
    if isinstance(file, (str, Path)):
        file = File(file, mode=mode)

    if file.mode != mode:
        raise ValueError(f"Can't set mode of H5 file to '{mode}'.")

    return TemporalDataFrame(file)


read_TDF = read_TemporalDataFrame
