# coding: utf-8
# Created on 31/03/2022 11:25
# Author : matteo

# ====================================================
# imports
import pandas as pd
from h5py import File
from pathlib import Path
from numbers import Number

from typing import Union, Optional, Collection, TYPE_CHECKING

from vdata.name_utils import H5Mode
from .name_utils import H5Data, DEFAULT_TIME_POINTS_COL_NAME
from .dataframe import TemporalDataFrame

if TYPE_CHECKING:
    from vdata import TimePoint


# ====================================================
# code
def read_TemporalDataFrame(file: Union[str, Path, H5Data],
                           mode: H5Mode = H5Mode.READ) -> TemporalDataFrame:
    if isinstance(file, (str, Path)):
        file = File(file, mode=mode)

    if file.file.mode != mode:
        raise ValueError(f"Can't set mode of H5 file to '{mode}'.")

    return TemporalDataFrame(file)


def read_TemporalDataFrame_from_csv(file: Path,
                                    sep: str = ',',
                                    time_list: Optional[Collection[Union[Number, str, 'TimePoint']]] = None,
                                    time_col_name: Optional[str] = None) -> 'TemporalDataFrame':
    """
    Read a .csv file into a TemporalDataFrame.

    Args:
        file: a path to the .csv file to read.
        sep: delimiter to use for reading the .csv file.
        time_list: time points for the dataframe's rows. (see TemporalDataFrame's documentation for more details.)
        time_col_name: if time points are not given explicitly with the 'time_list' parameter, a column name can be
            given. This column will be used as the time data.

    Returns:
        A TemporalDataFrame built from the .csv file.
    """
    df = pd.read_csv(file, index_col=0, sep=sep)

    if time_col_name is None:
        time_col_name = DEFAULT_TIME_POINTS_COL_NAME

    if time_list is None and time_col_name == DEFAULT_TIME_POINTS_COL_NAME:
        time_list = df[DEFAULT_TIME_POINTS_COL_NAME].values.tolist()
        del df[time_col_name]
        time_col_name = None

    return TemporalDataFrame(df, time_list=time_list, time_col_name=time_col_name)


read_TDF = read_TemporalDataFrame
