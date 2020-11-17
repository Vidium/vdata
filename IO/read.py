# coding: utf-8
# Created on 11/12/20 4:51 PM
# Author : matteo

# ====================================================
# imports
import os
import pickle
import pandas as pd
import numpy as np
from scipy import sparse
from pathlib import Path
from typing import Union, Optional, Dict, List, cast

from .errors import VValueError, VTypeError
from ..NameUtils import DType, ArrayLike_2D, LoggingLevel
from ..core.vdata import VData


# ====================================================
# code
def read(file: Union[Path, str]) -> VData:
    """
    Load a pickled VData object.
    Example :
    >>> import vdata
    >>> vdata.read("/path/to/file.p")

    :param file: path to a saved VData object
    """
    # make sure file is a path
    if not isinstance(file, Path):
        file = Path(file)

    # make sure the path exists
    if not os.path.exists(file):
        raise VValueError(f"The path {file} does not exist.")

    with open(file, 'rb') as save_file:
        vdata = pickle.load(save_file)

    return vdata


def read_from_GPU(data: Dict[str, Dict[Union[DType, str], ArrayLike_2D]], obs: Optional[pd.DataFrame] = None, var: Optional[pd.DataFrame] = None, time_points: Optional[pd.DataFrame] = None,
                  dtype: DType = np.float32, log_level: LoggingLevel = "INFO") -> VData:
    """
    Load a simulation's recorded information into a VData object.

    If time points are not given explicitly, this function will try to recover them from the time point names in the data.
    For this to work, time points names must be strings with :
        - last character in (s, m, h, D, M, Y)
        - first characters convertible to a float
    The last character indicates the unit:
        - s : second
        - m : minute
        - h : hour
        - D : day
        - M : month
        - Y : year

    :param data: a dictionary of data types (RNA, Proteins, etc.) linked to dictionaries of time points linked to matrices of cells x genes
    :param obs: a pandas DataFrame describing the observations (cells)
    :param var: a pandas DataFrame describing the variables (genes)
    :param time_points: a pandas DataFrame describing the time points
    :param dtype: the data type for the matrices in VData
    :param log_level: the logging level for the VData, in (DEBUG, INFO, WARNING, ERROR, CRITICAL)

    :return: a VData object containing the simulation's data
    """
    _data = {}
    _time_points = []
    check_tp = False

    if not isinstance(data, dict):
        raise VTypeError("Data should be a dictionary with format : {data type: {time point: matrix}}")

    else:
        for data_type, TP_matrices in data.items():
            if not isinstance(TP_matrices, dict):
                raise VTypeError(f"'{data_type}' in data should be a dictionary with format : {{time point: matrix}}")

            for matrix_index, matrix in TP_matrices.items():
                if not isinstance(matrix, (np.ndarray, sparse.spmatrix, pd.DataFrame)) or matrix.ndim != 2:
                    raise VTypeError(f"Item at time point '{matrix_index}' is not a 2D array-like object (numpy ndarray, scipy sparse matrix, pandas DatFrame).")

                elif check_tp:
                    if matrix_index not in _time_points:
                        raise VValueError("Time points do not match for all data types.")
                else:
                    _time_points.append(matrix_index)

            check_tp = True

            _data[data_type] = np.array([matrix for matrix in TP_matrices.values()])

        # if time points not given, try to guess them
        if time_points is None:
            if all([isinstance(_time_points[i], str) for i in range(len(_time_points))]):
                TP_df_data: Dict[str, List[Union[float, str]]] = {"value": [], "unit": []}
                del_unit = False

                for tp in _time_points:
                    tp = cast(str, tp)          # for typing
                    if tp.endswith("s"):
                        try:
                            TP_df_data["value"].append(float(tp[:-1]))
                            TP_df_data["unit"].append("second")
                        except ValueError:
                            del_unit = True
                            break
                    elif tp.endswith("m"):
                        try:
                            TP_df_data["value"].append(float(tp[:-1]))
                            TP_df_data["unit"].append("minute")
                        except ValueError:
                            del_unit = True
                            break
                    elif tp.endswith("h"):
                        try:
                            TP_df_data["value"].append(float(tp[:-1]))
                            TP_df_data["unit"].append("hour")
                        except ValueError:
                            del_unit = True
                            break
                    elif tp.endswith("D"):
                        try:
                            TP_df_data["value"].append(float(tp[:-1]))
                            TP_df_data["unit"].append("day")
                        except ValueError:
                            del_unit = True
                            break
                    elif tp.endswith("M"):
                        try:
                            TP_df_data["value"].append(float(tp[:-1]))
                            TP_df_data["unit"].append("month")
                        except ValueError:
                            del_unit = True
                            break
                    elif tp.endswith("Y"):
                        try:
                            TP_df_data["value"].append(float(tp[:-1]))
                            TP_df_data["unit"].append("year")
                        except ValueError:
                            del_unit = True
                            break
                    else:
                        del_unit = True
                        break

                if del_unit:
                    TP_df_data = {"value": _time_points}

                TP_df = pd.DataFrame(TP_df_data)

            else:
                TP_df = pd.DataFrame({"value": _time_points})

            return VData(_data, obs=obs, var=var, time_points=TP_df, dtype=dtype, log_level=log_level)

        else:
            return VData(_data, obs=obs, var=var, time_points=time_points, dtype=dtype, log_level=log_level)
