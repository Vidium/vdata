# coding: utf-8
# Created on 11/12/20 4:51 PM
# Author : matteo

# ====================================================
# imports
from __future__ import annotations

import json
import logging
import shutil
import warnings
from numbers import Number

import pandas as pd
import numpy as np
from time import perf_counter
from pathlib import Path
from h5py import Dataset
from h5py import Group
from typing import Any, Callable, Collection, cast, Literal

# This import is need when evaluating a string to get a type # fixme : is there a better way ?
from numpy import int8, int16, int32, int64, float16, float32, float64, float128  # noqa: F401

import vdata
from vdata.name_utils import DType
from vdata.h5pickle.name_utils import H5Mode
from vdata.utils import get_value, repr_array
from .utils import parse_path, H5GroupReader
from vdata.vdataframe import VDataFrame
from vdata.time_point import TimePoint
from vdata.IO import VValueError, VTypeError, ShapeError
from vdata.core.tdf import TemporalDataFrame, BackedTemporalDataFrame
from vdata.h5pickle import File, Dataset as pklDataset, Group as pklGroup
from vdata.core.tdf.name_utils import H5Data, DEFAULT_TIME_POINTS_COL_NAME
from ..core.backed_dict import BackedDict


# ====================================================
# code
logger = logging.getLogger("vdata")


def spacer(nb: int) -> str:
    return "  " * (nb - 1) + "  " + "\u21b3" + " " if nb else ""


def get_dtype_from_string(dtype_str: str) -> type:
    """
    Parse the given string into a data type.

    Args:
        dtype_str: string to parse.

    Returns:
        A parsed data type.
    """
    if dtype_str.startswith("<class '"):
        return eval(dtype_str[8:-2])

    elif dtype_str in ("object", "category"):
        return str

    else:
        return eval(dtype_str)


def deprecated(func):
    def wrapper(*args, **kwargs):
        warnings.warn(
            "Reading old-style TemporalDataFrame, this will be deprecated in future versions.",
            DeprecationWarning,
            stacklevel=2,
        )
        return func(*args, **kwargs)

    return wrapper


# region read VData
# CSV file format ---------------------------------------------------------------------------------
def read_from_csv(
    directory: Path | str,
    dtype: "DType" = np.float32,
    time_list: Collection | DType | Literal["*"] | None = None,
    time_col: str | None = None,
    name: Any | None = None,
) -> "vdata.VData":
    """
    Function for reading data from csv datasets and building a VData object.

    Args:
        directory: a path to a directory containing csv datasets.
            The directory should have the format, for any combination of the following datasets :
                ⊦ layers
                    ⊦ <...>.csv
                ⊦ obsm
                    ⊦ <...>.csv
                ⊦ obsp
                    ⊦ <...>.csv
                ⊦ varm
                    ⊦ <...>.csv
                ⊦ varp
                    ⊦ <...>.csv
                ⊦ obs.csv
                ⊦ timepoints.csv
                ⊦ var.csv
        dtype: data type to force on the newly built VData object.
        time_list: time points for the dataframe's rows. (see TemporalDataFrame's documentation for more details.)
        time_col: if time points are not given explicitly with the 'time_list' parameter, a column name can be
            given. This column will be used as the time data.
        name: an optional name for the loaded VData object.
    """
    parsed_directory = parse_path(directory)

    # make sure the path exists
    if not parsed_directory.exists():
        raise VValueError(f"The path {parsed_directory} does not exist.")

    # load metadata if possible
    metadata = None

    if (parsed_directory / ".metadata.json").is_file():
        with open(parsed_directory / ".metadata.json", "r") as metadata_file:
            metadata = json.load(metadata_file)

    data = {
        "obs": None,
        "var": None,
        "timepoints": None,
        "layers": None,
        "obsm": None,
        "obsp": None,
        "varm": None,
        "varp": None,
        "dtype": dtype,
    }

    # import the data
    for f in sorted(parsed_directory.iterdir()):
        if f.name != ".metadata.json":
            logger.info(f"Got key : '{f.name}'.")

            if f.suffix == ".csv":
                if f.name in ("var.csv", "timepoints.csv"):
                    logger.info(f"{spacer(1)}Reading pandas DataFrame '{f.name[:-4]}'.")
                    data[f.name[:-4]] = pd.read_csv(
                        parsed_directory / f.name, index_col=0
                    )

                elif f.name in ("obs.csv",):
                    logger.info(
                        f"{spacer(1)}Reading TemporalDataFrame '{f.name[:-4]}'."
                    )
                    if time_list is None and time_col is None:
                        if metadata is not None:
                            this_time_col = metadata["obs"]["timepoints_column_name"]
                        else:
                            this_time_col = None
                    else:
                        this_time_col = time_col

                    data[f.name[:-4]] = read_TemporalDataFrame_from_csv(
                        parsed_directory / f.name,
                        time_list=time_list,
                        time_col_name=this_time_col,
                    )

            else:
                dataset_dict = {}
                logger.info(f"{spacer(1)}Reading group '{f.name}'.")

                for dataset in sorted((parsed_directory / f.name).iterdir()):
                    dataset = dataset.name

                    if f.name in ("layers", "obsm"):
                        logger.info(
                            f"{spacer(2)} Reading TemporalDataFrame {str(dataset)[:-4]}"
                        )
                        if time_list is None and time_col is None:
                            if metadata is not None:
                                this_time_col = metadata[f.name][str(dataset)[:-4]][
                                    "timepoints_column_name"
                                ]
                            else:
                                this_time_col = None
                        else:
                            this_time_col = time_col

                        dataset_dict[str(dataset)[:-4]] = (
                            read_TemporalDataFrame_from_csv(
                                parsed_directory / f.name / dataset,
                                time_list=time_list,
                                time_col_name=this_time_col,
                            )
                        )

                    elif f.name in ("varm", "varp"):
                        logger.info(f"{spacer(2)} Reading pandas DataFrame {dataset}")
                        dataset_dict[str(dataset)[:-4]] = pd.read_csv(
                            parsed_directory / f.name, index_col=0
                        )

                    else:
                        raise NotImplementedError

                data[f.name] = dataset_dict

    return vdata.VData(
        data["layers"],
        data["obs"],
        data["obsm"],
        data["obsp"],
        data["var"],
        data["varm"],
        data["varp"],
        data["timepoints"],
        dtype=data["dtype"],
        name=name,
    )


def read_from_dict(
    data: dict[str, dict[DType | str, np.ndarray | pd.DataFrame]],
    obs: pd.DataFrame | TemporalDataFrame | None = None,
    var: pd.DataFrame | None = None,
    timepoints: pd.DataFrame | None = None,
    dtype: DType | None = None,
    name: Any | None = None,
) -> "vdata.VData":
    """
    Load a simulation's recorded information into a VData object.

    If time points are not given explicitly, this function will try to recover them from the time point names in
        the data.
    For this to work, time points names must be strings with :
        - last character in (s, m, h, D, M, Y)
        - first characters convertible to a float
    The last character indicates the unit:
        - s : seconds
        - m : minutes
        - h : hours
        - D : days
        - M : months
        - Y : years

    Args:
        data: a dictionary of data types (RNA, Proteins, etc.) linked to dictionaries of time points linked to
            matrices of cells x genes.
        obs: a pandas DataFrame describing the observations (cells).
        var: a pandas DataFrame describing the variables (genes).
        timepoints: a pandas DataFrame describing the time points.
        dtype: the data type for the matrices in VData.
        name: an optional name for the loaded VData object.

    Returns:
        A VData object containing the simulation's data
    """
    if not isinstance(data, dict):
        raise VTypeError(
            "Data should be a dictionary with format : {data type: {time point: matrix}}"
        )

    _data = {}
    _timepoints: list[TimePoint] = []
    check_tp = False

    _index = obs.index if isinstance(obs, (pd.DataFrame, TemporalDataFrame)) else None
    logger.debug(f"Found index is : {repr_array(_index)}.")

    _columns = var.index if isinstance(var, pd.DataFrame) else None
    logger.debug(f"Found columns is : {repr_array(_columns)}.")

    _repeating_index = None
    _time_list = None

    for data_index, (data_type, TP_matrices) in enumerate(data.items()):
        if not isinstance(TP_matrices, dict):
            raise VTypeError(
                f"'{data_type}' in data should be a dictionary with format : {{time point: matrix}}"
            )

        # ---------------------------------------------------------------------------
        logger.debug(f"Loading layer '{data_type}'.")

        for matrix_index, matrix in TP_matrices.items():
            matrix_TP = TimePoint(matrix_index)

            if not isinstance(matrix, (np.ndarray, pd.DataFrame)) or matrix.ndim != 2:
                raise VTypeError(
                    f"Item at time point '{matrix_TP}' is not a 2D array-like object "
                    f"(numpy ndarray, pandas DatFrame)."
                )

            elif check_tp:
                if matrix_TP not in _timepoints:
                    raise VValueError("Time points do not match for all data types.")
            else:
                _timepoints.append(matrix_TP)

        check_tp = True

        _layer_data = np.vstack(list(TP_matrices.values()))
        # check data matches columns shape
        if _columns is not None and _layer_data.shape[1] != len(_columns):
            raise ShapeError(
                f"Layer '{data_type}' has {_layer_data.shape[1]} columns , should have "
                f"{len(_columns)}."
            )

        _loaded_data = pd.DataFrame(
            np.vstack(list(TP_matrices.values())), columns=_columns
        )

        if _index is not None and (li := len(_index)) != (ld := len(_loaded_data)):
            if ld % li == 0:
                if _repeating_index is not None and not _repeating_index:
                    raise VValueError("Inconsistency in repeating index.")

                _repeating_index = True
                _data_index = np.concatenate([_index for _ in range(ld // li)])

            else:
                raise VValueError(
                    f"Index of length {li} is incompatible with data '{data_type}' of shape"
                    f" {_loaded_data.shape}."
                )

        else:
            if _repeating_index is not None and _repeating_index:
                raise VValueError("Inconsistency in repeating index.")

            _repeating_index = False
            _data_index = _index

        if _time_list is None:
            _time_list = np.concatenate(
                [
                    np.repeat(_timepoints[matrix_index], len(matrix))  # type: ignore
                    for matrix_index, matrix in enumerate(TP_matrices.values())
                ]
            )

        logger.debug(f"Computed time list to be : {repr_array(_time_list)}.")

        _data[data_type] = TemporalDataFrame(
            data=_loaded_data,
            time_list=_time_list,
            index=_data_index,
            repeating_index=_repeating_index,
            name=data_type,
        )

        logger.info(f"Loaded layer '{data_type}' ({data_index+1}/{len(data)})")

    if obs is not None and not isinstance(obs, TemporalDataFrame):
        if _time_list is not None and (lt := len(_time_list)) != (lo := len(obs)):
            if lt % lo:
                raise VValueError(
                    f"Index of length {lt} is incompatible with 'obs' DataFrame of shape {obs.shape}."
                )

            obs = pd.concat([obs for _ in range(lt // lo)])

        obs = TemporalDataFrame(
            data=obs, time_list=_time_list, repeating_index=_repeating_index, name="obs"
        )

    # if time points is not given, build a DataFrame from time points found in 'data'
    if timepoints is None:
        timepoints = pd.DataFrame({"value": _timepoints})

    return vdata.VData(
        _data, obs=obs, var=var, timepoints=timepoints, dtype=dtype, name=name
    )


# HDF5 file format --------------------------------------------------------------------------------
def read_VData(
    file: Path | str, mode: Literal["r", "r+"] = "r", backup: bool = False
) -> "vdata.VData":
    """
    Function for reading data from a h5 file and building a VData object from it.

    Args:
        file: path to a h5 file.
        mode: reading mode : 'r' (read only) or 'r+' (read and write).
        backup: create a backup copy of the read h5 file in case something goes wrong ?
    """
    logger.debug(
        "\u23be read VData : begin -------------------------------------------------------- "
    )
    file = parse_path(file)

    if file.suffix != ".vd":
        raise VValueError("Cannot read file with suffix != '.vd'.")

    # make sure the path exists
    if not file.exists():
        raise VValueError(f"The path {file} does not exist.")

    if backup:
        backup_file_suffix = ".backup"
        backup_nb = 0

        while Path(str(file) + backup_file_suffix).exists():
            backup_nb += 1
            backup_file_suffix = f"({backup_nb}).backup"

        shutil.copy(file, str(file) + backup_file_suffix)
        logger.info(f"Backup file '{str(file) + backup_file_suffix}' was created.")

    data = {
        "obs": None,
        "var": None,
        "timepoints": None,
        "layers": None,
        "obsm": None,
        "obsp": None,
        "varm": None,
        "varp": None,
        "uns": {},
    }

    # import data from file
    importFile = H5GroupReader(File(str(file), mode))

    try:
        for key in importFile.keys():
            logger.info(f"Got key : '{key}'.")

            if key in (
                "obs",
                "var",
                "timepoints",
                "layers",
                "obsm",
                "obsp",
                "varm",
                "varp",
                "uns",
            ):
                type_ = importFile[key].attrs("type")
                data[key] = func_[type_](importFile[key], 1, mode)

            else:
                logger.warning(
                    f"Unexpected data with key {key} while reading file, skipping."
                )

        data["timepoints"]["value"] = [
            TimePoint(tp) for tp in data["timepoints"]["value"]
        ]

        name = importFile.attrs("name")
        dtype = importFile.attrs("dtype")

        new_VData = vdata.VData(
            data["layers"],
            data["obs"],
            data["obsm"],
            data["obsp"],
            data["var"],
            data["varm"],
            data["varp"],
            data["timepoints"],
            data["uns"],
            dtype=dtype if dtype != "None" else None,
            name=name,
            file=importFile,
            no_check=True,
        )

        logger.debug(
            "\u23bf read VData : end -------------------------------------------------------- "
        )

        return new_VData

    except Exception:
        importFile.close()
        raise


read = read_VData

# endregion


# region read TemporalDataFrame
def read_TemporalDataFrame(
    file: str | Path | H5Data, mode: H5Mode = H5Mode.READ
) -> BackedTemporalDataFrame:
    if isinstance(file, (str, Path)):
        file = File(file, mode=mode)

    if file.file.mode != mode:
        raise ValueError(
            f"Can't set mode of H5 file in {file.file.mode} mode to '{mode}'."
        )

    return BackedTemporalDataFrame(file)


def read_TemporalDataFrame_from_csv(
    file: Path,
    sep: str = ",",
    time_list: Collection[Number | str | TimePoint] | None = None,
    time_col_name: str | None = None,
) -> "TemporalDataFrame":
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
read_TDF_from_csv = read_TemporalDataFrame_from_csv

# endregion


# region read objects
def read_h5_dict(
    group: H5GroupReader, level: int = 1, mode: Literal["r", "r+"] = "r"
) -> BackedDict:
    """
    Function for reading a dictionary from a h5 file.

    Args:
        group: a H5GroupReader from which to read a dictionary.
        level: for logging purposes, the recursion depth of calls to a read_h5 function.
        mode:  mode for reading the tdf.
    """
    logger.info(f"{spacer(level)}Reading dict {group.name}.")

    start = perf_counter()
    data = BackedDict(group.group)
    end = perf_counter()

    logger.info(f"{spacer(level)}(took {end - start:1.4f} seconds)).")
    return data


def read_h5_VDataFrame(
    group: H5GroupReader, level: int = 1, *args, **kwargs
) -> VDataFrame:
    """
    Function for reading a pandas DataFrame from a h5 file.

    Args:
        group: a H5GroupReader from which to read a DataFrame.
        level: for logging purposes, the recursion depth of calls to a read_h5 function.
    """
    logger.info(f"{spacer(level)}Reading VDataFrame {group.name}.")

    start = perf_counter()
    # get index
    dataset_type = cast(H5GroupReader, group["index"]).attrs("type")
    index = func_[dataset_type](group["index"], level=level + 1)

    # get columns
    dataset_type = cast(H5GroupReader, group["columns"]).attrs("type")
    columns = func_[dataset_type](group["columns"], level=level + 1)

    # get data
    data = None

    if "data_numeric" in group.keys():
        logger.info(f"{spacer(level + 1)}Reading numeric data.")
        dataset_type = cast(H5GroupReader, group["data_numeric"]["columns"]).attrs(
            "type"
        )
        data_numeric_columns = func_[dataset_type](
            group["data_numeric"]["columns"], level=level + 2
        )

        dataset_type = cast(H5GroupReader, group["data_numeric"]["data"]).attrs("type")
        data_numeric_data = func_[dataset_type](
            group["data_numeric"]["data"], level=level + 2
        )

        data = pd.DataFrame(
            data_numeric_data, columns=data_numeric_columns, index=index
        )

    if "data_str" in group.keys():
        logger.info(f"{spacer(level + 1)}Reading non numeric data.")
        dataset_type = cast(H5GroupReader, group["data_str"]["columns"]).attrs("type")
        data_str_columns = func_[dataset_type](
            group["data_str"]["columns"], level=level + 2
        )

        dataset_type = cast(H5GroupReader, group["data_str"]["data"]).attrs("type")
        data_str_data = func_[dataset_type](group["data_str"]["data"], level=level + 2)

        if data is not None:
            for col_i, col in enumerate(data_str_columns):
                col_index = int(np.where(columns == col)[0][0])
                data.insert(col_index, col, data_str_data[:, col_i])

        else:
            data = pd.DataFrame(data_str_data, columns=data_str_columns, index=index)

    end = perf_counter()
    logger.info(f"{spacer(level)}(took {end - start:1.4f} seconds)).")

    return VDataFrame(data=data, index=index, columns=columns, file=group.group)


def read_h5_TemporalDataFrame(
    group: H5GroupReader, level: int = 1, mode: H5Mode = "r"
) -> BackedTemporalDataFrame:
    """
    Function for reading a TemporalDataFrame from a h5 file.

    Args:
        group: a H5GroupReader from which to read a TemporalDataFrame.
        level: for logging purposes, the recursion depth of calls to a read_h5 function.
        mode: mode for reading the tdf.
    """
    return read_TDF(group.group, mode=mode)


# TODO : drop H5GroupReader class
def read_h5_series(
    group: H5GroupReader,
    index: list | None = None,
    level: int = 1,
    log_func: Literal["debug", "info"] = "info",
    *args,
    **kwargs,
) -> pd.Series:
    """
    Function for reading a pandas Series from a h5 file.

    Args:
        group: an H5GroupReader from which to read a Series.
        index: an optional list representing the indexes for the Series.
        level: for logging purposes, the recursion depth of calls to a read_h5 function.
        log_func: function to use with the logger. Either 'info' or 'debug'.
    """
    getattr(logger, log_func)(f"{spacer(level)}Reading Series {group.name}.")

    start = perf_counter()
    # simple Series
    if group.isinstance(Dataset) or group.isinstance(pklDataset):
        data_type = get_dtype_from_string(group.attrs("dtype"))
        if data_type == TimePoint:
            values = [
                TimePoint(tp)
                for tp in read_h5_array(group, level=level + 1, log_func=log_func)
            ]

        else:
            values = list(
                map(data_type, read_h5_array(group, level=level + 1, log_func=log_func))
            )

        end = perf_counter()
        getattr(logger, log_func)(f"{spacer(level)}(took {end - start:1.4f} seconds)).")

        return pd.Series(values, index=index)

    # categorical Series
    elif group.isinstance(Group) or group.isinstance(pklGroup):
        # get data
        data_type = get_dtype_from_string(group.attrs("dtype"))
        categories = read_h5_array(
            cast(H5GroupReader, group["categories"]), level=level + 1, log_func=log_func
        )
        ordered = get_value(group.attrs("ordered"))
        values = list(
            map(
                data_type,
                read_h5_array(
                    cast(H5GroupReader, group["values"]),
                    level=level + 1,
                    log_func=log_func,
                ),
            )
        )

        end = perf_counter()
        getattr(logger, log_func)(f"{spacer(level)}(took {end - start:1.4f} seconds)).")

        return pd.Series(
            pd.Categorical(values, categories, ordered=ordered), index=index
        )

    # unexpected type
    else:
        raise VTypeError(f"Unexpected type {type(group)} while reading h5 file.")


# TODO : should not be needed anymore
def read_h5_array(
    group: H5GroupReader,
    level: int = 1,
    log_func: Literal["debug", "info"] = "info",
    *args,
    **kwargs,
) -> np.ndarray:
    """
    Function for reading a numpy array from a h5 file.
    If the imported array contains strings, as they were stored as bytes, they are converted back to strings.

    Args:
        group: a H5GroupReader from which to read an array.
        level: for logging purposes, the recursion depth of calls to a read_h5 function.
        log_func: function to use with the logger. Either 'info' or 'debug'.
    """
    getattr(logger, log_func)(f"{spacer(level)}Reading array {group.name}.")

    start = perf_counter()
    arr = group[()]

    if isinstance(arr, np.ndarray):
        # fix string arrays (object type to strings)
        if arr.dtype.type is np.object_:
            try:
                end = perf_counter()
                getattr(logger, log_func)(
                    f"{spacer(level)}(took {end - start:1.4f} seconds))."
                )

                return arr.astype(float)

            except ValueError:
                end = perf_counter()
                getattr(logger, log_func)(
                    f"{spacer(level)}(took {end - start:1.4f} seconds))."
                )

                return arr.astype(np.str_)

        end = perf_counter()
        getattr(logger, log_func)(f"{spacer(level)}(took {end - start:1.4f} seconds)).")

        return arr

    else:
        raise VTypeError(f"Group is not an array (type is '{type(arr)}').")


def read_h5_value(
    group: H5GroupReader, level: int = 1, *args, **kwargs
) -> str | int | float | bool | type:
    """
    Function for reading a value from a h5 file.

    Args:
        group: a H5GroupReader from which to read a value.
        level: for logging purposes, the recursion depth of calls to a read_h5 function.
    """
    logger.info(f"{spacer(level)}Reading value {group.name}.")

    if group.is_string():
        return group.as_string()

    return get_value(group[()])


def read_h5_path(group: H5GroupReader, level: int = 1, *args, **kwargs) -> Path:
    """
    Function for reading a Path from a h5 file.

    Args:
        group: a H5GroupReader from which to read a Path.
        level: for logging purposes, the recursion depth of calls to a read_h5 function.
    """
    logger.info(f"{spacer(level)}Reading Path {group.name}.")

    return Path(group.as_string())


def read_h5_None(_: H5GroupReader, level: int = 1, *args, **kwargs) -> None:
    """
    Function for reading 'None' from a h5 file.

    Args:
        _: a H5GroupReader from which to read a value.
        level: for logging purposes, the recursion depth of calls to a read_h5 function.
    """
    logger.info(f"{spacer(level)}Reading None.")
    return None


func_: dict[str, Callable] = {
    "dict": read_h5_dict,
    "VDF": read_h5_VDataFrame,
    "tdf": read_h5_TemporalDataFrame,
    "TDF": deprecated(read_h5_TemporalDataFrame),  # for backwards compatibility
    "series": read_h5_series,
    "array": read_h5_array,
    "value": read_h5_value,
    "type": read_h5_value,
    "path": read_h5_path,
    "None": read_h5_None,
}

# endregion
