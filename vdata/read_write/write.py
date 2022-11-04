# coding: utf-8
# Created on 11/20/20 4:30 PM
# Author : matteo

# ====================================================
# imports
from __future__ import annotations

import os
import json
import shutil

import anndata.compat
import numpy as np
import pandas as pd
import scipy.sparse as sp
from tqdm.autonotebook import tqdm
from pathlib import Path
from h5py import string_dtype
from functools import singledispatch

from typing import TYPE_CHECKING

import vdata
from vdata.core.backed_dict import BackedDict
from vdata.h5pickle.name_utils import H5Mode
from vdata.read_write.utils import parse_path, H5GroupReader
from vdata.vdataframe import VDataFrame, ViewVDataFrame
from vdata.IO import generalLogger, VPathError, VValueError
from vdata.core.tdf.name_utils import DEFAULT_TIME_POINTS_COL_NAME, H5Data
from vdata.h5pickle import H5Group, File
from vdata.core.attribute_proxy.attribute import NONE_VALUE
from vdata.core.tdf.base import BaseTemporalDataFrame

if TYPE_CHECKING:
    from vdata import VData, ViewVData


# ====================================================
# code
def spacer(nb: int) -> str:
    return "  "*(nb-1) + "  " + u'\u21B3' + " " if nb else ''


# region write VData
def write_vdata(obj: VData | ViewVData,
                file: str | Path | None,
                show_progress: bool = True) -> None:
    """
    Save this VData object in HDF5 file format.

    Args:
        obj: VData object to save into an h5 file.
        file: path to save the VData.
        show_progress: print a progress bar while saving objects in this VData ? (default: True)
    """
    file = parse_path(file)
    if file is not None and file.suffix != '.vd':
        generalLogger.warning("File suffix must be '.vd', it has been changed.")
        file = file.with_suffix('.vd')

    if obj.is_backed:
        if obj.file.mode == 'r+':
            update_vdata(obj)

            if file is not None:
                shutil.copy(obj.file.filename, file)

        else:
            raise VValueError("Cannot save backed VData in 'r' mode !")

    else:
        # make sure the path exists
        if not os.path.exists(os.path.dirname(file)):
            os.makedirs(os.path.dirname(file))

        # TODO : better handling of already existing files
        if file.exists():
            os.remove(file)

        save_file = File(str(file), mode=H5Mode.READ_WRITE_CREATE)

        nb_items_to_write = len(obj.layers) + 1 + len(obj.obsm) + len(obj.obsp) + 1 + len(obj.varm) + len(obj.varp) + \
            1 + len(obj.uns)
        progressBar = tqdm(total=nb_items_to_write, desc=f'writing VData {obj.name}', unit='object') if show_progress \
            else None

        save_file.attrs['name'] = obj.name
        save_file.attrs['dtype'] = str(obj.dtype)

        # save layers
        write_data(obj.layers.data, save_file, 'layers', progress=progressBar)
        # save obs
        write_data(obj.obs, save_file, 'obs', progress=progressBar)
        write_data(obj.obsm.data, save_file, 'obsm', progress=progressBar)
        write_data(obj.obsp.data, save_file, 'obsp', progress=progressBar)
        # save var
        write_data(obj.var, save_file, 'var', progress=progressBar)
        write_data(obj.varm.data, save_file, 'varm', progress=progressBar)
        write_data(obj.varp.data, save_file, 'varp', progress=progressBar)
        # save time points
        write_data(obj.timepoints, save_file, 'timepoints', progress=progressBar)
        # save uns
        write_data(obj.uns, save_file, 'uns', progress=progressBar)

        if isinstance(obj, vdata.VData):
            obj.file = H5GroupReader(save_file)
            obj.uns = BackedDict(save_file['uns'])

        if show_progress:
            progressBar.clear()


# TODO : get rid of function
def update_vdata(obj: VData) -> None:
    """
    Update data from a backed VData object on the h5 file.

    Args:
        obj: VData object to save into a h5 file.
    """
    # save layers -------------------------------------------------------------
    generalLogger.info('Saving Layers')
    # remove deleted layers from the h5 file
    for layer_name in obj.file.group['layers'].keys():
        if layer_name not in obj.layers.keys():
            generalLogger.info(f"{spacer(1)}Removing layer {layer_name}")
            del obj.file.group['layers'][layer_name]
            obj.file.group.flush()

    # write new layers
    for layer_name in obj.layers.keys():
        if layer_name not in obj.file.group['layers'].keys():
            generalLogger.info(f"{spacer(1)}Saving layer {layer_name}")
            write_TemporalDataFrame(data=obj.layers[layer_name], group=obj.file.group['layers'], key=layer_name)

    # update obs --------------------------------------------------------------
    generalLogger.info('Saving obs')
    if obj.obs.empty or not obj.obs.is_backed:
        generalLogger.info(f"{spacer(1)}Removing obs")
        del obj.file.group['obs']
        obj.file.group.flush()

        write_TemporalDataFrame(data=obj.obs, group=obj.file.group, key='obs')

    # update obsm -------------------------------------------------------------
    generalLogger.info('Saving obsm')
    # remove deleted obsm datasets from the h5 file
    for obsm_name in obj.file.group['obsm'].keys():
        if obsm_name not in obj.obsm.keys():
            generalLogger.info(f"{spacer(1)}Removing obsm {obsm_name}")
            del obj.file.group['obsm'][obsm_name]
            obj.file.group.flush()

    # write new obsm datasets
    for obsm_name in obj.obsm.keys():
        if obsm_name not in obj.file.group['obsm'].keys():
            generalLogger.info(f"{spacer(1)}Saving obsm {obsm_name}")
            write_TemporalDataFrame(data=obj.obsm[obsm_name], group=obj.file.group['obsm'], key=obsm_name)

    # update obsp -------------------------------------------------------------
    del obj.file.group['obsp']
    obj.file.group.flush()
    write_data(obj.obsp.data, obj.file.group, 'obsp')

    # update var --------------------------------------------------------------
    del obj.file.group['var']
    obj.file.group.flush()
    write_data(obj.var, obj.file.group, 'var')

    del obj.file.group['varm']
    obj.file.group.flush()
    write_data(obj.varm.data, obj.file.group, 'varm')

    del obj.file.group['varp']
    obj.file.group.flush()
    write_data(obj.varp.data, obj.file.group, 'varp')

    # update time points ------------------------------------------------------
    del obj.file.group['timepoints']
    obj.file.group.flush()
    write_data(obj.timepoints, obj.file.group, 'timepoints')

    obj.file.group.flush()


def write_vdata_to_csv(obj: VData | ViewVData,
                       directory: str | Path,
                       sep: str = ",",
                       na_rep: str = "",
                       index: bool = True,
                       header: bool = True) -> None:
    """
    Save a VData object into csv files in a directory.

    Args:
        obj: a VData object to save into csv files.
        directory: path to a directory for saving the matrices
        sep: delimiter character
        na_rep: string to replace NAs
        index: write row names ?
        header: Write col names ?
    """
    directory = parse_path(directory)

    # make sure the directory exists and is empty
    if not os.path.exists(directory):
        os.makedirs(directory)

    if len(os.listdir(directory)):
        raise VPathError("The directory is not empty.")

    # save metadata
    with open(directory / ".metadata.json", 'w') as metadata:
        json.dump({"obs": {"timepoints_column_name": obj.obs.timepoints_column_name},
                   "obsm": {obsm_TDF_name:
                            {"timepoints_column_name": obsm_TDF.timepoints_column_name if
                                obsm_TDF.timepoints_column_name is not None else DEFAULT_TIME_POINTS_COL_NAME}
                            for obsm_TDF_name, obsm_TDF in obj.obsm.items()},
                   "layers": {layer_TDF_name:
                              {"timepoints_column_name": layer_TDF.timepoints_column_name if
                                  layer_TDF.timepoints_column_name is not None else DEFAULT_TIME_POINTS_COL_NAME}
                              for layer_TDF_name, layer_TDF in obj.layers.items()}}, metadata)

    # save matrices
    generalLogger.info(f"{spacer(1)}Saving TemporalDataFrame obs")
    obj.obs.to_csv(directory / "obs.csv", sep, na_rep, index=index, header=header)
    generalLogger.info(f"{spacer(1)}Saving TemporalDataFrame var")
    obj.var.to_csv(directory / "var.csv", sep, na_rep, index=index, header=header)
    generalLogger.info(f"{spacer(1)}Saving TemporalDataFrame time-points")
    obj.timepoints.to_csv(directory / "timepoints.csv", sep, na_rep, index=index, header=header)

    for dataset in (obj.layers, obj.obsm, obj.obsp, obj.varm, obj.varp):
        generalLogger.info(f"{spacer(1)}Saving {dataset.name}")
        dataset.to_csv(directory, sep, na_rep, index, header, spacer=spacer(2))

    if obj.uns is not None:
        generalLogger.warning(f"'uns' data stored in VData '{obj.name}' cannot be saved to a csv.")

# endregion


# region write TDF
def write_array_in_TDF(array: np.ndarray,
                       file: H5Data,
                       key: str) -> None:
    """
    Write a numpy array to a H5 file.
    """
    if np.issubdtype(array.dtype, np.number) or array.dtype == np.dtype('bool'):
        dtype = array.dtype

    else:
        dtype = string_dtype()
        array = array.astype(str).astype('O')

    if key in file.keys():
        if file[key].dtype == dtype:
            file[key].resize((len(array),))
            file[key].astype(dtype)
            file[key][()] = array
            return

        del file[key]

    file.create_dataset(key, data=array, dtype=dtype, chunks=True, maxshape=(None,))


def write_array_chunked_in_TDF(array: np.ndarray,
                               file: H5Data,
                               key: str) -> None:
    """
    Write a numpy array to a H5 file to create a chunked dataset.
    """
    if np.issubdtype(array.dtype, np.number) or array.dtype == np.dtype('bool'):
        dtype = array.dtype

    else:
        dtype = string_dtype()
        array = array.astype(str).astype('O')

    if key in file.keys():
        if file[key].chunks is None:
            del file[key]
            file.create_dataset(key, data=array, dtype=dtype, chunks=True, maxshape=(None, None))

        else:
            file[key].resize(array.shape)
            file[key].astype(dtype)
            file[key][()] = array

    else:
        file.create_dataset(key, data=array, dtype=dtype, chunks=True, maxshape=(None, None))


# TODO : move writing to Dataset to the dataset_proxy package
# TODO : smarter copies of h5 datasets to new h5 file
def write_TDF(TDF: BaseTemporalDataFrame,
              file: H5Data) -> None:
    """
    Write a TemporalDataFrame to a H5 file.

    Args:
        TDF: A TemporalDataFrame to write.
        file: A H5 File or Group in which to save the TemporalDataFrame.
    """
    # save attributes
    file.attrs['type'] = 'tdf'
    file.attrs['name'] = TDF.name
    file.attrs['locked_indices'] = TDF.has_locked_indices
    file.attrs['locked_columns'] = TDF.has_locked_columns
    file.attrs['repeating_index'] = TDF.has_repeating_index
    file.attrs['timepoints_column_name'] = NONE_VALUE if TDF.timepoints_column_name is None else \
        TDF.timepoints_column_name

    # save index
    write_array_in_TDF(TDF.index[:], file, 'index')

    # save columns numerical
    write_array_in_TDF(TDF.columns_num[:], file, 'columns_numerical')

    # save columns string
    write_array_in_TDF(TDF.columns_str[:], file, 'columns_string')

    # save timepoints data
    write_array_in_TDF(TDF.timepoints_column_str[:], file, 'timepoints')

    # save numerical data
    write_array_chunked_in_TDF(TDF.values_num, file, 'values_numerical')

    # save string data
    write_array_chunked_in_TDF(TDF.values_str, file, 'values_string')

# endregion


# region write objects

@singledispatch
def write_data(data,
               group: H5Group,
               key: str,
               key_level: int = 0,
               **kwargs) -> None:
    """
    This is the default function called for writing data to an h5 file.
    Using singledispatch, the correct write_<type> function is called depending on the type of the 'data' parameter.
    If no write_<type> implementation is found, this function defaults and raises a Warning indicating that the
    data could not be saved.

    Args:
        data: data to write.
        group: an h5py Group or File to write into.
        key: a string for identifying the data.
        key_level: for logging purposes, the recursion depth of calls to write_data.
    """
    generalLogger.info(f"{spacer(key_level)}Saving object {key}")
    generalLogger.warning(f"H5 writing not yet implemented for data of type '{type(data)}'.")


@write_data.register(dict)
@write_data.register(BackedDict)
@write_data.register(anndata.compat.OverloadedDict)
def write_Dict(data: dict | BackedDict | anndata.compat.OverloadedDict,
               group: H5Group,
               key: str,
               key_level: int = 0,
               progress: tqdm | None = None) -> None:
    """
    Function for writing dictionaries to the h5 file.
    It creates a group for storing the keys and recursively calls write_data to store them.
    """
    generalLogger.info(f"{spacer(key_level)}Saving dict {key}")

    data_dict = dict(data)

    if str(key) in group.keys():
        del group[str(key)]

    grp = group.create_group(str(key))
    grp.attrs['type'] = 'dict'

    for dict_key, value in data_dict.items():
        write_data(value, grp, dict_key, key_level=key_level+1, progress=progress)


@write_data.register(VDataFrame)
@write_data.register(pd.DataFrame)
def write_VDataFrame(data: VDataFrame | pd.DataFrame,
                     group: H5Group,
                     key: str,
                     key_level: int = 0,
                     progress: tqdm | None = None) -> None:
    """
    Function for writing VDataFrames to the h5 file. Each VDataFrame is stored in a group, containing the index and the
    columns as Series.
    Used for obs, var, time-points.
    """
    generalLogger.info(f"{spacer(key_level)}Saving VDataFrame {key}")

    # convert pandas DataFrames to VDataFrames for writing
    data = VDataFrame(data)

    df_group = group.create_group(str(key))
    df_group.attrs['type'] = 'VDF'

    # save index
    write_data(data.index, df_group, 'index', key_level=key_level + 1)

    # save columns
    write_data(data.columns, df_group, 'columns', key_level=key_level + 1)

    # save data
    data_numeric = data.select_dtypes(include=[np.number, bool])
    if not data_numeric.empty:
        generalLogger.info(f"{spacer(key_level + 1)}Saving numeric data")
        df_data_numeric_group = df_group.create_group('data_numeric')
        write_data(data_numeric.columns, df_data_numeric_group, 'columns', key_level=key_level + 2)
        write_data(data_numeric.values.astype(np.float64), df_data_numeric_group, 'data', key_level=key_level + 2)

    data_str = data.select_dtypes(exclude=[np.number, bool])
    if not data_str.empty:
        generalLogger.info(f"{spacer(key_level + 1)}Saving non numeric data")
        df_data_str_group = df_group.create_group('data_str')
        write_data(data_str.columns, df_data_str_group, 'columns', key_level=key_level + 2)
        write_data(data_str.values.astype(str), df_data_str_group, 'data', key_level=key_level + 2)

    if progress is not None:
        progress.update()


@write_data.register(ViewVDataFrame)
def write_ViewVDataFrame(data: ViewVDataFrame,
                         group: H5Group,
                         key: str,
                         key_level: int = 0,
                         progress: tqdm | None = None) -> None:
    generalLogger.info(f"{spacer(key_level)}Converting ViewVDataFrame {key} to VDataFrame")
    write_VDataFrame(data.to_pandas(), group, key, key_level)

    if progress is not None:
        progress.update()


@write_data.register(BaseTemporalDataFrame)
def write_TemporalDataFrame(data: BaseTemporalDataFrame,
                            group: H5Group,
                            key: str,
                            key_level: int = 0,
                            progress: tqdm | None = None) -> None:
    generalLogger.info(f"{spacer(key_level)}Saving TemporalDataFrame {key}")
    data.write(group.create_group(str(key)))

    if progress is not None:
        progress.update()


@write_data.register(pd.Series)
@write_data.register(pd.Index)
def write_series(series: pd.Series | pd.Index,
                 group: H5Group,
                 key: str,
                 key_level: int = 0,
                 progress: tqdm | None = None,
                 **kwargs) -> None:
    """
    Function for writing pd.Series to the h5 file. The Series are expected to belong to a group (a DataFrame or in uns).
    """
    generalLogger.info(f"{spacer(key_level)}Saving Series {key}")

    # Series of strings
    if series.dtype == object:
        group.create_dataset(str(key),
                             data=np.array(list(map(str, series.values)), dtype='object'),
                             dtype=string_dtype(encoding='utf-8'),
                             chunks=True,
                             maxshape=(None,))

    # Series of categorical data
    elif pd.api.types.is_categorical_dtype(series):
        series_group = group.create_group(str(key))
        # save values
        values = pd.Series(np.array(series.values))
        write_data(values, series_group, "values", key_level=key_level + 1)
        # save categories
        # noinspection PyUnresolvedReferences
        categories = np.array(series.values.categories, dtype='U')
        write_data(categories, series_group, "categories", key_level=key_level + 1)
        # save ordered
        # noinspection PyUnresolvedReferences
        series_group.attrs["ordered"] = series.values.ordered

    # Series of regular data
    else:
        group[str(key)] = series.values

    group[str(key)].attrs['type'] = 'series'

    if 'data_type' in kwargs:
        group[str(key)].attrs['dtype'] = str(kwargs['data_type'])

    else:
        group[str(key)].attrs['dtype'] = str(series.dtype)

    if progress is not None:
        progress.update()


@write_data.register
def write_array(data: np.ndarray,
                group: H5Group,
                key: str,
                key_level: int = 0,
                progress: tqdm = None) -> None:
    """
    Function for writing np.arrays to the h5 file.
    """
    generalLogger.info(f"{spacer(key_level)}Saving array {key}")

    if data.dtype.type == np.str_:
        group.create_dataset(str(key), data=data.astype('S'))
    else:
        group[str(key)] = data

    group[str(key)].attrs['type'] = 'array'

    if progress is not None:
        progress.update()


@write_data.register
def write_sparse_matrix(data: sp.spmatrix,
                        group: H5Group,
                        key: str,
                        key_level: int = 0,
                        progress: tqdm = None) -> None:
    """
    Function for writing scipy sparse matrices to the h5 file.
    """
    # TODO : fully handle sparse matrices in the future
    # cast matrix to dense
    data_dense = data.todense()

    write_data(data_dense, group, key, key_level=key_level)

    if progress is not None:
        progress.update()


@write_data.register(list)
def write_list(data: list,
               group: H5Group,
               key: str,
               key_level: int = 0,
               progress: tqdm = None) -> None:
    """
    Function for writing lists to the h5 file.
    """
    generalLogger.info(f"{spacer(key_level)}Saving list {key}")
    write_data(np.array(data), group, key, key_level=key_level+1)

    if progress is not None:
        progress.update()


@write_data.register(str)
@write_data.register(np.str_)
@write_data.register(int)
@write_data.register(np.integer)
@write_data.register(float)
@write_data.register(np.floating)
@write_data.register(bool)
@write_data.register(np.bool_)
def write_single_value(data: str | np.str_ | int | np.integer | float | np.floating | bool | np.bool_,
                       group: H5Group,
                       key: str,
                       key_level: int = 0,
                       progress: tqdm | None = None) -> None:
    """
    Function for writing a single value to the h5 file.
    """
    generalLogger.info(f"{spacer(key_level)}Saving single value {key}")
    group[str(key)] = data
    group[str(key)].attrs['type'] = 'value'

    if progress is not None:
        progress.update()


@write_data.register
def write_Type(data: type,
               group: H5Group,
               key: str,
               key_level: int = 0,
               progress: tqdm = None) -> None:
    """
    Function for writing a type to the h5 file.
    """
    generalLogger.info(f"{spacer(key_level)}Saving type {key}")
    group[str(key)] = data.__name__
    group[str(key)].attrs['type'] = 'type'

    if progress is not None:
        progress.update()


@write_data.register
def write_Path(data: Path,
               group: H5Group,
               key: str,
               key_level: int = 0,
               progress: tqdm = None) -> None:
    """
    Function for writing a Path to the h5 file.
    """
    generalLogger.info(f"{spacer(key_level)}Saving Path {key}")
    group[str(key)] = str(data)
    group[str(key)].attrs['type'] = 'path'

    if progress is not None:
        progress.update()


@write_data.register
def write_None(_: None,
               group: H5Group,
               key: str,
               key_level: int = 0,
               progress: tqdm = None) -> None:
    """
    Function for writing None to the h5 file.
    """
    generalLogger.info(f"{spacer(key_level)}Saving None value for {key}")
    _ = group.create_group(str(key))
    group[str(key)].attrs['type'] = 'None'

    if progress is not None:
        progress.update()


# endregion
