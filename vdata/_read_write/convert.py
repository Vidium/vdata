# coding: utf-8
# Created on 02/07/2021 11:36
# Author : matteo

# ====================================================
# imports
import os
import h5py
import shutil
import numpy as np
from pathlib import Path

from typing import Union, Optional

from .write import write_data
from ..time_point import TimePoint
from .._IO import generalLogger


# ====================================================
# code
def convert_anndata_to_vdata(file: Union[Path, str],
                             time_point: Union[int, float, str, TimePoint] = '0h',
                             time_column_name: Optional[str] = None,
                             inplace: bool = False) -> None:
    """
    Convert an anndata .h5 file into a valid vdata .h5 file.

    :param file: path to the anndata .h5 file to convert.
    :param time_point: a time point to set for the data in the anndata.
    :param time_column_name: the name of the column in anndata's obs to use as indicator of time point for the data.
    :param inplace: perform file conversion directly on the anndata .h5 file ? (default False)
        WARNING : if done inplace, you won't be able to open the file as an anndata anymore !
    """
    if not inplace:
        generalLogger.info('Working on file copy.')
        # copy file
        directory = os.path.split(file)[0]
        working_on_file = directory + '/' + Path(file).stem + '_vdata.h5'
        shutil.copy(file, working_on_file)

    else:
        generalLogger.info('Working on file inplace.')
        working_on_file = file

    # reformat copied file
    file = h5py.File(working_on_file, mode='a')

    # -------------------------------------------------------------------------
    # 1. remove X
    generalLogger.info("Removing 'X' layer.")
    del file['X']

    # -------------------------------------------------------------------------
    # 2. get time information
    valid_columns = list(set(file['obs'].keys()) - {'__categories', '_index'})

    if time_column_name is not None:
        if time_column_name not in valid_columns:
            raise ValueError(f"Could not find column '{time_column_name}' in obs ({valid_columns}).")

        time_points_in_data = set(file['obs'][time_column_name][()])
        time_points_masks = {tp: np.where(file['obs'][time_column_name][()] == tp)[0] for tp in time_points_in_data}

    else:
        time_points_masks = {time_point: np.arange(file['obs'][valid_columns[0]].shape[0])}

    # -------------------------------------------------------------------------
    # 3. convert layers to chunked TDFs
    # set group type
    file['layers'].attrs['type'] = 'dict'

    for layer in file['layers'].keys():
        generalLogger.info(f"Converting layer '{layer}'.")

        file['layers'].move(layer, f"{layer}_data")
        file['layers'].create_group(layer)

        # save index
        file['obs'].copy('_index', f'/layers/{layer}/index')
        file['layers'][layer]['index'].attrs['type'] = 'array'

        # save time_col_name
        write_data(time_column_name, file['layers'][layer], 'time_col_name', key_level=1)

        # create group for storing the data
        data_group = file['layers'][layer].create_group('data', track_order=True)

        # set group type
        file['layers'][layer].attrs['type'] = 'CHUNKED_TDF'

        # save columns
        file['var'].copy('_index', f'/layers/{layer}/columns')
        file['layers'][layer]['columns'].attrs['type'] = 'array'

        # save data, per time point, in DataSets
        for time_point in time_points_masks.keys():
            data_group.create_dataset(str(TimePoint(time_point)),
                                      data=file['layers'][f"{layer}_data"][time_points_masks[time_point][:, None], :],
                                      chunks=True, maxshape=(None, None))

        # remove old data
        del file['layers'][f"{layer}_data"]

    # -------------------------------------------------------------------------
    # 4.1 convert obs
    generalLogger.info(f"Converting 'obs'.")

    file.move('obs', 'obs_data')
    file.create_group('obs')

    # save index
    file['obs_data'].copy('_index', f'/obs/index')
    file['obs']['index'].attrs['type'] = 'array'

    # save time_col_name
    write_data(time_column_name, file['obs'], 'time_col_name', key_level=1)

    # save time_list
    write_data(np.repeat([str(TimePoint(tp)) for tp in time_points_masks.keys()],
                         [len(i) for i in time_points_masks.values()]),
               file['obs'], 'time_list', key_level=1)

    # create group for storing the data
    data_group = file['obs'].create_group('data', track_order=True)

    # set group type
    file['obs'].attrs['type'] = 'TDF'

    # save data, per column, in arrays
    for col in file['obs_data'].keys():
        if col in ('_index', '__categories'):
            continue

        values = file['obs_data'][col][()]

        write_data(values, data_group, col, key_level=1)

    # remove old data
    del file['obs_data']

    # -------------------------------------------------------------------------
    # 4.2 convert obsm
    generalLogger.info(f"Converting 'obsm'.")

    if 'obsm' in file.keys():
        file.move('obsm', 'obsm_data')
        file.create_group('obsm')

        # set group type
        file['obsm'].attrs['type'] = 'None'

        # remove old data
        del file['obsm_data']

    else:
        file.create_group('obsm')

        # set group type
        file['obsm'].attrs['type'] = 'None'

    # -------------------------------------------------------------------------
    # 4.3 convert obsp
    generalLogger.info(f"Converting 'obsp'.")

    if 'obsp' in file.keys():
        file.move('obsp', 'obsp_data')
        file.create_group('obsp')

        # set group type
        file['obsp'].attrs['type'] = 'None'

        # remove old data
        del file['obsp_data']

    else:
        file.create_group('obsp')

        # set group type
        file['obsp'].attrs['type'] = 'None'

    # -------------------------------------------------------------------------
    # 5.1 convert var
    generalLogger.info(f"Converting 'var'.")

    file.move('var', 'var_data')
    file.create_group('var')

    # save index
    file['var_data'].copy('_index', '/var/index')
    file['var']['index'].attrs['type'] = 'array'

    # create group for storing the data
    data_group = file['var'].create_group('data', track_order=True)

    # set group type
    file['var'].attrs['type'] = 'VDF'

    # save data, per column, in arrays
    for col in file['var_data'].keys():
        if col in ('_index', '__categories'):
            continue

        values = file['var_data'][col][()]

        write_data(values, data_group, col, key_level=1)

    # remove old data
    del file['var_data']

    # -------------------------------------------------------------------------
    # 5.2 convert varm
    convert_VDFs(file, 'varm')

    # -------------------------------------------------------------------------
    # 5.3 convert varp
    convert_VDFs(file, 'varp')

    # -------------------------------------------------------------------------
    # 6. copy uns
    generalLogger.info(f"Converting 'uns'.")

    set_type_to_dict(file['uns'])

    # -------------------------------------------------------------------------
    # 7. create time_points
    generalLogger.info(f"Creating 'time_points'.")

    file.create_group('time_points')

    # set group type
    file['time_points'].attrs['type'] = 'VDF'

    # create index
    write_data(np.arange(len(time_points_masks)), file['time_points'], 'index', key_level=1)

    # create data
    file['time_points'].create_group('data')

    values = [str(TimePoint(tp)) for tp in time_points_masks.keys()]

    write_data(values, file['time_points']['data'], 'value', key_level=1)

    # -------------------------------------------------------------------------
    file.close()


def convert_VDFs(file: h5py.Group, key: str) -> None:
    generalLogger.info(f"Converting '{key}'.")
    if key in file.keys():
        file.move(key, f'{key}_data')
        file.create_group(key)

        # set group type
        file[key].attrs['type'] = 'dict'

        for df_name in file[f'{key}_data'].keys():
            generalLogger.info(f"\tConverting dataframe '{df_name}'.")
            file[key].create_group(df_name)

            # save index
            file['var'].copy('index', f'/{key}/{df_name}/index')
            file[key][df_name]['index'].attrs['type'] = 'array'

            # create group for storing the data
            data_group = file[key][df_name].create_group('data', track_order=True)

            # set group type
            file[key][df_name].attrs['type'] = 'VDF'

            # save data, per column, in arrays
            for col in range(file[f'{key}_data'][df_name].shape[1]):
                values = file[f'{key}_data'][df_name][:, col]

                write_data(values, data_group, col, key_level=2)

        # remove old data
        del file[f'{key}_data']

    else:
        file.create_group(key)

        # set group type
        file[key].attrs['type'] = 'None'


def set_type_to_dict(group: h5py.Group):
    group.attrs['type'] = 'dict'

    for child in group.keys():
        if isinstance(group[child], h5py.Group):
            set_type_to_dict(group[child])

        elif isinstance(group[child], h5py.Dataset):
            if group[child].shape == ():
                group[child].attrs['type'] = 'value'

            else:
                group[child].attrs['type'] = 'array'

        else:
            del group[child]
