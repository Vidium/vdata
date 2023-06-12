from __future__ import annotations

# import os
# import shutil
from pathlib import Path
from typing import Collection

import numpy as np
from anndata import AnnData

import vdata
from vdata.IO.logger import generalLogger
from vdata.names import DEFAULT_TIME_COL_NAME

# import ch5mpy as ch
# import numpy as np
# from vdata.IO import generalLogger
from vdata.timepoint import TimePoint, TimePointArray, atleast_1d
from vdata.utils import as_tp_list, match_timepoints, repr_array


def convert_vdata_to_anndata(data: vdata.VData,
                             timepoints_list: str | TimePoint | Collection[str | TimePoint] | None = None,
                             into_one: bool = True,
                             with_timepoints_column: bool = True,
                             layer_as_X: str | None = None,
                             layers_to_export: list[str] | None = None) -> AnnData | list[AnnData]:
    """
    Convert a VData object to an AnnData object.

    Args:
        timepoints_list: a list of time points for which to extract data to build the AnnData. If set to
            None, all timepoints are selected.
        into_one: Build one AnnData, concatenating the data for multiple time points (True), or build one
            AnnData for each time point (False) ?
        with_timepoints_column: store time points data in the obs DataFrame. This is only used when
            concatenating the data into a single AnnData (i.e. into_one=True).
        layer_as_X: name of the layer to use as the X matrix. By default, the first layer is used.
        layers_to_export: if None export all layers

    Returns:
        An AnnData object with data for selected time points.
    """
    # TODO : obsp is not passed to AnnData

    generalLogger.debug(u'\u23BE VData conversion to AnnData : begin '
                        u'---------------------------------------------------------- ')

    if timepoints_list is None:
        _timepoints_list = data.timepoints_values

    else:
        _timepoints_list = TimePointArray(as_tp_list(timepoints_list))
        _timepoints_list = atleast_1d(_timepoints_list[np.where(match_timepoints(_timepoints_list, 
                                                                                 data.timepoints_values))])

    generalLogger.debug(f"Selected time points are : {repr_array(_timepoints_list)}")

    if into_one:
        return _convert_vdata_into_one_anndata(data, 
                                               with_timepoints_column,
                                               _timepoints_list,
                                               layer_as_X,
                                               layers_to_export)

    return _convert_vdata_into_many_anndatas(data, _timepoints_list, layer_as_X)
    
    
def _convert_vdata_into_one_anndata(data: vdata.VData,
                                    with_timepoints_column: bool,
                                    timepoints_list: TimePointArray,
                                    layer_as_X: str | None,
                                    layers_to_export: list[str] | None) -> AnnData:
    generalLogger.debug("Convert to one AnnData object.")

    if with_timepoints_column:
        tp_col_name = data.obs.timepoints_column_name if data.obs.timepoints_column_name is not None else \
            DEFAULT_TIME_COL_NAME
    else:
        tp_col_name = None

    view = data[timepoints_list]
    if layer_as_X is None:
        layer_as_X = list(view.layers.keys())[0]

    elif layer_as_X not in view.layers.keys():
        raise ValueError(f"Layer '{layer_as_X}' was not found.")

    X = view.layers[layer_as_X].to_pandas()
    X.index = X.index.astype(str)
    X.columns = X.columns.astype(str)
    if layers_to_export is None:
        layers_to_export = view.layers.keys()
        
    anndata = AnnData(X=X,
                    layers={key: view.layers[key].to_pandas(str_index=True) for key in layers_to_export},
                    obs=view.obs.to_pandas(with_timepoints=tp_col_name,
                                            str_index=True),
                    obsm={key: arr.values_num for key, arr in view.obsm.items()},
                    obsp={key: arr.values for key, arr in view.obsp.items()},
                    var=view.var.to_pandas(),
                    varm={key: arr for key, arr in view.varm.items()},
                    varp={key: arr for key, arr in view.varp.items()},
                    uns=view.uns.copy())
    
    generalLogger.debug('\u23BF VData conversion to AnnData : end '
                        '---------------------------------------------------------- ')
    
    return anndata
    


def _convert_vdata_into_many_anndatas(data: vdata.VData,
                                      timepoints_list: TimePointArray,
                                      layer_as_X: str | None) -> list[AnnData]:
    generalLogger.debug("Convert to many AnnData objects.")

    result = []
    for time_point in timepoints_list:
        view = data[time_point]
        
        if layer_as_X is None:
            layer_as_X = list(view.layers.keys())[0]

        elif layer_as_X not in view.layers.keys():
            raise ValueError(f"Layer '{layer_as_X}' was not found.")

        X = view.layers[layer_as_X].to_pandas()
        X.index = X.index.astype(str)
        X.columns = X.columns.astype(str)

        result.append(AnnData(X=X,
                              layers={key: layer.to_pandas(str_index=True)
                                      for key, layer in view.layers.items()},
                              obs=view.obs.to_pandas(str_index=True),
                              obsm={key: arr.to_pandas(str_index=True) for key, arr in view.obsm.items()},
                              var=view.var.to_pandas(),
                              varm={key: arr for key, arr in view.varm.items()},
                              varp={key: arr for key, arr in view.varp.items()},
                              uns=view.uns))

    generalLogger.debug(u'\u23BF VData conversion to AnnData : end '
                        u'---------------------------------------------------------- ')

    return result

def convert_anndata_to_vdata(file: Path | str,
                             time_point: int | float | str | TimePoint = '0h',
                             time_column_name: str | None = None,
                             inplace: bool = False) -> None:
    """
    Convert an anndata h5 file into a valid vdata h5 file.
    /!\\ WARNING : if done inplace, you won't be able to open the file as an anndata anymore !

    Args:
        file: path to the anndata h5 file to convert.
        time_point: a time point to set for the data in the anndata.
        time_column_name: the name of the column in anndata's obs to use as indicator of time point for the data.
        inplace: perform file conversion directly on the anndata h5 file ? (default False)
    """
    raise NotImplementedError
    
#     working_on_file = Path(file).with_suffix('.vd')

#     if not inplace:
#         generalLogger.info('Working on file copy.')
#         # copy file
#         shutil.copy(file, working_on_file)

#     else:
#         generalLogger.info('Working on file inplace.')
#         # rename file
#         os.rename(file, working_on_file)

#     # reformat copied file
#     data = ch.H5Dict.read(working_on_file, 
#                           mode=ch.H5Mode.READ_WRITE if inplace else ch.H5Mode.READ)

#     # -------------------------------------------------------------------------
#     # 1. remove X
#     generalLogger.info("Removing 'X' layer.")
#     if 'layers' not in data.keys():
#         data['layers'] = dict(X = data['X'])
        
#     del data['X']

#     # -------------------------------------------------------------------------
#     # 2. get time information
#     valid_columns = list(set(data['obs'].keys()) - {'__categories', '_index'})

#     if time_column_name is not None:
#         if time_column_name not in valid_columns:
#             raise ValueError(f"Could not find column '{time_column_name}' in obs ({valid_columns}).")

#         timepoints_in_data = set(data['obs'][time_column_name][()])
#         timepoints_masks = {tp: np.where(data['obs'][time_column_name][()] == tp)[0] for tp in timepoints_in_data}

#     else:
#         timepoints_masks = {time_point: np.arange(data['obs'][valid_columns[0]].shape[0])}

#     # -------------------------------------------------------------------------
#     # 3. convert layers to chunked TDFs
#     # set group type
#     obs_index_name = h5_file['obs'].attrs['_index']
#     obs_index = h5_file['obs'][obs_index_name]
#     var_index_name = h5_file['var'].attrs['_index']
#     var_index = h5_file['var'][var_index_name]

#     for layer in h5_file['layers'].keys():
#         generalLogger.info(f"Converting layer '{layer}'.")

#         h5_file['layers'].move(layer, f"{layer}_data")
#         h5_file['layers'].create_group(layer)

#         # save index
#         h5_file[f'layers/{layer}'].create_dataset_like('index', obs_index)
#         h5_file[f'layers/{layer}/index'][()] = obs_index
#         h5_file['layers'][layer]['index'].attrs['type'] = 'array'

#         # save time_col_name
#         write_data(time_column_name, h5_file['layers'][layer], 'time_col_name', key_level=1)

#         # create group for storing the data
#         data_group = h5_file['layers'][layer].create_group('data', track_order=True)

#         # set group type
#         h5_file['layers'][layer].attrs['type'] = 'CHUNKED_TDF'

#         # save columns
#         h5_file[f'layers/{layer}'].create_dataset_like('columns', var_index)
#         h5_file[f'layers/{layer}/columns'][()] = var_index
#         h5_file['layers'][layer]['columns'].attrs['type'] = 'array'

#         # save data, per time point, in DataSets
#         for time_point in timepoints_masks.keys():
#             # TODO : support for reading a sparse matrix
#             data_group.create_dataset(
#                 str(TimePoint(time_point)),
#                 data=h5_file['layers'][f"{layer}_data"][timepoints_masks[time_point]],
#                 chunks=True, maxshape=(None, None)
#             )

#         # remove old data
#         del h5_file['layers'][f"{layer}_data"]

#     # -------------------------------------------------------------------------
#     # 4.1 convert obs
#     generalLogger.info("Converting 'obs'.")

#     h5_file.move('obs', 'obs_data')
#     h5_file.create_group('obs')

#     # save index
#     h5_file['obs'].create_dataset_like('index', obs_index)
#     h5_file['obs/index'][()] = obs_index
#     h5_file['obs']['index'].attrs['type'] = 'array'

#     # save time_col_name
#     write_data(time_column_name, h5_file['obs'], 'time_col_name', key_level=1)

#     # save time_list
#     write_data(list(np.repeat([TimePoint(tp) for tp in timepoints_masks.keys()],
#                               [len(i) for i in timepoints_masks.values()])),
#                h5_file['obs'], 'time_list', key_level=1)

#     # create group for storing the data
#     data_group = h5_file['obs'].create_group('data', track_order=True)

#     # set group type
#     h5_file['obs'].attrs['type'] = 'tdf'

#     # save data, per column, in arrays
#     for col in h5_file['obs_data'].keys():
#         if col in ('_index', '__categories'):
#             continue

#         values = h5_file['obs_data'][col][()]

#         write_data(values, data_group, col, key_level=1)

#     # remove old data
#     del h5_file['obs_data']

#     # -------------------------------------------------------------------------
#     # 4.2 convert obsm
#     generalLogger.info("Converting 'obsm'.")

#     if 'obsm' in h5_file.keys():
#         h5_file.move('obsm', 'obsm_data')
#         h5_file.create_group('obsm')

#         # set group type
#         h5_file['obsm'].attrs['type'] = 'dict'

#         for df_name in h5_file['obsm_data'].keys():
#             generalLogger.info(f"\tConverting dataframe '{df_name}'.")
#             h5_file['obsm'].create_group(df_name)

#             # save index
#             h5_file['obs'].copy('index', f'/obsm/{df_name}/index')
#             h5_file['obsm'][df_name]['index'].attrs['type'] = 'array'

#             # save time_col_name
#             write_data(time_column_name, h5_file['obsm'][df_name], 'time_col_name', key_level=1)

#             # create group for storing the data
#             data_group = h5_file['obsm'][df_name].create_group('data', track_order=True)

#             # set group type
#             h5_file['obsm'][df_name].attrs['type'] = 'CHUNKED_TDF'

#             # save columns
#             write_data(np.arange(h5_file['obsm_data'][df_name].shape[1]), h5_file['obsm'][df_name], 'columns',
#                        key_level=1)

#             # save data, per time point, in DataSets
#             for time_point in timepoints_masks.keys():
#                 data_group.create_dataset(str(TimePoint(time_point)),
#                                          data=h5_file['obsm_data'][df_name][timepoints_masks[time_point][:, None], :],
#                                          chunks=True, maxshape=(None, None))

#         # remove old data
#         del h5_file['obsm_data']

#     else:
#         h5_file.create_group('obsm')

#         # set group type
#         h5_file['obsm'].attrs['type'] = 'None'

#     # -------------------------------------------------------------------------
#     # 4.3 convert obsp
#     generalLogger.info("Converting 'obsp'.")

#     if 'obsp' in h5_file.keys():
#         h5_file.move('obsp', 'obsp_data')
#         h5_file.create_group('obsp')

#         # set group type
#         h5_file['obsp'].attrs['type'] = 'None'
#         # h5_file['obsp'].attrs['type'] = 'dict'
#         #
#         # for df_name in h5_file['obsp_data'].keys():
#         #     generalLogger.info(f"\tConverting dataframe '{df_name}'.")
#         #     h5_file['obsp'].create_group(df_name)
#         #
#         #     # set group type
#         #     h5_file['obsp'][df_name].attrs['type'] = 'dict'
#         #
#         #     for tp in timepoints_masks.keys():
#         #         generalLogger.info(f"\t\tConverting for time point '{tp}'.")
#         #         h5_file['obsp'][df_name].create_group(str(TimePoint(tp)))
#         #
#         #         # save index
#         #         write_data(h5_file['obs']['index'][timepoints_masks[tp]],
#         #                    h5_file['obsp'][df_name][str(TimePoint(tp))], 'index', key_level=2)
#         #
#         #         # create group for storing the data
#         #         data_group = h5_file['obsp'][df_name][str(TimePoint(tp))].create_group('data', track_order=True)
#         #
#         #         # set group type
#         #         h5_file['obsp'][df_name][str(TimePoint(tp))].attrs['type'] = 'VDF'
#         #
#         #         # save data, per column, in arrays
#         #         for col in range(h5_file[f'obsp_data'][df_name].shape[1]):
#         #             pass
#         # TODO : here save the data (/!\ we need to handle sparse matrices)

#         # remove old data
#         del h5_file['obsp_data']

#     else:
#         h5_file.create_group('obsp')

#         # set group type
#         h5_file['obsp'].attrs['type'] = 'None'

#     # -------------------------------------------------------------------------
#     # 5.1 convert var
#     generalLogger.info("Converting 'var'.")

#     h5_file.move('var', 'var_data')
#     h5_file.create_group('var')

#     # save index
#     h5_file['var'].create_dataset_like('index', var_index)
#     h5_file['var/index'][()] = var_index
#     h5_file['var']['index'].attrs['type'] = 'array'

#     # create group for storing the data
#     data_group = h5_file['var'].create_group('data', track_order=True)

#     # set group type
#     h5_file['var'].attrs['type'] = 'VDF'

#     # save data, per column, in arrays
#     for col in h5_file['var_data'].keys():
#         if col in ('_index', '__categories'):
#             continue

#         values = h5_file['var_data'][col][()]

#         write_data(values, data_group, col, key_level=1)

#     # remove old data
#     del h5_file['var_data']

#     # -------------------------------------------------------------------------
#     # 5.2 convert varm
#     convert_VDFs(h5_file, 'varm')

#     # -------------------------------------------------------------------------
#     # 5.3 convert varp
#     convert_VDFs(h5_file, 'varp')

#     # -------------------------------------------------------------------------
#     # 6. copy uns
#     generalLogger.info("Converting 'uns'.")

#     if 'uns' in h5_file.keys():
#         set_type_to_dict(h5_file['uns'])

#     # -------------------------------------------------------------------------
#     # 7. create time-points
#     generalLogger.info("Creating 'time-points'.")

#     h5_file.create_group('timepoints')

#     # set group type
#     h5_file['timepoints'].attrs['type'] = 'VDF'

#     # create index
#     write_data(np.arange(len(timepoints_masks)), h5_file['timepoints'], 'index', key_level=1)

#     # create data
#     h5_file['timepoints'].create_group('data')

#     values = [str(TimePoint(tp)) for tp in timepoints_masks.keys()]

#     write_data(values, h5_file['timepoints']['data'], 'value', key_level=1)

#     # -------------------------------------------------------------------------
#     h5_file.close()


# def convert_VDFs(file: File, key: str) -> None:
#     generalLogger.info(f"Converting '{key}'.")
#     if key in file.keys():
#         file.move(key, f'{key}_data')
#         file.create_group(key)

#         # set group type
#         file[key].attrs['type'] = 'dict'

#         for df_name in file[f'{key}_data'].keys():
#             generalLogger.info(f"\tConverting dataframe '{df_name}'.")
#             file[key].create_group(df_name)

#             # save index
#             file['var'].copy('index', f'/{key}/{df_name}/index')
#             file[key][df_name]['index'].attrs['type'] = 'array'

#             # create group for storing the data
#             data_group = file[key][df_name].create_group('data', track_order=True)

#             # set group type
#             file[key][df_name].attrs['type'] = 'VDF'

#             # save data, per column, in arrays
#             for col in range(file[f'{key}_data'][df_name].shape[1]):
#                 values = file[f'{key}_data'][df_name][:, col]

#                 write_data(values, data_group, str(col), key_level=2)

#         # remove old data
#         del file[f'{key}_data']

#     else:
#         file.create_group(key)

#         # set group type
#         file[key].attrs['type'] = 'None'
