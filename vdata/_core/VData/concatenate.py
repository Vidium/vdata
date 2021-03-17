# coding: utf-8
# Created on 10/02/2021 16:06
# Author : matteo

# ====================================================
# imports
import pandas as pd
from typing import Optional

from .vdata import VData
from ..._IO import VValueError, VTypeError, generalLogger


# ====================================================
# code
def concatenate(*args: 'VData', name: Optional[str] = None) -> 'VData':
    """
    Concatenate together multiple VData objects, which share the same layer keys, vars and time points.
    :param args: list of at least 2 VData objects to concatenate.
    :param name: a name for the concatenated VData object.
    :return: a concatenated VData object.
    """
    if len(args) < 2:
        raise VValueError("At least 2 VData objects must be provided.")

    if not all(isinstance(arg, VData) for arg in args):
        raise VTypeError("Only Vdata objects are allowed.")

    generalLogger.debug(f"\u23BE Concatenation of VDatas : start "
                        f"---------------------------------------------------------- ")

    # get initial data
    first_VData = args[0]

    generalLogger.info(f"Using VData '{first_VData.name}' as first object.")

    _data = first_VData.layers.dict_copy()
    _obs = first_VData.obs
    _obsm = first_VData.obsm.dict_copy()
    _var = first_VData.var
    _varm = first_VData.varm.dict_copy()
    _varp = first_VData.varp.dict_copy()
    _time_points = first_VData.time_points
    _uns = first_VData.uns

    _obsp = first_VData.obsp.compact()

    # concatenate with data in other VData objects
    for next_VData_index, next_VData in enumerate(args[1:]):
        generalLogger.info(f"Working on VData '{next_VData.name}' ({next_VData_index + 1}/{len(args) - 1}).")

        # check var -----------------------------------------------------------
        generalLogger.info(f"  '\u21B3' merging 'var' DataFrame.")
        if not _var.index.equals(next_VData.var.index):
            raise VValueError("Cannot concatenate VData objects if 'var' indexes are different.")

        for column in next_VData.var.columns:
            if column in _var.columns:
                if not _var[column].equals(next_VData.var[column]):
                    raise VValueError(f"Values found in 'var' column '{column}' do not match.")

            else:
                _var[column] = next_VData.var[column]

        # check time points ---------------------------------------------------
        generalLogger.info(f"  '\u21B3' merging 'time_points' DataFrame.")
        if not _time_points.index.equals(next_VData.time_points.index):
            raise VValueError("Cannot concatenate VData objects if 'time_point' indexes are different.")

        for column in next_VData.time_points.columns:
            if column in _time_points.columns:
                if not _time_points[column].equals(next_VData.time_points[column]):
                    raise VValueError(f"Values found in 'time_points' column '{column}' do not match.")

            else:
                _time_points[column] = next_VData.time_points[column]

        # check layers keys ---------------------------------------------------
        generalLogger.info(f"  '\u21B3' merging layers.")
        if not _data.keys() == next_VData.layers.keys():
            raise VValueError("Cannot concatenate VData objects if 'layers' keys are different.")

        for key in _data.keys():
            generalLogger.info(f"    '\u21B3' merging layer '{key}' DataFrame.")
            _data[key] = _data[key].merge(next_VData.layers[key])

        # concat other data =============================================================
        # obs -----------------------------------------------------------------
        generalLogger.info(f"  '\u21B3' merging 'obs' TemporalDataFrame.")
        if any(_obs.index.isin(next_VData.obs.index)):
            raise VValueError("Cannot merge VData objects with common obs index values.")

        _obs = _obs.merge(next_VData.obs)

        # obsm ----------------------------------------------------------------
        generalLogger.info(f"  '\u21B3' merging obsm.")
        for key in _obsm.keys():
            generalLogger.info(f"    '\u21B3' merging obsm '{key}' TemporalDataFrame.")

            if key in next_VData.obsm.keys():
                _obsm[key] = _obsm[key].merge(next_VData.obsm[key])

            else:
                generalLogger.warning(f"Dropping 'obsm' '{key}' because it was not found in all VData objects.")
                del _obsm[key]

        # obsp ----------------------------------------------------------------
        generalLogger.info(f"  '\u21B3' merging obsp.")
        next_obsp = next_VData.obsp.compact()
        for key in _obsp.keys():
            generalLogger.info(f"    '\u21B3' merging obsp '{key}' DataFrame.")

            if key in next_obsp.keys():
                _index = _obsp[key].index.union(next_VData.obs.index, sort=False)
                result = pd.DataFrame(index=_index, columns=_index)

                result.iloc[0:len(_obsp[key].index), 0:len(_obsp[key].index)] = _obsp[key]
                result.iloc[len(_obsp[key].index):, len(_obsp[key].index):] = next_obsp[key]

                _obsp[key] = result

            else:
                generalLogger.warning(f"Dropping 'obsp' '{key}' because it was not found in all VData objects.")
                del _obsp[key]

        # varm ----------------------------------------------------------------
        generalLogger.info(f"  '\u21B3' merging varm.")
        for key in _varm.keys():
            if key in next_VData.varm.keys():
                generalLogger.info(f"    '\u21B3' merging varm '{key}' DataFrame.")

                _varm[key] = _varm[key].reset_index().merge(next_VData.varm[key].reset_index(),
                                                            how='outer').set_index('index')

            else:
                generalLogger.warning(f"Dropping 'varm' '{key}' because it was not found in all VData objects.")
                del _varm[key]

        # varp ----------------------------------------------------------------
        generalLogger.info(f"  '\u21B3' merging varp.")
        for key in _varp.keys():
            if key in next_VData.varp.keys():
                generalLogger.info(f"    '\u21B3' merging varp '{key}' DataFrame.")

                _varp[key] = _varp[key].reset_index().merge(next_VData.varp[key].reset_index(),
                                                            how='outer').set_index('index')

            else:
                generalLogger.warning(f"Dropping 'varp' '{key}' because it was not found in all VData objects.")
                del _varp[key]

        # uns -----------------------------------------------------------------
        generalLogger.info(f"  '\u21B3' merging uns.")
        for key, value in next_VData.uns.items():
            generalLogger.info(f"    '\u21B3' merging uns '{key}'.")

            if key not in _uns:
                _uns[key] = value

            elif _uns[key] != value:
                generalLogger.warning(f"Found different values for key '{key}' in 'uns', keeping first found value.")

    concatenated_VData = VData(data=_data, obs=_obs, obsm=_obsm, obsp=_obsp,
                               var=_var, varm=_varm, varp=_varp,
                               time_points=_time_points, uns=_uns,
                               name=name)

    generalLogger.debug(f"\u23BF Concatenation of VDatas : end "
                        f"---------------------------------------------------------- ")

    return concatenated_VData