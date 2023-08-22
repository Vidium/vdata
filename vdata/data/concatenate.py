from __future__ import annotations

from copy import deepcopy
from typing import Sequence

import numpy as np
import pandas as pd
from h5dataframe import H5DataFrame

from vdata.data.vdata import VData
from vdata.IO import generalLogger


def concatenate(arr: Sequence[VData], name: str = "") -> "VData":
    """
    Concatenate together multiple VData objects, which share the same layer keys, vars and time points.

    Args:
        arr: sequence of at least 2 VData objects to concatenate.
        name: a name for the concatenated VData object.

    Returns:
        A concatenated VData object.
    """
    if len(arr) < 2:
        raise ValueError("At least 2 VData objects must be provided.")

    if not all(isinstance(arg, VData) for arg in arr):
        raise TypeError("Only Vdata objects are allowed.")

    generalLogger.debug(
        "\u23BE Concatenation of VDatas : start " "---------------------------------------------------------- "
    )

    # get initial data
    first_VData = arr[0]

    generalLogger.info(lambda: f"Using VData '{first_VData.name}' as first object.")

    _data = first_VData.layers.dict_copy()
    _obs = first_VData.obs.copy()
    _obsm = first_VData.obsm.dict_copy()
    _obsp = first_VData.obsp.dict_copy()
    _var = first_VData.var.copy()
    _varm = first_VData.varm.dict_copy()
    _varp = first_VData.varp.dict_copy()
    _timepoints = first_VData.timepoints.copy()
    _uns = deepcopy(first_VData.uns)

    # concatenate with data in other VData objects
    for next_VData_index, next_VData in enumerate(arr[1:]):
        generalLogger.info(lambda: f"Working on VData '{next_VData.name}' ({next_VData_index + 1}/{len(arr) - 1}).")

        # check var -----------------------------------------------------------
        generalLogger.info("  '\u21B3' merging 'var' DataFrame.")
        if not _var.index.equals(next_VData.var.index):
            raise ValueError("Cannot concatenate VData objects if 'var' indexes are different.")

        for column in next_VData.var.columns:
            if column in _var.columns:
                if not _var[column].equals(next_VData.var[column]):
                    raise ValueError(f"Values found in 'var' column '{column}' do not match.")

            else:
                _var[column] = next_VData.var[column]

        # check time points ---------------------------------------------------
        generalLogger.info("  '\u21B3' merging 'time-points' DataFrame.")
        if not _timepoints.index.equals(next_VData.timepoints.index):
            raise ValueError("Cannot concatenate VData objects if 'time_point' indexes are different.")

        for column in next_VData.timepoints.columns:
            if column in _timepoints.columns:
                if not _timepoints[column].equals(next_VData.timepoints[column]):
                    raise ValueError(f"Values found in 'time-points' column '{column}' do not match.")

            else:
                _timepoints[column] = next_VData.timepoints[column]

        # check layers keys ---------------------------------------------------
        generalLogger.info("  '\u21B3' merging layers.")
        if not _data.keys() == next_VData.layers.keys():
            raise ValueError("Cannot concatenate VData objects if 'layers' keys are different.")

        for key in _data.keys():
            generalLogger.info(lambda: f"    '\u21B3' merging layer '{key}' DataFrame.")
            _data[key] = _data[key].merge(next_VData.layers[key])

        # concat other data =============================================================
        # obs -----------------------------------------------------------------
        generalLogger.info("  '\u21B3' merging 'obs' TemporalDataFrame.")
        if np.any(np.isin(_obs.index, next_VData.obs.index)):
            raise ValueError("Cannot merge VData objects with common obs index values.")

        _obs = _obs.merge(next_VData.obs)

        # obsm ----------------------------------------------------------------
        generalLogger.info("  '\u21B3' merging obsm.")
        for key in _obsm.keys():
            generalLogger.info(lambda: f"    '\u21B3' merging obsm '{key}' TemporalDataFrame.")

            if key in next_VData.obsm.keys():
                _obsm[key] = _obsm[key].merge(next_VData.obsm[key])

            else:
                generalLogger.warning(lambda: f"Dropping 'obsm' '{key}' because it was not found in all VData objects.")
                del _obsm[key]

        # obsp ----------------------------------------------------------------
        generalLogger.info("  '\u21B3' merging obsp.")
        next_obsp = next_VData.obsp
        for key in _obsp.keys():
            generalLogger.info(lambda: f"    '\u21B3' merging obsp '{key}' DataFrame.")

            if key in next_obsp.keys():
                _index = _obsp[key].index.union(pd.Index(next_VData.obs.index.values), sort=False)
                result = H5DataFrame(index=_index, columns=_index)

                result.iloc[0 : len(_obsp[key].index), 0 : len(_obsp[key].index)] = _obsp[key]
                result.iloc[len(_obsp[key].index) :, len(_obsp[key].index) :] = next_obsp[key]

                _obsp[key] = result

            else:
                generalLogger.warning(lambda: f"Dropping 'obsp' '{key}' because it was not found in all VData objects.")
                del _obsp[key]

        # varm ----------------------------------------------------------------
        generalLogger.info("  '\u21B3' merging varm.")
        for key in _varm.keys():
            if key in next_VData.varm.keys():
                generalLogger.info(lambda: f"    '\u21B3' merging varm '{key}' DataFrame.")

                _varm[key] = H5DataFrame(
                    _varm[key].reset_index().merge(next_VData.varm[key].reset_index(), how="outer").set_index("index")
                )

            else:
                generalLogger.warning(lambda: f"Dropping 'varm' '{key}' because it was not found in all VData objects.")
                del _varm[key]

        # varp ----------------------------------------------------------------
        generalLogger.info("  '\u21B3' merging varp.")
        for key in _varp.keys():
            if key in next_VData.varp.keys():
                generalLogger.info(lambda: f"    '\u21B3' merging varp '{key}' DataFrame.")

                _varp[key] = H5DataFrame(
                    _varp[key].reset_index().merge(next_VData.varp[key].reset_index(), how="outer").set_index("index")
                )

            else:
                generalLogger.warning(lambda: f"Dropping 'varp' '{key}' because it was not found in all VData objects.")
                del _varp[key]

        # uns -----------------------------------------------------------------
        generalLogger.info("  '\u21B3' merging uns.")
        for key, value in next_VData.uns.items():
            generalLogger.info(lambda: f"    '\u21B3' merging uns '{key}'.")

            if key not in _uns:
                _uns[key] = value

            elif _uns[key] != value:
                generalLogger.warning(
                    lambda: f"Found different values for key '{key}' in 'uns', keeping first found value."
                )

    concatenated_VData = VData(
        data=_data,
        obs=_obs,
        obsm=_obsm,
        obsp=_obsp,
        var=_var,
        varm=_varm,
        varp=_varp,
        timepoints=_timepoints,
        uns=_uns,
        name=name,
    )

    generalLogger.debug(
        "\u23BF Concatenation of VDatas : end " "---------------------------------------------------------- "
    )

    return concatenated_VData
