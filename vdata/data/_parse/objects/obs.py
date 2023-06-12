from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from vdata._typing import NDArray_IFS
from vdata.data._parse.time import check_time_match
from vdata.data._parse.utils import log_timepoints
from vdata.IO.logger import generalLogger
from vdata.names import NO_NAME
from vdata.tdf import TemporalDataFrame, TemporalDataFrameBase
from vdata.utils import first
from vdata.vdataframe import VDataFrame

if TYPE_CHECKING:
    from vdata.data._parse.data import ParsingDataIn


def _index(obj: pd.DataFrame | VDataFrame | TemporalDataFrameBase) -> tuple[NDArray_IFS, bool]:
    _repeating_index = obj.has_repeating_index if isinstance(obj, TemporalDataFrameBase) else False
    return np.array(obj.index), _repeating_index
    

def get_obs_index(data: pd.DataFrame | 
                        VDataFrame | 
                        TemporalDataFrameBase | 
                        dict[str, pd.DataFrame | VDataFrame | TemporalDataFrameBase] | 
                        None,
                  obs: pd.DataFrame | VDataFrame | TemporalDataFrameBase | None) \
    -> tuple[NDArray_IFS | None, bool]:
    if obs is not None:
        return _index(obs)
    
    if isinstance(data, (pd.DataFrame, VDataFrame, TemporalDataFrameBase)):
        return _index(data)
        
    if isinstance(data, dict):
        return _index(first(data))
    
    return None, False
    

def parse_obs(data: ParsingDataIn) -> TemporalDataFrameBase:
    # find time points list
    check_time_match(data)
    log_timepoints(data.timepoints)
    
    return data.obs


def parse_obsm(data: ParsingDataIn) -> dict[str, TemporalDataFrame]:
    if not len(data.obsm):
        generalLogger.debug("    3. \u2717 'obsm' was not given.")
        return {}
        
    generalLogger.debug(f"    3. \u2713 'obsm' is a {type(data.obsm).__name__}.")

    if data.obs is None and not len(data.layers):
        raise ValueError("'obsm' parameter cannot be set unless either 'data' or 'obs' are set.")

    if not isinstance(data.obsm, dict):
        raise TypeError("'obsm' must be a dictionary of DataFrames.")

    valid_obsm: dict[str, TemporalDataFrame] = {}

    for key, value in data.obsm.items():
        if isinstance(value, (pd.DataFrame, VDataFrame)):
            if data.time_list is None:
                if data.obs is not None:
                    data.time_list = data.obs.timepoints_column
                else:
                    data.time_list = first(data.layers).timepoints_column

            valid_obsm[str(key)] = TemporalDataFrame(value, 
                                                     time_list=data.time_list,
                                                     name=str(key))

        elif isinstance(value, TemporalDataFrame):
            value.unlock_indices()
            value.unlock_columns()

            if value.name != str(key):
                value.name = str(key) if value.name == NO_NAME else f"{value.name}_{key}"
                
            valid_obsm[str(key)] = value
                
        else:
            raise TypeError(f"'obsm' '{key}' must be a TemporalDataFrame or a pandas DataFrame.")

        if not np.all(np.isin(valid_obsm[str(key)].index, data.obs.index)):
            raise ValueError(f"Index of 'obsm' '{key}' does not match 'obs' and 'layers' indexes.")
        
        valid_obsm[str(key)].reindex(data.obs.index)

    return valid_obsm


def parse_obsp(data: ParsingDataIn) -> dict[str, VDataFrame]:
    if not len(data.obsp):
        generalLogger.debug("    4. \u2717 'obsp' was not given.")
        return {}
    
    generalLogger.debug(f"    4. \u2713 'obsp' is a {type(data.obsp).__name__}.")

    if data.obs is None and not len(data.layers):
        raise ValueError("'obsp' parameter cannot be set unless either 'data' or 'obs' are set.")

    if not isinstance(data.obsp, dict):
        raise TypeError("'obsp' must be a dictionary of 2D numpy arrays or pandas DataFrames.")
    
    valid_obsp: dict[str, VDataFrame] = {}

    for key, value in data.obsp.items():
        if not isinstance(value, (np.ndarray, pd.DataFrame, VDataFrame)) or value.ndim != 2:
            raise TypeError(f"'obsp' '{key}' must be a 2D numpy array or pandas DataFrame.")

        if isinstance(value, (pd.DataFrame, VDataFrame)):
            if not all(value.index.isin(data.obs.index)):
                raise ValueError(f"Index of 'obsp' '{key}' does not match 'obs' and 'layers' indexes.")
            
            if not all(value.columns.isin(data.obs.index)):
                raise ValueError("Column names of 'obsp' do not match 'obs' and 'layers' indexes.")
            
            value.reindex(data.obs.index)
            value = value[data.obs.index]

        else:
            value = VDataFrame(value, index=data.obs.index, columns=data.obs.index)

        valid_obsp[str(key)] = value

    return valid_obsp
