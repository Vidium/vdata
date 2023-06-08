from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

import numpy as np
import pandas as pd

from vdata.data._parse.utils import log_timepoints
from vdata.IO.logger import generalLogger
from vdata.tdf import TemporalDataFrameBase
from vdata.timepoint import TimePoint, TimePointArray
from vdata.utils import as_tp_list, first, match_timepoints
from vdata.vdataframe import VDataFrame

if TYPE_CHECKING:
    from vdata.data._parse.data import ParsingDataIn


def parse_time_list(time_list: Sequence[str | TimePoint] | None,
                    time_col_name: str | None,
                    obs: pd.DataFrame | VDataFrame | TemporalDataFrameBase | None) -> TimePointArray | None:
    if time_list is not None:
        return TimePointArray([TimePoint(tp) for tp in time_list])

    elif obs is not None and time_col_name is not None:
        if time_col_name not in obs.columns:
            raise ValueError(f"Could not find column '{time_col_name}' in obs.")
        
        return TimePointArray([TimePoint(tp) for tp in obs[time_col_name]])
    
    return None
        
    # TODO : could also get time_list from obsm and obsp
        

def parse_timepoints(timepoints: pd.DataFrame | VDataFrame | None) -> VDataFrame:
    if timepoints is None:
        generalLogger.debug("  'time points' DataFrame was not given.")
        return VDataFrame(columns=['value'])
        
    if not isinstance(timepoints, (pd.DataFrame, VDataFrame)):
        raise TypeError(f"'time points' must be a DataFrame, got '{type(timepoints).__name__}'.")

    if 'value' not in timepoints.columns:
        raise ValueError("'time points' must have at least a column 'value' to store time points value.")

    timepoints["value"] = sorted(as_tp_list(timepoints["value"]))

    timepoints = VDataFrame(timepoints)
    log_timepoints(timepoints)
    
    return timepoints


def check_time_match(data: ParsingDataIn) -> None:
    """
    Build timepoints DataFrame if it was not given by the user but 'time_list' or 'time_col_name' were given.
    Otherwise, if both timepoints and 'time_list' or 'time_col_name' were given, check that they match.
    """
    if data.timepoints.empty and data.time_list is None and data.time_col_name is None:
        # timepoints cannot be guessed
        return
    
    # build timepoints DataFrame from time_list or time_col_name
    if data.timepoints.empty and data.time_list is not None:
        data.timepoints = VDataFrame({'value': np.unique(data.time_list)})
        return

    if data.timepoints.empty and len(data.layers):
        data.timepoints = VDataFrame({'value': first(data.layers).timepoints})
        return

    # check that timepoints and _time_list and _time_col_name match
    if data.time_list is not None and not all(match_timepoints(data.time_list, data.timepoints.value)):
        raise ValueError("There are values in 'time_list' unknown in 'timepoints'.")

    elif data.time_col_name is not None and not all(match_timepoints(data.obs.timepoints, data.timepoints.value)):
        raise ValueError(f"There are values in obs['{data.time_col_name}'] unknown in 'timepoints'.")

    data.timepoints = VDataFrame(data.timepoints)