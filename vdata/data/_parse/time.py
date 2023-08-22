from __future__ import annotations

from typing import TYPE_CHECKING, Sequence, cast

import numpy as np
import pandas as pd
from h5dataframe import H5DataFrame

import vdata.timepoint as tp
from vdata.array_view import NDArrayView
from vdata.data._parse.utils import log_timepoints
from vdata.IO.logger import generalLogger
from vdata.tdf import TemporalDataFrameBase, TemporalDataFrameView
from vdata.utils import first_in

if TYPE_CHECKING:
    from vdata.data._parse.data import ParsingDataIn


def parse_timepoints_list(
    timepoints_list: Sequence[str | tp.TimePoint] | tp.TimePointArray | None,
    time_col_name: str | None,
    obs: pd.DataFrame | H5DataFrame | TemporalDataFrameBase | None,
) -> tp.TimePointArray | NDArrayView[tp.TimePoint] | None:
    if timepoints_list is not None:
        return tp.as_timepointarray(timepoints_list)

    elif obs is not None and time_col_name is not None:
        if time_col_name not in obs.columns:
            raise ValueError(f"Could not find column '{time_col_name}' in obs.")

        if isinstance(obs, TemporalDataFrameBase):
            column = cast(TemporalDataFrameView, obs[time_col_name])
            return tp.as_timepointarray(column.values)

        return tp.as_timepointarray(obs[time_col_name])

    return None

    # TODO : could also get timepoints_list from obsm and obsp


def parse_timepoints(timepoints: pd.DataFrame | H5DataFrame | None) -> H5DataFrame:
    if timepoints is None:
        generalLogger.debug("  'time points' DataFrame was not given.")
        return H5DataFrame(pd.DataFrame(columns=["value"]))

    if not isinstance(timepoints, (pd.DataFrame, H5DataFrame)):
        raise TypeError(f"'time points' must be a DataFrame, got '{type(timepoints).__name__}'.")

    if "value" not in timepoints.columns:
        raise ValueError("'time points' must have at least a column 'value' to store time points value.")

    timepoints = H5DataFrame(timepoints)
    timepoints["value"] = sorted(tp.as_timepointarray(timepoints["value"]))
    log_timepoints(timepoints)

    return timepoints


def check_time_match(data: ParsingDataIn) -> None:
    """
    Build timepoints DataFrame if it was not given by the user but 'timepoints_list' or 'time_col_name' were given.
    Otherwise, if both timepoints and 'timepoints_list' or 'time_col_name' were given, check that they match.
    """
    if data.timepoints.empty and data.timepoints_list is None and data.time_col_name is None:
        # timepoints cannot be guessed
        return

    # build timepoints DataFrame from timepoints_list or time_col_name
    if data.timepoints.empty and data.timepoints_list is not None:
        data.timepoints["value"] = np.unique(data.timepoints_list, equal_nan=False)
        return

    if data.timepoints.empty and len(data.layers):
        data.timepoints["value"] = first_in(data.layers).timepoints
        return

    # check that timepoints and _time_list and _time_col_name match
    if data.timepoints_list is not None and not np.all(
        np.in1d(data.timepoints_list, tp.as_timepointarray(data.timepoints.value))
    ):
        raise ValueError("There are values in 'timepoints_list' unknown in 'timepoints'.")

    elif data.time_col_name is not None and not np.all(
        np.in1d(tp.as_timepointarray(data.obs.timepoints), tp.as_timepointarray(data.timepoints.value))
    ):
        raise ValueError(f"There are values in obs['{data.time_col_name}'] unknown in 'timepoints'.")
