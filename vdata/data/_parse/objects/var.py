from __future__ import annotations

from typing import TYPE_CHECKING, Mapping

import numpy as np
import pandas as pd
from h5dataframe import H5DataFrame

from vdata._typing import NDArray_IFS
from vdata.IO.logger import generalLogger
from vdata.tdf import TemporalDataFrameBase
from vdata.utils import first_in, obj_as_str

if TYPE_CHECKING:
    from vdata.data._parse.data import ParsingDataIn


def get_var_index(
    data: pd.DataFrame
    | H5DataFrame
    | TemporalDataFrameBase
    | Mapping[str, pd.DataFrame | H5DataFrame | TemporalDataFrameBase]
    | None,
    var: pd.DataFrame | H5DataFrame | None,
) -> NDArray_IFS | None:
    if var is not None:
        return obj_as_str(np.array(var.index))

    if isinstance(data, (pd.DataFrame, H5DataFrame, TemporalDataFrameBase)):
        return obj_as_str(np.array(data.columns))

    if isinstance(data, dict):
        return obj_as_str(np.array(first_in(data).columns))

    return None


def parse_varm(data: ParsingDataIn) -> dict[str, H5DataFrame]:
    if not len(data.varm):
        generalLogger.debug("    6. \u2717 'varm' was not given.")
        return {}

    if not isinstance(data.varm, dict):
        raise TypeError("'varm' must be a dictionary of pandas DataFrames.")

    generalLogger.debug(lambda: f"    6. \u2713 'varm' is a {type(data.varm).__name__}.")

    if data.var is None and not len(data.layers):
        raise ValueError("'varm' parameter cannot be set unless either 'data' or 'var' are set.")

    valid_varm: dict[str, H5DataFrame] = {}

    for key, value in data.varm.items():
        if not isinstance(value, pd.DataFrame):
            raise TypeError(f"'varm' '{key}' must be a pandas DataFrame.")

        if not np.all(np.isin(value.index, data.var.index)):
            raise ValueError("Index of 'varm' does not match 'var' and 'layers' column names.")

        valid_varm[str(key)] = H5DataFrame(value)
        valid_varm[str(key)].reindex(data.var.index)

    return valid_varm


def parse_varp(data: ParsingDataIn) -> dict[str, H5DataFrame]:
    if not len(data.varp):
        generalLogger.debug("    7. \u2717 'varp' was not given.")
        return {}

    if not isinstance(data.varp, dict):
        raise TypeError("'varp' must be a dictionary of 2D numpy arrays or of pandas DataFrames.")

    if data.var is None and not len(data.layers):
        raise ValueError("'varp' parameter cannot be set unless either 'data' or 'var' are set.")

    generalLogger.debug(lambda: f"    7. \u2713 'varp' is a {type(data.varp).__name__}.")

    valid_varp: dict[str, H5DataFrame] = {}

    for key, value in data.varp.items():
        if not isinstance(value, (np.ndarray, pd.DataFrame)) or value.ndim != 2:
            raise TypeError(f"'varp' '{key}' must be a 2D numpy array or a pandas DataFrame.")

        if isinstance(value, pd.DataFrame):
            if not all(value.index.isin(data.var.index)):
                raise ValueError(f"Index of 'varp' '{key}' does not match 'var' and 'layers' column names.")

            if not all(value.columns.isin(data.var.index)):
                raise ValueError(f"Column names of 'varp' '{key}' do not match 'var' and 'layers' column names.")

            value.reindex(data.var.index)
            value = value[data.var.index]

        else:
            value = pd.DataFrame(value, index=data.var.index, columns=data.var.index)

        valid_varp[str(key)] = H5DataFrame(value)

    return valid_varp
