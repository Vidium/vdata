import numpy as np
import pandas as pd
from h5dataframe import H5DataFrame

from vdata.data._parse.data import ParsingDataIn
from vdata.IO.logger import generalLogger
from vdata.names import NO_NAME
from vdata.tdf import TemporalDataFrame


def _parse_data_from_dataframe(df: pd.DataFrame | H5DataFrame, data: ParsingDataIn) -> TemporalDataFrame:
    if len(data.timepoints) > 1:
        raise TypeError("'data' is a 2D pandas DataFrame but more than 1 time-point were provided.")

    tdf = TemporalDataFrame(
        df,
        timepoints=data.timepoints_list,
        time_col_name=data.time_col_name,
        name="data",
    )

    if data.obs is not None and not isinstance(data.obs, TemporalDataFrame) and data.timepoints_list is None:
        data.timepoints_list = tdf.timepoints_column

    return tdf


def _parse_data_from_tdf(tdf: TemporalDataFrame, data: ParsingDataIn) -> TemporalDataFrame:
    tdf.unlock_indices()
    tdf.unlock_columns()

    if data.timepoints.empty:
        data.timepoints = pd.DataFrame({"value": tdf.timepoints})

    elif np.any(data.timepoints.value.values != tdf.timepoints):
        raise ValueError("'time points' found in DataFrame do not match 'layers' time points.")

    if data.obs is not None and not isinstance(data.obs, TemporalDataFrame) and data.timepoints_list is None:
        data.timepoints_list = tdf.timepoints_column

    return tdf.copy()


def parse_layers(data: ParsingDataIn) -> None:
    if data.data is None:
        generalLogger.debug("    1. \u2717 'data' was not given.")
        return

    if isinstance(data.data, (pd.DataFrame, H5DataFrame)):
        generalLogger.debug("    1. \u2713 'data' is a DataFrame.")
        data.layers["data"] = _parse_data_from_dataframe(data.data, data)
        return

    if isinstance(data.data, TemporalDataFrame):
        generalLogger.debug("    1. \u2713 'data' is a TemporalDataFrame.")
        data.layers["data"] = _parse_data_from_tdf(data.data, data)
        return

    if isinstance(data.data, dict):
        generalLogger.debug("    1. \u2713 'data' is a dictionary.")

        for key, value in data.data.items():
            if isinstance(value, (pd.DataFrame, H5DataFrame)):
                generalLogger.debug(lambda: f"        \u2713 '{key}' is DataFrame.")
                data.layers[str(key)] = _parse_data_from_dataframe(value, data)

            elif isinstance(value, TemporalDataFrame):
                generalLogger.debug(lambda: f"        \u2713 '{key}' is TemporalDataFrame.")
                _layer = _parse_data_from_tdf(value, data)

                if _layer.name != str(key):
                    _layer.name = str(key) if _layer.name == NO_NAME else f"{_layer.name}_{key}"

                data.layers[str(key)] = _layer

            else:
                raise TypeError(
                    f"Layer '{key}' must be a TemporalDataFrame or a pandas DataFrame, " f"it is a {type(value)}."
                )

        return

    raise TypeError(
        f"Type '{type(data.data)}' is not allowed for 'data' parameter, should be a dict,"
        f"a pandas DataFrame, a TemporalDataFrame or an AnnData object."
    )
