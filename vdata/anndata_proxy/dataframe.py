import numpy as np
import pandas as pd

from vdata.tdf import TemporalDataFrameBase


class DataFrameproxy:

    # region magic methods
    def __init__(self, tdf: TemporalDataFrameBase) -> None:
        self._tdf = tdf

    # endregion

    # region attributes
    @property
    def index(self) -> pd.Index:
        return pd.Index(self._tdf.index)

    @property
    def columns(self) -> pd.Index:
        return pd.Index(np.concatenate((np.array([self._tdf.get_timepoints_column_name()]), self._tdf.columns)))

    # endregion
