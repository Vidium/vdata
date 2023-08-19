from __future__ import annotations

from typing import Collection

import numpy as np
import pandas as pd

from vdata._typing import IFS, Slicer
from vdata.tdf import TemporalDataFrameBase


class DataFrameProxy_TDF:

    __slots__ = ("_tdf",)

    # region magic methods
    def __init__(self, tdf: TemporalDataFrameBase) -> None:
        self._tdf = tdf

    def __repr__(self) -> str:
        return f"Proxy<TDF -> DataFrame> for\n{self._tdf}"

    def __getitem__(self, key: Slicer) -> pd.Series[int | float | str]:
        if key == self._tdf.get_timepoints_column_name():
            return self._tdf.timepoints_column

        return pd.Series(self._tdf[:, :, key].values.flatten(), index=self._tdf.index)

    def __setitem__(
        self,
        key: Slicer,
        values: IFS | Collection[IFS],
    ) -> None:
        self._tdf[:, :, key] = values

    # endregion

    # region attributes
    @property
    def index(self) -> pd.Index:
        return pd.Index(self._tdf.index)

    @property
    def columns(self) -> pd.Index:
        return pd.Index(np.concatenate((np.array([self._tdf.get_timepoints_column_name()]), self._tdf.columns)))

    # endregion

    # region methods
    def keys(self) -> pd.Index:
        return self.columns

    # endregion
