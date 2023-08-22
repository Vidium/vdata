from __future__ import annotations

from typing import Collection, cast

import numpy as np
import pandas as pd

from vdata._typing import IFS, Slicer
from vdata.tdf import TemporalDataFrameBase, TemporalDataFrameView
from vdata.timepoint import TimePointArray


class DataFrameProxy_TDF:

    __slots__ = ("_tdf",)

    # region magic methods
    def __init__(self, tdf: TemporalDataFrameBase) -> None:
        self._tdf = tdf

    def __repr__(self) -> str:
        return f"Proxy<TDF -> DataFrame> for\n{self._tdf}"

    def __getitem__(self, key: Slicer) -> pd.Series[int | float | str] | TimePointArray:
        if key == self._tdf.get_timepoints_column_name():
            return self._tdf.timepoints_column

        column = cast(TemporalDataFrameView, self._tdf[:, :, key])
        return pd.Series(column.values.flatten(), index=self._tdf.index.values)

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
        return pd.Index(self._tdf.index.values)

    @property
    def columns(self) -> pd.Index:
        return pd.Index(np.concatenate((np.array([self._tdf.get_timepoints_column_name()]), self._tdf.columns)))

    # endregion

    # region methods
    def keys(self) -> pd.Index:
        return self.columns

    # endregion
