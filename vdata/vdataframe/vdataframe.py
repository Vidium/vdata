"""
VDataFrame wrapper around pandas DataFrames.
"""

from __future__ import annotations

from typing import Any, Collection, Iterable

import ch5mpy as ch
import numpy as np
import numpy.typing as npt
import pandas as pd
from pandas._typing import Axes, Dtype

from vdata.vdataframe.indexing import _iLocIndexer, _LocIndexer


def _get_column(
    name: np.int_ | np.float_ | np.str_,
    order: ch.H5Array[np.int_ | np.float_ | np.str_],
    data_num: ch.H5Array[np.int_ | np.float_],
    data_str: ch.H5Array[np.str_],
) -> ch.H5Array[np.int_ | np.float_ | np.str_]:
    index = np.where(order == name)[0][0]

    if index < data_num.shape[1]:
        return data_num[:, index]  # type: ignore[return-value]

    return data_str[:, index - data_num.shape[1]]  # type: ignore[return-value]


def _get_column_ordered(
    index: int, data_num: ch.H5Array[np.int_ | np.float_], data_str: ch.H5Array[np.str_]
) -> ch.H5Array[np.int_ | np.float_ | np.str_]:
    if index < data_num.shape[1]:
        return data_num[:, index]  # type: ignore[return-value]
    return data_str[:, index - data_num.shape[1]]  # type: ignore[return-value]


class VDataFrame(pd.DataFrame):
    """
    Simple wrapper around pandas DataFrames for managing index and columns modification when the DataFrame is read
    from a h5 file.
    """

    _internal_names_set = {"_file"} | pd.DataFrame._internal_names_set  # type: ignore[attr-defined]

    # region magic methods
    def __init__(
        self,
        data: npt.NDArray[Any] | Iterable[Any] | dict[Any, Any] | pd.DataFrame | None = None,
        index: Axes | None = None,
        columns: Axes | None = None,
        dtype: Dtype | None = None,
        copy: bool = False,
        file: ch.Group | None = None,
    ):
        """
        Args:
            file: an optional h5py group where this VDataFrame is read from.
        """
        super().__init__(data=data, index=index, columns=columns, dtype=dtype, copy=copy)  # type: ignore[call-arg]

        self._file = file

    def __h5_write__(self, group: ch.Group) -> None:
        if self._file is not None:
            return

        data_numeric = self.select_dtypes(include=[np.number, bool])  # type: ignore[list-item]
        data_string = self.select_dtypes(exclude=[np.number, bool])  # type: ignore[list-item]

        ch.write_objects(
            group,
            data_numeric=data_numeric.values.astype(np.float64),
            data_string=data_string.values.astype(str),
            index=np.array(self.index),
            columns=np.array(self.columns),
            columns_stored_order=np.concatenate((data_numeric.columns, data_string.columns)),
        )

    @classmethod
    def __h5_read__(cls, group: ch.Group) -> VDataFrame:
        if np.array_equal(group["columns"], group["columns_stored_order"]):
            return VDataFrame(
                data={
                    c: _get_column_ordered(i, group["data_numeric"], group["data_string"])
                    for i, c in enumerate(group["columns"])
                },
                index=group["index"],
                file=group,
            )

        return VDataFrame(
            data={
                c: _get_column(c, group["columns_stored_order"], group["data_numeric"], group["data_string"])
                for c in group["columns"]
            },
            index=group["index"],
            file=group,
        )

    # endregion

    # region attributes
    @property
    def is_backed(self) -> bool:
        """
        Is this VDataFrame backed on a h5 file ?

        Returns:
            Is this VDataFrame backed on a h5 file ?
        """
        return self._file is not None

    @property
    def file(self) -> ch.Group | None:
        """
        Get the h5 file this VDataFrame is backed on.
        :return: the h5 file this VDataFrame is backed on.
        """
        return self._file

    @property
    def index(self) -> pd.Index:
        """
        Get the index.
        """
        return super().index

    @index.setter
    def index(self, values: Collection[int | np.int_ | float | np.float_ | str | np.str_]) -> None:
        """
        Set the index (and write modifications to h5 file if backed).
        Args:
            values: new index to set.
        """
        self._set_axis(1, pd.Index(values))  # type: ignore[operator]

        if self._file is not None and self._file.file.mode == ch.H5Mode.READ_WRITE:
            self._file["index"][()] = list(values)
            self._file.file.flush()

    @property
    def columns(self) -> pd.Index:
        """
        Get the columns.
        """
        return super().columns

    @columns.setter
    def columns(self, values: Collection[int | np.int_ | float | np.float_ | str | np.str_]) -> None:
        """
        Set the columns (and write modifications to h5 file if backed).

        Args:
            values: new column names to set.
        """
        self._set_axis(0, pd.Index(values))  # type: ignore[operator]

        if self._file is not None and self._file.file.mode == ch.H5Mode.READ_WRITE:
            self._file.attrs["column_order"] = list(values)

            for col_index, col in enumerate(values):
                self._file.move(str(self.axes[1][col_index]), str(col))

            self._file.file.flush()

    @property
    def loc(self) -> _LocIndexer:  # type: ignore[override]
        return _LocIndexer("loc", self)

    @property
    def iloc(self) -> _iLocIndexer:  # type: ignore[override]
        return _iLocIndexer("iloc", self)

    # endregion
