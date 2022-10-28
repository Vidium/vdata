# coding: utf-8
# Created on 12/04/2022 15:05
# Author : matteo

# ====================================================
# imports
from __future__ import annotations

import numpy as np
from numbers import Number

from typing import Any, TYPE_CHECKING, Collection, Union

from vdata.utils import isCollection

if TYPE_CHECKING:
    from .base import BaseTemporalDataFrame, BaseTemporalDataFrameView


# ====================================================
# code
class VAtIndexer:
    """
    Access a single value in a TemporalDataFrame, from a pair of row and column labels.
    """

    __slots__ = '_TDF'

    # region magic methods
    def __init__(self,
                 TDF: BaseTemporalDataFrame):
        self._TDF = TDF

    def __getitem__(self,
                    key: tuple[Any, Any]) -> float | str:
        index, column = key

        if column in self._TDF.columns_num:
            a = self._TDF[:, index, column].values_num
            return a[0, 0]

        return self._TDF[:, index, column].values_str[0, 0]

    def __setitem__(self,
                    key: tuple[Any, Any],
                    value: Number | np.number | str) -> None:
        index, column = key

        self._TDF[:, index, column] = value

    # endregion


class ViAtIndexer:
    """
    Access a single value in a TemporalDataFrame, from a pair of row and column indices.
    """

    __slots__ = '_TDF'

    # region magic methods
    def __init__(self,
                 TDF: BaseTemporalDataFrame):
        self._TDF = TDF

    def __getitem__(self,
                    key: tuple[int, int]) -> BaseTemporalDataFrameView:
        index_id, column_id = key
        column = self._TDF.columns[column_id]

        if column in self._TDF.columns_num:
            return self._TDF[:, self._TDF.index[index_id], column].values_num[0, 0]

        return self._TDF[:, self._TDF.index[index_id], column].values_str[0, 0]

    def __setitem__(self,
                    key: tuple[int, int],
                    value: Number | np.number | str) -> None:
        index_id, column_id = key

        self._TDF[:, self._TDF.index[index_id], self._TDF.columns[column_id]] = value

    # endregion


class VLocIndexer:
    """
    Access a group of rows and columns by label(s) or a boolean array.

    Allowed inputs are:
        - A single label, e.g. 5 or 'a', (note that 5 is interpreted as a label of the index, and never as an
        integer position along the index).
        - A list or array of labels, e.g. ['a', 'b', 'c'].
        - A slice object with labels, e.g. 'a':'f'.
        - A boolean array of the same length as the axis being sliced, e.g. [True, False, True].
        - A callable function with one argument (the calling Series or DataFrame) and that returns valid output
        for indexing (one of the above)
    """

    __slots__ = '_TDF'

    # region magic methods
    def __init__(self,
                 TDF: BaseTemporalDataFrame):
        self._TDF = TDF

    @staticmethod
    def __parse_slicer(values: Any | Collection[Any],
                       reference: np.ndarray) -> np.ndarray:
        if not isCollection(values):
            return values

        values = np.array(values)

        if values.dtype != bool:
            return values

        return reference[values]

    def __getitem__(self,
                    key: Any | Collection[Any] | tuple[Any | Collection[Any], Any | Collection[Any]]) \
            -> BaseTemporalDataFrameView:
        if isinstance(key, tuple):
            indices, columns = key

        else:
            indices, columns = key, slice(None)

        # parse indices
        indices = self.__parse_slicer(indices, self._TDF.index)

        # parse columns
        columns = self.__parse_slicer(columns, self._TDF.columns)

        return self._TDF[:, indices, columns]

    def __setitem__(self,
                    key: Any | Collection[Any] | tuple[Any | Collection[Any], Any | Collection[Any]],
                    value: Number | np.number | str | Collection) -> None:
        if isinstance(key, tuple):
            indices, columns = key

        else:
            indices, columns = key, slice(None)

        # parse indices
        indices = self.__parse_slicer(indices, self._TDF.index)

        # parse columns
        columns = self.__parse_slicer(columns, self._TDF.columns)

        self._TDF[:, indices, columns] = value

    # endregion


I_LOC_INDEX = Union[int, slice, Collection[int], Collection[bool]]


class ViLocIndexer:
    """
    Purely integer-location based indexing for selection by position (from 0 to length-1 of the axis).

    Allowed inputs are:
        - An integer, e.g. 5.
        - A list or array of integers, e.g. [4, 3, 0].
        - A slice object with ints, e.g. 1:7.
        - A boolean array.
        - A callable function with one argument (the calling Series or DataFrame) and that returns valid output
        for indexing (one of the above). This is useful in method chains, when you don’t have a reference to the
        calling object, but would like to base your selection on some value.
    """

    __slots__ = '_TDF'

    # region magic methods
    def __init__(self,
                 TDF: BaseTemporalDataFrame):
        self._TDF = TDF

    @staticmethod
    def __parse_slicer(values_index: I_LOC_INDEX,
                       reference: np.ndarray) -> np.ndarray:
        if not isCollection(values_index):
            return reference[values_index]

        return reference[np.array(values_index)]

    def __getitem__(self,
                    key: I_LOC_INDEX | tuple[I_LOC_INDEX, I_LOC_INDEX]) -> BaseTemporalDataFrameView:
        if isinstance(key, tuple):
            indices, columns = key

        else:
            indices, columns = key, slice(None)

        # parse indices
        indices = self.__parse_slicer(indices, self._TDF.index)

        # parse columns
        columns = self.__parse_slicer(columns, self._TDF.columns)

        return self._TDF[:, indices, columns]

    def __setitem__(self,
                    key: I_LOC_INDEX | tuple[I_LOC_INDEX, I_LOC_INDEX],
                    value: Number | np.number | str | Collection) -> None:
        if isinstance(key, tuple):
            indices, columns = key

        else:
            indices, columns = key, slice(None)

        # parse indices
        indices = self.__parse_slicer(indices, self._TDF.index)

        # parse columns
        columns = self.__parse_slicer(columns, self._TDF.columns)

        self._TDF[:, indices, columns] = value

    # endregion
