# coding: utf-8
# Created on 12/04/2022 15:05
# Author : matteo

# ====================================================
# imports
import numpy as np
from numbers import Number

from typing import Any, Union, TYPE_CHECKING

from .view import ViewTemporalDataFrame

if TYPE_CHECKING:
    from .dataframe import TemporalDataFrame


# ====================================================
# code
class VAtIndexer:
    """
    Access a single value in a TemporalDataFrame, from a pair of row and column labels.
    """

    __slots__ = '_TDF'

    def __init__(self,
                 TDF: Union['TemporalDataFrame', 'ViewTemporalDataFrame']):
        self._TDF = TDF

    def __getitem__(self,
                    key: tuple[Any, Any]) -> Union[float, str]:
        index, column = key

        if column in self._TDF.columns_num:
            return self._TDF[:, index, column].values_num[0, 0]

        return self._TDF[:, index, column].values_str[0, 0]

    def __setitem__(self,
                    key: tuple[Any, Any],
                    value: Union[Number, np.number, str]) -> None:
        index, column = key

        self._TDF[:, index, column] = value


class ViAtIndexer:
    """
    Access a single value in a TemporalDataFrame, from a pair of row and column indices.
    """

    __slots__ = '_TDF'

    def __init__(self,
                 TDF: Union['TemporalDataFrame', 'ViewTemporalDataFrame']):
        self._TDF = TDF

    def __getitem__(self,
                    key: tuple[int, int]) -> 'ViewTemporalDataFrame':
        index_id, column_id = key
        column = self._TDF.columns[column_id]

        if column in self._TDF.columns_num:
            return self._TDF[:, self._TDF.index[index_id], column].values_num[0, 0]

        return self._TDF[:, self._TDF.index[index_id], column].values_str[0, 0]

    def __setitem__(self,
                    key: tuple[int, int],
                    value: Union[Number, np.number, str]) -> None:
        index_id, column_id = key

        self._TDF[:, self._TDF.index[index_id], self._TDF.columns[column_id]] = value


class VLocIndexer:
    """"""


class ViLocIndexer:
    """"""
