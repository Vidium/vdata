# coding: utf-8
# Created on 09/02/2021 15:37
# Author : matteo

# ====================================================
# imports
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from typing import Iterable, Generator, Union, Tuple, Any, List, Collection, Optional, Dict
from typing_extensions import Literal

import vdata
from vdata.NameUtils import PreSlicer
from vdata.utils import TimePoint


# ====================================================
# code
class BaseTemporalDataFrame(ABC):
    """
    Base abstract class for TemporalDataFrames and ViewTemporalDataFrames.
    """

    __base_repr_str = ''

    @property
    @abstractmethod
    def index(self) -> pd.Index:
        """
        Get the full index of this TemporalDataFrame (concatenated over all time points).
        :return: the full index of this TemporalDataFrame.
        """
        pass

    @property
    def n_index_total(self) -> int:
        """
        Get the number of indexes.
        :return: the number of indexes.
        """
        return len(self.index)

    def __len__(self) -> int:
        """
        Returns the length of info axis.
        :return: the length of info axis.
        """
        return self.n_index_total

    @abstractmethod
    def __getitem__(self, index: Union[PreSlicer,
                                       Tuple[PreSlicer],
                                       Tuple[PreSlicer, PreSlicer],
                                       Tuple[PreSlicer, PreSlicer, PreSlicer]]) \
            -> 'views.ViewTemporalDataFrame':
        """
        Get a view from this TemporalDataFrame using an index with the usual sub-setting mechanics.
        :param index: A sub-setting index. It can be a single index, a 2-tuple or a 3-tuple of indexes.
            An index can be a string, an int, a float, a sequence of those, a range, a slice or an ellipsis ('...').
            Indexes are converted to a 3-tuple :
                * TDF[index]            --> (index, :, :)
                * TDF[index1, index2]   --> (index1, index2, :)

            The first element in the 3-tuple is the list of time points to select, the second element is a
            collection of rows to select, the third element is a collection of columns to select.

            The values ':' or '...' are shortcuts for 'take all values'.

            Example:
                * TemporalDataFrame[:] or TemporalDataFrame[...]    --> select all data
                * TemporalDataFrame[0]                              --> select all data from time point 0
                * TemporalDataFrame[[0, 1], [1, 2, 3], 'col1']      --> select data from time points 0 and 1 for rows
                                                                    with index in [1, 2, 3] and column 'col1'
        :return: a view on a sub-set of a TemporalDataFrame
        """
        pass

    def __setitem__(self, index: Union[PreSlicer, Tuple[PreSlicer], Tuple[PreSlicer, Collection[bool]]],
                    values: Any) -> None:
        """
        Set values in the TemporalDataFrame.
        The columns and the rows must match.
        :param index: a sub-setting index. (see __getitem__ for more details)
        :param values: a DataFrame with values to set.
        """
        self[index].set(values)

    @property
    @abstractmethod
    def time_points(self) -> List[TimePoint]:
        """
        Get the list of time points in this TemporalDataFrame.
        :return: the list of time points in this TemporalDataFrame.
        """
        pass

    @property
    def n_time_points(self) -> int:
        """
        Get the number of distinct time points in this TemporalDataFrame.
        :return: the number of time points.
        """
        return len(self.time_points)

    @abstractmethod
    def index_at(self, time_point: TimePoint) -> pd.Index:
        """
        Get the index of this TemporalDataFrame.
        :param time_point: a time point in this TemporalDataFrame.
        :return: the index of this TemporalDataFrame.
        """
        pass

    def n_index_at(self, time_point: TimePoint) -> int:
        """
        Get the length of the index at a given time point.
        :param time_point: a time point in this TemporalDataFrame.
        :return: the length of the index at a given time point.
        """
        return len(self.index_at(time_point))

    @property
    def time_points_column(self) -> pd.Series:
        """
        Get the time points data for all rows in this TemporalDataFrame.
        :return: the time points data.
        """
        _data = pd.Series([])

        for time_point in self.time_points:
            # noinspection PyTypeChecker
            _data = pd.concat((_data, pd.Series(np.repeat(time_point, self.n_index_at(time_point)))))

        _data.index = self.index

        return _data

    @property
    @abstractmethod
    def time_points_column_name(self) -> Optional[str]:
        """
        Get the name of the column with time points data. Returns None if '__TPID' is used.
        :return: the name of the column with time points data.
        """
        pass

    @property
    @abstractmethod
    def columns(self) -> pd.Index:
        """
        Get the columns of this TemporalDataFrame.
        :return: the column names of this TemporalDataFrame.
        """
        pass

    @property
    def n_columns(self) -> int:
        """
        Get the number of columns in this TemporalDataFrame
        :return: the number of columns
        """
        return len(self.columns)

    @property
    @abstractmethod
    def name(self) -> str:
        """
        Get the name of this TemporalDataFrame.
        :return: the name of this TemporalDataFrame.
        """
        pass

    @abstractmethod
    def to_pandas(self, with_time_points: bool = False) -> pd.DataFrame:
        """
        Get the data in a pandas format.
        :param with_time_points: add a column with time points data ?
        :return: the data in a pandas format.
        """
        pass

    def _asmd_func(self, operation: Literal['__add__', '__sub__', '__mul__', '__truediv__'],
                   value: Union[int, float]) -> 'vdata.TemporalDataFrame':
        """
        Common function for modifying all values in this TemporalDataFrame through the common operation (+, -, *, /).
        :param operation: the operation to apply on the TemporalDataFrame.
        :param value: an int or a float to add to values.
        :return: a TemporalDataFrame with new values.
        """
        _data = self.to_pandas()

        # avoid working on time points
        if self.time_points_column_name is not None:
            _data = _data.loc[:, _data.columns != self.time_points_column_name]

        # transform the data by the operation and the value
        _data = getattr(_data, operation)(value)

        # insert back the time points
        if self.time_points_column_name is not None:
            _data.insert(list(self.columns).index(self.time_points_column_name), self.time_points_column_name,
                         self.time_points_column)

        time_col = self.time_points_column_name
        time_list = self.time_points_column if time_col is None else None

        return vdata.TemporalDataFrame(data=_data,
                                       time_list=time_list,
                                       time_col=time_col,
                                       time_points=self.time_points,
                                       index=self.index,
                                       name=self.name)

    @property
    def values(self) -> np.ndarray:
        """
        Return a Numpy representation of this TemporalDataFrame.
        :return: a Numpy representation of the DataFrame.
        """
        return self.to_pandas().values

    @property
    def axes(self) -> List[pd.Index]:
        """
        Return a list of the row axis labels.
        :return: a list of the row axis labels.
        """
        return [self.index, self.columns]

    @property
    def ndim(self) -> Literal[3]:
        """
        Return an int representing the number of axes / array dimensions.
        :return: 3
        """
        return 3

    @property
    def size(self) -> int:
        """
        Return the number of rows times number of columns.
        :return: an int representing the number of elements in this object.
        """
        return len(self.columns) * self.n_index_total

    @property
    def shape(self) -> Tuple[int, List[int], int]:
        """
        Return a tuple representing the dimensionality of this TemporalDataFrame
        (nb_time_points, [n_index_at(time point) for all time points], nb_col).
        :return: a tuple representing the dimensionality of this TemporalDataFrame
        """
        return self.n_time_points, [self.n_index_at(TP) for TP in self.time_points], self.n_columns

    @property
    def empty(self) -> bool:
        """
        Indicator whether this TemporalDataFrame is empty.
        :return: True if this TemporalDataFrame is empty.
        """
        if not self.n_time_points or not self.n_columns or not self.n_index_total:
            return True

        return False

    # TODO : these two functions should print and not return, to be usable by the user
    # TODO : they could be used better with time points in one_tp_repr()
    def head(self, n: int = 5, time_points: PreSlicer = slice(None, None, None)) -> str:
        """
        This function returns the first n rows for the object based on position.

        For negative values of n, this function returns all rows except the last n rows.
        :return: the first n rows.
        """
        sub_TDF = self[time_points]

        if sub_TDF.n_time_points:
            repr_str = ""
            for TP in sub_TDF.time_points:
                repr_str += f"\033[4mTime point : {TP}\033[0m\n"
                repr_str += f"{sub_TDF[TP].one_TP_repr(TP, n)}\n"

        else:
            repr_str = f"Empty {self.__class__.__base_repr_str}\n" \
                       f"Columns: {[col for col in sub_TDF.columns]}\n" \
                       f"Index: {[idx for idx in sub_TDF.index]}"

        return repr_str

    def tail(self, n: int = 5, time_points: PreSlicer = slice(None, None, None)) -> str:
        """
        This function returns the last n rows for the object based on position.

        For negative values of n, this function returns all rows except the first n rows.
        :return: the last n rows.
        """
        sub_TDF = self[time_points]

        if sub_TDF.n_time_points:
            repr_str = ""
            for TP in sub_TDF.time_points:
                repr_str += f"\033[4mTime point : {TP}\033[0m\n"
                repr_str += f"{sub_TDF[TP].one_TP_repr(TP, n, func='tail')}\n"

        else:
            repr_str = f"Empty {self.__class__.__base_repr_str}\n" \
                       f"Columns: {[col for col in sub_TDF.columns]}\n" \
                       f"Index: {[idx for idx in sub_TDF.index]}"

        return repr_str

    def keys(self) -> pd.Index:
        """
        Get the ‘info axis’.
        :return: the ‘info axis’.
        """
        return self.columns

    def items(self) -> Generator[Tuple[str, pd.Series], None, None]:
        """
        Iterate over (column name, Series) pairs.
        :return: a tuple with the column name and the content as a Series.
        """
        _data = self.to_pandas()

        for column in self.columns:
            yield column, _data[column]

    def isin(self, values: Union[Iterable, pd.Series, pd.DataFrame, Dict]) -> 'vdata.TemporalDataFrame':
        """
        Whether each element in the TemporalDataFrame is contained in values.
        :return: whether each element in the DataFrame is contained in values.
        """
        _time_col = self.time_points_column_name
        _time_list = self.time_points_column if _time_col is None else None

        return vdata.TemporalDataFrame(self.to_pandas().isin(values), time_list=_time_list, time_col=_time_col)

    def eq(self, other: Any, axis: Literal[0, 1, 'index', 'column'] = 'columns',
           level: Any = None) -> 'vdata.TemporalDataFrame':
        """
        Get Equal to of TemporalDataFrame and other, element-wise (binary operator eq).
        Equivalent to '=='.
        :param other: Any single or multiple element data structure, or list-like object.
        :param axis: {0 or ‘index’, 1 or ‘columns’}
        :param level: int or label
        """
        _time_col = self.time_points_column_name
        _time_list = self.time_points_column if self.time_points_column_name is None else None

        return vdata.TemporalDataFrame(self.to_pandas().eq(other, axis, level), time_list=_time_list,
                                       time_col=_time_col)