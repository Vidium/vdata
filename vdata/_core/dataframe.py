# coding: utf-8
# Created on 12/9/20 9:17 AM
# Author : matteo

# ====================================================
# imports
import pandas as pd
import numpy as np
from typing import Dict, Union, Optional, Collection, Tuple, Any, List, IO
from typing_extensions import Literal

from .._IO.errors import VValueError, VTypeError, ShapeError
from .._IO.logger import generalLogger
from ..NameUtils import DType, PreSlicer
from ..utils import slice_to_range


# ====================================================
# code
class TemporalDataFrame:
    """
    An extension of pandas DataFrames to include a notion of time on the rows.
    An hidden column '__TPID' contains for each row the list of time points this row appears in.
    This class implements a modified sub-setting mechanism to subset on time points and on the regular conditional
    selection.
    """

    def __init__(self, data: Optional[Union[Dict, pd.DataFrame]] = None,
                 time_points: Optional[Union[Collection, DType, Literal['*']]] = None,
                 time_col: Optional[str] = None,
                 index: Optional[Collection] = None,
                 columns: Optional[Collection] = None,
                 dtype: Optional[DType] = None):
        """
        :param data: data to store as a dataframe
        :param time_points: time points for the dataframe's rows. The value indicates at which time point a given row
            exists in the dataframe.
            It can be :
                - a collection of values of the same length as the number of rows.
                - a single value to set for all rows.

            In any case, the values can be :
                - a single time point (indicating that the row only exists at that given time point)
                - a collection of time points (indicating that the row exists at all those time points)
                - the character '*' (indicating that the row exists at all time points)

        :param time_col: if time points are not given explicitly with the 'time_points' parameter, a column name can be
            given. This column will be used as the time data.
        :param index: indexes for the dataframe's rows
        :param columns: column labels
        :param dtype: data type to force
        """
        self._time_points_col = '__TPID'

        if time_points is not None:
            time_points = list(map(str, time_points))

        if columns is not None:
            columns = ['__TPID'] + list(columns)

        # no data given, empty DataFrame is created
        if data is None or (isinstance(data, dict) and not len(data.keys())) or (isinstance(data, pd.DataFrame) and
                                                                                 not len(data.columns)):
            generalLogger.debug('Setting empty TemporalDataFrame.')

            if time_points is None:
                time_points = np.repeat('0', len(index)) if index is not None else []

            if time_col is not None:
                generalLogger.warning("Both 'time_points' and 'time_col' parameters were set, 'time_col' will be "
                                      "ignored.")

            self._df = pd.DataFrame({'__TPID': time_points}, index=index, columns=columns, dtype=dtype)

        # data given
        elif isinstance(data, (dict, pd.DataFrame)):
            # if data is a dict, check that the dict can be converted to a DataFrame
            if isinstance(data, dict):
                # get number of rows in data
                data_len = 1
                values = data.values() if isinstance(data, dict) else data.values
                for value in values:
                    value_len = len(value) if hasattr(value, '__len__') else 1

                    if value_len != data_len and data_len != 1 and value_len != 1:
                        raise ShapeError("All items in 'data' must have the same length "
                                         "(or be a unique value to set for all rows).")

                    if data_len == 1:
                        data_len = value_len

            else:
                data_len = len(data)

            # no time points given
            if time_points is None:
                # no column to use as time point : all time points set to 0 by default
                if time_col is None:
                    generalLogger.info(f"Setting all time points to default value '0'.")
                    time_points = ['0' for _ in range(data_len)]

                # a column has been given to use as time point : check that it exists
                elif (isinstance(data, dict) and time_col in data.keys()) or \
                        (isinstance(data, pd.DataFrame) and time_col in data.columns):
                    generalLogger.info(f"Using '{time_col}' as time points data.")
                    time_points = data[time_col]
                    self._time_points_col = time_col

                else:
                    raise VValueError(f"'{time_col}' could not be found in the supplied DataFrame's columns.")

            # time points given, check length is correct
            else:
                if time_col is not None:
                    generalLogger.warning("Both 'time_points' and 'time_col' parameters were set, 'time_col' will be "
                                          "ignored.")

                if len(time_points) != data_len:
                    raise VValueError(f"Supplied time points must be of length {data_len}.")

            if isinstance(data, pd.DataFrame):
                data = {col: data[col].values for col in data.columns}

            if '__TPID' in data.keys():
                raise VValueError("'__TPID' key is reserved and cannot be used in 'data'.")

            # check that all values in time_points are hashable
            hashable_time_points = []
            for tp in time_points:
                try:
                    hash(tp)
                except TypeError:
                    if isinstance(tp, (list, np.ndarray)):
                        hashable_time_points.append(tuple(tp))
                    else:
                        raise VTypeError(f"Un-hashable type '{type(tp)}' cannot be used for time points.")
                else:
                    hashable_time_points.append(tp)

            generalLogger.debug('Setting TemporalDataFrame from data.')
            self._df = pd.DataFrame(dict({'__TPID': hashable_time_points}, **data), index=index, columns=columns,
                                    dtype=dtype)

        else:
            raise VTypeError(f"Type {type(data)} is not handled for 'data' parameter.")

    def __repr__(self) -> str:
        """
        Description for this TemporalDataFrame object to print.
        :return: a description of this TemporalDataFrame object
        """
        if self.n_time_points:
            repr_str = ""
            for TP in self.time_points:
                repr_str += f"\033[4mTime point : {TP}\033[0m\n"
                repr_str += f"{self[TP]._one_TP_repr(TP)}\n"

        else:
            repr_str = f"Empty TemporalDataFrame\n" \
                       f"Columns: {[col for col in self.columns]}\n" \
                       f"Index: {[idx for idx in self.index]}"

        return repr_str

    @staticmethod
    def match(tp_list: pd.Series, tp_index: Collection[str]) -> List[bool]:
        mask = [False for _ in range(len(tp_list))]

        for tp_i, tp_obs in enumerate(tp_list):
            if tp_obs == '*':
                mask[tp_i] = True

            else:
                if issubclass(type(tp_obs), str) or not hasattr(tp_obs, '__iter__'):
                    tp_obs_iter = (tp_obs,)
                else:
                    tp_obs_iter = map(str, tp_obs)

                for one_tp_obs in tp_obs_iter:
                    if one_tp_obs in tp_index:
                        mask[tp_i] = True
                        break
        return mask

    def _one_TP_repr(self, time_point: str, n: Optional[int] = None):
        """
        Representation of a single time point in this TemporalDataFrame to print.
        :param time_point: the time point to represent.
        :param n: the number of rows to print. Defaults to all.
        :return: a representation of a single time point in this TemporalDataFrame object
        """
        mask = self.match(self._df['__TPID'], time_point)
        return repr(self._df[mask][self.columns].head(n=n))

    def __getitem__(self, index: Union[PreSlicer, Tuple[PreSlicer], Tuple[PreSlicer, Collection[bool]]]) -> \
            'TemporalDataFrame':
        """
        Get data from the DataFrame using an index with the usual sub-setting mechanics.
        :param index: A sub-setting index. It can be a single index or a 2-tuple of indexes.
            An index can be a string, an int, a float, a sequence of those, a range, a slice or an ellipsis ('...').
            Single indexes are converted to a 2-tuple :
                * single index --> (index, :)

            The first element in the 2-tuple is the list of time points to select, the second element is a
            collection of bool that can be obtained from conditions on the TemporalDataFrame as it is done with
            pandas DataFrames.

            The values ':' or '...' are shortcuts for 'take all values'.

            Example:
                * TemporalDataFrame[:] or TemporalDataFrame[...]    --> select all data
                * TemporalDataFrame[0]                              --> select all data from time point 0
                * TemporalDataFrame[[0, 1], <condition>]            --> select all data from time points 0 and 1 which
                                                                        match the condition. <condition> takes the form
                                                                        of a list of booleans indicating the rows to
                                                                        select with 'True'.
        :return: a sub-set TemporalDataFrame
        """
        # TODO : is this satisfying ? we copy the data to a completely new TDF, this can be long and mem-intensive ...
        # TODO : use views ?
        if not isinstance(index, tuple):
            index = [index, slice(None, None, None)]
        elif isinstance(index, tuple) and len(index) == 1:
            index = [index[0], slice(None, None, None)]
        else:
            index = [index[0], index[1]]

        # check first index (subset on time points)
        if isinstance(index[0], type(Ellipsis)) or index[0] == slice(None, None, None):
            index[0] = list(map(str, pd.Categorical(self._df['__TPID'].values).categories.values))

        elif isinstance(index[0], slice):
            max_value = max(self._df['__TPID'].values)
            if not issubclass(type(max_value), (int, np.int_)):
                raise VTypeError("Cannot slice on time points since time points are not ints.")

            index[0] = list(map(str, slice_to_range(index[0], max_value)))

        elif not hasattr(index[0], '__len__') or issubclass(type(index[0]), np.str_):
            index[0] = [str(index[0])]

        else:
            str_index = []
            for v in index[0]:
                if not hasattr(v, '__iter__') or issubclass(type(v), str):
                    str_index.append(str(v))

                else:
                    str_index.append(list(map(str, v)))

            index[0] = str_index

        data_for_TP = self._df[self.match(self._df['__TPID'], index[0])]
        index_conditions = index[1][data_for_TP.index] if not isinstance(index[1], slice) else index[1]

        # get attributes
        data = data_for_TP[index_conditions]
        time_points = data['__TPID'] if self._time_points_col == '__TPID' else None
        time_col = self._time_points_col if self._time_points_col != '__TPID' else None
        index = data.index

        return TemporalDataFrame(data=data[self._df.columns[1:]], time_points=time_points, time_col=time_col, index=index)

    def __getattr__(self, attr: str) -> Any:
        """
        Get columns in the DataFrame (this is done for maintaining the pandas DataFrame behavior).
        :return: a column with name <attr> from the DataFrame
        """
        return getattr(self._df, attr)

    def __setitem__(self, key, value) -> None:
        """
        TODO
        """
        print('hi')
        pass

    @property
    def time_points(self) -> List[str]:
        """
        Get the list of time points in this TemporalDataFrame
        :return: the list of time points in this TemporalDataFrame
        """
        all_values = set(self._df[self._time_points_col].values)

        unique_values = set()
        for value in all_values:
            if isinstance(value, tuple):
                unique_values.union(set(value))

            else:
                unique_values.add(value)

        return sorted(map(str, unique_values - {'*'}))

    @property
    def columns(self) -> pd.Index:
        """
        Get the columns of the DataFrame (but mask the reserved __TPID column).
        :return: the column names of the DataFrame
        """
        return self._df.columns[1:]

    @property
    def index(self) -> pd.Index:
        """
        Get the index of the DataFrame.
        :return: the index of the DataFrame
        """
        return self._df.index

    @property
    def n_time_points(self) -> int:
        """
        :return: the number of time points
        """
        return len(self.time_points)

    def len_index(self, time_point: str) -> int:
        """
        :return: the length of the index at a given time point
        """
        return len(self[time_point].index)

    @property
    def n_columns(self) -> int:
        """
        :return: the number of columns
        """
        return len(self.columns)

    @property
    def dtypes(self) -> None:
        """
        Return the dtypes in the DataFrame.
        :return: the dtypes in the DataFrame.
        """
        return self._df[self.columns].dtypes

    def info(self, verbose: Optional[bool] = None, buf: Optional[IO[str]] = None, max_cols: Optional[int] = None,
             memory_usage: Optional[Union[bool, str]] = None, null_counts: Optional[bool] = None) -> None:
        """
        This method prints information about a DataFrame including the index dtype and columns, non-null values and
        memory usage.
        :return: a concise summary of a DataFrame.
        """
        return self._df.info(verbose=verbose, buf=buf, max_cols=max_cols, memory_usage=memory_usage,
                             null_counts=null_counts)

    @property
    def values(self) -> np.ndarray:
        """
        Return a Numpy representation of the DataFrame.
        :return: a Numpy representation of the DataFrame.
        """
        return self._df[self.columns].values

    @property
    def axes(self) -> List[pd.Index]:
        """
        Return a list of the row axis labels.
        :return: a list of the row axis labels.
        """
        return self._df[self.columns].axes

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
        return self._df[self.columns].size

    @property
    def shape(self) -> Tuple[int, List[int], int]:
        """
        Return a tuple representing the dimensionality of the DataFrame
        (nb_time_points, [len_index for all time points], nb_col)
        :return: a tuple representing the dimensionality of the DataFrame
        """
        return self.n_time_points, [self.len_index(TP) for TP in self.time_points], self.n_columns

    def memory_usage(self, index: bool = True, deep: bool = False):
        """
        Return the memory usage of each column in bytes.
        The memory usage can optionally include the contribution of the index and elements of object dtype.
        :return: the memory usage of each column in bytes.
        """
        return self._df.memory_usage(index=index, deep=deep)

    @property
    def empty(self) -> bool:
        """
        Indicator whether DataFrame is empty.
        :return: True if this TemporalDataFrame is empty.
        """
        return True if not self.n_time_points else all([self[TP]._df.empty for TP in self.time_points])

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
                repr_str += f"{sub_TDF[TP]._one_TP_repr(TP, n)}\n"

        else:
            repr_str = f"Empty TemporalDataFrame\n" \
                       f"Columns: {[col for col in sub_TDF.columns]}\n" \
                       f"Index: {[idx for idx in sub_TDF.index]}"

        return repr_str

