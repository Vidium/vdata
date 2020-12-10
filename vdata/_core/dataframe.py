# coding: utf-8
# Created on 12/9/20 9:17 AM
# Author : matteo

# ====================================================
# imports
import pandas as pd
import numpy as np
from typing import Dict, Union, Optional, Collection, Tuple, Any, List
from typing_extensions import Literal

from .._IO.errors import VValueError, VTypeError, ShapeError
from .._IO.logger import generalLogger
from .._core import vdata
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

    def __init__(self, parent: 'vdata', data: Optional[Union[Dict, pd.DataFrame]] = None,
                 time_points: Optional[Union[Collection, DType, Literal['*']]] = None,
                 time_col: Optional[str] = None,
                 index: Optional[Collection] = None):
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
        """
        self._parent = parent

        # no data given, empty DataFrame is created
        if data is None or not len(data.keys()):
            if time_points is not None:
                raise VValueError("Cannot set time points if no data is supplied.")

            generalLogger.debug('Setting empty TemporalDataFrame.')
            self._df = pd.DataFrame()

        # data given
        elif isinstance(data, (dict, pd.DataFrame)):
            # get number of rows in data
            data_len = 1
            values = data.values() if isinstance(data, dict) else data.values
            for value in values:
                value_len = len(value) if hasattr(value, '__len__') else 1

                if value_len != data_len and data_len != 1 and value_len != 1:
                    raise ShapeError('All items in data must have the same length (or be a unique value to set for '
                                     'all rows).')

                if data_len == 1:
                    data_len = value_len

            # no time points given
            if time_points is None:
                # all set to 0 by default
                if time_col is None:
                    generalLogger.info(f"Setting all time points to default value '0'.")
                    time_points = [0 for _ in range(data_len)]

                elif (isinstance(data, dict) and time_col in data.keys()) or \
                        (isinstance(data, pd.DataFrame) and time_col in data.columns):
                    generalLogger.info(f"Using '{time_col}' as time points data.")
                    time_points = data[time_col]

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
                data = data.to_dict()

            if '__TPID' in data.keys():
                raise VValueError("'__TPID' key is reserved and cannot be used in 'data'.")

            generalLogger.debug('Setting TemporalDataFrame from data.')
            self._df = pd.DataFrame(dict({'__TPID': time_points}, **data), index=index)

        else:
            raise VTypeError(f"Type {type(data)} is not handled.")

    def __repr__(self) -> str:
        """
        Description for this TemporalDataFrame object to print.
        :return: a description of this TemporalDataFrame object
        """
        repr_str = ""
        for TP_i, TP in enumerate(self.time_point_indexes):
            print(TP)
            repr_str += f"\033[4m{self.time_points[TP_i]}\033[0m\n"
            repr_str += f"{repr(self[TP])}\n"

        return repr_str
        # return repr(self._df[self._df.columns[1:]])

    def __getitem__(self, index: Union[PreSlicer, Tuple[PreSlicer, Any]]) -> pd.DataFrame:
        """
        Get data from the DataFrame using an index with the usual sub-setting mechanics.
        :param index: A sub-setting index. It can be a single index or a 2-tuple of indexes.
            An index can be a string, an int, a float, a sequence of those, a range, a slice or an ellipsis ('...').
            Single indexes are converted to a 2-tuple :
                * single index --> (index, :)

            The first element in the 2-tuple is the list of time points to select, the second element is the list of
            conditions for selection on the DataFrame as it is done with pandas DataFrames.

            The values ':' or '...' are shortcuts for 'take all values'.

            Example:
                * TemporalDataFrame[:] or TemporalDataFrame[...]    --> select all data
                * TemporalDataFrame[0]                              --> select all data from time point 0
                * TemporalDataFrame[[0, 1], <condition>]            --> select all data from time points 0 and 1 which
                                                                        match the condition. <condition> takes the form
                                                                        of a list of booleans indicating the rows to
                                                                        select with 'True'.
        :return: a selection on this TemporalDataFrame
        """
        if not isinstance(index, tuple):
            index = [index, slice(None, None, None)]
        else:
            index = [index[0], index[1]]

        # check first index (subset on time points)
        if isinstance(index[0], type(Ellipsis)) or index[0] == slice(None, None, None):
            index[0] = pd.Categorical(self._df['__TPID'].values).categories

        elif isinstance(index[0], slice):
            max_value = max(self._df['__TPID'].values)
            if not issubclass(type(max_value), (int, np.int_)):
                raise VTypeError("Cannot slice on time points since time points are not ints.")

            index[0] = slice_to_range(index[0], max_value)

        elif not hasattr(index[0], '__len__'):
            index[0] = [index[0]]

        data_for_TP = self._df[self._df['__TPID'].isin(index[0])]
        index_conditions = index[1][data_for_TP.index] if not isinstance(index[1], slice) else index[1]

        return data_for_TP[index_conditions][self._df.columns[1:]]

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
        if 'unit' in self._parent.time_points.columns:
            return [f"{i} : {v} {u}" for i, v, u in zip(self.time_point_indexes,
                                                        self._parent.time_points.value.values,
                                                        self._parent.time_points.unit.values)]

        else:
            return [f"{i} : {v}" for i, v in zip(self.time_point_indexes,
                                                 self._parent.time_points.value.values)]

    @property
    def time_point_indexes(self) -> List[int]:
        """
        Get the list of time point indexes.
        :return: the list of time point indexes
        """
        return self._parent.time_points.index.values

    @property
    def columns(self) -> pd.Index:
        """
        Get the columns of the DataFrame (but mask the reserved __TPID column).
        :return: the column names of the DataFrame
        """
        return self._df.columns[1:]
