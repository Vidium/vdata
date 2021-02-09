# coding: utf-8
# Created on 09/02/2021 15:24
# Author : matteo

# ====================================================
# imports
import pandas as pd
from typing import Any, Dict, Tuple, Union

from vdata.utils import TimePoint
from . import dataframe
from . import views
from .._IO import generalLogger


# ====================================================
# code
class _VAtIndexer:
    """
    Wrapper around pandas _AtIndexer object for use in TemporalDataFrames.
    The .at can access elements by indexing with :
        - a single element (TDF.loc[<element0>])    --> on indexes

    Allowed indexing elements are :
        - a single label
    """

    def __init__(self, parent: 'dataframe.TemporalDataFrame', data: Dict[TimePoint, pd.DataFrame]):
        """
        :param parent: a parent TemporalDataFrame.
        :param data: the parent TemporalDataFrame's data to work on.
        """
        self.__parent = parent
        self.__data = data
        self.__pandas_data = parent.to_pandas()

    def __getitem__(self, key: Tuple[Any, Any]) -> Any:
        """
        Get values using the _AtIndexer.
        :param key: a tuple of row index and column name.
        :return: the value stored at the row index and column name.
        """
        return self.__pandas_data.at[key]

    def __setitem__(self, key: Tuple[Any, Any], value: Any) -> None:
        """
        Set values using the _AtIndexer.
        :param key: a tuple of row index and column name.
        :param value: a value to set.
        """
        row, col = key[0], key[1]
        target_tp = None

        for tp in self.__parent.time_points:
            if row in self.__data[tp].index:
                target_tp = tp
                break

        self.__data[target_tp].at[key] = value


class _ViAtIndexer:
    """
    Wrapper around pandas _iAtIndexer object for use in TemporalDataFrames.
    The .iat can access elements by indexing with :
        - a 2-tuple of elements (TDF.loc[<element0>, <element1>])             --> on indexes and columns

    Allowed indexing elements are :
        - a single integer
    """

    def __init__(self, parent: 'dataframe.TemporalDataFrame', data: Dict[TimePoint, pd.DataFrame]):
        """
        :param parent: a parent TemporalDataFrame.
        :param data: the parent TemporalDataFrame's data to work on.
        """
        self.__parent = parent
        self.__data = data
        self.__pandas_data = parent.to_pandas()

    def __getitem__(self, key: Tuple[int, int]) -> Any:
        """
        Get values using the _AtIndexer.
        :param key: a tuple of row # and column #
        :return: the value stored at the row # and column #.
        """
        return self.__pandas_data.iat[key]

    def __setitem__(self, key: Tuple[int, int], value: Any) -> None:
        """
        Set values using the _AtIndexer.
        :param key: a tuple of row # and column #.
        :param value: a value to set.
        """
        row, col = key[0], key[1]
        target_tp = None

        row_cumul = 0
        for tp in self.__parent.time_points:

            if row_cumul + len(self.__data[tp]) >= row:
                target_tp = tp
                break

            else:
                row_cumul += len(self.__data[tp])

        self.__data[target_tp].iat[key[0] - row_cumul, key[1]] = value


class _VLocIndexer:
    """
    Wrapper around pandas _LocIndexer object for use in TemporalDataFrames.
    The .loc can access elements by indexing with :
        - a single element (TDF.loc[<element0>])                              --> on indexes
        - a 2-tuple of elements (TDF.loc[<element0>, <element1>])             --> on indexes and columns

    Allowed indexing elements are :
        - a single label
        - a list of labels
        - a slice of labels
        - a boolean array of the same length as the axis
    """

    def __init__(self, parent: 'dataframe.TemporalDataFrame', data: Dict[TimePoint, pd.DataFrame]):
        """
        :param parent: a parent TemporalDataFrame.
        :param data: the parent TemporalDataFrame's data to work on.
        """
        self.__parent = parent
        self.__data = data
        self.__pandas_data = parent.to_pandas()

    def __getitem__(self, key: Union[Any, Tuple[Any, Any]]) -> 'views.ViewTemporalDataFrame':
        """
        Get rows and columns from the loc.
        :param key: loc index.
        :return: TemporalDataFrame or single value.
        """
        generalLogger.debug(u'\u23BE .loc access : begin ------------------------------------------------------- ')

        result = self.__pandas_data.loc[key]
        generalLogger.debug(f'.loc data is : \n{result}')

        if isinstance(result, pd.DataFrame):
            generalLogger.debug('.loc data is a DataFrame.')
            final_result = views.ViewTemporalDataFrame(self.__parent, self.__data,
                                                       self.__parent.time_points, result.index, result.columns)

        elif isinstance(result, pd.Series):
            generalLogger.debug('.loc data is a Series.')

            if result.name in self.__parent.columns:
                result = result.to_frame()

            else:
                result = result.to_frame().T

            final_result = views.ViewTemporalDataFrame(self.__parent, self.__data,
                                                       self.__parent.time_points, result.index, result.columns)

        else:
            generalLogger.debug('.loc data is a single value.')
            # in this case, the key has to be a tuple of 2 single values

            final_result = views.ViewTemporalDataFrame(self.__parent, self.__data,
                                                       self.__parent.time_points, [key[0]], [key[1]])

        generalLogger.debug(u'\u23BF .loc access : end --------------------------------------------------------- ')
        return final_result

    def __setitem__(self, key: Union[Any, Tuple[Any, Any]], value: Any) -> None:
        """
        Set rows and columns from the loc.
        :param key: loc index.
        :param value: pandas DataFrame, Series or a single value to set.
        """
        self[key].set(value)


class _ViLocIndexer:
    """
    Wrapper around pandas _iLocIndexer object for use in TemporalDataFrames.
    The .iloc can access elements by indexing with :
        - a single element (TDF.loc[<element0>])                              --> on indexes
        - a 2-tuple of elements (TDF.loc[<element0>, <element1>])             --> on indexes and columns

    Allowed indexing elements are :
        - a single integer
        - a list of integers
        - a slice of integers
        - a boolean array of the same length as the axis
    """

    def __init__(self, parent: 'dataframe.TemporalDataFrame', data: Dict[TimePoint, pd.DataFrame]):
        """
        :param parent: a parent TemporalDataFrame.
        :param data: the parent TemporalDataFrame's data to work on.
        """
        self.__parent = parent
        self.__data = data
        self.__pandas_data = parent.to_pandas()

    def __getitem__(self, key: Union[Any, Tuple[Any, Any]]) -> 'views.ViewTemporalDataFrame':
        """
        Get rows and columns from the loc.
        :param key: loc index.
        :return: TemporalDataFrame or single value.
        """
        generalLogger.debug(u'\u23BE .iloc access : begin ------------------------------------------------------- ')

        result = self.__pandas_data.iloc[key]
        generalLogger.debug(f'.iloc data is : \n{result}')

        if isinstance(result, pd.DataFrame):
            generalLogger.debug('.iloc data is a DataFrame.')
            final_result = views.ViewTemporalDataFrame(self.__parent, self.__data,
                                                       self.__parent.time_points, result.index, result.columns)

        elif isinstance(result, pd.Series):
            generalLogger.debug('.iloc data is a Series.')

            if result.name in self.__parent.columns:
                result = result.to_frame()

            else:
                result = result.to_frame().T

            final_result = views.ViewTemporalDataFrame(self.__parent, self.__data,
                                                       self.__parent.time_points, result.index, result.columns)

        else:
            generalLogger.debug('.iloc data is a single value.')
            # in this case, the key has to be a tuple of 2 single values

            final_result = views.ViewTemporalDataFrame(self.__parent, self.__data,
                                                       self.__parent.time_points, [key[0]], [key[1]])

        generalLogger.debug(u'\u23BF .iloc access : end --------------------------------------------------------- ')
        return final_result

    def __setitem__(self, key: Union[Any, Tuple[Any, Any]], value: Any) -> None:
        """
        Set rows and columns from the loc.
        :param key: loc index.
        :param value: pandas DataFrame, Series or a single value to set.
        """
        self[key].set(value)
