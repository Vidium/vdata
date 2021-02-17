# coding: utf-8
# Created on 09/02/2021 15:24
# Author : matteo

# ====================================================
# imports
import numpy as np
import pandas as pd
from typing import Any, Dict, Tuple, Union

from vdata.utils import TimePoint
from . import dataframe
from . import views
from .._IO import generalLogger, VValueError


# ====================================================
# code
class _VAtIndexer:
    """
    Wrapper around pandas _AtIndexer object for use in TemporalDataFrames.
    The .at can access elements by indexing with :
        - a single element (TDF.at[<element0>])    --> on indexes

    Allowed indexing elements are :
        - a single label
    """

    def __init__(self, parent: 'dataframe.TemporalDataFrame', data: Dict[TimePoint, np.ndarray]):
        """
        :param parent: a parent TemporalDataFrame.
        :param data: the parent TemporalDataFrame's data to work on.
        """
        self.__parent = parent
        self.__data = data

    def __getitem__(self, key: Tuple[Any, Any]) -> 'views.ViewTemporalDataFrame':
        """
        Get values using the _AtIndexer.
        :param key: a tuple of row index and column name.
        :return: the value stored at the row index and column name.
        """
        return self.__parent[:, key[0], key[1]]

    def __setitem__(self, key: Tuple[Any, Any], value: Any) -> None:
        """
        Set values using the _AtIndexer.
        :param key: a tuple of row index and column name.
        :param value: a value to set.
        """
        self.__parent[:, key[0], key[1]] = value


class _ViAtIndexer:
    """
    Wrapper around pandas _iAtIndexer object for use in TemporalDataFrames.
    The .iat can access elements by indexing with :
        - a 2-tuple of elements (TDF.iat[<element0>, <element1>])             --> on indexes and columns

    Allowed indexing elements are :
        - a single integer
    """

    def __init__(self, parent: 'dataframe.TemporalDataFrame', data: Dict[TimePoint, np.ndarray]):
        """
        :param parent: a parent TemporalDataFrame.
        :param data: the parent TemporalDataFrame's data to work on.
        """
        self.__parent = parent
        self.__data = data

    def __get_target_tp(self, index_key: int) -> Tuple[TimePoint, int]:
        """
        Get the time point where the data needs to be accessed.
        :return: the time point and the index offset.
        """
        target_tp = None
        cnt = 0
        for time_point in self.__parent.time_points:
            if index_key < cnt + len(self.__data[time_point]):
                target_tp = time_point
                break

            cnt += len(self.__data[time_point])

        if target_tp is None:
            raise VValueError("Index out of range.")

        return target_tp, cnt

    def __getitem__(self, key: Tuple[int, int]) -> Any:
        """
        Get values using the _AtIndexer.
        :param key: a tuple of row # and column #
        :return: the value stored at the row # and column #.
        """
        target_tp, offset = self.__get_target_tp(key[0])

        return self.__data[target_tp][key[0] - offset, key[1]]

    def __setitem__(self, key: Tuple[int, int], value: Any) -> None:
        """
        Set values using the _AtIndexer.
        :param key: a tuple of row # and column #.
        :param value: a value to set.
        """
        target_tp, offset = self.__get_target_tp(key[0])

        self.__data[target_tp][key[0] - offset, key[1]] = value


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

    def __init__(self, parent: Union['dataframe.TemporalDataFrame', 'views.ViewTemporalDataFrame'],
                 data: Dict[TimePoint, np.ndarray]):
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

        # print(self.__pandas_data)
        # print(key)

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
        - a single element (TDF.iloc[<element0>])                              --> on indexes
        - a 2-tuple of elements (TDF.iloc[<element0>, <element1>])             --> on indexes and columns

    Allowed indexing elements are :
        - a single integer
        - a list of integers
        - a slice of integers
        - a boolean array of the same length as the axis
    """

    def __init__(self, parent: Union['dataframe.TemporalDataFrame', 'views.ViewTemporalDataFrame'],
                 data: Dict[TimePoint, np.ndarray]):
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
