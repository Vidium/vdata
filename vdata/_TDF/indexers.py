# coding: utf-8
# Created on 09/02/2021 15:24
# Author : matteo

# ====================================================
# imports
import pandas as pd
from typing import Any, Dict, Tuple, Union

from vdata.utils import TimePoint
from . import dataframe
from .views import dataframe as vdf
from .._IO import generalLogger


# ====================================================
# code
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

    def __init__(self, parent: TemporalDataFrame, data: Dict[TimePoint, pd.DataFrame]):
        """
        :param parent: a parent TemporalDataFrame.
        :param data: the parent TemporalDataFrame's data to work on.
        """
        self.__parent = parent
        self.__data = data
        self.__pandas_data = parent.to_pandas()

    def __getitem__(self, key: Union[Any, Tuple[Any, Any]]) -> 'ViewTemporalDataFrame':
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
            final_result = ViewTemporalDataFrame(self.__parent, self.__data,
                                                 self.__parent.time_points, result.index, result.columns)

        elif isinstance(result, pd.Series):
            generalLogger.debug('.loc data is a Series.')

            if result.name in self.__parent.columns:
                result = result.to_frame()

            else:
                result = result.to_frame().T

            final_result = ViewTemporalDataFrame(self.__parent, self.__data,
                                                 self.__parent.time_points, result.index, result.columns)

        else:
            generalLogger.debug('.loc data is a single value.')
            # in this case, the key has to be a tuple of 2 single values

            final_result = ViewTemporalDataFrame(self.__parent, self.__data,
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

    def __init__(self, parent: 'TemporalDataFrame', data: Dict[TimePoint, pd.DataFrame]):
        """
        :param parent: a parent TemporalDataFrame.
        :param data: the parent TemporalDataFrame's data to work on.
        """
        self.__parent = parent
        self.__data = data
        self.__pandas_data = parent.to_pandas()

    def __getitem__(self, key: Union[Any, Tuple[Any, Any]]) -> 'ViewTemporalDataFrame':
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
            final_result = ViewTemporalDataFrame(self.__parent, self.__data,
                                                 self.__parent.time_points, result.index, result.columns)

        elif isinstance(result, pd.Series):
            generalLogger.debug('.iloc data is a Series.')

            if result.name in self.__parent.columns:
                result = result.to_frame()

            else:
                result = result.to_frame().T

            final_result = ViewTemporalDataFrame(self.__parent, self.__data,
                                                 self.__parent.time_points, result.index, result.columns)

        else:
            generalLogger.debug('.iloc data is a single value.')
            # in this case, the key has to be a tuple of 2 single values

            final_result = ViewTemporalDataFrame(self.__parent, self.__data,
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

