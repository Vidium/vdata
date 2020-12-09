# coding: utf-8
# Created on 12/9/20 9:17 AM
# Author : matteo

# ====================================================
# imports
import pandas as pd
from .._IO.errors import VValueError, VTypeError, ShapeError
from ..NameUtils import DType, PreSlicer

from typing import Dict, Union, Optional, Collection
from typing_extensions import Literal


# ====================================================
# code
class TemporalDataFrame:
    """
    TODO
    """

    def __init__(self, data: Optional[Union[Dict, pd.DataFrame]] = None,
                 time_points: Optional[Union[Collection, DType, Literal['*']]] = None,
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

        :param index: indexes for the dataframe's rows
        """
        # no data given, empty DataFrame is created
        if data is None or not len(data.keys()):
            if time_points is not None:
                raise VValueError("Cannot set time points if no data is supplied.")

            self._df = pd.DataFrame()

        # data given
        elif isinstance(data, (dict, pd.DataFrame)):
            # get number of rows in data
            data_len = 1
            for value in data.values():
                value_len = len(value) if hasattr(value, '__len__') else 1

                if value_len != data_len and data_len != 1 and value_len != 1:
                    raise ShapeError('All items in data must have the same length (or be a unique value to set for '
                                     'all rows).')

                if data_len == 1:
                    data_len = value_len

            # no time points given, all set to 0 by default
            if time_points is None:
                time_points = [0 for _ in range(data_len)]

            # time points given, check length is correct
            else:
                if len(time_points) != data_len:
                    raise VValueError(f"Supplied time points must be of length {data_len}.")

            if isinstance(data, pd.DataFrame):
                data = data.to_dict()

            if '__TPID' in data.keys():
                raise VValueError("'__TPID' key is reserved and cannot be used in 'data'.")

            self._df = pd.DataFrame(dict({'__TPID': time_points}, **data), index=index)

        else:
            raise VTypeError(f"Type {type(data)} is not handled.")

    def __repr__(self) -> str:
        """
        Description for this TemporalDataFrame object to print.
        :return: a description of this TemporalDataFrame object
        """
        return repr(self._df[self._df.columns[1:]])

    def __getitem__(self, index: Union[PreSlicer, Tuple[PreSlicer, Any]]) -> pd.DataFrame:
        """
        TODO
        """
        if not isinstance(index, tuple):
            index = (index, slice(None, None, None))

        if isinstance(index[0], type(Ellipsis)) or index[0] == slice[None, None, None]:


        self._df[self._df.__TPID]

    def __getattr__(self, item):
        """
        TODO
        """
        pass

    def __setitem__(self, key, value) -> None:
        """
        TODO
        """
        pass
