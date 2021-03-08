# coding: utf-8
# Created on 09/02/2021 18:24
# Author : matteo

# ====================================================
# imports
from typing import Union

from . import dataframe
from . import views


# ====================================================
# code
def copy_TemporalDataFrame(TDF: Union['dataframe.TemporalDataFrame', 'views.ViewTemporalDataFrame']) \
        -> 'dataframe.TemporalDataFrame':
    """
    Create a new copy of a TemporalDataFrame.
    :return: a copy of a TemporalDataFrame.
    """
    _time_list = TDF.time_points_column if TDF.time_points_column_name is None else None

    new_TDF = dataframe.TemporalDataFrame(data=TDF.to_pandas(),
                                          time_list=_time_list,
                                          time_col_name=TDF.time_points_column_name,
                                          time_points=TDF.time_points,
                                          index=TDF.index.copy(),
                                          columns=TDF.columns.copy(),
                                          name=TDF.name)
    new_TDF.lock(TDF.is_locked)

    return new_TDF
