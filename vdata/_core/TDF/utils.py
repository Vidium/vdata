# coding: utf-8
# Created on 01/04/2022 08:52
# Author : matteo

# ====================================================
# imports
import numpy as np
import numpy_indexed as npi
from numbers import Number
from collections import Collection
import numpy_indexed as npi

from typing import Any, Union, Type, Optional, TYPE_CHECKING

from .name_utils import SLICER
from vdata.utils import repr_array
from vdata.time_point import TimePoint, TimePointRange

if TYPE_CHECKING:
    from .dataframe import TemporalDataFrame
    from .view import ViewTemporalDataFrame


# ====================================================
# code
def is_collection(obj: Any) -> bool:
    return isinstance(obj, Collection) and not isinstance(obj, (str, bytes, bytearray))


def _expand_slicer(s: Union[SLICER,
                            tuple[SLICER, SLICER],
                            tuple[SLICER, SLICER, SLICER]]) \
        -> tuple[SLICER, SLICER, SLICER]:
    """TODO"""
    if isinstance(s, tuple) and len(s) == 3:
        return s

    elif isinstance(s, tuple) and len(s) == 2:
        return s[0], s[1], slice(None)

    elif isinstance(s, tuple) and len(s) == 1:
        return s[0], slice(None), slice(None)

    elif isinstance(s, (Number, np.number, str, TimePoint, range, slice)) \
            or s is Ellipsis \
            or is_collection(s) and all([isinstance(e, (Number, np.number, str, TimePoint)) for e in s]):
        return s, slice(None), slice(None)

    else:
        raise ValueError("Invalid slicer.")


def parse_axis_slicer(axis_slicer: SLICER,
                      cast_type: Union[np.dtype, Type[TimePoint]],
                      range_function: Union[Type[range], Type[TimePointRange]],
                      possible_values: np.ndarray) -> Optional[np.ndarray]:
    """TODO"""
    if axis_slicer is Ellipsis or (isinstance(axis_slicer, slice) and axis_slicer == slice(None)):
        return None

    elif isinstance(axis_slicer, slice):
        start = possible_values[0] if axis_slicer.start is None else cast_type(axis_slicer.start)
        stop = possible_values[-1] if axis_slicer.stop is None else cast_type(axis_slicer.stop)

        if axis_slicer.step is None:
            step = cast_type(1, start.unit) if cast_type == TimePoint else cast_type(1)

        else:
            step = cast_type(axis_slicer.step)

        return np.array(list(iter(range_function(start, stop, step))))

    elif isinstance(axis_slicer, range) or is_collection(axis_slicer):
        return np.array(list(map(cast_type, axis_slicer)))

    return np.array([cast_type(axis_slicer)])


def parse_slicer(TDF: Union['TemporalDataFrame', 'ViewTemporalDataFrame'],
                 slicer: Union[SLICER,
                               tuple[SLICER, SLICER],
                               tuple[SLICER, SLICER, SLICER]]) \
        -> tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Given a TemporalDataFrame and a slicer, get the list of indices, columns str and num and the sliced index and
    columns.
    """
    tp_slicer, index_slicer, column_slicer = _expand_slicer(slicer)

    # convert slicers to simple lists of values
    tp_array = parse_axis_slicer(tp_slicer, TimePoint, TimePointRange, TDF.timepoints)
    index_array = parse_axis_slicer(index_slicer, TDF.index.dtype.type, range, TDF.index)
    columns_array = parse_axis_slicer(column_slicer, TDF.columns.dtype.type, range, TDF.columns)

    if tp_array is None and index_array is None:
        selected_index_pos = np.arange(len(TDF.index))

    elif tp_array is None:
        valid_index = np.in1d(index_array, TDF.index)

        if not np.all(valid_index):
            raise ValueError(f"Some indices were not found in this {TDF.__class__.__name__} "
                             f"({repr_array(index_array[~valid_index])})")

        uniq, indices = np.unique(np.concatenate([index_array[np.in1d(index_array, TDF.index_at(tp))]
                                                  for tp in TDF.timepoints]), return_index=True)
        selected_index_pos = npi.indices(uniq[indices.argsort()], TDF.index)

    elif index_array is None:
        valid_tp = np.in1d(tp_array, TDF.timepoints)

        if not np.all(valid_tp):
            raise ValueError(f"Some time-points were not found in this {TDF.__class__.__name__} "
                             f"({repr_array(tp_array[~valid_tp])})")

        selected_index_pos = np.where([np.in1d(TDF.timepoints_column, tp_array)])[0]

    else:
        valid_tp = np.in1d(tp_array, TDF.timepoints)

        if not np.all(valid_tp):
            raise ValueError(f"Some time-points were not found in this {TDF.__class__.__name__} "
                             f"({repr_array(tp_array[~valid_tp])})")

        valid_index = np.in1d(index_array, TDF.index)

        if not np.all(valid_index):
            raise ValueError(f"Some indices were not found in this {TDF.__class__.__name__} "
                             f"({repr_array(index_array[~valid_index])})")

        selected_index = np.concatenate([index_array[np.in1d(index_array, TDF.index_at(tp))]
                                         for tp in tp_array])
        selected_index_pos = npi.indices(TDF.index, selected_index)

    if columns_array is None:
        selected_columns_num = TDF.columns_num
        selected_columns_str = TDF.columns_str
        columns_array = TDF.columns

    else:
        valid_columns = np.in1d(columns_array, TDF.columns)

        if not np.all(valid_columns):
            raise ValueError(f"Some columns were not found in this {TDF.__class__.__name__} "
                             f"({repr_array(columns_array[~valid_columns])})")

        selected_columns_num = columns_array[np.in1d(columns_array, TDF.columns_num)]
        selected_columns_str = columns_array[np.in1d(columns_array, TDF.columns_str)]

    return selected_index_pos, selected_columns_num, selected_columns_str, index_array, columns_array
