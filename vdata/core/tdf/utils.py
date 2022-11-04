# coding: utf-8
# Created on 01/04/2022 08:52
# Author : matteo

# ====================================================
# imports
from __future__ import annotations

import numpy as np
import pandas as pd
import numpy_indexed as npi
from numbers import Number

from typing import Any, Type, TYPE_CHECKING

from vdata.h5pickle import Dataset

from .name_utils import SLICER
from vdata.utils import repr_array, isCollection
from vdata.time_point import TimePoint, TimePointRange

if TYPE_CHECKING:
    from .base import BaseTemporalDataFrame


# ====================================================
# code
def _expand_slicer(s: SLICER | tuple[SLICER, SLICER] | tuple[SLICER, SLICER, SLICER]) \
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
            or isCollection(s) and all([isinstance(e, (Number, np.number, str, TimePoint)) for e in s]):
        return s, slice(None), slice(None)

    else:
        raise ValueError("Invalid slicer.")


def parse_axis_slicer(axis_slicer: SLICER,
                      cast_type: np.dtype | Type[TimePoint],
                      range_function: Type[range] | Type[TimePointRange],
                      possible_values: np.ndarray) -> np.ndarray | None:
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

    elif isinstance(axis_slicer, np.ndarray) and axis_slicer.dtype == bool:
        return possible_values[axis_slicer.flatten()]

    elif isinstance(axis_slicer, range) or isCollection(axis_slicer):
        return np.array(list(map(cast_type, axis_slicer)))

    return np.array([cast_type(axis_slicer)])


def parse_slicer(TDF: BaseTemporalDataFrame,
                 slicer: SLICER | tuple[SLICER, SLICER] | tuple[SLICER, SLICER, SLICER]) \
        -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]]:
    """
    Given a TemporalDataFrame and a slicer, get the list of indices, columns str and num and the sliced index and
    columns.
    """
    tp_slicer, index_slicer, column_slicer = _expand_slicer(slicer)

    view = 'view of a ' if TDF.is_view else ''
    backed = 'backed ' if TDF.is_backed else ''

    # convert slicers to simple lists of values
    tp_array = parse_axis_slicer(tp_slicer, TimePoint, TimePointRange, TDF.timepoints)
    index_array = parse_axis_slicer(index_slicer, TDF.index.dtype.type, range, TDF.index)
    columns_array = parse_axis_slicer(column_slicer, TDF.columns.dtype.type, range, TDF.columns)

    if tp_array is None and index_array is None:
        selected_index_pos = np.arange(len(TDF.index))

    elif tp_array is None:
        valid_index = np.in1d(index_array, TDF.index[:])

        if not np.all(valid_index):
            raise ValueError(f"Some indices were not found in this {view}{backed}TemporalDataFrame "
                             f"({repr_array(index_array[~valid_index])})")

        indices = []
        cumulated_length = 0

        for tp in TDF.timepoints:
            itp = TDF.index_at(tp)
            indices.append(npi.indices(itp, index_array[np.isin(index_array, itp)]) + cumulated_length)
            cumulated_length += len(itp)

        selected_index_pos = np.concatenate(indices)

    elif index_array is None:
        valid_tp = np.in1d(tp_array, TDF.timepoints)

        if not np.all(valid_tp):
            raise ValueError(f"Some time-points were not found in this {view}{backed}TemporalDataFrame "
                             f"({repr_array(tp_array[~valid_tp])})")

        selected_index_pos = np.where(np.in1d(TDF.timepoints_column[:], tp_array))[0]

    else:
        valid_tp = np.in1d(tp_array, TDF.timepoints)

        if not np.all(valid_tp):
            raise ValueError(f"Some time-points were not found in this {view}{backed}TemporalDataFrame "
                             f"({repr_array(tp_array[~valid_tp])})")

        valid_index = np.in1d(index_array, TDF.index[:])

        if not np.all(valid_index):
            raise ValueError(f"Some indices were not found in this {view}{backed}TemporalDataFrame "
                             f"({repr_array(index_array[~valid_index])})")

        indices = []
        cumulated_length = 0

        for tp in TDF.timepoints:
            itp = TDF.index_at(tp)

            if tp in tp_array:
                indices.append(npi.indices(itp, index_array[np.isin(index_array, itp)]) + cumulated_length)

            cumulated_length += len(itp)

        selected_index_pos = np.concatenate(indices)

    if columns_array is None:
        selected_columns_num = TDF.columns_num[:]
        selected_columns_str = TDF.columns_str[:]

    else:
        valid_columns = np.in1d(columns_array, TDF.columns)

        if not np.all(valid_columns):
            raise ValueError(f"Some columns were not found in this {view}{backed}TemporalDataFrame "
                             f"({repr_array(columns_array[~valid_columns])})")

        selected_columns_num = columns_array[np.in1d(columns_array, TDF.columns_num[:])]
        selected_columns_str = columns_array[np.in1d(columns_array, TDF.columns_str[:])]

    return selected_index_pos, selected_columns_num, selected_columns_str, (tp_array, index_array, columns_array)


def parse_values(values: Any,
                 len_index: int,
                 len_columns: int) -> np.ndarray:
    """TODO"""
    from .base import BaseTemporalDataFrame
    if isinstance(values, BaseTemporalDataFrame):
        if values.n_columns_num:
            if values.n_columns_str:
                values = np.concatenate((values.values_num, values.values_str), axis=1)

            else:
                values = values.values_num

        elif values.n_columns_str:
            values = values.values_str

        else:
            values = []

    if not isCollection(values):
        return np.full((len_index, len_columns), values)

    if isinstance(values, pd.Series):
        if len_index == 1 and len_columns == len(values):
            return values.values.reshape((1, len(values)))

        if len_index == len(values) and len_columns == 1:
            return values.values.reshape((len(values), 1))

        raise ValueError(f"Can't set {len_index} x {len_columns} values from 1 x {len(values)} array.")

    if isinstance(values, list):
        if len_index == 1 and len_columns == len(values):
            return np.array(values).reshape((1, len(values)))

        if len_index == len(values) and len_columns == 1:
            return np.array(values).reshape((len(values), 1))

        raise ValueError(f"Can't set {len_index} x {len_columns} values from 1 x {len(values)} array.")

    if values.ndim == 1:
        if len_index == 1 and len_columns == len(values):
            return values.reshape((1, len(values)))

        if len_index == len(values) and len_columns == 1:
            return values.reshape((len(values), 1))

        raise ValueError(f"Can't set {len_index} x {len_columns} values from 1 x {len(values)} array.")

    if values.shape != (len_index, len_columns):
        raise ValueError(f"Can't set {len_index} x {len_columns} values from "
                         f"{values.shape[0]} x {values.shape[1]} array.")

    return values


def _fast_compare(timepoint_masks: dict[TimePoint | bytes, np.ndarray],
                  timepoints_array: np.ndarray | Dataset,
                  comparison_tp: TimePoint | bytes) -> np.ndarray:
    if not (ltpm := len(timepoint_masks)):
        return np.equal(timepoints_array, comparison_tp)

    tp_mask = np.zeros(len(timepoints_array), dtype=bool)

    if ltpm == 1:
        not_already_computed = ~next(iter(timepoint_masks.values()))

    else:
        not_already_computed = np.logical_and.reduce([~mask for mask in timepoint_masks.values()])

    tp_mask[not_already_computed] = np.equal(timepoints_array[not_already_computed], comparison_tp)
    return tp_mask
