from __future__ import annotations

from dataclasses import dataclass
from itertools import takewhile
from typing import Collection, Literal

import ch5mpy.indexing as ci
import numpy as np
import numpy_indexed as npi

import vdata.tdf as tdf
import vdata.timepoint as tp
from vdata._typing import IFS, AnyNDArrayLike_IFS, NDArray_IFS, Slicer
from vdata.array_view import NDArrayView
from vdata.names import Number
from vdata.utils import isCollection


@dataclass
class SlicerData:
    tp: Slicer
    idx: Slicer = slice(None)
    col: Slicer = slice(None)

    def __post_init__(self) -> None:
        for s in (self.tp, self.idx, self.col):
            if (
                not isinstance(s, (Number, str, tp.TimePoint, range, slice))
                and s is not Ellipsis
                and not (isCollection(s) and all([isinstance(e, (Number, str, tp.TimePoint)) for e in s]))
            ):
                raise ValueError(f"Invalid slicing element '{s}'.")

    def targets_single_column(self) -> bool:
        return (
            isinstance(self.tp, slice)
            and self.tp == slice(None)
            and isinstance(self.idx, slice)
            and self.idx == slice(None)
            and isinstance(self.col, (str, int, float, np.int_, np.float_))
        )


def as_slicer(slicers: Slicer | tuple[Slicer, Slicer] | tuple[Slicer, Slicer, Slicer]) -> SlicerData:
    if isinstance(slicers, tuple):
        return SlicerData(*slicers)
    return SlicerData(slicers)


def _parse_timepoints_slicer(slicer: Slicer, TDF: tdf.TemporalDataFrameBase) -> ci.FullSlice | ci.ListIndex:
    if slicer is Ellipsis or (isinstance(slicer, slice) and slicer == slice(None)):
        return ci.FullSlice.whole_axis(TDF.n_index)

    if isinstance(slicer, slice):
        start = TDF.timepoints[0] if slicer.start is None else tp.TimePoint(slicer.start, TDF.timepoints.unit)
        stop = TDF.timepoints[-1] if slicer.stop is None else tp.TimePoint(slicer.stop, TDF.timepoints.unit)
        step = (
            tp.TimePoint(1, TDF.timepoints.unit)
            if slicer.step is None
            else tp.TimePoint(slicer.step, TDF.timepoints.unit)
        )

        selected_timepointrange = tp.TimePointRange(start, stop, step)

        len_index_before_start = sum(
            TDF.n_index_at(tp) for tp in takewhile(lambda tp: tp not in selected_timepointrange, TDF.timepoints)
        )
        len_slice = sum(TDF.n_index_at(tp) for tp in selected_timepointrange)

        return ci.FullSlice(len_index_before_start, len_index_before_start + len_slice, 1, max=TDF.n_index)

    if isinstance(slicer, np.ndarray) and slicer.dtype == bool:
        if slicer.ndim != 1 or len(slicer) != TDF.n_timepoints:
            raise KeyError(
                f"Boolean mask for timepoints has incorrect shape {slicer.shape}, " f"expected ({TDF.n_timepoints},)."
            )

        selected_timepoints = tp.as_timepointarray(TDF.timepoints[slicer])

    else:
        selected_timepoints = tp.as_timepointarray(slicer)

    if np.any(~np.isin(selected_timepoints, TDF.timepoints)):
        raise KeyError(
            f"Could not find {selected_timepoints[~np.isin(selected_timepoints, TDF.timepoints)]} "
            "in TemporalDataFrame's timepoints."
        )

    return ci.ListIndex(
        np.where(np.logical_or.reduce(tuple(TDF.get_timepoint_mask(tp) for tp in selected_timepoints)))[0],
        max=TDF.n_index,
    )


def _parse_axis_slicer(
    slicer: Slicer, TDF: tdf.TemporalDataFrameBase, axis: Literal["index", "columns"]
) -> ci.FullSlice | ci.ListIndex | ci.EmptyList:
    if axis == "index":
        values, len_values = TDF.index.values, TDF.n_index
    else:
        values, len_values = TDF.columns, TDF.n_columns

    if slicer is Ellipsis or (isinstance(slicer, slice) and slicer == slice(None)):
        return ci.FullSlice.whole_axis(len_values) if len_values else ci.EmptyList(max=len_values)

    if isinstance(slicer, slice):
        start = values[0] if slicer.start is None else slicer.start
        stop = values[-1] if slicer.stop is None else slicer.stop
        step = 1 if slicer.step is None else int(slicer.step)

        try:
            index_start = np.where(values == start)[0][0]
        except IndexError:
            raise KeyError(f"Could not find '{start}' in TemporalDataFrame's {axis}.")

        try:
            index_stop = np.where(values == stop)[0][0]
        except IndexError:
            raise KeyError(f"Could not find '{stop}' in TemporalDataFrame's {axis}.")

        if index_start <= index_stop:
            return ci.FullSlice(index_start, index_stop, step, max=len_values)

        selected_indices = np.arange(start, stop, step)

    elif isinstance(slicer, np.ndarray) and slicer.dtype == bool:
        if slicer.ndim != 1 or len(slicer) != len_values:
            raise KeyError(f"Boolean mask for index has incorrect shape {slicer.shape}, " f"expected ({len_values},).")

        selected_indices = values[slicer]

    elif isinstance(slicer, range) or isCollection(slicer):
        selected_indices = np.array(list(slicer), dtype=values.dtype)

    else:
        selected_indices = np.array([slicer], dtype=values.dtype)

    if not selected_indices.size:
        return ci.EmptyList(max=len_values)

    # get indices of selected values in `values` **while maintaining the selection order**
    idx_selected_indices = np.where(values == selected_indices[:, None])[1]

    if not len(idx_selected_indices) or len(idx_selected_indices) % len(selected_indices) > 0:
        raise KeyError(
            f"Could not find {selected_indices[~np.isin(selected_indices, values)]} " f"in TemporalDataFrame's {axis}."
        )

    idx_selected_indices = idx_selected_indices.reshape((len(selected_indices), -1)).flatten(order="F")

    if axis == "index":
        timepoint_per_index = TDF.timepoints_column[idx_selected_indices]
        assert isinstance(timepoint_per_index, (tp.TimePointArray, NDArrayView))

        timepoints_appearance_order = timepoint_per_index[
            np.sort(np.unique(timepoint_per_index, return_index=True, equal_nan=False)[1].astype(int))
        ]
        idx_selected_indices = idx_selected_indices[
            np.where(timepoint_per_index == timepoints_appearance_order[:, None])[1]
        ]

    return ci.ListIndex(idx_selected_indices, max=len_values)


def _merge_index_selection(
    tp_selection: ci.LengthedIndexer,
    index_selection: ci.LengthedIndexer,
) -> ci.LengthedIndexer:
    if tp_selection.is_whole_axis:
        return index_selection

    return ci.ListIndex(
        np.array(index_selection)[np.isin(np.array(index_selection), np.array(tp_selection))], max=index_selection.max
    )


def _split_columns_selection(
    columns_selection: ci.LengthedIndexer,
    columns_numerical: AnyNDArrayLike_IFS,
    columns_string: AnyNDArrayLike_IFS,
) -> tuple[ci.LengthedIndexer, ci.LengthedIndexer]:
    n_num, n_str = len(columns_numerical), len(columns_string)

    if isinstance(columns_selection, (ci.ListIndex, ci.EmptyList)):
        return (
            ci.as_indexer(
                [int(i) for i in columns_selection.as_array() if i < n_num],
                max=n_num,
            ),
            ci.as_indexer(
                [int(i - n_num) for i in columns_selection.as_array() if i >= n_num],
                max=n_str,
            ),
        )

    if columns_selection.is_whole_axis:
        return (
            ci.FullSlice.whole_axis(n_num) if n_num else ci.EmptyList(max=0),
            ci.FullSlice.whole_axis(n_str) if n_str else ci.EmptyList(max=0),
        )

    if columns_selection.start <= n_num and columns_selection.true_stop <= n_num:
        return (
            ci.FullSlice(columns_selection.start, columns_selection.stop, columns_selection.step, max=n_num),
            ci.EmptyList(max=n_str),
        )

    if columns_selection.start > n_num:
        return (
            ci.EmptyList(max=n_num),
            ci.FullSlice(
                columns_selection.start - n_num,
                columns_selection.stop - n_num,
                columns_selection.step,
                max=n_str,
            ),
        )

    offset = (columns_selection.step - n_num % columns_selection.step) % columns_selection.step

    return (
        ci.FullSlice(columns_selection.start, n_num, columns_selection.step, max=n_num),
        ci.FullSlice(offset, columns_selection.stop - n_num, columns_selection.step, max=n_str),
    )


def ix_(
    index_selection: ci.LengthedIndexer,
    columns_selection: ci.LengthedIndexer,
) -> tuple[ci.LengthedIndexer, ci.LengthedIndexer]:
    if isinstance(index_selection, ci.FullSlice) or isinstance(columns_selection, ci.FullSlice):
        return index_selection, columns_selection

    return ci.ListIndex(np.array(index_selection).reshape(-1, 1), max=index_selection.max), columns_selection


def parse_slicer(TDF: tdf.TemporalDataFrameBase, slicer: SlicerData) -> tuple[ci.Selection, ci.Selection]:
    """
    Given a TemporalDataFrame and a slicer, get the list of indices, columns str and num and the sliced index and
    columns.
    """
    if TDF.is_inverted:
        raise NotImplementedError

    tp_selection = _parse_timepoints_slicer(slicer.tp, TDF)
    index_selection = _parse_axis_slicer(slicer.idx, TDF, "index")
    columns_selection = _parse_axis_slicer(slicer.col, TDF, "columns")

    combined_index_selection = _merge_index_selection(tp_selection, index_selection)
    columns_numerical_selection, columns_string_selection = _split_columns_selection(
        columns_selection, TDF.columns_num, TDF.columns_str
    )

    return (
        ci.Selection(
            ix_(combined_index_selection, columns_numerical_selection),
            shape=(TDF.n_index, TDF.n_columns_num),
        ),
        ci.Selection(
            ix_(combined_index_selection, columns_string_selection),
            shape=(TDF.n_index, TDF.n_columns_str),
        ),
    )


def parse_values(
    values: NDArray_IFS,
    slicer_columns: slice | IFS | Collection[IFS],
    TDF: tdf.TemporalDataFrameBase,
    numerical_selection: ci.Selection,
    string_selection: ci.Selection,
) -> tuple[NDArray_IFS | None, NDArray_IFS | None]:
    num_shape, str_shape = numerical_selection.out_shape, string_selection.out_shape
    values_shape = (num_shape[0], num_shape[1] + str_shape[1])

    try:
        values = values.reshape(values_shape)

    except ValueError:
        try:
            values = np.broadcast_to(values, values_shape)

        except ValueError:
            raise ValueError(f"Can't set {values_shape} values from {values.shape} array.")

    selected_num_columns = TDF.columns_num[numerical_selection[1].as_numpy_index()]
    selected_str_columns = TDF.columns_str[string_selection[1].as_numpy_index()]

    sliced_columns = (
        TDF.columns[slicer_columns] if isinstance(slicer_columns, slice) else np.atleast_1d(slicer_columns)
    )  # type: ignore[arg-type]

    num_columns_indices = npi.indices(sliced_columns, selected_num_columns)
    str_columns_indices = npi.indices(sliced_columns, selected_str_columns)

    values_num = values[:, num_columns_indices] if len(selected_num_columns) else None  # type: ignore[arg-type]
    values_str = values[:, str_columns_indices] if len(selected_str_columns) else None  # type: ignore[arg-type]

    return values_num, values_str
