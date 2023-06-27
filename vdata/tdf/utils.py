from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import numpy.typing as npt
import numpy_indexed as npi

import vdata.tdf
import vdata.timepoint as tp
from vdata._typing import AnyNDArrayLike_IFS, NDArrayLike_IFS, Slicer
from vdata.names import Number
from vdata.utils import isCollection, repr_array


@dataclass
class SlicerData:
    tp: Slicer
    idx: Slicer = slice(None)
    col: Slicer = slice(None)
    
    def __post_init__(self) -> None:
        for s in (self.tp, self.idx, self.col):
            if not isinstance(s, (Number, str, tp.TimePoint, range, slice)) \
                and s is not Ellipsis \
                and not (isCollection(s) and all([isinstance(e, (Number, str, tp.TimePoint)) for e in s])):
                raise ValueError(f"Invalid slicing element '{s}'.")


def _parse_timepoints_slicer(slicer: Slicer, 
                             timepoints: tp.TimePointArray) -> tp.TimePointArray | None:
    if slicer is Ellipsis or (isinstance(slicer, slice) and slicer == slice(None)):
        return None
    
    if isinstance(slicer, slice):
        start: tp.TimePoint = timepoints[0] if slicer.start is None else tp.TimePoint(slicer.start)
        stop: tp.TimePoint = timepoints[-1] if slicer.stop is None else tp.TimePoint(slicer.stop)
        step = tp.TimePoint(1, start.unit) if slicer.step is None else tp.TimePoint(slicer.step)

        return tp.as_timepointarray(tp.TimePointRange(start, stop, step))

    if isinstance(slicer, np.ndarray) and slicer.dtype == bool:
        return tp.atleast_1d(timepoints[slicer.flatten()])

    return tp.as_timepointarray(slicer)


def _parse_axis_slicer(slicer: Slicer,
                       possible_values: AnyNDArrayLike_IFS) -> AnyNDArrayLike_IFS | None:
    cast_type = possible_values.dtype.type
    
    if slicer is Ellipsis or (isinstance(slicer, slice) and slicer == slice(None)):
        return None

    if isinstance(slicer, slice):
        if np.issubdtype(cast_type, (np.str_, np.float_)):
            raise ValueError(f'Cannot take slice from axis with {cast_type} dtype.')
        
        start = cast(np.int_, possible_values[0] if slicer.start is None else cast_type(slicer.start))
        stop = cast(np.int_, possible_values[-1] if slicer.stop is None else cast_type(slicer.stop))
        step = 1 if slicer.step is None else int(slicer.step)
            
        return np.array(list(range(start, stop, step)))

    if isinstance(slicer, np.ndarray) and slicer.dtype == bool:
        return np.atleast_1d(possible_values[slicer.flatten()])

    if isinstance(slicer, range) or isCollection(slicer):
        return cast(NDArrayLike_IFS, np.array(list(slicer)).astype(cast_type))

    return cast(NDArrayLike_IFS, np.array([slicer]).astype(cast_type))


def parse_slicer_full(TDF: vdata.tdf.TemporalDataFrameBase,
                 slicers: Slicer | tuple[Slicer, Slicer] | tuple[Slicer, Slicer, Slicer]) \
    -> tuple[
        npt.NDArray[np.int_], 
        AnyNDArrayLike_IFS,
        AnyNDArrayLike_IFS,
        tuple[tp.TimePointArray | None, AnyNDArrayLike_IFS | None, AnyNDArrayLike_IFS | None]
    ]:
    """
    Given a TemporalDataFrame and a slicer, get the list of indices, columns str and num and the sliced index and
    columns.
    """
    slicer = SlicerData(*slicers) if isinstance(slicers, tuple) else SlicerData(slicers)
    
    # convert slicers to simple lists of values              
    tp_array = _parse_timepoints_slicer(slicer.tp, TDF.timepoints)
    index_array = _parse_axis_slicer(slicer.idx, TDF.index)
    columns_array = _parse_axis_slicer(slicer.col, TDF.columns)

    if tp_array is None:
        if index_array is None:
            selected_index_pos = np.arange(len(TDF.index))

        else:
            valid_index = np.in1d(index_array, TDF.index)

            if not np.all(valid_index):
                raise ValueError(f"Some indices were not found in this TemporalDataFrame "
                                f"({repr_array(index_array[~valid_index])})")

            indices = []
            cumulated_length = 0

            for timepoint in TDF.timepoints:
                itp = TDF.index_at(timepoint)
                indices.append(npi.indices(itp, index_array[np.isin(index_array, itp)]) + cumulated_length)
                cumulated_length += len(itp)

            selected_index_pos = np.concatenate(indices)

    elif index_array is None:
        valid_tp = np.in1d(tp_array, TDF.timepoints)

        if not np.all(valid_tp):
            raise ValueError(f"Some time-points were not found in this TemporalDataFrame "
                             f"({repr_array(tp_array[~valid_tp])})")

        selected_index_pos = np.where(np.in1d(TDF.timepoints_column[:], tp_array))[0]

    else:        
        valid_tp = np.in1d(tp_array, TDF.timepoints)

        if not np.all(valid_tp):
            raise ValueError(f"Some time-points were not found in this TemporalDataFrame "
                             f"({repr_array(tp_array[~valid_tp])})")

        valid_index = np.in1d(index_array, TDF.index)

        if not np.all(valid_index):
            raise ValueError(f"Some indices were not found in this TemporalDataFrame "
                             f"({repr_array(index_array[~valid_index])})")

        indices = []
        cumulated_length = 0

        for timepoint in TDF.timepoints:
            itp = TDF.index_at(timepoint)

            if timepoint in tp_array:
                indices.append(npi.indices(itp, index_array[np.isin(index_array, itp)]) + cumulated_length)

            cumulated_length += len(itp)

        selected_index_pos = np.concatenate(indices)

    if columns_array is None:
        selected_columns_num = TDF.columns_num
        selected_columns_str = TDF.columns_str

    else:
        valid_columns = np.in1d(columns_array, TDF.columns)

        if not np.all(valid_columns):
            raise ValueError(f"Some columns were not found in this TemporalDataFrame "
                             f"({repr_array(columns_array[~valid_columns])})")

        selected_columns_num = np.atleast_1d(columns_array[np.in1d(columns_array, TDF.columns_num)])
        selected_columns_str = np.atleast_1d(columns_array[np.in1d(columns_array, TDF.columns_str)])

    return selected_index_pos, selected_columns_num, selected_columns_str, (tp_array, index_array, columns_array)


def parse_slicer(TDF: vdata.tdf.TemporalDataFrameBase,
                   slicers: Slicer | tuple[Slicer, Slicer] | tuple[Slicer, Slicer, Slicer]) \
        -> tuple[npt.NDArray[np.int_], AnyNDArrayLike_IFS, AnyNDArrayLike_IFS]:
    return parse_slicer_full(TDF, slicers)[:3]


def parse_values(values: Any,
                 len_index: int,
                 len_columns: int) -> npt.NDArray[Any]:
    """
    When setting values in a TemporalDataFrame, check those values roughly have the correct shape and reshape 
    them if possible.
    """    
    if isinstance(values, vdata.tdf.TemporalDataFrameBase):
        if values.n_columns_num and values.n_columns_str:
            values = np.concatenate((values.values_num, values.values_str), axis=1)

        elif values.n_columns_num:
            values = values.values_num

        elif values.n_columns_str:
            values = values.values_str

        else:
            values = []

    if not isCollection(values):
        return np.full((len_index, len_columns), values)

    values = np.array(values)

    if values.shape == (len_index, len_columns):
        return values
    
    if len_index == 1 and len_columns == len(values):
        return values.reshape((1, len(values)))

    if len_index == len(values) and len_columns == 1:
        return values.reshape((len(values), 1))

    raise ValueError(f"Can't set {len_index} x {len_columns} values from {values.shape} array.")
