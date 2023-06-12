from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import numpy_indexed as npi

import vdata.tdf
from vdata._typing import NDArray_IFS, NDArrayLike_IFS, Slicer
from vdata.names import Number
from vdata.timepoint import TimePoint, TimePointArray, TimePointRange, atleast_1d
from vdata.utils import isCollection, repr_array


@dataclass
class SlicerData:
    tp: Slicer
    idx: Slicer = slice(None)
    col: Slicer = slice(None)
    
    def __post_init__(self) -> None:
        for s in (self.tp, self.idx, self.col):
            if not isinstance(s, (Number, str, TimePoint, range, slice)) \
                and s is not Ellipsis \
                and not (isCollection(s) and all([isinstance(e, (Number, str, TimePoint)) for e in s])):
                raise ValueError(f"Invalid slicing element '{s}'.")


def _parse_timepoints_slicer(slicer: Slicer, 
                             timepoints: TimePointArray) -> TimePointArray | None:
    if slicer is Ellipsis or (isinstance(slicer, slice) and slicer == slice(None)):
        return None
    
    if isinstance(slicer, slice):
        start: TimePoint = timepoints[0] if slicer.start is None else TimePoint(slicer.start)
        stop: TimePoint = timepoints[-1] if slicer.stop is None else TimePoint(slicer.stop)
        step = TimePoint(1, start.unit) if slicer.step is None else TimePoint(slicer.step)

        return TimePointArray(TimePointRange(start, stop, step))

    if isinstance(slicer, np.ndarray) and slicer.dtype == bool:
        return atleast_1d(timepoints[slicer.flatten()])

    if isinstance(slicer, range) or isCollection(slicer):
        return TimePointArray(slicer)
    
    return TimePointArray([slicer])


def _parse_axis_slicer(slicer: Slicer,
                       possible_values: NDArrayLike_IFS) -> NDArrayLike_IFS | None:
    cast_type = possible_values.dtype.type
    
    if slicer is Ellipsis or (isinstance(slicer, slice) and slicer == slice(None)):
        return None

    if isinstance(slicer, slice):
        if cast_type == np.str_:
            raise ValueError(f'Cannot take slice from axis with {cast_type} dtype.')
        
        start = cast_type(possible_values[0] if slicer.start is None else slicer.start)
        stop = cast_type(possible_values[-1] if slicer.stop is None else slicer.stop)
        step = 1 if slicer.step is None else int(slicer.step)
            
        return np.array(list(range(start, stop, step)))

    if isinstance(slicer, np.ndarray) and slicer.dtype == bool:
        return np.atleast_1d(possible_values[slicer.flatten()])

    if isinstance(slicer, range) or isCollection(slicer):
        return np.array(list(slicer)).astype(cast_type)

    return np.array([slicer]).astype(cast_type)


def parse_slicer(TDF: vdata.tdf.TemporalDataFrameBase,
                 slicers: Slicer | tuple[Slicer, Slicer] | tuple[Slicer, Slicer, Slicer]) \
    -> tuple[
        npt.NDArray[np.int_], 
        NDArray_IFS,
        NDArray_IFS,
        tuple[TimePointArray | None, NDArray_IFS | None, NDArray_IFS | None]
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

            for tp in TDF.timepoints:
                itp = TDF.index_at(tp)
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

        for tp in TDF.timepoints:
            itp = TDF.index_at(tp)

            if tp in tp_array:
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

        selected_columns_num = columns_array[np.in1d(columns_array, TDF.columns_num)]
        selected_columns_str = columns_array[np.in1d(columns_array, TDF.columns_str)]

    return selected_index_pos, selected_columns_num, selected_columns_str, (tp_array, index_array, columns_array)


def parse_slicer_s(TDF: vdata.tdf.TemporalDataFrameBase,
                   slicers: Slicer | tuple[Slicer, Slicer] | tuple[Slicer, Slicer, Slicer]) \
        -> tuple[npt.NDArray[np.int_], NDArray_IFS, NDArray_IFS]:
    return parse_slicer(TDF, slicers)[:3]


def parse_values(values: Any,
                 len_index: int,
                 len_columns: int) -> np.ndarray:
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


def underlined(text: str) -> str:
    return text + "\n" + "\u203e" * len(text)


def equal_paths(p1: str | Path, p2: str | Path) -> bool:
    return Path(p1).expanduser().resolve() == Path(p2).expanduser().resolve()
