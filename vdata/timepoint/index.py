from __future__ import annotations

from typing import Any, Iterator, Literal, cast, overload

import ch5mpy as ch
import numpy as np
import numpy.typing as npt

from vdata.array_view import NDArrayView
from vdata.timepoint._typing import _TIME_UNIT
from vdata.timepoint.array import TimePointArray
from vdata.timepoint.timepoint import TimePoint


class TimePointIndex:
    # region magic methods
    def __init__(self, timepoints: TimePointArray, ranges: npt.NDArray[np.int_]) -> None:
        """
        Args:
            timepoints: list of UNIQUE and ORDERED timepoints
            ranges: list of indices defining the ranges where to repeat timepoints
        """
        assert len(timepoints) == len(ranges)

        self._timepoints = timepoints
        self._ranges = ranges

    def __repr__(self) -> str:
        if self.is_empty:
            return "TimePointIndex[]"

        return "TimePointIndex[0" + "".join([f" --{t}--> {i}" for t, i in zip(self._timepoints, self._ranges)]) + "]"

    @overload
    def __getitem__(self, key: int) -> TimePoint: ...

    @overload
    def __getitem__(self, key: slice | ch.indexing.Indexer) -> TimePointIndex: ...

    def __getitem__(self, key: int | slice | ch.indexing.Indexer) -> TimePoint | TimePointIndex:
        if isinstance(key, (ch.indexing.NewAxisType, ch.indexing.EmptyList)):
            raise ValueError(f"Cannot subset TimePointIndex with {key}")

        if isinstance(key, (ch.indexing.FullSlice, ch.indexing.SingleIndex)):
            key = key.as_slice()

        if isinstance(key, ch.indexing.ListIndex):
            key = key.flatten()

        if isinstance(key, (slice, ch.indexing.ListIndex)):
            return TimePointIndex.from_array(self.as_array()[key])

        if isinstance(key, int):
            if key < 0:
                key = len(self) + key

            if key < 0 or key >= len(self):
                raise IndexError(f"index {key} is out of range")

            return self._timepoints[np.argmax(self._ranges > key)]

        raise NotImplementedError

    def __len__(self) -> int:
        return 0 if self.is_empty else int(self._ranges[-1])

    def __iter__(self) -> Iterator[TimePoint]:
        return iter(self._timepoints)

    def __eq__(self, index: object) -> bool:
        if not isinstance(index, TimePointIndex):
            return False

        return np.array_equal(self._timepoints, index.timepoints) and np.array_equal(self._ranges, index.ranges)

    def __h5_write__(self, values: ch.H5Dict[Any]) -> None:
        ch.write_object(self._timepoints, values, "timepoints")
        ch.write_dataset(self._ranges, values, "ranges")

    @classmethod
    def __h5_read__(cls, values: ch.H5Dict[Any]) -> TimePointIndex:
        with ch.options(error_mode="raise"):
            return TimePointIndex(timepoints=values["timepoints"], ranges=values["ranges"])

    # endregion

    # region predicates
    @property
    def is_empty(self) -> bool:
        return len(self._ranges) == 0

    # endregion

    # region attributes
    @property
    def timepoints(self) -> TimePointArray:
        return self._timepoints

    @property
    def ranges(self) -> npt.NDArray[np.int_]:
        return self._ranges

    @property
    def unit(self) -> _TIME_UNIT:
        return self._timepoints.unit

    # endregion

    # region methods
    @classmethod
    def from_array(cls, array: TimePointArray | NDArrayView[TimePoint]) -> TimePointIndex:
        timepoints = cast(
            TimePointArray | NDArrayView[TimePoint],
            array[np.sort(np.unique(array, return_index=True, equal_nan=False)[1].astype(int))],
        )

        ranges = np.append(np.argmax(array == timepoints[1:][:, None], axis=1), len(array))

        return TimePointIndex(timepoints, ranges)

    def as_array(self) -> TimePointArray:
        return TimePointArray(np.repeat(self._timepoints, np.diff(self._ranges, prepend=0)))

    def len(self, timepoint: TimePoint) -> int:
        index_tp = np.where(self._timepoints == timepoint)[0][0]
        start = np.r_[0, self._ranges][index_tp]
        stop = self._ranges[index_tp]

        return int(stop - start)

    def where(self, *timepoints: TimePoint) -> npt.NDArray[np.bool_]:
        mask = np.zeros(len(self), dtype=bool)
        for tp in timepoints:
            mask[self.at(tp)] = True

        return mask

    @overload
    def sort(self, return_indices: Literal[False]) -> TimePointIndex: ...

    @overload
    def sort(self, return_indices: Literal[True]) -> tuple[TimePointIndex, npt.NDArray[np.int_]]: ...

    def sort(self, return_indices: bool = False) -> TimePointIndex | tuple[TimePointIndex, npt.NDArray[np.int_]]:
        order = np.argsort(self._timepoints)
        ranges = np.cumsum(np.ediff1d(np.r_[0, self._ranges])[order])

        sorted = TimePointIndex(self._timepoints[order], ranges)

        if return_indices:
            return sorted, np.concatenate([self.at(tp) for tp in sorted])

        return sorted

    def at(self, timepoint: TimePoint) -> npt.NDArray[np.int_]:
        """Get indices for a given time point"""
        tp_index = np.where(self._timepoints == timepoint)[0][0]
        return np.arange(*np.r_[0, self._ranges][tp_index : tp_index + 2])

    def n_at(self, timepoint: TimePoint) -> int:
        """Get the number of indices at a given time point"""
        tp_index = np.where(self._timepoints == timepoint)[0][0]
        return int(np.diff(np.r_[0, self._ranges][tp_index : tp_index + 2]))

    @classmethod
    def read(cls, values: ch.H5Dict[Any]) -> TimePointIndex:
        return TimePointIndex.__h5_read__(values)

    # endregion
