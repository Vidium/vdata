from __future__ import annotations

from typing import Iterator, Any

import numpy as np
import numpy.typing as npt
import ch5mpy as ch

from vdata.array_view import NDArrayView
from vdata.timepoint.array import TimePointArray
from vdata.timepoint.timepoint import TimePoint
from vdata.timepoint._typing import _TIME_UNIT


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

    def __getitem__(self, key: int | slice) -> TimePoint | TimePointIndex:
        if isinstance(key, slice):
            start = 0 if key.start is None else key.start
            stop = len(self) if key.stop is None else key.stop
            step = 1 if key.step is None else key.step

            if start < 0:
                start = len(self) + start

            if stop < 0:
                stop = len(self) + stop

            if step != 1:
                raise NotImplementedError

            if stop <= start or start >= len(self):
                return TimePointIndex(TimePointArray([], unit=self.timepoints.unit), np.array([]))

            start_index = np.argmax(self._ranges > start)
            stop_index = np.argmax(self._ranges > stop) if np.any(self._ranges > stop) else len(self._ranges)

            ranges = self._ranges[start_index:stop_index]

            if stop > ranges[-1]:
                ranges = np.append(ranges, stop)
                timepoints = self._timepoints[start_index : stop_index + 1]

            else:
                timepoints = self._timepoints[start_index:stop_index]

            ranges -= start

            return TimePointIndex(timepoints, ranges)

        if key < 0:
            key = len(self) + key

        if key < 0 or key >= len(self):
            raise IndexError(f"index {key} is out of range")

        return self._timepoints[np.argmax(self._ranges > key)]

    def __len__(self) -> int:
        return 0 if self.is_empty else int(self._ranges[-1])

    def __iter__(self) -> Iterator[TimePoint]:
        return iter(self._timepoints)

    def __eq__(self, index: object) -> bool:
        if not isinstance(index, TimePointIndex):
            return False

        return np.array_equal(self._timepoints, index.timepoints) and np.array_equal(self._ranges, index.ranges)

    def __h5_write__(self, values: ch.H5Dict[Any]) -> None:
        ch.write_datasets(values, timepoints=self._timepoints, ranges=self._ranges)

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
        timepoints = array[np.sort(np.unique(array, return_index=True, equal_nan=False)[1].astype(int))]
        ranges = np.append(np.argmax(array == timepoints[1:][:, None], axis=1), len(array))

        return TimePointIndex(timepoints, ranges)

    def as_array(self) -> TimePointArray:
        return TimePointArray(np.repeat(self._timepoints, np.diff(self._ranges, prepend=0)))

    def len(self, timepoint: TimePoint) -> int:
        index_tp = np.where(self._timepoints == timepoint)[0][0]
        start = np.r_[0, self._ranges][index_tp]
        stop = self._ranges[index_tp]

        return int(stop - start)

    def where(self, timepoint: TimePoint, n_max: int | None = None) -> npt.NDArray[np.int_]:
        index_tp = np.where(self._timepoints == timepoint)[0][0]
        start = np.r_[0, self._ranges][index_tp]
        stop = self._ranges[index_tp]

        if n_max is not None and n_max <= stop - start:
            stop -= stop - start - n_max

        return np.arange(start, stop)

    def sort(self, return_indices: bool = False) -> TimePointIndex | tuple[TimePointIndex, npt.NDArray[np.int_]]:
        order = np.argsort(self._timepoints)
        ranges = np.cumsum(np.ediff1d(np.r_[0, self._ranges])[order])

        sorted = TimePointIndex(self._timepoints[order], ranges)

        if return_indices:
            return sorted, np.concatenate([self.where(tp) for tp in sorted])

        return sorted

    @classmethod
    def read(cls, values: ch.H5Dict[Any]) -> TimePointIndex:
        return TimePointIndex.__h5_read__(values)

    # endregion
