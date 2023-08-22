from __future__ import annotations

from typing import TYPE_CHECKING, Any, Collection, Iterable, SupportsIndex, Union, cast

import numpy as np

from vdata._typing import IFS, AnyNDArrayLike_IFS
from vdata.tdf.index import Index
from vdata.utils import isCollection

if TYPE_CHECKING:
    from vdata.tdf.base import TemporalDataFrameBase
    from vdata.tdf.view import TemporalDataFrameView


_I_LOC_INDEX = Union[int, slice, SupportsIndex, Iterable[int], Iterable[bool]]


class VAtIndexer:
    """
    Access a single value in a TemporalDataFrame, from a pair of row and column labels.
    """

    __slots__ = "_TDF"

    # region magic methods
    def __init__(self, TDF: TemporalDataFrameBase):
        self._TDF = TDF

    def __getitem__(self, key: tuple[Any, Any]) -> IFS:
        index, column = key

        return cast(IFS, self._TDF[:, index, column])

    def __setitem__(self, key: tuple[Any, Any], value: IFS) -> None:
        index, column = key

        self._TDF[:, index, column] = value

    # endregion


class ViAtIndexer:
    """
    Access a single value in a TemporalDataFrame, from a pair of row and column indices.
    """

    __slots__ = "_TDF"

    # region magic methods
    def __init__(self, TDF: TemporalDataFrameBase):
        self._TDF = TDF

    def __getitem__(self, key: tuple[int, int]) -> IFS:
        index_id, column_id = key
        row = self._TDF.index[index_id]
        column = self._TDF.columns[column_id]

        return cast(IFS, self._TDF[:, row, column])

    def __setitem__(self, key: tuple[int, int], value: IFS) -> None:
        index_id, column_id = key

        self._TDF[:, self._TDF.index[index_id], self._TDF.columns[column_id]] = value

    # endregion


class VLocIndexer:
    """
    Access a group of rows and columns by label(s) or a boolean array.

    Allowed inputs are:
        - A single label, e.g. 5 or 'a', (note that 5 is interpreted as a label of the index, and never as an
        integer position along the index).
        - A list or array of labels, e.g. ['a', 'b', 'c'].
        - A slice object with labels, e.g. 'a':'f'.
        - A boolean array of the same length as the axis being sliced, e.g. [True, False, True].
        - A callable function with one argument (the calling Series or DataFrame) and that returns valid output
        for indexing (one of the above)
    """

    __slots__ = "_TDF"

    # region magic methods
    def __init__(self, TDF: TemporalDataFrameBase):
        self._TDF = TDF

    @staticmethod
    def _parse_slicer(values: IFS | Collection[IFS], reference: AnyNDArrayLike_IFS | Index) -> AnyNDArrayLike_IFS | IFS:
        if not isCollection(values):
            return cast(IFS, values)

        values = np.array(values)

        if values.dtype != bool:
            return values

        return cast(Union[AnyNDArrayLike_IFS, IFS], reference[values])

    def __getitem__(
        self, key: Any | Collection[Any] | tuple[Any | Collection[Any], Any | Collection[Any]]
    ) -> TemporalDataFrameView:
        if isinstance(key, tuple):
            indices, columns = key

        else:
            indices, columns = key, slice(None)

        # parse indices
        indices = self._parse_slicer(indices, self._TDF.index)

        # parse columns
        columns = self._parse_slicer(columns, self._TDF.columns)

        return self._TDF[:, indices, columns]

    def __setitem__(
        self,
        key: Any | Collection[Any] | tuple[Any | Collection[Any], Any | Collection[Any]],
        value: IFS | Collection[IFS],
    ) -> None:
        if isinstance(key, tuple):
            indices, columns = key

        else:
            indices, columns = key, slice(None)

        # parse indices
        indices = self._parse_slicer(indices, self._TDF.index)

        # parse columns
        columns = self._parse_slicer(columns, self._TDF.columns)

        self._TDF[:, indices, columns] = value

    # endregion


class ViLocIndexer:
    """
    Purely integer-location based indexing for selection by position (from 0 to length-1 of the axis).

    Allowed inputs are:
        - An integer, e.g. 5.
        - A list or array of integers, e.g. [4, 3, 0].
        - A slice object with ints, e.g. 1:7.
        - A boolean array.
        - A callable function with one argument (the calling Series or DataFrame) and that returns valid output
        for indexing (one of the above). This is useful in method chains, when you don’t have a reference to the
        calling object, but would like to base your selection on some value.
    """

    __slots__ = "_TDF"

    # region magic methods
    def __init__(self, TDF: TemporalDataFrameBase):
        self._TDF = TDF

    @staticmethod
    def _parse_slicer(values_index: _I_LOC_INDEX, reference: AnyNDArrayLike_IFS) -> AnyNDArrayLike_IFS | IFS:
        if isCollection(values_index):
            return reference[np.array(values_index)]

        return reference[values_index]  # type: ignore[index]

    def __getitem__(
        self, key: _I_LOC_INDEX | tuple[_I_LOC_INDEX, _I_LOC_INDEX]
    ) -> TemporalDataFrameView | np.int_ | np.float_ | np.str_:
        if isinstance(key, tuple):
            indices, columns = key

        else:
            indices, columns = key, slice(None)

        # parse indices
        indices = self._parse_slicer(indices, self._TDF.index)

        # parse columns
        columns = self._parse_slicer(columns, self._TDF.columns)

        return self._TDF[:, indices, columns]

    def __setitem__(self, key: _I_LOC_INDEX | tuple[_I_LOC_INDEX, _I_LOC_INDEX], value: IFS | Collection[IFS]) -> None:
        if isinstance(key, tuple):
            indices, columns = key

        else:
            indices, columns = key, slice(None)

        # parse indices
        indices = self._parse_slicer(indices, self._TDF.index)

        # parse columns
        columns = self._parse_slicer(columns, self._TDF.columns)

        self._TDF[:, indices, columns] = value

    # endregion
