from __future__ import annotations

from typing import Any, Collection, Literal, NoReturn, cast

import ch5mpy as ch
import numpy as np
import numpy.typing as npt
import numpy_indexed as npi

from vdata._typing import (
    IFS,
    AnyNDArrayLike,
    AnyNDArrayLike_IFS,
    AttrDict,
    NDArray_IFS,
    Slicer,
)
from vdata.array_view import NDArrayView
from vdata.tdf.base import TemporalDataFrameBase
from vdata.tdf.dataframe import TemporalDataFrame
from vdata.tdf.utils import parse_slicer, parse_values
from vdata.timepoint import TimePointArray


def _indices(this: AnyNDArrayLike[Any], 
             that: AnyNDArrayLike[Any]) -> npt.NDArray[np.int_]:
    if not len(that):
        return np.array([], dtype=np.int64)
    
    return npi.indices(np.array(this), np.array(that))      # type: ignore[no-any-return]


def _array_view(container: Any,
                accession: str,
                index: AnyNDArrayLike[np.int_] | tuple[AnyNDArrayLike[np.int_], ...]) \
        -> ch.H5Array[Any] | NDArrayView[Any]:
    if isinstance(getattr(container, accession), ch.H5Array):
        return getattr(container, accession)[index]                                     # type: ignore[no-any-return]
    
    return NDArrayView(container, accession, index)
    

class TemporalDataFrameView(TemporalDataFrameBase):
    """
    Abstract base class for views on a TemporalDataFrame.
    """
    _attributes = TemporalDataFrameBase._attributes.union({'_parent', '_index_positions', '_inverted'})

    # region magic methods
    def __init__(self,
                 parent: TemporalDataFrame,
                 index_positions: AnyNDArrayLike[np.int_],
                 columns_numerical: AnyNDArrayLike_IFS,
                 columns_string: AnyNDArrayLike_IFS,
                 *,
                 inverted: bool = False):
        super().__init__(
            index=_array_view(parent, 'index', index_positions),
            timepoints_array=_array_view(parent, 'timepoints_column', index_positions),  # type: ignore[arg-type]
            numerical_array=_array_view(parent, 'values_num', 
                                        np.ix_(index_positions, 
                                               _indices(parent.columns_num, columns_numerical))),
            string_array=_array_view(parent, 'values_str',
                                     np.ix_(index_positions, 
                                            _indices(parent.columns_str, columns_string))),
            columns_numerical=columns_numerical,
            columns_string=columns_string,
            attr_dict=AttrDict(name=parent.name,
                               timepoints_column_name=parent.timepoints_column_name,
                               locked_indices=parent.has_locked_indices,
                               locked_columns=parent.has_locked_columns,
                               repeating_index=parent.has_repeating_index),
            data=parent.data
        )
        
        self._parent = parent
        self._index_positions = index_positions
        self._inverted = inverted

    def __getattr__(self,
                    column_name: str) -> TemporalDataFrameView:
        """
        Get a single column from this view of a TemporalDataFrame.
        """
        return self._get_view(column_name, self._parent, self._index_positions)

    def __delattr__(self,
                    _column_name: str) -> NoReturn:
        raise TypeError('Cannot delete columns from a view.')

    def __getitem__(self,
                    slicer: Slicer | tuple[Slicer, Slicer] | tuple[Slicer, Slicer, Slicer]) -> TemporalDataFrameView:
        """Get a subset."""
        _index_positions, _columns_numerical, _columns_string = self._parse_inverted(*parse_slicer(self, slicer))
        return type(self)(parent=self._parent,
                          index_positions=_index_positions,
                          columns_numerical=_columns_numerical,
                          columns_string=_columns_string,
                          inverted=self._inverted)

    def __setitem__(self,
                    slicer: Slicer | tuple[Slicer, Slicer] | tuple[Slicer, Slicer, Slicer],
                    values: IFS | Collection[IFS] | TemporalDataFrameBase) -> None:
        """
        Set values in a subset.
        """
        index_positions, column_num_slicer, column_str_slicer, (tp_array, index_array, columns_array) = \
            parse_slicer(self, slicer)

        index_positions, columns_numerical, columns_string = self._parse_inverted(
            index_positions,
            column_num_slicer,
            column_str_slicer,
            (tp_array, index_array, columns_array)
        )

        if self._inverted:
            columns_array = np.concatenate((columns_numerical, columns_string))

        elif columns_array is None:
            columns_array = self.columns

        # parse values
        lcn, lcs = len(columns_numerical), len(columns_string)

        parsed_values = parse_values(values, len(index_positions), lcn + lcs)

        if not lcn + lcs:
            return

        # reorder values to match original index
        if index_array is None:
            index_array = self.index_at(self.tp0) if self.has_repeating_index else self.index

        index_positions.sort()

        original_positions = self._parent._get_index_positions(index_array)
        parsed_values = parsed_values[
            np.argsort(npi.indices(index_positions,
                                   original_positions[np.isin(original_positions, index_positions)]))
        ]

        if len(columns_numerical):
            self._parent.values_num[np.ix_(index_positions, 
                                           npi.indices(self._parent.columns_num, columns_numerical))] = \
                parsed_values[:, npi.indices(columns_array, columns_numerical)]
            
        if len(columns_string):
            values_str = parsed_values[:, npi.indices(columns_array, columns_string)].astype(str)
            if values_str.dtype > self._parent.values_str.dtype:
                self._parent._string_array = self._parent.values_str.astype(values_str.dtype)
            
            self._parent.values_str[np.ix_(index_positions,
                                           npi.indices(self._parent.columns_str, columns_string))] = \
                values_str

    def __invert__(self) -> TemporalDataFrameView:
        """
        Invert the getitem selection behavior : all elements NOT present in the slicers will be selected.
        """
        return type(self)(parent=self._parent,
                          index_positions=self._index_positions, 
                          columns_numerical=self._columns_numerical,
                          columns_string=self._columns_string,
                          inverted=not self._inverted)

    # endregion

    # region attributes
    @property
    def full_name(self) -> str:
        """
        Get the full name.
        """
        parts = []
        if self.empty:
            parts.append('empty')

        if self.is_inverted:
            parts.append('inverted')

        parent_full_name = self._parent.full_name
        if not parent_full_name.startswith('TemporalDataFrame'):
            parent_full_name = parent_full_name[0].lower() + parent_full_name[1:]

        parts += ['view of', parent_full_name]

        parts[0] = parts[0].capitalize()

        return ' '.join(parts)

    @property
    def parent(self) -> TemporalDataFrame:
        """Get the parent TemporalDataFrame of this view."""
        return self._parent

    @property
    def index(self) -> AnyNDArrayLike_IFS:
        """
        Get the index across all time-points.
        """
        return super().index

    @index.setter
    def index(self,
              values: NDArray_IFS) -> None:
        """
        Set the index for rows across all time-points.
        """
        new_index = self._parent.index.copy()
        new_index[self.index_positions] = values
        
        self._parent.index = new_index

    # endregion

    # region predicates
    @property
    def is_view(self) -> Literal[True]:
        """
        Is this a view on a TemporalDataFrame ?
        """
        return True

    @property
    def is_inverted(self) -> bool:
        """
        Whether this view of a TemporalDataFrame is inverted or not.
        """
        return self._inverted

    # endregion

    # region methods
    def lock_indices(self) -> None:
        """Lock the "index" axis to prevent modifications."""
        self._parent.lock_indices()

    def unlock_indices(self) -> None:
        """Unlock the "index" axis to allow modifications."""
        self._parent.unlock_indices()

    def lock_columns(self) -> None:
        """Lock the "columns" axis to prevent modifications."""
        self._parent.lock_columns()

    def unlock_columns(self) -> None:
        """Unlock the "columns" axis to allow modifications."""
        self._parent.unlock_columns()
    
    def set_index(self,
                  values: NDArray_IFS,
                  repeating_index: bool = False) -> None:
        """Set new index values."""
        raise NotImplementedError
    
    def reindex(self,
                order: NDArray_IFS,
                repeating_index: bool = False) -> None:
        """Re-order rows in this TemporalDataFrame so that their index matches the new given order."""
        raise NotImplementedError
    
    def _parse_inverted(self,
                        index_slicer: npt.NDArray[np.int_],
                        column_num_slicer: AnyNDArrayLike_IFS,
                        column_str_slicer: AnyNDArrayLike_IFS,
                        arrays: tuple[TimePointArray | None, AnyNDArrayLike_IFS | None, AnyNDArrayLike_IFS | None]) \
            -> tuple[npt.NDArray[np.int_], AnyNDArrayLike_IFS, AnyNDArrayLike_IFS]:
        tp_array, index_array, columns_array = arrays

        if self._inverted:
            if tp_array is None and index_array is None:
                _index_positions = self._index_positions[index_slicer]

            else:
                _index_positions = self._index_positions[~np.isin(self._index_positions, index_slicer)]

            if columns_array is None:
                _columns_numerical: AnyNDArrayLike_IFS = column_num_slicer
                _columns_string: AnyNDArrayLike_IFS = column_str_slicer

            else:
                _columns_numerical = cast(AnyNDArrayLike_IFS,
                                          self._columns_numerical[~np.isin(self._columns_numerical, column_num_slicer)])
                _columns_string = cast(AnyNDArrayLike_IFS,
                                       self._columns_string[~np.isin(self._columns_string, column_str_slicer)])

            return np.array(_index_positions), _columns_numerical, _columns_string

        return np.array(self._index_positions[index_slicer]), column_num_slicer, column_str_slicer

    # endregion

    # region data methods
    def merge(self,
              other: TemporalDataFrameBase,
              name: str | None = None) -> TemporalDataFrame:
        """
        Merge two TemporalDataFrames together, by rows. The column names and time points must match.

        Args:
            other: a TemporalDataFrame to merge with this one.
            name: a name for the merged TemporalDataFrame.

        Returns:
            A new merged TemporalDataFrame.
        """
        raise NotImplementedError

    # endregion
