from __future__ import annotations

import re
from abc import ABC, abstractmethod
from itertools import chain
from pathlib import Path
from typing import TYPE_CHECKING, Any, Collection, Iterable, Literal, cast

import ch5mpy as ch
import numpy as np
import numpy.typing as npt
import pandas as pd

import vdata.tdf as tdf
import vdata.timepoint as tp
from vdata._typing import (
    IF,
    IFS,
    AnyNDArrayLike,
    AnyNDArrayLike_IF,
    AnyNDArrayLike_IFS,
    AttrDict,
    NDArray_IFS,
    NDArrayLike_IFS,
    Slicer,
)
from vdata.IO import VLockError
from vdata.names import DEFAULT_TIME_COL_NAME
from vdata.tdf import indexer
from vdata.tdf._parse import parse_data_h5
from vdata.tdf.utils import equal_paths, underlined
from vdata.utils import are_equal, first_in, repr_array

if TYPE_CHECKING:
    from vdata.tdf.dataframe import TemporalDataFrame
    from vdata.tdf.indexer import (
        VAtIndexer,
        ViAtIndexer,
        ViLocIndexer,
        VLocIndexer,
    )
    from vdata.tdf.view import TemporalDataFrameView


class TemporalDataFrameBase(ABC):
    """
    Abstract base class for all TemporalDataFrames.
    """
    _attributes = {'_attr_dict', '_index', '_timepoints_array', '_columns_numerical', '_columns_string', 
                   '_numerical_array', '_string_array', '_data', '_timepoint_masks'}

    # region magic methods
    def __init__(self, 
                 index: AnyNDArrayLike_IFS,
                 timepoints_array: tp.TimePointArray,
                 numerical_array: AnyNDArrayLike_IF,
                 string_array: AnyNDArrayLike[np.str_],
                 columns_numerical: AnyNDArrayLike_IFS,
                 columns_string: AnyNDArrayLike_IFS,
                 attr_dict: AttrDict | ch.AttributeManager, 
                 data: ch.H5Dict[Any] | None = None):
        self._attr_dict = attr_dict
        self._index = index
        self._timepoints_array = timepoints_array
        self._columns_numerical = columns_numerical
        self._columns_string = columns_string
        self._numerical_array = numerical_array
        self._string_array = string_array
        self._data = data
        
        self._timepoint_masks: dict[tp.TimePoint, npt.NDArray[np.bool_]] = dict()
    
    def __repr__(self) -> str:       
        return f"{self.full_name}\n" \
               f"{self.head()}"               

    def __dir__(self) -> Iterable[str]:
        return chain(super().__dir__(), map(str, self.columns))

    def _get_view(self,
                  column_name: str,
                  parent: TemporalDataFrame,
                  index_positions: AnyNDArrayLike[np.int_]) -> TemporalDataFrameView:
        if column_name in self._columns_numerical:
            return tdf.TemporalDataFrameView(parent=parent, 
                                             index_positions=index_positions, 
                                             columns_numerical=np.array([column_name]),
                                             columns_string=np.array([]))

        elif column_name in self._columns_string:
            return tdf.TemporalDataFrameView(parent=parent, 
                                             index_positions=index_positions, 
                                             columns_numerical=np.array([]), 
                                             columns_string=np.array([column_name]))

        raise AttributeError(f"'{column_name}' not found in this TemporalDataFrame.")
    
    @abstractmethod
    def __getattr__(self,
                    column_name: str) -> TemporalDataFrameView:
        """
        Get a single column.
        """

    @abstractmethod
    def __getitem__(self,
                    slicer: Slicer | tuple[Slicer, Slicer] | tuple[Slicer, Slicer, Slicer]) \
            -> TemporalDataFrameView:
        """
        Get a subset.
        """

    @abstractmethod
    def __setitem__(self,
                    slicer: Slicer | tuple[Slicer, Slicer] | tuple[Slicer, Slicer, Slicer],
                    values: IFS | Collection[IFS] | TemporalDataFrameBase) -> None:
        """
        Set values in a subset.
        """

    def _check_compatibility(self,
                             value: TemporalDataFrameBase) -> None:
        # time-points column and nb of columns must be identical
        if not np.array_equal(self.timepoints_column, value.timepoints_column):
            raise ValueError("Time-points do not match.")
        if not np.array_equal(self.n_columns_num, value.n_columns_num):
            raise ValueError("Columns numerical do not match.")
        if not np.array_equal(self.n_columns_str, value.n_columns_str):
            raise ValueError("Columns string do not match.")

    def _add_core(self,
                  value: IFS | TemporalDataFrameBase) -> TemporalDataFrame:
        """
        Internal function for adding a value, called from __add__. Do not use directly.
        """
        if isinstance(value, (int, float, np.int_, np.float_)):
            if self._numerical_array.size == 0:
                raise ValueError("No numerical data to add to.")

            values_num = cast(AnyNDArrayLike_IF, self._numerical_array + value)
            values_str = self._string_array
            value_name: IFS = value

        elif isinstance(value, tdf.TemporalDataFrameBase):
            self._check_compatibility(value)

            values_num = cast(AnyNDArrayLike_IF, self._numerical_array + value.values_num)
            values_str = np.char.add(self._string_array, value.values_str)
            value_name = value.full_name

        elif isinstance(value, (str, np.str_)):
            if self._string_array.size == 0:
                raise ValueError("No string data to add to.")

            values_num = self._numerical_array
            values_str = np.char.add(self._string_array, value)
            value_name = value
            
        else:
            raise ValueError(f"Cannot add value with unknown type '{type(value)}'.")

        if self.timepoints_column_name is None:
            df_data = pd.concat((
                pd.DataFrame(np.array(values_num), 
                             index=np.array(self._index),
                             columns=np.array(self._columns_numerical)),
                pd.DataFrame(np.array(values_str),
                             index=np.array(self._index),
                             columns=np.array(self._columns_string))
            ), axis=1)

            return tdf.TemporalDataFrame(df_data,
                                         time_list=self.timepoints_column,
                                         lock=self.lock,
                                         name=f"{self.full_name} + {value_name}")

        else:
            df_data = pd.concat((
                pd.DataFrame(self.timepoints_column_str[:, None], 
                             index=np.array(self._index), 
                             columns=[str(self.timepoints_column_name)]),
                pd.DataFrame(np.array(values_num), 
                             index=np.array(self._index),
                             columns=np.array(self._columns_numerical)),
                pd.DataFrame(np.array(values_str), 
                             index=np.array(self._index),
                             columns=np.array(self._columns_string))
            ), axis=1)

            return tdf.TemporalDataFrame(df_data,
                                         time_col_name=self.timepoints_column_name,
                                         lock=self.lock,
                                         name=f"{self.full_name} + {value_name}")

    def __add__(self,
                value: IFS | TemporalDataFrameBase) -> TemporalDataFrame:
        """
        Get a copy with :
            - numerical values incremented by <value> if <value> is a number
            - <value> appended to string values if <value> is a string
        """
        return self._add_core(value)

    def __radd__(self,                      # type: ignore[misc]
                 value: IFS | TemporalDataFrameBase) -> TemporalDataFrame:
        """
        Get a copy with :
            - numerical values incremented by <value> if <value> is a number
            - <value> appended to string values if <value> is a string
        """
        return self.__add__(value)

    def _iadd_str(self,
                  value: str) -> TemporalDataFrameBase:
        """Inplace modification of the string values."""
        self._string_array = np.char.add(self._string_array, value)
        return self

    def __iadd__(self,
                 value: IFS | TemporalDataFrameBase) -> TemporalDataFrameBase:
        """
        Modify inplace the values :
            - numerical values incremented by <value> if <value> is a number.
            - <value> appended to string values if <value> is a string.
        """
        if isinstance(value, (str, np.str_)):
            if self._string_array.size == 0:
                raise ValueError("No string data to add to.")

            return self._iadd_str(value)

        elif isinstance(value, (int, float, np.int_, np.float_)):
            if self._numerical_array.size == 0:
                raise ValueError("No numerical data to add to.")

            self._numerical_array += value      # type: ignore[assignment]
            return self

        raise NotImplementedError

    def _op_core(self,
                 value: IF | TemporalDataFrameBase,
                 operation: Literal['sub', 'mul', 'div']) -> TemporalDataFrame:
        """
        Internal function for subtracting, multiplying by and dividing by a value, called from __add__. Do not use
        directly.
        """
        if operation == 'sub':
            if self._numerical_array.size == 0:
                raise ValueError("No numerical data to subtract.")
            op = '-'

            if isinstance(value, tdf.TemporalDataFrameBase):
                self._check_compatibility(value)

                values_num = self._numerical_array - value.values_num
                value_name = value.full_name

            elif isinstance(value, (int, float, np.int_, np.float_)):
                values_num = self._numerical_array - value
                value_name = str(value)
                
            else:
                raise ValueError(f"Cannot subtract value with unknown type '{type(value)}'.")

        elif operation == 'mul':
            if self._numerical_array.size == 0:
                raise ValueError("No numerical data to multiply.")
            op = '*'

            if isinstance(value, tdf.TemporalDataFrameBase):
                self._check_compatibility(value)

                values_num = self._numerical_array * value.values_num
                value_name = value.full_name

            elif isinstance(value, (int, float, np.int_, np.float_)):
                values_num = self._numerical_array * value
                value_name = str(value)
                
            else:
                raise ValueError(f"Cannot multiply by value with unknown type '{type(value)}'.")

        elif operation == 'div':
            if self._numerical_array.size == 0:
                raise ValueError("No numerical data to divide.")
            op = '/'

            if isinstance(value, tdf.TemporalDataFrameBase):
                self._check_compatibility(value)

                values_num = self._numerical_array / value.values_num
                value_name = value.full_name

            elif isinstance(value, (int, float, np.int_, np.float_)):
                values_num = self._numerical_array / value
                value_name = str(value)
                
            else:
                raise ValueError(f"Cannot divide by value with unknown type '{type(value)}'.")

        else:
            raise ValueError(f"Unknown operation '{operation}'.")

        if self.timepoints_column_name is None:
            df_data = pd.concat((
                pd.DataFrame(np.array(values_num), 
                             index=np.array(self._index),
                             columns=np.array(self._columns_numerical)),
                pd.DataFrame(np.array(self._string_array), 
                             index=np.array(self._index),
                             columns=np.array(self._columns_string))
            ), axis=1)

            return tdf.TemporalDataFrame(df_data,
                                         time_list=self.timepoints_column,
                                         lock=self.lock,
                                         name=f"{self.full_name} {op} {value_name}")

        else:
            df_data = pd.concat((
                pd.DataFrame(self.timepoints_column_str[:, None],
                             index=np.array(self._index), 
                             columns=[str(self.timepoints_column_name)]),
                pd.DataFrame(np.array(values_num),
                             index=np.array(self._index),
                             columns=np.array(self._columns_numerical)),
                pd.DataFrame(np.array(self._string_array),
                             index=np.array(self._index),
                             columns=np.array(self._columns_string))
            ), axis=1)

            return tdf.TemporalDataFrame(df_data,
                                                 time_col_name=self.timepoints_column_name,
                                                 lock=self.lock,
                                                 name=f"{self.full_name} {op} {value_name}")

    def __sub__(self,
                value: IF | TemporalDataFrameBase) -> TemporalDataFrame:
        """
        Get a copy with :
            - numerical values decremented by <value>.
        """
        return self._op_core(value, 'sub')

    def __rsub__(self,                      # type: ignore[misc]
                 value: IF | TemporalDataFrameBase) -> TemporalDataFrame:
        """
        Get a copy with :
            - numerical values decremented by <value>.
        """
        return self.__sub__(value)

    def __isub__(self,
                 value: IF | TemporalDataFrameBase) -> TemporalDataFrameBase:
        """
        Modify inplace the values :
            - numerical values decremented by <value>.
        """
        if self._numerical_array.size == 0:
            raise ValueError("No numerical data to subtract.")

        if isinstance(value, (int, float, np.int_, np.float_)):
            self._numerical_array -= value          # type: ignore[assignment]
            return self
        
        raise ValueError(f"Cannot subtract value with unknown type '{type(value)}'.")

    def __mul__(self,
                value: IF | TemporalDataFrameBase) -> TemporalDataFrame:
        """
        Get a copy with :
            - numerical values multiplied by <value>.
        """
        return self._op_core(value, 'mul')

    def __rmul__(self,                      # type: ignore[misc]
                 value: IF | TemporalDataFrameBase) -> TemporalDataFrame:
        """
        Get a copy with :
            - numerical values multiplied by <value>.
        """
        return self.__mul__(value)

    def __imul__(self,
                 value: IF | TemporalDataFrameBase) -> TemporalDataFrameBase:
        """
        Modify inplace the values :
            - numerical values multiplied by <value>.
        """
        if self._numerical_array.size == 0:
            raise ValueError("No numerical data to multiply.")

        if isinstance(value, (int, float, np.int_, np.float_)):
            self._numerical_array *= value          # type: ignore[assignment]
            return self
        
        raise ValueError(f"Cannot subtract value with unknown type '{type(value)}'.")

    def __truediv__(self,
                    value: IF | TemporalDataFrameBase) -> TemporalDataFrame:
        """
        Get a copy with :
            - numerical values divided by <value>.
        """
        return self._op_core(value, 'div')

    def __rtruediv__(self,                      # type: ignore[misc]
                     value: IF | TemporalDataFrameBase) -> TemporalDataFrame:
        """
        Get a copy with :
            - numerical values divided by <value>.
        """
        return self.__truediv__(value)

    def __itruediv__(self,
                     value: IF | TemporalDataFrameBase) -> TemporalDataFrameBase:
        """
        Modify inplace the values :
            - numerical values divided by <value>.
        """
        if self._numerical_array.size == 0:
            raise ValueError("No numerical data to divide.")

        if isinstance(value, (int, float, np.int_, np.float_)):
            self._numerical_array /= value          # type: ignore[assignment]
            return self
        
        raise ValueError(f"Cannot subtract value with unknown type '{type(value)}'.")

    def __eq__(self,                                                # type: ignore[override]
               other: Any) -> bool | npt.NDArray[np.bool_]:
        """
        Test for equality with :
            - another TemporalDataFrame or view of a TemporalDataFrame
            - a single value (either numerical or string)
        """
        if isinstance(other, tdf.TemporalDataFrameBase):
            for attr in ['timepoints_column_name', 'has_locked_indices', 'has_locked_columns', 'columns',
                         'timepoints_column', 'index', 'values_num', 'values_str']:
                if not are_equal(getattr(self, attr), getattr(other, attr)):
                    return False

            return True

        if isinstance(other, (int, float, np.number)):
            return cast(npt.NDArray[np.bool_],
                        self._numerical_array == other)

        elif isinstance(other, (str, np.str_)):
            return cast(npt.NDArray[np.bool_],
                        self._string_array == other)

        return False

    @abstractmethod
    def __invert__(self) -> TemporalDataFrameView:
        """
        Invert the getitem selection behavior : all elements NOT present in the slicers will be selected.
        """

    def __h5_write__(self, values: ch.H5Dict[Any]) -> None:
        if self._data is not None and equal_paths(self._data.filename, values.filename):
            return
        
        values.attributes.set(name=self.name,
                              locked_indices=self.has_locked_indices,
                              locked_columns=self.has_locked_columns,
                              repeating_index=self.has_repeating_index,
                              timepoints_column_name=self.timepoints_column_name)
        
        ch.write_datasets(values, 
                          index=self.index, 
                          columns_numerical=self.columns_num,
                          columns_string=self.columns_str,
                          timepoints_array=self.timepoints_column_str)
        
        ch.write_datasets(values, 
                          numerical_array=self.values_num,
                          string_array=self.values_str,
                          chunks=True)
    
    # endregion

    # region attributes
    @property
    def name(self) -> str:
        """
        Get the name.
        """
        return self._attr_dict['name']

    @property
    @abstractmethod
    def full_name(self) -> str:
        """
        Get the full name.
        """

    @property
    def data(self) -> ch.H5Dict[Any] | None:
        """Get the file backing this TemporalDataFrame."""
        return self._data

    @property
    def lock(self) -> tuple[bool, bool]:
        """
        Get the index and columns lock state.
        """
        return self.has_locked_indices, self.has_locked_columns

    @property
    def shape(self) -> tuple[int, list[int], int]:
        """
        Get the shape of this TemporalDataFrame as a 3-tuple of :
            - number of time-points
            - number of rows per time-point
            - number of columns
        """
        return self.n_timepoints, \
            [self.n_index_at(tp) for tp in self.timepoints], \
            self.n_columns_num + self.n_columns_str

    @property
    def timepoints(self) -> tp.TimePointArray:
        """
        Get the list of unique time points in this TemporalDataFrame.
        """
        unique_timepoints_idx = np.unique(self._timepoints_array, equal_nan=False, return_index=True)[1].astype(int)
        return self._timepoints_array[sorted(unique_timepoints_idx)]

    @property
    def timepoints_column(self) -> tp.TimePointArray:
        """
        Get the column of time-point values.
        """
        return self._timepoints_array

    @property
    def n_timepoints(self) -> int:
        return len(self.timepoints)

    @property
    def timepoints_column_str(self) -> npt.NDArray[np.str_]:
        """
        Get the column of time-point values cast as strings.
        """
        return self.timepoints_column.astype(str)

    @property
    def timepoints_column_numerical(self) -> npt.NDArray[np.float_]:
        """
        Get the column of time-point values cast as floats.
        """
        return tp.TimePointArray(self._timepoints_array).astype(np.float64)

    @property
    def timepoints_column_name(self) -> str | None:
        """
        Get the name of the column containing the time-points values.
        """
        return self._attr_dict['timepoints_column_name']

    @property
    def index(self) -> AnyNDArrayLike_IFS:
        """
        Get the index across all time-points.
        """
        return self._index
    
    @index.setter
    @abstractmethod
    def index(self,
              values: NDArray_IFS) -> None:
        """
        Set the index for rows across all time-points.
        """
    
    @property
    def n_index(self) -> int:
        return len(self._index)

    @property
    def columns_num(self) -> AnyNDArrayLike_IFS:
        """
        Get the list of column names for numerical data.
        """
        return self._columns_numerical

    @columns_num.setter
    def columns_num(self,
                    values: AnyNDArrayLike_IFS) -> None:
        """
        Set the list of column names for numerical data.
        """
        if self.has_locked_columns:
            raise VLockError("Cannot set columns in tdf with locked columns.")

        self._columns_numerical[:] = values
    
    @property
    def n_columns_num(self) -> int:
        """
        Get the number of numerical data columns.
        """
        return len(self._columns_numerical)

    @property
    def columns_str(self) -> AnyNDArrayLike_IFS:
        """
        Get the list of column names for string data.
        """
        return self._columns_string

    @columns_str.setter
    def columns_str(self,
                    values: AnyNDArrayLike_IFS) -> None:
        """
        Set the list of column names for string data.
        """
        if self.has_locked_columns:
            raise VLockError("Cannot set columns in tdf with locked columns.")

        self._columns_string[:] = values

    @property
    def n_columns_str(self) -> int:
        """
        Get the number of string data columns.
        """
        return len(self._columns_string)

    @property
    def columns(self) -> NDArray_IFS:
        """Get the list of all column names."""
        return np.concatenate((self._columns_numerical, self._columns_string))
    
    @columns.setter
    def columns(self, values: NDArray_IFS) -> None:
        """Set new column names."""
        if len(values) != self.n_columns:
            raise ValueError(f'Cannot set {self.n_columns}, {len(values)} provided.')
        
        self._columns_numerical[:] = values[:self.n_columns_num]
        self._columns_string[:] = values[self.n_columns_num:]

    @property
    def n_columns(self) -> int:
        return self.n_columns_num + self.n_columns_str

    @property
    def values_num(self) -> AnyNDArrayLike_IF:
        """
        Get the numerical data.
        """
        return self._numerical_array
    
    @values_num.setter
    def values_num(self,
                   values: AnyNDArrayLike_IF) -> None:
        """
        Set the numerical data.
        """
        self._numerical_array[:] = values
        
    @property
    def values_str(self) -> AnyNDArrayLike[np.str_]:
        """
        Get the string data.
        """
        return self._string_array
    
    @values_str.setter
    def values_str(self,
                   values: AnyNDArrayLike[np.str_]) -> None:
        """
        Set the string data.
        """
        self._string_array[:] = values
        
    @property
    def values(self) -> npt.NDArray[np.object_]:
        """
        Get all the data (num and str concatenated).
        """
        return np.hstack((self._numerical_array.astype(object), self._string_array))

    @property
    def tp0(self) -> tp.TimePoint:
        return self.timepoints[0]

    @property
    def at(self) -> VAtIndexer:
        """
        Access a single value from a pair of row and column labels.
        """
        return indexer.VAtIndexer(self)

    @property
    def iat(self) -> ViAtIndexer:
        """
        Access a single value from a pair of row and column indices.
        """
        return indexer.ViAtIndexer(self)

    @property
    def loc(self) -> VLocIndexer:
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
        return indexer.VLocIndexer(self)

    @property
    def iloc(self) -> ViLocIndexer:
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
        return indexer.ViLocIndexer(self)

    # endregion

    # region predicates
    @property
    def has_locked_indices(self) -> bool:
        """
        Is the "index" axis locked for modification ?
        """
        return self._attr_dict['locked_indices']

    @property
    def has_locked_columns(self) -> bool:
        """
        Is the "columns" axis locked for modification ?
        """
        return self._attr_dict['locked_columns']

    @property
    def has_repeating_index(self) -> bool:
        """
        Is the index repeated at each time-point ?
        """
        return self._attr_dict['repeating_index']

    def _empty_numerical(self) -> bool:
        return self._numerical_array.size == 0

    def _empty_string(self) -> bool:
        return self._string_array.size == 0

    @property
    def empty(self) -> bool:
        """
        Whether this TemporalDataFrame is empty (no numerical data and no string data).
        """
        return self._empty_numerical() and self._empty_string()

    @property
    @abstractmethod
    def is_view(self) -> bool:
        """
        Is this a view on a TemporalDataFrame ?
        """

    @property
    def is_backed(self) -> bool:
        """
        Is this TemporalDataFrame backed on a file ?
        """
        return self._data is not None

    @property
    def is_closed(self) -> bool:
        """
        Is the h5 file (this TemporalDataFrame is backed on) closed ?
        """
        return self._data is not None and not bool(self._data.is_closed)

    # endregion

    # region methods
    @abstractmethod
    def lock_indices(self) -> None:
        """Lock the "index" axis to prevent modifications."""

    @abstractmethod
    def unlock_indices(self) -> None:
        """Unlock the "index" axis to allow modifications."""

    @abstractmethod
    def lock_columns(self) -> None:
        """Lock the "columns" axis to prevent modifications."""

    @abstractmethod
    def unlock_columns(self) -> None:
        """Unlock the "columns" axis to allow modifications."""
    
    @abstractmethod
    def set_index(self,
                  values: NDArray_IFS,
                  repeating_index: bool = False) -> None:
        """Set new index values."""
    
    @abstractmethod
    def reindex(self,
                order: NDArray_IFS,
                repeating_index: bool = False) -> None:
        """Re-order rows in this TemporalDataFrame so that their index matches the new given order."""
    
    def _repr_single_array(self,
                           tp: tp.TimePoint,
                           n: int,
                           array: AnyNDArrayLike_IFS, 
                           columns_: AnyNDArrayLike_IFS) \
    -> tuple[pd.DataFrame, tuple[int, ...]]:
        tp_data_array_ = cast(NDArrayLike_IFS, array[self.get_timepoint_mask(tp)])

        tp_array_ = np.array([[tp] for _ in range(min(n, tp_data_array_.shape[0]))])

        spacer_ = np.array([['\uff5c'] for _ in range(min(n, tp_data_array_.shape[0]))])

        columns_ = np.concatenate(([self.timepoints_column_name, ''], columns_)) \
            if self.timepoints_column_name is not None else \
            np.concatenate(([DEFAULT_TIME_COL_NAME, ''], columns_))

        tp_df_ = pd.DataFrame(np.concatenate((tp_array_,
                                              spacer_,
                                              np.array(tp_data_array_[:n])), axis=1),
                                index=self.index_at(tp)[:n],
                                columns=columns_)

        return tp_df_, tp_data_array_.shape
        
    def _head_tail(self,
                   n: int) -> str:
        """
        Common function for getting a head or tail representation of this TemporalDataFrame.

        Args:
            n: number of rows to print.

        Returns:
            A short string representation of the first/last n rows in this TemporalDataFrame.
        """
        if not len(self.timepoints):
            return f"Time points: []\n" \
                   f"Columns: {[col for col in self.columns]}\n" \
                   f"Index: {[idx for idx in self._index]}"

        repr_string = ""

        for timepoint in self.timepoints[:5]:
            # display the first n rows of the first 5 timepoints in this TemporalDataFrame
            repr_string += underlined(f"Time point : {repr(timepoint)}") + "\n"

            if not self._empty_numerical() and not self._empty_string():
                tp_mask = self.get_timepoint_mask(timepoint)

                tp_numerical_array = cast(AnyNDArrayLike_IF, self._numerical_array[tp_mask])
                tp_string_array = cast(AnyNDArrayLike[np.str_], self._string_array[tp_mask])

                tp_array = np.tile(timepoint, (min(n, tp_numerical_array.shape[0]), 1))
                spacer = np.tile('\uff5c', (min(n, tp_numerical_array.shape[0]), 1))

                tp_col_name = DEFAULT_TIME_COL_NAME if self.timepoints_column_name is None else \
                    self.timepoints_column_name
                columns: npt.NDArray[np.str_] = np.concatenate(([tp_col_name, ''], 
                                                                self._columns_numerical[:],
                                                                [''], 
                                                                self._columns_string[:]))

                tp_df = pd.DataFrame(np.concatenate((tp_array,
                                                     spacer,
                                                     np.array(tp_numerical_array[:n]),
                                                     spacer,
                                                     np.array(tp_string_array[:n])), axis=1),
                                     index=self.index_at(timepoint)[:n],
                                     columns=columns)
                tp_shape: tuple[int, ...] = (tp_numerical_array.shape[0], 
                                             tp_numerical_array.shape[1] + tp_string_array.shape[1])

            elif not self._empty_numerical():
                tp_df, tp_shape = self._repr_single_array(timepoint, n, self._numerical_array, self._columns_numerical)

            elif not self._empty_string():
                tp_df, tp_shape = self._repr_single_array(timepoint, n, self._string_array, self._columns_string)

            else:
                nb_rows_at_tp = int(np.sum(self.get_timepoint_mask(timepoint)))

                tp_array_ = np.array([[timepoint] for _ in range(min(n, nb_rows_at_tp))])

                spacer = np.array([['\uff5c'] for _ in range(min(n, nb_rows_at_tp))])

                columns_ = [self.timepoints_column_name, ''] if self.timepoints_column_name is not None \
                    else [DEFAULT_TIME_COL_NAME, '']

                tp_df = pd.DataFrame(np.concatenate((tp_array_, spacer), axis=1),
                                     index=self.index_at(timepoint)[:n],
                                     columns=columns_)

                tp_shape = (tp_df.shape[0], 0)

            # remove unwanted shape display by pandas and replace it by our own
            repr_tp_df = repr(tp_df)
            
            if re.search(r'\n\[.*$', repr_tp_df) is None:
                repr_string += f'{repr(tp_df)}\n[{tp_shape[0]} rows x {tp_shape[1]} columns]\n\n'
                
            else:
                repr_string += re.sub(r'\n\[.*$', 
                                      f'[{tp_shape[0]} rows x {tp_shape[1]} columns]\n\n', 
                                      repr(tp_df))

        # then display only the list of remaining timepoints
        if len(self.timepoints) > 5:
            repr_string += f"\nSkipped time points {repr_array(self.timepoints[5:])} ...\n\n\n"

        return repr_string

    def head(self,
             n: int = 5) -> str:
        """
        Get a short representation of the first n rows in this TemporalDataFrame.

        Args:
            n: number of rows to print.

        Returns:
            A short string representation of the first n rows in this TemporalDataFrame.
        """
        return self._head_tail(n)

    def tail(self,
             n: int = 5) -> str:
        """
        Get a short representation of the last n rows in this TemporalDataFrame.

        Args:
            n: number of rows to print.

        Returns:
            A short string representation of the last n rows in this TemporalDataFrame.
        """
        # TODO : negative n not handled
        return self._head_tail(-n)

    def _fast_compare(self,
                      comparison_tp: tp.TimePoint) -> npt.NDArray[np.bool_]:       
        if not len(self._timepoint_masks):
            return self._timepoints_array == comparison_tp              # type: ignore[no-any-return]

        tp_mask = np.zeros(len(self._timepoints_array), dtype=bool)

        if len(self._timepoint_masks) == 1:
            not_already_computed = ~first_in(self._timepoint_masks)

        else:
            not_already_computed = np.logical_and.reduce([~mask for mask in self._timepoint_masks.values()])

        tp_mask[not_already_computed] = self._timepoints_array[not_already_computed] == comparison_tp
        return tp_mask

    def get_timepoint_mask(self,
                           timepoint: tp.TimePoint | str) -> npt.NDArray[np.bool_]:
        """
        Get a boolean mask indicating where in this TemporalDataFrame's the rows' time-point are equal to <timepoint>.

        Args:
            timepoint: A time-point (str or tp.TimePoint object) to get a mask for.

        Returns:
            A boolean mask for rows matching the time-point.
        """
        timepoint = tp.TimePoint(timepoint)
        
        # cache masks for performance, cache is reinitialized when _timepoints_array changes
        if timepoint not in self._timepoint_masks.keys():
            self._timepoint_masks[timepoint] = self._fast_compare(timepoint)

        return self._timepoint_masks[timepoint]

    def index_at(self,
                 timepoint: str | tp.TimePoint) -> NDArray_IFS:
        """
        Get the index of rows existing at the given time-point.

        Args:
            timepoint: time_point for which to get the index.

        Returns:
            The index of rows existing at that time-point.
        """
        return np.atleast_1d(self._index[self.get_timepoint_mask(timepoint)])

    def n_index_at(self,
                   timepoint: str | tp.TimePoint) -> int:
        """
        Get the number of rows existing at the given time-point.
        """
        return len(self.index_at(timepoint))

    def _min_max_mean(self,
                      axis: int | None,
                      func: Literal['min', 'max', 'mean']) -> float | TemporalDataFrame:
        if axis is None:
            return float(getattr(self._numerical_array, func)())

        elif axis == 0:
            i0 = self.index_at(self.tp0)
            for timepoint in self.timepoints[1:]:
                if not np.array_equal(i0, self.index_at(timepoint)):
                    raise ValueError(f"Can't take '{func}' along axis 0 if indices are not the same at all "
                                     f"time-points.")

            mmm_tp = getattr(self.timepoints, func)()

            return tdf.TemporalDataFrame(data=pd.DataFrame(
                getattr(np, func)([self._numerical_array[self.get_timepoint_mask(timepoint)] 
                                   for timepoint in self.timepoints], axis=0),
                index=i0,
                columns=np.array(self._columns_numerical),
            ),
                time_list=[mmm_tp for _ in enumerate(i0)],
                time_col_name=self.timepoints_column_name)

        elif axis == 1:
            return tdf.TemporalDataFrame(data=pd.DataFrame([
                    getattr(np, func)(self._numerical_array[self.get_timepoint_mask(timepoint)], axis=0) 
                    for timepoint in self.timepoints
                ],
                index=[func for _ in enumerate(self.timepoints)],
                columns=np.array(self._columns_numerical)
            ),
                repeating_index=True,
                time_list=self.timepoints,
                time_col_name=self.timepoints_column_name)

        elif axis == 2:
            return tdf.TemporalDataFrame(data=pd.DataFrame(
                getattr(np, func)(self._numerical_array, axis=1),
                index=np.array(self._index),
                columns=[func],
            ),
                time_list=self.timepoints_column,
                time_col_name=self.timepoints_column_name)

        raise ValueError(f"Invalid axis '{axis}', should be in [0, 1, 2].")

    def min(self,
            axis: int | None = None) -> float | TemporalDataFrame:
        """
        Get the min value along the specified axis.

        Args:
            axis: Can be 0 (time-points), 1 (rows), 2 (columns) or None (global min). (default: None)
        """
        return self._min_max_mean(axis, 'min')

    def max(self,
            axis: int | None = None) -> float | TemporalDataFrame:
        """
        Get the max value along the specified axis.

        Args:
            axis: Can be 0 (time-points), 1 (rows), 2 (columns) or None (global max). (default: None)
        """
        return self._min_max_mean(axis, 'max')

    def mean(self,
             axis: int | None = None) -> float | TemporalDataFrame:
        """
        Get the mean value along the specified axis.

        Args:
            axis: Can be 0 (time-points), 1 (rows), 2 (columns) or None (global mean). (default: None)
        """
        return self._min_max_mean(axis, 'mean')

    # endregion

    # region data methods
    def _convert_to_pandas(self,
                           with_timepoints: str | None = None,
                           timepoints_type: Literal['string', 'numerical'] = 'string',
                           str_index: bool = False) -> pd.DataFrame:
        """
        Internal function for converting to a pandas DataFrame. Do not use directly, it is called by '.to_pandas()'.

        Args:
            with_timepoints: Name of the column containing time-points data to add to the DataFrame. If left to None,
                no column is created.
            timepoints_type: if <with_timepoints> if True, type of the timepoints that will be added (either 'string'
                or 'numerical'). (default: 'string')
            str_index: cast index as string ?
        """
        index_ = np.array(self._index.astype(str) if str_index else self._index)
        
        if with_timepoints is None:
            return pd.concat((pd.DataFrame(np.array(self._numerical_array) if self._numerical_array.size else None,
                                           index=index_, 
                                           columns=np.array(self._columns_numerical)),
                              pd.DataFrame(np.array(self._string_array) if self._string_array.size else None,
                                           index=index_, 
                                           columns=np.array(self._columns_string))),
                             axis=1)

        if timepoints_type == 'string':
            return pd.concat((
                pd.DataFrame(self.timepoints_column_str[:, None], 
                             index=index_, 
                             columns=[str(with_timepoints)]),
                pd.DataFrame(np.array(self._numerical_array) if self._numerical_array.size else None,
                             index=index_, 
                             columns=np.array(self._columns_numerical)),
                pd.DataFrame(np.array(self._string_array) if self._string_array.size else None,
                             index=index_, 
                             columns=np.array(self._columns_string))
            ), axis=1)

        elif timepoints_type == 'numerical':
            return pd.concat((
                pd.DataFrame(self.timepoints_column_numerical[:, None], 
                             index=index_, 
                             columns=[str(with_timepoints)]),
                pd.DataFrame(np.array(self._numerical_array) if self._numerical_array.size else None,
                             index=index_, 
                             columns=np.array(self._columns_numerical)),
                pd.DataFrame(np.array(self._string_array) if self._numerical_array.size else None,
                             index=index_, 
                             columns=np.array(self._columns_string))
            ), axis=1)

        raise ValueError(f"Invalid timepoints_type argument '{timepoints_type}'. Should be 'string' or 'numerical'.")

    def to_pandas(self,
                  with_timepoints: str | None = None,
                  timepoints_type: Literal['string', 'numerical'] = 'string',
                  str_index: bool = False) -> pd.DataFrame:
        """
        Convert to a pandas DataFrame.

        Args:
            with_timepoints: Name of the column containing time-points data to add to the DataFrame. If left to None,
                no column is created.
            timepoints_type: if <with_timepoints> if True, type of the timepoints that will be added (either 'string'
                or 'numerical'). (default: 'string')
            str_index: cast index as string ?
        """
        return self._convert_to_pandas(with_timepoints=with_timepoints,
                                       timepoints_type=timepoints_type,
                                       str_index=str_index)

    def write(self,
              file: str | Path | ch.File | ch.Group | ch.H5Dict[Any]) -> None:
        """
        Save in HDF5 file format.

        Args:
            file: path to save the data.
        """
        if not isinstance(file, (ch.H5Dict, ch.Group)):
            file = ch.File(file, mode=ch.H5Mode.WRITE_TRUNCATE)
            
        if not isinstance(file, ch.H5Dict):
            file = ch.H5Dict(file)
            
        ch.write_object(file, '', self)
        
        if not self.is_backed:
            self._attr_dict = parse_data_h5(file, self.lock, self.name)
            self._index = cast(AnyNDArrayLike_IFS, file['index'])
            self._timepoints_array = cast(tp.TimePointArray, file["timepoints_array"].maptype(tp.TimePoint))
            self._numerical_array = cast(AnyNDArrayLike_IF, file["numerical_array"])
            self._string_array = cast(AnyNDArrayLike[np.str_], file["string_array"])
            self._columns_numerical = cast(AnyNDArrayLike_IFS, file["columns_numerical"])
            self._columns_string = cast(AnyNDArrayLike_IFS, file["columns_string"])
            self._data = file
        
    def to_csv(self, 
               path: str | Path,
               sep: str = ",", na_rep: str = "",
               index: bool = True,
               header: bool = True) -> None:
        """
        Save this TemporalDataFrame in a csv file.

        Args:
            path: a path to the csv file.
            sep: String of length 1. Field delimiter for the output file.
            na_rep: Missing data representation.
            index: Write row names (index) ?
            header: Write out the column names ? If a list of strings is given it is assumed to be aliases for the
                column names.
        """
        tp_col_name = self.timepoints_column_name if self.timepoints_column_name is not None else \
            DEFAULT_TIME_COL_NAME

        self.to_pandas(with_timepoints=tp_col_name).to_csv(path, sep=sep, na_rep=na_rep, index=index, header=header)

    def copy(self) -> TemporalDataFrame:
        """
        Get a copy.
        """
        if self.timepoints_column_name is None:
            return tdf.TemporalDataFrame(self.to_pandas(),
                                         repeating_index=self.has_repeating_index,
                                         time_list=self.timepoints_column,
                                         lock=self.lock,
                                         name=f"copy of {self.name}")

        return tdf.TemporalDataFrame(self.to_pandas(with_timepoints=self.timepoints_column_name),
                                     repeating_index=self.has_repeating_index,
                                     time_col_name=self.timepoints_column_name,
                                     lock=self.lock,
                                     name=f"copy of {self.name}")

    @abstractmethod
    def merge(self,
              other: TemporalDataFrame,
              name: str | None = None) -> TemporalDataFrame:
        """
        Merge two TemporalDataFrames together, by rows. The column names and time points must match.

        Args:
            other: a TemporalDataFrame to merge with this one.
            name: a name for the merged TemporalDataFrame.

        Returns:
            A new merged TemporalDataFrame.
        """

    # endregion
