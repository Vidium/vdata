# coding: utf-8
# Created on 28/03/2022 11:22
# Author : matteo

# ====================================================
# imports
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Collection, Iterable, Literal, cast

import ch5mpy as ch
import numpy as np
import numpy.typing as npt
import numpy_indexed as npi
import pandas as pd

import vdata.tdf as tdf
import vdata.timepoint as tp
from vdata._typing import IFS, AnyNDArrayLike_IFS, AttrDict, Collection_IFS, NDArray_IFS, NDArrayLike_IFS, Slicer
from vdata.IO import VLockError
from vdata.names import DEFAULT_TIME_COL_NAME, NO_NAME
from vdata.tdf._parse import parse_data
from vdata.tdf.base import TemporalDataFrameBase
from vdata.tdf.utils import parse_slicer, parse_slicer_full, parse_values
from vdata.utils import isCollection

if TYPE_CHECKING:
    from vdata.tdf.view import TemporalDataFrameView


# ====================================================
# code
class TemporalDataFrame(TemporalDataFrameBase):
    """
    An equivalent to pandas DataFrames that includes the notion of time on the rows.
    This class implements a modified sub-setting mechanism to subset on time points, rows and columns
    """
    
    # region magic methods
    def __init__(self,
                 data: dict[str, NDArray_IFS] | pd.DataFrame | None = None,
                 index: Collection_IFS | None = None,
                 repeating_index: bool = False,
                 columns: Collection[IFS] | None = None,
                 time_list: Collection[IFS | tp.TimePoint] | IFS | tp.TimePoint | None = None,
                 time_col_name: str | None = None,
                 lock: tuple[bool, bool] | None = None,
                 name: str = NO_NAME):
        """
        Args:
            data: Optional object containing the data to store in this TemporalDataFrame. It can be :
                - a dictionary of ['column_name': [values]], where [values] has always the same length
                - a pandas DataFrame
                - a single value to fill the data with
            index: Optional collection of indices. Must match the total number of rows in this TemporalDataFrame,
                over all time-points.
            repeating_index: Is the index repeated at all time-points ?
                If False, the index must contain unique values.
                If True, the index must be exactly equal at all time-points.
            columns: Optional column names.
            time_list: Optional list of time values of the same length as the index, indicating for each row at which
                time point it exists.
            time_col_name: Optional column name in data (if data is a dictionary or a pandas DataFrame) to use as
                time list. This parameter will be ignored if the 'time_list' parameter was set.
            lock: Optional 2-tuple of booleans indicating which axes (index, columns) are locked.
                If 'index' is locked, .index.setter() and .reindex() cannot be used.
                If 'columns' is locked, .__delattr__(), .columns.setter() and .insert() cannot be used.
            name: a name for this TemporalDataFrame.
        """        
        _numerical_array, _string_array, _timepoints_array, _index, _columns_numerical, _columns_string, \
            _lock, _timepoints_column_name, _name, repeating_index = \
            parse_data(data, index, repeating_index, columns, time_list, time_col_name, lock, name)
            
        super().__init__(index=_index, 
                         timepoints_array=_timepoints_array,
                         numerical_array=_numerical_array,
                         string_array=_string_array,
                         columns_numerical=_columns_numerical,
                         columns_string=_columns_string,
                         attr_dict=AttrDict(name=_name,
                                            timepoints_column_name=_timepoints_column_name,
                                            locked_indices=_lock[0],
                                            locked_columns=_lock[1],
                                            repeating_index=repeating_index))
        
    def __getattr__(self,
                    column_name: str) -> TemporalDataFrameView:
        """
        Get a single column from this TemporalDataFrame.
        """
        return self._get_view(column_name, self, np.arange(self.n_index))

    def __setattr__(self,
                    name: IFS,
                    values: NDArray_IFS) -> None:
        """
        Set values of a single column. If the column does not already exist in this TemporalDataFrame, it is created
            at the end.
        """
        if isinstance(name, str) and (name in self._attributes or name in object.__dir__(self)):
            object.__setattr__(self, name, values)
            return

        values = np.array(values)

        if len(values) != self.n_index:
            raise ValueError(f"Wrong number of values ({len(values)}) for column '{name}', expected {self.n_index}.")

        if name in self.columns_num:
            # set values for numerical column
            self._numerical_array[:, np.where(self._columns_numerical == name)[0][0]] = \
                values.astype(self._numerical_array.dtype)

        elif name in self.columns_str:
            # set values for string column
            self._string_array[:, np.where(self._columns_string == name)[0][0]] = \
                values.astype(str)

        else:
            if np.issubdtype(values.dtype, np.number):
                # create numerical column
                self._numerical_array = np.insert(self._numerical_array, self.n_columns_num, values, axis=1)
                self._columns_numerical = np.insert(self._columns_numerical, self.n_columns_num, name)
                
            else:
                # create string column
                self._string_array = np.insert(self._string_array, self.n_columns_str, values, axis=1)
                self._columns_string = np.insert(self._columns_string, self.n_columns_str, name)

    def __delattr__(self,
                    column_name: str) -> None:
        """Drop a column."""
        if self.has_locked_columns:
            raise VLockError("Cannot delete column from tdf with locked columns.")

        if column_name in self._columns_numerical:
            item_index = np.where(self._columns_numerical == column_name)[0][0]
            self._numerical_array = np.delete(self._numerical_array, item_index, axis=1)
            self._columns_numerical = np.delete(self._columns_numerical, item_index)

        elif column_name in self.columns_str:
            item_index = np.where(self._columns_string == column_name)[0][0]
            self._string_array = np.delete(self._string_array, item_index, axis=1)
            self._columns_string = np.delete(self._columns_string, item_index)

        else:
            raise AttributeError(f"'{column_name}' not found in this TemporalDataFrame.")

    def __getitem__(self,
                    slicer: Slicer | tuple[Slicer, Slicer] | tuple[Slicer, Slicer, Slicer]) -> TemporalDataFrameView:
        index_slicer, column_num_slicer, column_str_slicer = parse_slicer(self, slicer)

        return tdf.TemporalDataFrameView(parent=self, 
                                         index_positions=index_slicer,
                                         columns_numerical=column_num_slicer,
                                         columns_string=column_str_slicer)

    def __setitem__(self,
                    slicer: Slicer | tuple[Slicer, Slicer] | tuple[Slicer, Slicer, Slicer],
                    values: IFS | Collection[IFS] | TemporalDataFrameBase) -> None:
        """
        Set values in a subset.
        """
        # TODO : setattr if setting a single column

        index_positions, column_num_slicer, column_str_slicer, (_, index_array, columns_array) = \
            parse_slicer_full(self, slicer)

        if columns_array is None:
            columns_array = self.columns

        # parse values
        lcn, lcs = len(column_num_slicer), len(column_str_slicer)

        parsed_values = parse_values(values, len(index_positions), lcn + lcs)

        if not lcn + lcs:
            return

        # reorder values to match original index
        if index_array is None:
            index_array = self.index_at(self.tp0) if self.has_repeating_index else self.index
                
        index_positions.sort()

        original_positions = self._get_index_positions(index_array)
        parsed_values = parsed_values[
            np.argsort(npi.indices(index_positions,
                                   original_positions[np.isin(original_positions, index_positions)]))
        ]

        if len(column_num_slicer):
            self._numerical_array[np.ix_(index_positions, npi.indices(self._columns_numerical, column_num_slicer))] = \
                parsed_values[:, npi.indices(columns_array, column_num_slicer)]

        if len(column_str_slicer):
            values_str = parsed_values[:, npi.indices(columns_array, column_str_slicer)].astype(str)
            if values_str.dtype > self._string_array.dtype:
                self._string_array = self._string_array.astype(values_str.dtype)

            self.values_str[np.ix_(index_positions, npi.indices(self._columns_string, column_str_slicer))] = \
                values_str

    def __invert__(self) -> TemporalDataFrameView:
        """
        Invert the getitem selection behavior : all elements NOT present in the slicers will be selected.
        """
        return tdf.TemporalDataFrameView(parent=self, 
                                         index_positions=np.arange(self.n_index), 
                                         columns_numerical=self._columns_numerical, 
                                         columns_string=self._columns_string,
                                         inverted=True)

    # endregion

    # region class methods
    @classmethod
    def __h5_read__(cls, values: ch.H5Dict[Any]) -> TemporalDataFrame:
        obj = cls.__new__(cls)
               
        super().__init__(obj,
                         index=values['index'], 
                         timepoints_array=tp.TimePointArray(values['timepoints_array']),
                         numerical_array=values['numerical_array'],
                         string_array=values['string_array'],
                         columns_numerical=values['columns_numerical'],
                         columns_string=values['columns_string'],
                         attr_dict=values.attributes,
                         data=values)
        return obj
    
    @classmethod
    def read(cls,
             file: str | Path | ch.Group | ch.H5Dict[Any],
             mode: Literal[ch.H5Mode.READ, ch.H5Mode.READ_WRITE] | None = None) -> TemporalDataFrame:        
        _mode = _get_valid_mode(file, mode)

        if isinstance(file, ch.Group) and file.file.mode != _mode:
            raise ValueError(f'Cannot read TemporalDataFrame in {_mode} mode from file open in {file.file.mode} mode.')

        if not isinstance(file, ch.H5Dict):
            file = ch.H5Dict.read(file, mode=_mode)
            
        return TemporalDataFrame.__h5_read__(file)
    
    @classmethod
    def read_from_csv(cls,
                      file: Path,
                      sep: str = ',',
                      time_list: Collection[IFS | tp.TimePoint] | IFS | tp.TimePoint | None = None,
                      time_col_name: str | None = None) -> TemporalDataFrame:
        """
        Read a .csv file into a TemporalDataFrame.

        Args:
            file: a path to the .csv file to read.
            sep: delimiter to use for reading the .csv file.
            time_list: time points for the dataframe's rows. (see TemporalDataFrame's documentation for more details.)
            time_col_name: if time points are not given explicitly with the 'time_list' parameter, a column name can be
                given. This column will be used as the time data.

        Returns:
            A TemporalDataFrame built from the .csv file.
        """
        df = pd.read_csv(file, index_col=0, sep=sep)

        time_col_name = DEFAULT_TIME_COL_NAME if time_col_name is None else time_col_name

        if time_list is None and time_col_name == DEFAULT_TIME_COL_NAME:
            time_list = df[DEFAULT_TIME_COL_NAME].values.tolist()
            del df[time_col_name]
            time_col_name = None

        return TemporalDataFrame(df, time_list=time_list, time_col_name=time_col_name)
    
    # endregion

    # region attributes
    @property
    def name(self) -> str:
        """
        Get the name.
        """
        return super().name

    @name.setter
    def name(self,
             name: str) -> None:
        """Set the name."""
        self._attr_dict['name'] = str(name)

    @property
    def full_name(self) -> str:
        """
        Get the full name.
        """
        parts = []
        if self.empty:
            parts.append('empty')

        if self.is_backed:
            parts.append('backed')

        if len(parts):
            parts[0] = parts[0].capitalize()

        parts += ['TemporalDataFrame', self.name]

        return ' '.join(parts)

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
        if self.has_locked_indices:
            raise VLockError("Cannot set index in tdf with locked index.")

        self._check_valid_index(values, self.has_repeating_index)

        self.index[:] = values

    @property
    def columns(self) -> NDArray_IFS:
        """
        Get the list of all column names.
        """
        return super().columns

    @columns.setter
    def columns(self,
                values: NDArrayLike_IFS) -> None:
        """
        Set the list of all column names.
        """
        if self.has_locked_columns:
            raise VLockError("Cannot set columns in tdf with locked columns.")

        if not (vs := len(values)) == (s := self.n_columns_num + self.n_columns_str):
            raise ValueError(f"Shape mismatch, new 'columns_num' values have shape {vs}, expected {s}.")

        self.columns_num[:] = values[:self.n_columns_num]
        self.columns_str[:] = values[self.n_columns_num:]

    # endregion

    # region predicates
    @property
    def is_view(self) -> Literal[False]:
        """
        Is this a view on a TemporalDataFrame ?
        """
        return False

    # endregion

    # region methods    
    def lock_indices(self) -> None:
        """Lock the "index" axis to prevent modifications."""
        self._attr_dict['locked_indices'] = True

    def unlock_indices(self) -> None:
        """Unlock the "index" axis to allow modifications."""
        self._attr_dict['locked_indices'] = False

    def lock_columns(self) -> None:
        """Lock the "columns" axis to prevent modifications."""
        self._attr_dict['locked_columns'] = True

    def unlock_columns(self) -> None:
        """Unlock the "columns" axis to allow modifications."""
        self._attr_dict['locked_columns'] = False

    def _check_valid_index(self,
                           values: NDArray_IFS,
                           repeating_index: bool) -> None:
        if not values.shape == self._index.shape:
            raise ValueError(f"Shape mismatch, new 'index' values have shape {values.shape}, "
                             f"expected {self._index.shape}.")

        if repeating_index:
            first_index = values[self.timepoints_column == self.timepoints[0]]

            for timepoint in self.timepoints[1:]:
                index_tp = values[self.timepoints_column == timepoint]

                if not len(first_index) == len(index_tp) or not np.all(first_index == index_tp):
                    raise ValueError(f"Index at time-point {timepoint} is not equal to index at time-point "
                                     f"{self.timepoints[0]}.")

        elif not self.n_index == len(np.unique(values)):
            raise ValueError("Index values must be all unique.")

    def set_index(self,
                  values: NDArray_IFS,
                  repeating_index: bool = False) -> None:
        """Set new index values."""
        if self.has_locked_indices:
            raise VLockError("Cannot set index in TemporalDataFrame with locked index.")

        self._check_valid_index(values, repeating_index)

        if values.dtype == self._index.dtype:
            self._index[:] = values

        elif self._data is not None:
            self._data['index'] = values
            
        else:
            self._index = values
            
        self._attr_dict['repeating_index'] = repeating_index

    def _get_index_positions(self,
                             index_: AnyNDArrayLike_IFS,
                             repeating_values: bool = False) -> NDArray_IFS:
        if not self.has_repeating_index:
            return cast(npt.NDArray[np.int_], 
                        npi.indices(self.index, index_))
            
        if repeating_values:
            index_offset = 0
            index_0 = self.index_at(self.tp0)
            index_positions = np.zeros(len(index_), dtype=int)
            first_positions = npi.indices(index_0, index_[:len(index_0)])

            for tpi in range(self.n_timepoints):
                index_positions[tpi * len(index_0):(tpi + 1) * len(index_0)] = first_positions + index_offset
                index_offset += len(index_0)

            return index_positions

        index_len_count = 0
        total_index = np.zeros((self.n_timepoints, len(index_)), dtype=int)

        for tpi, timepoint in enumerate(self.timepoints):
            i_at_tp = self.index_at(timepoint)
            total_index[tpi] = npi.indices(i_at_tp, index_) + index_len_count
            index_len_count += len(i_at_tp)

        return total_index.flatten()

    def reindex(self,
                order: NDArray_IFS,
                repeating_index: bool = False) -> None:
        """Re-order rows in this TemporalDataFrame so that their index matches the new given order."""
        # check all values in index
        if not np.all(np.isin(order, self.index)):
            raise ValueError("New index contains values which are not in the current index.")

        if repeating_index and not self.has_repeating_index:
            raise ValueError("Cannot set repeating index on tdf with non-repeating index.")

        elif not repeating_index and self.has_repeating_index:
            raise ValueError("Cannot set non-repeating index on tdf with repeating index.")

        # re-arrange rows to conform to new index
        index_positions = self._get_index_positions(order, repeating_values=True)

        self.set_index(order, repeating_index)
        self.values_num[:] = self.values_num[index_positions]
        self.values_str[:] = self.values_str[index_positions]

    def _check_before_insert(self,
                             name: IFS,
                             values: NDArray_IFS | Iterable[IFS] | IFS) -> NDArray_IFS:
        if self.has_locked_columns:
            raise VLockError("Cannot insert columns in tdf with locked columns.")

        if not isCollection(values):
            values = np.repeat(values, self.n_index)                # type: ignore[arg-type]

        values = np.array(values)

        if len(values) != self.n_index:
            raise ValueError(f"Wrong number of values ({len(values)}), expected {self.n_index}.")

        if name in self.columns:
            raise ValueError(f"A column named '{name}' already exists.")
        
        return values

    def insert(self,
               loc: int,
               name: IFS,
               values: NDArray_IFS | Iterable[IFS] | IFS) -> None:
        """
        Insert a column in either the numerical data or the string data, depending on the type of the <values> array.
            The column is inserted at position <loc> with name <name>.
        """
        values = self._check_before_insert(name, values)
        
        if np.issubdtype(values.dtype, (np.int_, np.float_)):
            # create numerical column
            if loc < 0:
                loc += self.n_columns_num + 1

            self._numerical_array = np.insert(self._numerical_array, loc, values, axis=1)
            self._columns_numerical = np.insert(self._columns_numerical, loc, name)  
            
        else:
            # create string column
            if loc < 0:
                loc += self.n_columns_str + 1

            self._string_array = np.insert(self._string_array, loc, values, axis=1)
            self._columns_string = np.insert(self._columns_string, loc, name)    

    def close(self) -> None:
        """Close the file this TemporalDataFrame is backed on."""
        if self._data is not None:
            self._data.close()

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
        if not np.all(self.timepoints == other.timepoints):
            raise ValueError("Cannot merge TemporalDataFrames with different time points.")

        if not np.all(self.columns_num == other.columns_num) \
            and not np.all(self.columns_str == other.columns_str):
            raise ValueError("Cannot merge TemporalDataFrames with different columns.")

        if not self.timepoints_column_name == other.timepoints_column_name:
            raise ValueError("Cannot merge TemporalDataFrames with different 'timepoints_column_name'.")

        if self.has_repeating_index is not other.has_repeating_index:
            raise ValueError('Cannot merge TemporalDataFrames if one has repeating index while the other has not.')

        if self.empty:
            combined_index = np.array([])
            for timepoint in self.timepoints:
                combined_index = np.concatenate((combined_index,
                                                 self.index_at(timepoint),
                                                 other.index_at(timepoint)))

            _data = pd.DataFrame(index=combined_index, columns=self.columns)

        else:
            _check_merge_index(self, other, self.tp0)
            _data = pd.concat((self[self.tp0].to_pandas(), 
                               other[self.tp0].to_pandas()))

            for time_point in self.timepoints[1:]:
                _check_merge_index(self, other, time_point)
                _data = pd.concat((_data,
                                   self[time_point].to_pandas(), 
                                   other[time_point].to_pandas()))

            _data.columns = _data.columns.astype(self.columns.dtype)

        if self.timepoints_column_name is None:
            _time_list = [time_point for time_point in self.timepoints
                          for _ in range(self.n_index_at(time_point) + other.n_index_at(time_point))]

        else:
            _time_list = None

        return TemporalDataFrame(data=_data,
                                 repeating_index=self.has_repeating_index,
                                 columns=self.columns,
                                 time_list=_time_list,
                                 time_col_name=self.timepoints_column_name,
                                 name=name or f"{self.name} + {other.name}")

    # endregion


def _get_valid_mode(file: str | Path | ch.Group | ch.H5Dict[Any],
                mode: Literal[ch.H5Mode.READ, ch.H5Mode.READ_WRITE] | None) \
        -> Literal[ch.H5Mode.READ, ch.H5Mode.READ_WRITE]:
    if mode is None:
        if isinstance(file, (str, Path)):
            return ch.H5Mode.READ
            
        return file.file.file.mode

    elif mode not in (ch.H5Mode.READ, ch.H5Mode.READ_WRITE):
        raise ValueError("Only 'r' and 'r+' are valid modes.")
    
    return mode


def _check_merge_index(tdf1: TemporalDataFrameBase, tdf2: TemporalDataFrameBase, timepoint: tp.TimePoint) -> None:
    if np.any(np.isin(tdf1.index_at(timepoint), tdf2.index_at(timepoint))):
        raise ValueError(f"TemporalDataFrames to merge have index values in common at time point '{timepoint}'.")
