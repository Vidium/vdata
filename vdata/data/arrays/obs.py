from __future__ import annotations

from pathlib import Path
from typing import Any, Collection

import ch5mpy as ch
import numpy as np
import numpy.typing as npt
import pandas as pd

import vdata
from vdata._typing import IFS_NP
from vdata.data.arrays.base import (
    TimedDict,
    VBase3DArrayContainer,
    VBaseArrayContainer,
    VBaseArrayContainerView,
    VTDFArrayContainerView,
)
from vdata.IO import (
    IncoherenceError,
    ShapeError,
    VClosedFileError,
    VReadOnlyError,
    generalLogger,
)
from vdata.tdf import TemporalDataFrame
from vdata.vdataframe import VDataFrame


class VObsmArrayContainer(VBase3DArrayContainer):
    """
    Class for obsm.
    This object contains any number of TemporalDataFrames, with shape (n_timepoints, n_obs, any).
    The TemporalDataFrames can be accessed from the parent VData object by :
        VData.obsm[<array_name>])
    """

    # region magic methods
    def __init__(self, 
                 parent: vdata.VData, 
                 data: dict[str, TemporalDataFrame] | None = None):
        """
        Args:
            parent: the parent VData object this VObsmArrayContainer is linked to.
            data: a dictionary of TemporalDataFrames in this VObsmArrayContainer.
        """
        super().__init__(parent, data)
        self._file = parent.file

    def _check_init_data(self, data: dict[str, TemporalDataFrame] | None) -> dict[str, TemporalDataFrame]:
        """
        Function for checking, at VObsmArrayContainer creation, that the supplied data has the correct format :
            - the shape of the TemporalDataFrames in 'data' match the parent VData object's shape (except for the
            number of columns).
            - the index of the TemporalDataFrames in 'data' match the index of the parent VData's obs TemporalDataFrame.
            - the time points of the TemporalDataFrames in 'data' match the index of the parent VData's timepoints
            DataFrame.

        Args:
            data: optional dictionary of TemporalDataFrames.

        Returns:
            The data (dictionary of TemporalDataFrames), if correct.
        """
        if data is None or not len(data):
            generalLogger.debug("  No data was given.")
            return dict()

        else:
            generalLogger.debug("  Data was found.")
            _data = {}
            _shape = (self._parent.timepoints.shape[0],
                      self._parent.obs.shape[1],
                      'Any')
            _index = self._parent.obs.index
            _timepoints = self._parent.timepoints.value.values

            generalLogger.debug(f"  Reference shape is {_shape}.")

            for TDF_index, tdf in data.items():
                TDF_shape = tdf.shape

                generalLogger.debug(f"  Checking TemporalDataFrame '{TDF_index}' with shape {TDF_shape}.")

                if _shape != TDF_shape:

                    # check that shapes match
                    if _shape[0] != TDF_shape[0]:
                        raise IncoherenceError(f"TemporalDataFrame '{TDF_index}' has {TDF_shape[0]} "
                                               f"time point{'s' if TDF_shape[0] > 1 else ''}, "
                                               f"should have {_shape[0]}.")

                    elif _shape[1] != TDF_shape[1]:
                        for i in range(len(tdf.timepoints)):
                            if _shape[1][i] != TDF_shape[1][i]:
                                raise IncoherenceError(f"TemporalDataFrame '{TDF_index}' at time point {i} has"
                                                       f" {TDF_shape[1][i]} rows, "
                                                       f"should have {_shape[1][i]}.")

                    else:
                        pass

                # check that indexes match
                if np.any(_index != tdf.index):
                    raise IncoherenceError(f"Index of TemporalDataFrame '{TDF_index}' ({tdf.index}) does not match "
                                           f"obs' index. ({_index})")

                if np.any(_timepoints != tdf.timepoints):
                    raise IncoherenceError(f"Time points of TemporalDataFrame '{TDF_index}' ({tdf.timepoints}) "
                                           f"do not match time_point's index. ({_timepoints})")

                # checks passed, store the TemporalDataFrame
                if not tdf.has_locked_indices:
                    tdf.lock_indices()
                _data[str(TDF_index)] = tdf

            generalLogger.debug("  Data was OK.")
            return _data

    def __setitem__(self, key: str, value: TemporalDataFrame) -> None:
        """
        Set a specific TemporalDataFrame in _data. The given TemporalDataFrame must have the correct shape.

        Args:
            key: key for storing a TemporalDataFrame in this VObsmArrayContainer.
            value: a TemporalDataFrame to store.
        """
        if not isinstance(value, TemporalDataFrame):
            raise TypeError(f"Cannot set {self.name} '{key}' from non TemporalDataFrame object.")

        if not self.empty and not self.shape[1:3] == value.shape[:2]:
            raise ShapeError(f"Cannot set {self.name} '{key}' because of shape mismatch.")

        value_copy = value.copy()
        value_copy.name = key

        value_copy.lock_indices()
        super().__setitem__(key, value_copy)

        if isinstance(self._file, ch.Group) and self._file.mode == ch.H5Mode.READ_WRITE:
            if key not in self._file['obsm'].keys():
                self._file['obsm'].create_group(key)

            value_copy.write(self._file['obsm'][key])

    # endregion


class VObsmArrayContainerView(VTDFArrayContainerView):
    """Class for views on obsm."""


class VObspArrayContainer(VBaseArrayContainer[VDataFrame]):
    """
    Class for obsp.
    This object contains sets of <nb time points> 2D square DataFrames of shapes (<n_obs>, <n_obs>) for each time point.
    The DataFrames can be accessed from the parent VData object by :
        VData.obsp[<array_name>][<time point>]
    """

    # region magic methods
    def __init__(self,
                 parent: vdata.VData,
                 data: dict[str, pd.DataFrame] | None,
                 file: ch.File | ch.Group | None = None):
        """
        Args:
            parent: the parent VData object this VObspArrayContainer is linked to.
            data: a dictionary of array-like objects to store in this VObspArrayContainer.
        """
        self._file = file

        super().__init__(parent, data)

    def _check_init_data(self,
                         data: dict[str, pd.DataFrame] | None) -> TimedDict:
        """
        Function for checking, at VObspArrayContainer creation, that the supplied data has the correct format :
            - the shape of the DataFrames in 'data' match the parent VData object's index length.
            - the index and columns names of the DataFrames in 'data' match the index of the parent VData's obs
            TemporalDataFrame.
            - the time points of the dictionaries of DataFrames in 'data' match the index of the parent VData's
            time-points DataFrame.

        Args:
            data: dictionary of dictionaries (TimePoint: DataFrame (n_obs x n_obs))

        Returns:
            The data (dictionary of dictionaries of DataFrames), if correct.
        """
        if data is None or not len(data):
            generalLogger.debug("  No data was given.")
            return TimedDict(parent=self._parent)

        else:
            generalLogger.debug("  Data was found.")
            _data = TimedDict(parent=self._parent)

            for key, df in data.items():
                generalLogger.debug(f"  Checking DataFrame at key '{key}' with shape {df.shape}.")

                _index = self._parent.obs.index
                file = self._file[key] if self._file is not None else None

                # check that square
                if df.shape[0] != df.shape[1]:
                    raise ShapeError(f"DataFrame at key '{key}' should be square.")

                # check that indexes match
                if not np.all(_index == df.index):
                    raise IncoherenceError(f"Index of DataFrame at key '{key}' ({df.index}) does not "
                                           f"match obs' index. ({_index})")

                if not np.all(_index == df.columns):
                    raise IncoherenceError(f"Column names of DataFrame at key '{key}' ({df.columns}) "
                                           f"do not match obs' index. ({_index})")

                # checks passed, store as VDataFrame
                _data[key] = VDataFrame(df, file=file)

            generalLogger.debug("  Data was OK.")
            return _data

    def __getitem__(self,
                    item: str) -> VDataFrame:
        """
        Get a specific set VDataFrame stored in this VObspArrayContainer.

        Args:
            item: key in _data linked to a set of DataFrames.

        Returns:
            A VDataFrame stored in _data under the given key.
        """
        if self.is_closed:
            raise VClosedFileError

        if not len(self) or item not in self.keys():
            raise AttributeError(f"{self.name} ArrayContainer has no attribute '{item}'")

        # FIXME
        return self._data[(slice(None), item)]

    def __setitem__(self,
                    key: str,
                    value: VDataFrame | pd.DataFrame) -> None:
        """
        Set a specific DataFrame in _data. The given DataFrame must have the correct shape.

        Args:
            key: key for storing a set of DataFrames in this VObspArrayContainer.
            value: a set of DataFrames to store.
        """
        if self.is_closed:
            raise VClosedFileError

        if self.is_read_only:
            raise VReadOnlyError

        if not isinstance(value, (pd.DataFrame, VDataFrame)):
            raise TypeError("The value should be a pandas DataFrame or a VDataFrame.")

        if isinstance(value, pd.DataFrame):
            value = VDataFrame(value)

        _index = self._parent.obs.index

        if not value.shape == (len(_index), len(_index)):
            raise ShapeError(f"DataFrame should have shape ({len(_index)}, {len(_index)}).")

        if not np.all(value.index == _index):
            raise ValueError("The index of the DataFrame does not match the index of the parent VData.")

        if not np.all(value.columns == _index):
            raise ValueError("The column names the DataFrame do not match the index of the parent VData.")

        self._data[key] = value

    # endregion
    
    # region attributes
    @property
    def data(self) -> dict[str, VDataFrame]:
        """
        Data of this VObspArrayContainer.

        Returns:
            The data of this VObspArrayContainer.
        """
        if self.is_closed:
            raise VClosedFileError

        return self._data

    @property
    def empty(self) -> bool:
        """
        Whether this VObspArrayContainer is empty or not.

        Returns:
            Is this VObspArrayContainer empty ?
        """
        if not len(self) or all([vdf.empty for vdf in self.data.values()]):
            return True
        return False

    def update_dtype(self,
                     type_: npt.DTypeLike) -> None:
        """
        Update the data type of VDataFrames stored in this VObspArrayContainer.

        Args:
            type_: the new data type.
        """
        for vdf_name in self.keys():
            self[vdf_name] = self[vdf_name].astype(type_)

    @property
    def shape(self) -> tuple[int, int, int]:
        """
        The shape of the VObspArrayContainer is computed from the shape of the Arrays it contains.
        See __len__ for getting the number of Arrays it contains.

        Returns:
            The shape of this VObspArrayContainer.
        """
        len_index = self._parent.n_obs_total

        if len(self):
            return len(self), len_index, len_index

        else:
            return 0, len_index, len_index

    # endregion
    
    # region methods
    def dict_copy(self) -> dict[str, VDataFrame]:
        """
        Dictionary of keys and copied data items in this ArrayContainer.

        Returns:
            A dictionary copy of this ArrayContainer.
        """
        return {key: vdf.copy() for key, vdf in self.items()}

    def to_csv(self,
               directory: Path,
               sep: str = ",",
               na_rep: str = "",
               index: bool = True,
               header: bool = True,
               spacer: str = '') -> None:
        """
        Save this VObspArrayContainer in CSV file format.

        Args:
            directory: path to a directory for saving the Array
            sep: delimiter character
            na_rep: string to replace NAs
            index: write row names ?
            header: Write col names ?
            spacer: for logging purposes, the recursion depth of calls to a read_h5 function.
        """
        if self.is_closed:
            raise VClosedFileError

        # create subdirectory for storing sets
        (directory / self.name).mkdir(parents=True)

        for vdf_name, vdf in self.items():
            generalLogger.info(f"{spacer}Saving {vdf_name}")

            # save array
            vdf.to_csv(f"{directory / self.name / vdf_name}.csv", sep, na_rep, index=index, header=header)

    def set_index(self,
                  values: Collection[Any]) -> None:
        """
        Set a new index for rows and columns.

        Args:
            values: collection of new index values.
        """
        for vdf_name in self.keys():

            self[vdf_name].index = values
            self[vdf_name].columns = values

    def set_file(self,
                 file: ch.File | ch.Group) -> None:
        """
        Set the file to back the VDataFrames in this VObspArrayContainer.

        Args:
            file: a h5 file to back the VDataFrames on.
        """
        if not isinstance(file, (ch.File, ch.Group)):
            raise TypeError(f"Cannot back VDataFrames in this VObspArrayContainer with an object of type '"
                             f"{type(file)}'.")

        for vdf_name, vdf in self.items():
            vdf.file = file[vdf_name]

    # endregion
    

class VObspArrayContainerView(VBaseArrayContainerView[VDataFrame]):
    """
    Class for views of obsp.
    """

    # region magic methods
    def __init__(self,
                 array_container: VBaseArrayContainer[VDataFrame],
                 obs_slicer: npt.NDArray[IFS_NP]):
        """
        Args:
            array_container: a VBaseArrayContainer object to build a view on.
            obs_slicer: the list of observations to view.
        """
        super().__init__(array_container)

        self._obs_slicer = obs_slicer

    def __getitem__(self,
                    item: str) -> VDataFrame:
        """
        Get a specific data item stored in this view.

        Args:
            item: key in _data linked to a data item.

        Returns:
            Data item stored in _data under the given key.
        """
        return self._array_container[item].loc[self._obs_slicer, self._obs_slicer]

    def __setitem__(self, key: str, value: VDataFrame) -> None:
        """
        Set a specific data item in this view. The given data item must have the correct shape.

        Args:
            key: key for storing a data item in this view.
            value: a data item to store.
        """
        if not isinstance(value, (pd.DataFrame, VDataFrame)):
            raise TypeError(f"Value at key '{key}' should be a pandas DataFrame or a VDataFrame.")

        _index = self._array_container[key].loc[self._obs_slicer, self._obs_slicer].index

        if not value.shape == (len(_index), len(_index)):
            raise ShapeError(f"DataFrame at key '{key}' should have shape ({len(_index)}, {len(_index)}).")

        if not value.index.equals(_index):
            raise ValueError(f"Index of DataFrame at key '{key}' does not match previous index.")

        if not value.columns.equals(_index):
            raise ValueError(f"Column names of DataFrame at key '{key}' do not match previous names.")

        self._array_container[key].loc[self._obs_slicer, self._obs_slicer] = value

    # endregion
    
    # region attributes
    @property
    def empty(self) -> bool:
        """
        Whether this view is empty or not.

        Returns:
            Is this view empty ?
        """
        if not len(self) or all([vdf.empty for vdf in self.values()]):
            return True
        return False

    @property
    def shape(self) -> tuple[int, int, int]:
        """
        The shape of this view is computed from the shape of the Arrays it contains.
        See __len__ for getting the number of Arrays it contains.

        Returns:
            The shape of this view.
        """
        len_index = len(self._obs_slicer)

        if len(self):
            return len(self), len_index, len_index

        else:
            return 0, len_index, len_index

    @property
    def data(self) -> dict[str, VDataFrame]:
        """
        Data of this view.

        Returns:
            The data of this view.
        """
        return {key: vdf.loc[self._obs_slicer, self._obs_slicer] for key, vdf in self._array_container.items()}

    # endregion
    
    # region methods
    def dict_copy(self) -> dict[str, VDataFrame]:
        """
        Dictionary of keys and data items in this view.

        Returns:
            Dictionary of this view.
        """
        return {key: vdf.loc[self._obs_slicer, self._obs_slicer].copy() for key, vdf in self.items()}

    def to_csv(self,
               directory: Path,
               sep: str = ",",
               na_rep: str = "",
               index: bool = True,
               header: bool = True,
               spacer: str = '') -> None:
        """
        Save this view in CSV file format.

        Args:
            directory: path to a directory for saving the Array
            sep: delimiter character
            na_rep: string to replace NAs
            index: write row names ?
            header: Write col names ?
            spacer: for logging purposes, the recursion depth of calls to a read_h5 function.
        """
        # create sub directory for storing sets
        (directory / self.name).mkdir(parents=True)

        for vdf_name, vdf in self.data.items():
            generalLogger.info(f"{spacer}Saving {vdf_name}")

            # save array
            vdf.to_csv(f"{directory / self.name / vdf_name}.csv", sep, na_rep, index=index, header=header)

    def set_index(self, values: Collection[Any]) -> None:
        """
        Set a new index for rows and columns.

        Args:
            values: collection of new index values.
        """
        for vdf in self.data.values():
            vdf.lock = (False, False)
            vdf.index = values
            vdf.columns = values
            vdf.lock = (True, True)

    # endregion
    