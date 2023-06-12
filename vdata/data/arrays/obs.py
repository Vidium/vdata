from __future__ import annotations

from typing import Any, Collection

import numpy as np
import pandas as pd

from vdata._typing import IFS, DictLike, NDArray_IFS
from vdata.data.arrays.base import VBaseArrayContainer, VTDFArrayContainer
from vdata.data.arrays.view import VBaseArrayContainerView, VTDFArrayContainerView
from vdata.IO import IncoherenceError, ShapeError, generalLogger
from vdata.tdf import TemporalDataFrame, TemporalDataFrameBase
from vdata.timedict import TimeDict
from vdata.vdataframe import VDataFrame


class VObsmArrayContainer(VTDFArrayContainer):
    """
    Class for obsm.
    This object contains any number of TemporalDataFrames, with shape (n_timepoints, n_obs, any).
    The TemporalDataFrames can be accessed from the parent VData object by :
        VData.obsm[<array_name>])
    """

    # region magic methods
    def _check_init_data(self, data: DictLike[TemporalDataFrame]) -> DictLike[TemporalDataFrame]:
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
        if not len(data):
            generalLogger.debug("  No data was given.")
            return dict()

        generalLogger.debug("  Data was found.")
        
        _shape = (self._vdata.timepoints.shape[0], self._vdata.obs.shape[1], 'Any')
        _data: DictLike[TemporalDataFrame] = {} if isinstance(data, dict) else data

        generalLogger.debug(f"  Reference shape is {_shape}.")

        for TDF_index, tdf in data.items():
            TDF_shape = tdf.shape

            generalLogger.debug(f"  Checking TemporalDataFrame '{TDF_index}' with shape {TDF_shape}.")

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

            # check that indexes match
            if np.any(self._vdata.obs.index != tdf.index):
                raise IncoherenceError(f"Index of TemporalDataFrame '{TDF_index}' ({tdf.index}) does not match "
                                        f"obs' index. ({self._vdata.obs.index})")

            if np.any(self._vdata.timepoints.value.values != tdf.timepoints):
                raise IncoherenceError(f"Time points of TemporalDataFrame '{TDF_index}' ({tdf.timepoints}) "
                                        f"do not match time_point's index. ({self._vdata.timepoints.value.values})")

            tdf.lock_indices()
            
            if isinstance(data, dict):
                _data[str(TDF_index)] = tdf
                
        generalLogger.debug("  Data was OK.")
        return _data

    def __setitem__(self, key: str, value: TemporalDataFrameBase) -> None:
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

    # endregion
    
    # region magic methods
    def set_index(self, values: Collection[IFS], repeating_index: bool) -> None:
        for TDF in self.values():
            TDF.unlock_indices()
            TDF.set_index(np.array(values), repeating_index)
            TDF.lock_indices()
    
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
    def _check_init_data(self, data: DictLike[VDataFrame]) -> TimeDict:
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
        if not len(data):
            generalLogger.debug("  No data was given.")
            return TimeDict(vdata=self._vdata)

        generalLogger.debug("  Data was found.")
        _data: TimeDict = TimeDict(vdata=self._vdata)

        for key, df in data.items():
            generalLogger.debug(f"  Checking DataFrame at key '{key}' with shape {df.shape}.")

            _index = self._vdata.obs.index
            # file = self._file[key] if self._file is not None else None

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
            _data[str(key)] = df

        generalLogger.debug("  Data was OK.")
        return _data

    def __getitem__(self, key: str) -> VDataFrame:
        """
        Get a specific set VDataFrame stored in this VObspArrayContainer.

        Args:
            item: key in _data linked to a set of DataFrames.

        Returns:
            A VDataFrame stored in _data under the given key.
        """
        if key not in self.keys():
            raise AttributeError(f"{self.name} ArrayContainer has no attribute '{key}'")

        return self.data[key]

    def __setitem__(self, key: str, value: VDataFrame | pd.DataFrame) -> None:
        """
        Set a specific DataFrame in _data. The given DataFrame must have the correct shape.

        Args:
            key: key for storing a set of DataFrames in this VObspArrayContainer.
            value: a set of DataFrames to store.
        """
        if not isinstance(value, VDataFrame):
            value = VDataFrame(value)

        _index = self._vdata.obs.index

        if not value.shape == (len(_index), len(_index)):
            raise ShapeError(f"DataFrame should have shape ({len(_index)}, {len(_index)}).")

        if not np.all(value.index == _index):
            raise ValueError("The index of the DataFrame does not match the index of the parent VData.")

        if not np.all(value.columns == _index):
            raise ValueError("The column names the DataFrame do not match the index of the parent VData.")

        self.data[key] = value

    # endregion
    
    # region attributes
    @property
    def shape(self) -> tuple[int, int, int]:
        """
        The shape of the VObspArrayContainer is computed from the shape of the Arrays it contains.
        See __len__ for getting the number of Arrays it contains.

        Returns:
            The shape of this VObspArrayContainer.
        """
        len_index = self._vdata.n_obs_total

        if not len(self):
            return 0, len_index, len_index
            
        return len(self), len_index, len_index

    # endregion
    
    # region methods
    def set_index(self, values: Collection[IFS]) -> None:
        """
        Set a new index for rows and columns.

        Args:
            values: collection of new index values.
        """
        for vdf in self.values():
            vdf.index = pd.Index(values)
            vdf.columns = pd.Index(values)

    # endregion
    

class VObspArrayContainerView(VBaseArrayContainerView[VDataFrame, VDataFrame]):
    """
    Class for views of obsp.
    """

    # region magic methods
    def __init__(self,
                 array_container: VObspArrayContainer,
                 obs_slicer: NDArray_IFS):
        """
        Args:
            array_container: a VBaseArrayContainer object to build a view on.
            obs_slicer: the list of observations to view.
        """
        super().__init__(data={key: VDataFrame(vdf.loc[obs_slicer, obs_slicer])
                               for key, vdf in array_container.items()},
                         array_container=array_container)
        
        self._obs_slicer = obs_slicer

    def __setitem__(self, key: str, value: VDataFrame | pd.DataFrame) -> None:
        """
        Set a specific data item in this view. The given data item must have the correct shape.

        Args:
            key: key for storing a data item in this view.
            value: a data item to store.
        """
        if not isinstance(value, VDataFrame):
            value = VDataFrame(value)
            
        _index = self._array_container[key].loc[self._obs_slicer, self._obs_slicer].index

        if not value.shape == (len(_index), len(_index)):
            raise ShapeError(f"DataFrame at key '{key}' should have shape ({len(_index)}, {len(_index)}).")

        if not value.index.equals(_index):
            raise ValueError(f"Index of DataFrame at key '{key}' does not match previous index.")

        if not value.columns.equals(_index):
            raise ValueError(f"Column names of DataFrame at key '{key}' do not match previous names.")

        self.data[key] = value

    # endregion
    
    # region attributes
    @property
    def shape(self) -> tuple[int, int, int]:
        """
        The shape of this view is computed from the shape of the Arrays it contains.
        See __len__ for getting the number of Arrays it contains.

        Returns:
            The shape of this view.
        """
        return len(self), len(self._obs_slicer), len(self._obs_slicer)

    # endregion
    
    # region methods
    def set_index(self, values: Collection[Any]) -> None:
        """
        Set a new index for rows and columns.

        Args:
            values: collection of new index values.
        """
        for vdf in self.values():
            vdf.index = pd.Index(values)
            vdf.columns = pd.Index(values)

    # endregion
    