from __future__ import annotations

import ch5mpy as ch
import numpy as np
import numpy.typing as npt

import vdata
from vdata._typing import IFS_NP
from vdata.data.arrays.base import VBase3DArrayContainer, VTDFArrayContainerView
from vdata.data.file import NoFile
from vdata.IO import IncoherenceError, ShapeError, generalLogger
from vdata.tdf import TemporalDataFrame, TemporalDataFrameBase
from vdata.timepoint import TimePointArray


class VLayersArrayContainer(VBase3DArrayContainer):
    """
    Class for layers.
    This object contains any number of TemporalDataFrames, with shapes (n_timepoints, n_obs, n_var).
    The arrays-like objects can be accessed from the parent VData object by :
        VData.layers[<array_name>]
    """

    # region magic methods
    def __init__(self, 
                 parent: vdata.VData,
                 data: dict[str, TemporalDataFrame] | ch.H5Dict[TemporalDataFrame]):
        """
        Args:
            parent: the parent VData object this VLayerArrayContainer is linked to.
            data: a dictionary of TemporalDataFrames in this VLayerArrayContainer.
        """
        super().__init__(parent, data)

    def _check_init_data(self,
                         data: dict[str, TemporalDataFrame] | ch.H5Dict[TemporalDataFrame]) \
        -> dict[str, TemporalDataFrame] | ch.H5Dict[TemporalDataFrame]:
        """
        Function for checking, at VLayerArrayContainer creation, that the supplied data has the correct format :
            - the shape of the TemporalDataFrames in 'data' match the parent VData object's shape.
            - the index of the TemporalDataFrames in 'data' match the index of the parent VData's obs TemporalDataFrame.
            - the column names of the TemporalDataFrames in 'data' match the index of the parent VData's var DataFrame.
            - the time points of the TemporalDataFrames in 'data' match the index of the parent VData's time-points
            DataFrame.

        Args:
            data: optional dictionary of TemporalDataFrames.

        Returns:
            The data (dictionary of TemporalDataFrames), if correct.
        """
        if not len(data):
            generalLogger.debug("  No data was given.")
            return {}

        generalLogger.debug("  Data was found.")

        _shape = (self._parent.timepoints.shape[0], self._parent.obs.shape[1], self._parent.var.shape[0])
        generalLogger.debug(f"  Reference shape is {_shape}.")

        _index = self._parent.obs.index
        _columns = self._parent.var.index
        _timepoints: npt.NDArray[IFS_NP] = self._parent.timepoints.value.values

        _data = {} if isinstance(data, dict) else data

        for TDF_index, tdf in data.items():
            TDF_shape = tdf.shape

            generalLogger.debug(f"  Checking TemporalDataFrame '{TDF_index}' with shape {TDF_shape}.")

            # check that shapes match
            if _shape != TDF_shape:
                if _shape[0] != TDF_shape[0]:
                    raise IncoherenceError(f"Layer '{TDF_index}' has {TDF_shape[0]} "
                                           f"time point{'s' if TDF_shape[0] > 1 else ''}, "
                                           f"should have {_shape[0]}.")

                elif _shape[1] != TDF_shape[1]:
                    for i in range(len(tdf.timepoints)):
                        if _shape[1][i] != TDF_shape[1][i]:
                            raise IncoherenceError(f"Layer '{TDF_index}' at time point {i} has"
                                                   f" {TDF_shape[1][i]} observations, "
                                                   f"should have {_shape[1][i]}.")

                raise IncoherenceError(f"Layer '{TDF_index}' has  {TDF_shape[2]} variables, "
                                       f"should have {_shape[2]}.")

            # check that indexes match
            if np.any(_index != tdf.index):
                raise IncoherenceError(f"Index of layer '{TDF_index}' ({tdf.index}) does not match obs' index. ("
                                       f"{_index})")

            if np.any(_columns != tdf.columns):
                raise IncoherenceError(f"Column names of layer '{TDF_index}' ({tdf.columns}) do not match var's "
                                       f"index. ({_columns})")

            if np.any(_timepoints != tdf.timepoints):
                raise IncoherenceError(f"Time points of layer '{TDF_index}' ({tdf.timepoints}) do not match "
                                       f"time_point's index. ({_timepoints})")

            # checks passed, store the TemporalDataFrame
            if tdf.file is None or tdf.file.mode == ch.H5Mode.READ_WRITE:
                tdf.lock_indices()
                tdf.lock_columns()

            assert tdf.has_locked_indices and tdf.has_locked_columns

            if isinstance(data, dict):
                _data[str(TDF_index)] = tdf

        generalLogger.debug("  Data was OK.")
        return _data

    def __setitem__(self, 
                    key: str,
                    value: TemporalDataFrameBase) -> None:
        """
        Set a specific TemporalDataFrame in _data. The given TemporalDataFrame must have the correct shape.

        Args:
            key: key for storing a TemporalDataFrame in this VObsmArrayContainer.
            value: a TemporalDataFrame to store.
        """
        if not isinstance(value, TemporalDataFrame):
            raise TypeError(f"Cannot set {self.name} '{key}' from non TemporalDataFrame object.")

        if not self.one_shape == value.shape:
            raise ShapeError(f"Cannot set {self.name} '{key}' because of shape mismatch.")

        if not np.array_equal(self._parent.var.index, value.columns):
            raise ValueError("Column names do not match.")

        value_copy = value.copy()
        value_copy.name = key
        value_copy.lock_indices()
        value_copy.lock_columns()
        
        if self._parent.file is NoFile._:
            return super().__setitem__(key, value_copy)

        if key not in self._parent.file['layers'].keys():
            self._parent.file['layers'].create_group(key)
            
        value_copy.write(self._parent.file['layers'][key])
        super().__setitem__(key, value_copy)
        
    # endregion
    
    # region methods
    @property
    def one_shape(self) -> tuple[int, list[int], int]:
        """Shape of one layer."""
        _shape = self.shape
        return _shape[1], _shape[2], _shape[3][0]
    
    # endregion


class VLayersArrayContainerView(VTDFArrayContainerView):
    """View on a layer container."""

    # region magic methods
    def __init__(self,
                 array_container: VLayersArrayContainer,
                 timepoints_slicer: TimePointArray,
                 obs_slicer: npt.NDArray[IFS_NP],
                 var_slicer: npt.NDArray[IFS_NP] | slice):
        """
        Args:
            array_container: a VLayerArrayContainer object to build a view on.
            obs_slicer: the list of observations to view.
            var_slicer: the list of variables to view.
            timepoints_slicer: the list of time points to view.
        """
        super().__init__(array_container, timepoints_slicer, obs_slicer, var_slicer)

        self._parent_var_hash: int = hash(tuple(self._array_container._parent.var.index))
        
    # endregion
