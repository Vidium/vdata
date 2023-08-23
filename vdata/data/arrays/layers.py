from __future__ import annotations

from typing import Union, cast

import ch5mpy as ch
import numpy as np

from vdata._typing import AnyDictLike, DictLike
from vdata.data.arrays.base import VTDFArrayContainer
from vdata.data.arrays.view import VTDFArrayContainerView
from vdata.IO import IncoherenceError, ShapeError, generalLogger
from vdata.tdf import TemporalDataFrame, TemporalDataFrameBase, TemporalDataFrameView


class VLayersArrayContainer(VTDFArrayContainer):
    """
    Class for layers.
    This object contains any number of TemporalDataFrames, with shapes (n_timepoints, n_obs, n_var).
    The arrays-like objects can be accessed from the parent VData object by :
        VData.layers[<array_name>]
    """

    # region magic methods
    def _check_init_data(
        self, data: AnyDictLike[TemporalDataFrame | TemporalDataFrameView]
    ) -> DictLike[TemporalDataFrame | TemporalDataFrameView]:
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
            return data if isinstance(data, ch.H5Dict) else {}

        generalLogger.debug("  Data was found.")

        _shape = (self._vdata.timepoints.shape[0], self._vdata.obs.shape[1], self._vdata.var.shape[0])
        _data: DictLike[TemporalDataFrame | TemporalDataFrameView] = {} if not isinstance(data, ch.H5Dict) else data

        generalLogger.debug(lambda: f"  Reference shape is {_shape}.")

        for TDF_index, tdf in data.items():
            tdf = cast(Union[TemporalDataFrame, TemporalDataFrameView], tdf)
            TDF_shape = tdf.shape

            generalLogger.debug(lambda: f"  Checking TemporalDataFrame '{TDF_index}' with shape {TDF_shape}.")

            # check that shapes match
            if _shape[0] != TDF_shape[0]:
                raise IncoherenceError(
                    f"Layer '{TDF_index}' has {TDF_shape[0]} "
                    f"time point{'s' if TDF_shape[0] > 1 else ''}, "
                    f"should have {_shape[0]}."
                )

            elif _shape[1] != TDF_shape[1]:
                for i in range(len(tdf.timepoints)):
                    if _shape[1][i] != TDF_shape[1][i]:
                        raise IncoherenceError(
                            f"Layer '{TDF_index}' at time point {i} has"
                            f" {TDF_shape[1][i]} observations, "
                            f"should have {_shape[1][i]}."
                        )

            elif _shape[2] != TDF_shape[2]:
                raise IncoherenceError(
                    f"Layer '{TDF_index}' has  {TDF_shape[2]} variables, " f"should have {_shape[2]}."
                )

            # check that indexes match
            if np.any(self._vdata.obs.index != tdf.index):
                raise IncoherenceError(
                    f"Index of layer '{TDF_index}' ({tdf.index}) does not match obs' index. ("
                    f"{self._vdata.obs.index})"
                )

            if np.any(self._vdata.var.index != tdf.columns):
                raise IncoherenceError(
                    f"Column names of layer '{TDF_index}' ({tdf.columns}) do not match var's "
                    f"index. ({self._vdata.var.index})"
                )

            if np.any(self._vdata.timepoints.value.values != tdf.timepoints):
                raise IncoherenceError(
                    f"Time points of layer '{TDF_index}' ({tdf.timepoints}) do not match "
                    f"time_point's index. ({self._vdata.timepoints.value.values})"
                )

            if tdf.data is None or ch.H5Mode.has_write_intent(tdf.data.mode):
                tdf.lock_indices()
                tdf.lock_columns()

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

        if not self.one_shape == value.shape:
            raise ShapeError(f"Cannot set {self.name} '{key}' because of shape mismatch.")

        if not np.array_equal(self._vdata.var.index, value.columns):
            raise ValueError("Column names do not match.")

        value_copy = value.copy()
        value_copy.name = key
        value_copy.lock_indices()
        value_copy.lock_columns()

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
