from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Callable, Iterator, Literal, cast

import ch5mpy as ch
import numpy as np
import pandas as pd
from h5dataframe import H5DataFrame

import vdata
import vdata.timepoint as tp
from vdata._typing import DictLike, NDArray_IFS, PreSlicer
from vdata.data._file import NoData
from vdata.data._indexing import reformat_index
from vdata.data.arrays import (
    VLayersArrayContainerView,
    VObsmArrayContainerView,
    VObspArrayContainerView,
    VVarmArrayContainerView,
    VVarpArrayContainerView,
)
from vdata.data.hash import VDataHash
from vdata.data.write import write_vdata, write_vdata_in_h5dict, write_vdata_to_csv
from vdata.IO import IncoherenceError, ShapeError, generalLogger
from vdata.names import NO_NAME
from vdata.tdf import TemporalDataFrame, TemporalDataFrameBase, TemporalDataFrameView
from vdata.utils import repr_array, repr_index


def _check_parent_has_not_changed(func: Callable[..., Any]) -> Callable[..., Any]:
    def wrapper(self: VDataView, *args: Any, **kwargs: Any) -> Any:
        try:
            self._hash.assert_unchanged()
        except AssertionError:
            raise ValueError("View no longer valid since parent's VData has changed.")

        return func(self, *args, **kwargs)

    return wrapper


class VDataView:
    """
    A view of a VData object.
    """

    __slots__ = (
        "name",
        "_parent",
        "_hash",
        "_timepoints",
        "_timepoints_slicer",
        "_obs",
        "_obs_slicer",
        "_obs_slicer_flat",
        "_var",
        "_var_slicer",
        "_layers",
        "_obsm",
        "_obsp",
        "_varm",
        "_varp",
        "_uns",
    )

    # region magic methods
    def __init__(
        self,
        parent: vdata.VData,
        timepoints_slicer: tp.TimePointArray | None,
        obs_slicer: NDArray_IFS | None,
        var_slicer: NDArray_IFS | None,
    ):
        """
        Args:
            parent: a VData object to build a view of
            obs_slicer: the list of observations to view
            var_slicer: the list of variables to view
            timepoints_slicer: the list of time points to view
        """
        self.name = f"{parent.name}_view"
        generalLogger.debug("\u23BE ViewVData creation : start ----------------------------------------------------- ")

        self._parent = parent

        self._hash = VDataHash(parent, timepoints=True, obs=True, var=True)

        # first store obs : we get a sub-set of the parent's obs TemporalDataFrame
        # this is needed here because obs will be needed to recompute the time points and obs slicers
        self._obs = cast(
            TemporalDataFrameView,
            self._parent.obs[
                slice(None) if timepoints_slicer is None else timepoints_slicer,
                slice(None) if obs_slicer is None else obs_slicer,
            ],
        )

        # recompute time points and obs slicers since there could be empty subsets
        _tp_slicer = tp.as_timepointarray(parent.timepoints.value) if timepoints_slicer is None else timepoints_slicer
        self._timepoints_slicer = tp.atleast_1d(_tp_slicer[np.in1d(_tp_slicer, self._obs.timepoints)])
        self._timepoints = cast(
            H5DataFrame, self._parent.timepoints[np.in1d(self._parent.timepoints.value, self._timepoints_slicer)]
        )

        generalLogger.debug(
            f"  1'. Recomputed time points slicer to : {repr_array(self._timepoints_slicer)} "
            f"({len(self._timepoints_slicer)} value{'' if len(self._timepoints_slicer) == 1 else 's'}"
            f" selected)"
        )

        if obs_slicer is None:
            self._obs_slicer: list[NDArray_IFS] = [self._obs.index_at(tp) for tp in self._obs.timepoints]

        else:
            self._obs_slicer = cast(
                list[NDArray_IFS],
                [obs_slicer[np.isin(obs_slicer, self._obs.index_at(tp))] for tp in self._obs.timepoints],
            )

        self._obs_slicer_flat: NDArray_IFS = np.concatenate(self._obs_slicer)

        generalLogger.debug(
            f"  2'. Recomputed obs slicer to : {repr_array(self._obs_slicer_flat)} "
            f"({len(self._obs_slicer_flat)} value{'' if len(self._obs_slicer_flat) == 1 else 's'}"
            f" selected)"
        )

        self._var: H5DataFrame = self._parent.var.loc[
            slice(None) if var_slicer is None else var_slicer
        ]  # type: ignore[call-overload]
        self._var_slicer = np.array(self._var.index)

        # subset and store arrays
        _obs_slicer_flat = self._obs_slicer[0] if self.obs.index.is_repeating else self._obs_slicer_flat

        self._layers = VLayersArrayContainerView(
            self._parent.layers, self._timepoints_slicer, _obs_slicer_flat, self._var_slicer
        )
        self._obsm = VObsmArrayContainerView(self._parent.obsm, self._timepoints_slicer, _obs_slicer_flat, slice(None))
        self._obsp = VObspArrayContainerView(self._parent.obsp, np.array(self._obs.index))
        self._varm = VVarmArrayContainerView(self._parent.varm, self._var_slicer)
        self._varp = VVarpArrayContainerView(self._parent.varp, self._var_slicer)

        # uns is not subset
        self._uns = self._parent.uns

        generalLogger.debug(lambda: f"Guessed dimensions are : {self.shape}")

        generalLogger.debug("\u23BF ViewVData creation : end ------------------------------------------------------- ")

    @_check_parent_has_not_changed
    def __repr__(self) -> str:
        """
        Description for this view of a Vdata object to print.
        :return: a description of this view
        """
        _n_obs = self.n_obs if len(self.n_obs) > 1 else self.n_obs[0]

        if self.empty:
            repr_str = (
                f"Empty view of VData '{self._parent.name}' ({_n_obs} obs x {self.n_var} vars over "
                f"{self.n_timepoints} time point{'' if self.n_timepoints == 1 else 's'})."
            )

        else:
            repr_str = (
                f"View of VData '{self._parent.name}' ({_n_obs} obs x {self.n_var} vars over "
                f"{self.n_timepoints} time point{'' if self.n_timepoints == 1 else 's'})."
            )

        for attr_name in ["layers", "obs", "var", "timepoints", "obsm", "varm", "obsp", "varp"]:
            attr = getattr(self, attr_name)
            if isinstance(attr, TemporalDataFrameBase):
                keys = attr.columns
            else:
                keys = attr.keys()

            if len(keys) > 0:
                repr_str += f"\n\t{attr_name}: {str(list(keys))[1:-1]}"

        if len(self.uns):
            repr_str += f"\n\tuns: {str(list(self.uns.keys()))[1:-1]}"

        return repr_str

    @_check_parent_has_not_changed
    def __getitem__(
        self, index: PreSlicer | tuple[PreSlicer, PreSlicer] | tuple[PreSlicer, PreSlicer, PreSlicer]
    ) -> VDataView:
        """
        Get a subset of a view of a VData object.
        :param index: A sub-setting index. It can be a single index, a 2-tuple or a 3-tuple of indexes.
        """
        generalLogger.debug("ViewVData sub-setting - - - - - - - - - - - - - - ")
        generalLogger.debug(lambda: f"  Got index \n{repr_index(index)}")

        # convert to a 3-tuple
        _index = reformat_index(index, self._timepoints_slicer, self._obs_slicer_flat, self._var_slicer)

        generalLogger.debug(lambda: f"  1. Refactored index to \n{repr_index(_index)}")

        if _index is None:
            return self

        return VDataView(self._parent, _index[0], _index[1], _index[2])

    def __h5_write__(self, values: ch.H5Dict[Any]) -> None:
        write_vdata_in_h5dict(self, values, verbose=False)

    # endregion

    # region predicates
    @property
    def is_backed(self) -> Literal[False]:
        """
        For compliance with VData's API.

        Returns:
            False
        """
        return False

    @property
    def is_backed_w(self) -> Literal[False]:
        """
        For compliance with VData's API.

        Returns:
            False
        """
        return False

    @property
    @_check_parent_has_not_changed
    def empty(self) -> bool:
        """
        Is this view of a Vdata object empty ? (no obs or no vars)

        Returns:
            Is view empty ?
        """
        if not len(self.layers) or not self.n_timepoints or not self.n_obs_total or not self.n_var:
            return True
        return False

    @property
    def is_view(self) -> Literal[True]:
        return True

    # endregion

    # region attributes
    # Shapes -------------------------------------------------------------
    @property
    @_check_parent_has_not_changed
    def n_timepoints(self) -> int:
        """
        Number of time points in this view of a VData object.

        Returns:
            The number of time points in this view
        """
        return len(self._timepoints_slicer)

    @property
    @_check_parent_has_not_changed
    def n_obs(self) -> list[int]:
        """
        Number of observations in this view of a VData object.

        Returns:
            The number of observations in this view
        """
        return [len(slicer) for slicer in self._obs_slicer]

    @property
    def n_obs_total(self) -> int:
        """
        Get the total number of observations across all time points.
        :return: the total number of observations across all time points.
        """
        return sum(self.n_obs)

    @property
    @_check_parent_has_not_changed
    def n_var(self) -> int:
        """
        Number of variables in this view of a VData object.
        :return: number of variables in this view
        """
        return len(self._var_slicer)

    @property
    def shape(self) -> tuple[int, int, list[int], int]:
        """
        Shape of this view of a VData object.
        :return: view's shape
        """
        return len(self.layers), self.n_timepoints, self.n_obs, self.n_var

    # DataFrames ---------------------------------------------------------
    @property
    @_check_parent_has_not_changed
    def timepoints(self) -> H5DataFrame:
        """
        Get a view on the time points DataFrame in this ViewVData.
        :return: a view on the time points DataFrame.
        """
        return self._timepoints

    @property
    def timepoints_values(self) -> tp.TimePointArray:
        """
        Get the list of time points values (with the unit if possible).

        :return: the list of time points values (with the unit if possible).
        """
        return tp.TimePointArray(self.timepoints.value.values)

    @property
    def timepoints_strings(self) -> Iterator[str]:
        """
        Get the list of time points as strings.

        :return: the list of time points as strings.
        """
        return map(str, self.timepoints.value.values)

    @property
    def timepoints_numerical(self) -> list[float]:
        """
        Get the list of bare values from the time points.

        :return: the list of bare values from the time points.
        """
        return [tp.value for tp in self.timepoints.value]

    @property
    @_check_parent_has_not_changed
    def obs(self) -> TemporalDataFrameView:
        """
        Get a view on the obs in this ViewVData.
        :return: a view on the obs.
        """
        return self._obs

    @obs.setter
    @_check_parent_has_not_changed
    def obs(self, df: TemporalDataFrameBase) -> None:
        if not isinstance(df, TemporalDataFrameBase):
            raise TypeError("'obs' must be a TemporalDataFrame.")

        elif df.columns != self._parent.obs.columns:
            raise IncoherenceError("'obs' must have the same column names as the original 'obs' it replaces.")

        elif df.shape[0] != self.n_obs:
            raise ShapeError(f"'obs' has {df.shape[0]} lines, it should have {self.n_obs}.")

        df.index = cast(TemporalDataFrameView, self._parent.obs[self._obs_slicer_flat]).index
        self._parent.obs[self._obs_slicer_flat] = df

    @property
    @_check_parent_has_not_changed
    def var(self) -> H5DataFrame:
        """
        Get a view on the var DataFrame in this ViewVData.
        :return: a view on the var DataFrame.
        """
        return self._var

    @var.setter
    @_check_parent_has_not_changed
    def var(self, df: pd.DataFrame) -> None:
        if not isinstance(df, pd.DataFrame):
            raise TypeError("'var' must be a pandas DataFrame.")

        elif df.columns != self._parent.var.columns:
            raise IncoherenceError("'var' must have the same column names as the original 'var' it replaces.")

        elif df.shape[0] != self.n_var:
            raise ShapeError(f"'var' has {df.shape[0]} lines, it should have {self.n_var}.")

        else:
            df.index = cast(H5DataFrame, self._parent.var.loc[self._var_slicer]).index
            self._parent.var.loc[self._var_slicer] = df

    @property
    @_check_parent_has_not_changed
    def uns(self) -> DictLike[Any]:
        """
        Get a view on the uns dictionary in this ViewVData.
        :return: a view on the uns dictionary in this ViewVData.
        """
        return self._uns

    # Array containers ---------------------------------------------------
    @property
    @_check_parent_has_not_changed
    def layers(self) -> VLayersArrayContainerView:
        """
        Get a view on the layers in this ViewVData.
        :return: a view on the layers.
        """
        return self._layers

    @property
    @_check_parent_has_not_changed
    def obsm(self) -> VObsmArrayContainerView:
        """
        Get a view on the obsm in this ViewVData.
        :return: a view on the obsm.
        """
        return self._obsm

    @property
    @_check_parent_has_not_changed
    def obsp(self) -> VObspArrayContainerView:
        """
        Get a view on the obsp in this ViewVData.
        :return: a view on the obsp.
        """
        return self._obsp

    @property
    @_check_parent_has_not_changed
    def varm(self) -> VVarmArrayContainerView:
        """
        Get a view on the varm in this ViewVData.
        :return: a view on the varm.
        """
        return self._varm

    @property
    @_check_parent_has_not_changed
    def varp(self) -> VVarpArrayContainerView:
        """
        Get a view on the varp in this ViewVData.
        :return: a view on the varp.
        """
        return self._varp

    # Special ------------------------------------------------------------
    @property
    def data(self) -> ch.H5Dict[H5DataFrame | TemporalDataFrame] | NoData:
        return self._parent.data

    # Aliases ------------------------------------------------------------
    cells = obs
    genes = var

    # endregion

    # region methods
    # functions ----------------------------------------------------------
    def _mean_min_max_func(
        self, func: Literal["mean", "min", "max"], axis: Literal[0, 1]
    ) -> tuple[dict[str, TemporalDataFrame], tp.TimePointArray, NDArray_IFS]:
        """
        Compute mean, min or max of the values over the requested axis.
        """
        if axis == 0:
            _data: dict[str, TemporalDataFrame] = {
                layer: getattr(self.layers[layer], func)(axis=axis).T for layer in self.layers
            }
            _time_list = self.timepoints_values
            _index = np.array(["mean" for _ in range(self.n_timepoints)])

        elif axis == 1:
            _data = {layer: getattr(self.layers[layer], func)(axis=axis) for layer in self.layers}
            _time_list = self.obs.timepoints_column
            _index = np.array(self.obs.index)

        else:
            raise ValueError(f"Invalid axis '{axis}', should be 0 (on columns) or 1 (on rows).")

        return _data, _time_list, _index

    @_check_parent_has_not_changed
    def mean(self, axis: Literal[0, 1] = 0) -> vdata.VData:
        """
        Return the mean of the values over the requested axis.

        :param axis: compute mean over columns (0: default) or over rows (1).
        :return: a TemporalDataFrame with mean values.
        """
        _data, _time_list, _index = self._mean_min_max_func("mean", axis)

        _name = f"Mean of {self.name}" if self.name != NO_NAME else "Mean"
        return vdata.VData(data=_data, obs=pd.DataFrame(index=_index), timepoints_list=_time_list, name=_name)

    @_check_parent_has_not_changed
    def min(self, axis: Literal[0, 1] = 0) -> vdata.VData:
        """
        Return the minimum of the values over the requested axis.

        :param axis: compute minimum over columns (0: default) or over rows (1).
        :return: a TemporalDataFrame with minimum values.
        """
        _data, _time_list, _index = self._mean_min_max_func("min", axis)

        _name = f"Minimum of {self.name}" if self.name != NO_NAME else "Minimum"
        return vdata.VData(data=_data, obs=pd.DataFrame(index=_index), timepoints_list=_time_list, name=_name)

    @_check_parent_has_not_changed
    def max(self, axis: Literal[0, 1] = 0) -> vdata.VData:
        """
        Return the maximum of the values over the requested axis.

        :param axis: compute maximum over columns (0: default) or over rows (1).
        :return: a TemporalDataFrame with maximum values.
        """
        _data, _time_list, _index = self._mean_min_max_func("max", axis)

        _name = f"Maximum of {self.name}" if self.name != NO_NAME else "Maximum"
        return vdata.VData(data=_data, obs=pd.DataFrame(index=_index), timepoints_list=_time_list, name=_name)

    # writing ------------------------------------------------------------
    @_check_parent_has_not_changed
    def write(self, file: str | Path, verbose: bool = True) -> None:
        """
        Save this view of a VData in HDF5 file format.

        Args:
            file: path to save the VData
            verbose: print a progress bar while saving objects in this VData ? (default: True)
        """
        write_vdata(self, file)

    @_check_parent_has_not_changed
    def write_to_csv(
        self, directory: str | Path, sep: str = ",", na_rep: str = "", index: bool = True, header: bool = True
    ) -> None:
        """
        Save layers, timepoints, obs, obsm, obsp, var, varm and varp to csv files in a directory.

        Args:
            directory: path to a directory for saving the matrices
            sep: delimiter character
            na_rep: string to replace NAs
            index: write row names ?
            header: Write col names ?
        """
        write_vdata_to_csv(self, directory, sep, na_rep, index, header)

    # copy ---------------------------------------------------------------
    @_check_parent_has_not_changed
    def copy(self) -> vdata.VData:
        """
        Build an actual VData object from this view.
        """
        return vdata.VData(
            data=self.layers.dict_copy(),
            obs=self.obs.copy(),
            obsm=self.obsm.dict_copy(),
            obsp=self.obsp.dict_copy(),
            var=self.var.copy(),
            varm=self.varm.dict_copy(),
            varp=self.varp.dict_copy(),
            timepoints=self.timepoints.copy(),
            uns=deepcopy(self.uns),
            name=f"{self.name}_copy",
        )

    # endregion
