# coding: utf-8
# Created on 15/01/2021 12:58
# Author : matteo

# ====================================================
# imports
import numpy as np
import pandas as pd
from pathlib import Path

from typing import Union, Iterator, Literal, Optional

import vdata
from .arrays import ViewVLayerArrayContainer, ViewVObsmArrayContainer, ViewVObspArrayContainer, \
    ViewVVarmArrayContainer, ViewVVarpArrayContainer
from vdata.core.utils import reformat_index, repr_index
from vdata.core.tdf import TemporalDataFrame, TemporalDataFrameView
from vdata.core.name_utils import PreSlicer
from vdata.utils import repr_array
from vdata.time_point import TimePoint
from vdata.vdataframe import ViewVDataFrame
from vdata.IO import generalLogger, VTypeError, IncoherenceError, ShapeError, VValueError
from vdata.read_write.write import write_vdata, write_vdata_to_csv
from ...tdf.base import BaseTemporalDataFrame


# ====================================================
# code
def _check_parent_has_not_changed(func):
    def wrapper(self, *args, **kwargs):
        if hash(tuple(self._parent.timepoints.value.values)) != self._timepoints_hash or \
                hash(tuple(self._parent.obs.index)) != self._obs_hash or \
                hash(tuple(self._parent.var.index)) != self._var_hash:
            raise VValueError("View no longer valid since parent's VData has changed.")

        return func(self, *args, **kwargs)

    return wrapper


class ViewVData:
    """
    A view of a VData object.
    """

    __slots__ = 'name', '_parent', '_timepoints_hash', '_obs_hash', '_var_hash', \
                '_timepoints', '_timepoints_slicer', '_obs', '_obs_slicer', '_obs_slicer_flat', '_var', '_var_slicer',\
                '_layers', '_obsm', '_obsp', '_varm', '_varp', '_uns'

    # region magic methods
    def __init__(self,
                 parent: 'vdata.VData',
                 timepoints_slicer: Optional[np.ndarray],
                 obs_slicer: Optional[np.ndarray],
                 var_slicer: Optional[np.ndarray]):
        """
        Args:
            parent: a VData object to build a view of
            obs_slicer: the list of observations to view
            var_slicer: the list of variables to view
            timepoints_slicer: the list of time points to view
        """
        self.name = f"{parent.name}_view"
        generalLogger.debug(u'\u23BE ViewVData creation : start ----------------------------------------------------- ')

        self._parent = parent

        self._timepoints_hash = hash(tuple(parent.timepoints.value.values))
        self._obs_hash = hash(tuple(parent.obs.index))
        self._var_hash = hash(tuple(parent.var.index))

        # first store obs : we get a sub-set of the parent's obs TemporalDataFrame
        # this is needed here because obs will be needed to recompute the time points and obs slicers
        _tp_slicer = slice(None) if timepoints_slicer is None else timepoints_slicer
        _obs_slicer = slice(None) if obs_slicer is None else obs_slicer
        self._obs = self._parent.obs[_tp_slicer, _obs_slicer]

        # recompute time points and obs slicers since there could be empty subsets
        _tp_slicer = parent.timepoints.value.values if timepoints_slicer is None else timepoints_slicer
        self._timepoints_slicer = np.array([e for e in _tp_slicer if e in self._obs.timepoints])
        self._timepoints = ViewVDataFrame(self._parent.timepoints,
                                          index_slicer=self._parent.timepoints.value.isin(self._timepoints_slicer))

        generalLogger.debug(f"  1'. Recomputed time points slicer to : {repr_array(self._timepoints_slicer)} "
                            f"({len(self._timepoints_slicer)} value{'' if len(self._timepoints_slicer) == 1 else 's'}"
                            f" selected)")

        if obs_slicer is None:
            self._obs_slicer = [self._obs.index_at(tp) for tp in self._obs.timepoints]

        else:
            self._obs_slicer = [np.array(obs_slicer)[np.isin(obs_slicer, self._obs.index_at(tp))]
                                for tp in self._obs.timepoints]

        self._obs_slicer_flat = np.concatenate(self._obs_slicer)

        generalLogger.debug(f"  2'. Recomputed obs slicer to : {repr_array(self._obs_slicer_flat)} "
                            f"({len(self._obs_slicer_flat)} value{'' if len(self._obs_slicer_flat) == 1 else 's'}"
                            f" selected)")

        # then store var : we get a sub-set of the parent's var VDataFrame
        # this is needed to recompute the var slicer
        _var_slicer = slice(None) if var_slicer is None else var_slicer
        self._var = ViewVDataFrame(self._parent.var, index_slicer=_var_slicer)

        # recompute var slicer
        # TODO : check the order is maintained
        if var_slicer is None:
            self._var_slicer = self._var.index

        else:
            self._var_slicer = np.array(var_slicer)[np.isin(var_slicer, self._var.index)]

        # subset and store arrays
        _obs_slicer_flat = self._obs_slicer[0] if self.has_repeated_obs_index else self._obs_slicer_flat

        self._layers = ViewVLayerArrayContainer(self._parent.layers,
                                                self._timepoints_slicer, _obs_slicer_flat, self._var_slicer)

        self._obsm = ViewVObsmArrayContainer(self._parent.obsm, self._timepoints_slicer, _obs_slicer_flat, slice(None))
        self._obsp = ViewVObspArrayContainer(self._parent.obsp, np.array(self._obs.index))
        self._varm = ViewVVarmArrayContainer(self._parent.varm, self._var_slicer)
        self._varp = ViewVVarpArrayContainer(self._parent.varp, self._var_slicer)

        # uns is not subset
        self._uns = self._parent.uns

        generalLogger.debug(f"Guessed dimensions are : {self.shape}")

        generalLogger.debug(u'\u23BF ViewVData creation : end ------------------------------------------------------- ')

    @_check_parent_has_not_changed
    def __repr__(self) -> str:
        """
        Description for this view of a Vdata object to print.
        :return: a description of this view
        """
        _n_obs = self.n_obs if len(self.n_obs) > 1 else self.n_obs[0]

        if self.empty:
            repr_str = f"Empty view of VData '{self._parent.name}' ({_n_obs} obs x {self.n_var} vars over " \
                       f"{self.n_timepoints} time point{'' if self.n_timepoints == 1 else 's'})."

        else:
            repr_str = f"View of VData '{self._parent.name}' with n_obs x n_var = {_n_obs} x {self.n_var} over " \
                       f"{self.n_timepoints} time point{'' if self.n_timepoints == 1 else 's'}"

        for attr_name in ["layers", "obs", "var", "timepoints", "obsm", "varm", "obsp", "varp"]:
            attr = getattr(self, attr_name)
            if isinstance(attr, BaseTemporalDataFrame):
                keys = attr.columns
            else:
                keys = attr.keys()

            if len(keys) > 0:
                repr_str += f"\n\t{attr_name}: {str(list(keys))[1:-1]}"

        if len(self.uns):
            repr_str += f"\n\tuns: {str(list(self.uns.keys()))[1:-1]}"

        return repr_str

    @_check_parent_has_not_changed
    def __getitem__(self, index: Union['PreSlicer',
                                       tuple['PreSlicer', 'PreSlicer'],
                                       tuple['PreSlicer', 'PreSlicer', 'PreSlicer']]) \
            -> 'ViewVData':
        """
        Get a subset of a view of a VData object.
        :param index: A sub-setting index. It can be a single index, a 2-tuple or a 3-tuple of indexes.
        """
        generalLogger.debug('ViewVData sub-setting - - - - - - - - - - - - - - ')
        generalLogger.debug(f'  Got index \n{repr_index(index)}')

        # convert to a 3-tuple
        index = reformat_index(index, self._timepoints_slicer, self._obs_slicer_flat, self._var_slicer)

        generalLogger.debug(f"  1. Refactored index to \n{repr_index(index)}")

        return ViewVData(self._parent, index[0], index[1], index[2])

    # endregion

    # region predicates
    @property                                                                           # type: ignore
    def is_backed(self) -> Literal[False]:
        """
        For compliance with VData's API.

        Returns:
            False
        """
        return False

    @property                                                                           # type: ignore
    def is_backed_w(self) -> Literal[False]:
        """
        For compliance with VData's API.

        Returns:
            False
        """
        return False

    @property                                                                           # type: ignore
    @_check_parent_has_not_changed
    def has_repeated_obs_index(self) -> bool:
        return self._parent.has_repeated_obs_index
    # endregion

    # region attributes
    # Shapes -------------------------------------------------------------
    @property                                                                           # type: ignore
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

    @property                                                                           # type: ignore
    @_check_parent_has_not_changed
    def n_timepoints(self) -> int:
        """
        Number of time points in this view of a VData object.

        Returns:
            The number of time points in this view
        """
        return len(self._timepoints_slicer)

    @property                                                                           # type: ignore
    @_check_parent_has_not_changed
    def n_obs(self) -> list[int]:
        """
        Number of observations in this view of a VData object.

        Returns:
            The number of observations in this view
        """
        return [len(slicer) for slicer in self._obs_slicer]

    @property                                                                           # type: ignore
    def n_obs_total(self) -> int:
        """
        Get the total number of observations across all time points.
        :return: the total number of observations across all time points.
        """
        return sum(self.n_obs)

    @property                                                                           # type: ignore
    @_check_parent_has_not_changed
    def n_var(self) -> int:
        """
        Number of variables in this view of a VData object.
        :return: number of variables in this view
        """
        return len(self._var_slicer)

    @property                                                                           # type: ignore
    def shape(self) -> tuple[int, int, list[int], int]:
        """
        Shape of this view of a VData object.
        :return: view's shape
        """
        return len(self.layers), self.n_timepoints, self.n_obs, self.n_var

    # DataFrames ---------------------------------------------------------
    @property                                                                           # type: ignore
    @_check_parent_has_not_changed
    def timepoints(self) -> ViewVDataFrame:
        """
        Get a view on the time points DataFrame in this ViewVData.
        :return: a view on the time points DataFrame.
        """
        return self._timepoints

    @property                                                                           # type: ignore
    def timepoints_values(self) -> list['TimePoint']:
        """
        Get the list of time points values (with the unit if possible).

        :return: the list of time points values (with the unit if possible).
        """
        return self.timepoints.value.values

    @property                                                                           # type: ignore
    def timepoints_strings(self) -> Iterator[str]:
        """
        Get the list of time points as strings.

        :return: the list of time points as strings.
        """
        return map(str, self.timepoints.value.values)

    @property                                                                           # type: ignore
    def timepoints_numerical(self) -> list[float]:
        """
        Get the list of bare values from the time points.

        :return: the list of bare values from the time points.
        """
        return [tp.value for tp in self.timepoints.value]

    @property                                                                           # type: ignore
    @_check_parent_has_not_changed
    def obs(self) -> TemporalDataFrameView:
        """
        Get a view on the obs in this ViewVData.
        :return: a view on the obs.
        """
        return self._obs

    @obs.setter                                                                           # type: ignore
    @_check_parent_has_not_changed
    def obs(self, df: Union['TemporalDataFrame', 'TemporalDataFrameView']) -> None:
        if not isinstance(df, (TemporalDataFrame, TemporalDataFrameView)):
            raise VTypeError("'obs' must be a TemporalDataFrame.")

        elif df.columns != self._parent.obs.columns:
            raise IncoherenceError("'obs' must have the same column names as the original 'obs' it replaces.")

        elif df.shape[0] != self.n_obs:
            raise ShapeError(f"'obs' has {df.shape[0]} lines, it should have {self.n_obs}.")

        else:
            df.index = self._parent.obs[self._obs_slicer_flat].index
            self._parent.obs[self._obs_slicer_flat] = df

    @property                                                                           # type: ignore
    @_check_parent_has_not_changed
    def var(self) -> ViewVDataFrame:
        """
        Get a view on the var DataFrame in this ViewVData.
        :return: a view on the var DataFrame.
        """
        return self._var

    @var.setter                                                                           # type: ignore
    @_check_parent_has_not_changed
    def var(self, df: pd.DataFrame) -> None:
        if not isinstance(df, pd.DataFrame):
            raise VTypeError("'var' must be a pandas DataFrame.")

        elif df.columns != self._parent.var.columns:
            raise IncoherenceError("'var' must have the same column names as the original 'var' it replaces.")

        elif df.shape[0] != self.n_var:
            raise ShapeError(f"'var' has {df.shape[0]} lines, it should have {self.n_var}.")

        else:
            df.index = self._parent.var[self._var_slicer].index
            self._parent.var[self._var_slicer] = df

    @property                                                                           # type: ignore
    @_check_parent_has_not_changed
    def uns(self) -> dict:
        """
        Get a view on the uns dictionary in this ViewVData.
        :return: a view on the uns dictionary in this ViewVData.
        """
        return self._uns

    # Array containers ---------------------------------------------------
    @property                                                                           # type: ignore
    @_check_parent_has_not_changed
    def layers(self) -> ViewVLayerArrayContainer:
        """
        Get a view on the layers in this ViewVData.
        :return: a view on the layers.
        """
        return self._layers

    @property                                                                           # type: ignore
    @_check_parent_has_not_changed
    def obsm(self) -> ViewVObsmArrayContainer:
        """
        Get a view on the obsm in this ViewVData.
        :return: a view on the obsm.
        """
        return self._obsm

    @property                                                                           # type: ignore
    @_check_parent_has_not_changed
    def obsp(self) -> ViewVObspArrayContainer:
        """
        Get a view on the obsp in this ViewVData.
        :return: a view on the obsp.
        """
        return self._obsp

    @property                                                                           # type: ignore
    @_check_parent_has_not_changed
    def varm(self) -> ViewVVarmArrayContainer:
        """
        Get a view on the varm in this ViewVData.
        :return: a view on the varm.
        """
        return self._varm

    @property                                                                           # type: ignore
    @_check_parent_has_not_changed
    def varp(self) -> ViewVVarpArrayContainer:
        """
        Get a view on the varp in this ViewVData.
        :return: a view on the varp.
        """
        return self._varp

    # Special ------------------------------------------------------------
    @property                                                                           # type: ignore
    def dtype(self) -> np.dtype:
        """
        Get the data type of this ViewVData object.
        """
        return self._parent.dtype

    # Aliases ------------------------------------------------------------
    @property                                                                           # type: ignore
    def cells(self) -> TemporalDataFrameView:
        """
        Alias for the obs attribute.
        :return: a view on the obs TemporalDataFrame.
        """
        return self.obs

    @cells.setter
    def cells(self, df: Union['TemporalDataFrame', 'TemporalDataFrameView']) -> None:
        self.obs = df

    @property
    def genes(self) -> ViewVDataFrame:
        """
        Alias for the var attribute.
        :return: a view on the var DataFrame.
        """
        return self.var

    @genes.setter
    def genes(self, df: pd.DataFrame) -> None:
        self.var = df

    # endregion

    # region methods
    # functions ----------------------------------------------------------
    def __mean_min_max_func(self, func: Literal['mean', 'min', 'max'], axis) \
            -> tuple[dict[str, TemporalDataFrame], list[TimePoint], pd.Index]:
        """
        Compute mean, min or max of the values over the requested axis.
        """
        if axis == 0:
            _data = {layer: getattr(self.layers[layer], func)(axis=axis).T for layer in self.layers}
            _time_list = self.timepoints_values
            _index = pd.Index(['mean' for _ in range(self.n_timepoints)])

        elif axis == 1:
            _data = {layer: getattr(self.layers[layer], func)(axis=axis) for layer in self.layers}
            _time_list = self.obs.timepoints_column
            _index = self.obs.index

        else:
            raise VValueError(f"Invalid axis '{axis}', should be 0 (on columns) or 1 (on rows).")

        return _data, _time_list, _index

    @_check_parent_has_not_changed
    def mean(self, axis: Literal[0, 1] = 0) -> 'vdata.VData':
        """
        Return the mean of the values over the requested axis.

        :param axis: compute mean over columns (0: default) or over rows (1).
        :return: a TemporalDataFrame with mean values.
        """
        _data, _time_list, _index = self.__mean_min_max_func('mean', axis)

        _name = f"Mean of {self.name}" if self.name != 'No_Name' else None
        return vdata.VData(data=_data, obs=pd.DataFrame(index=_index), time_list=_time_list, name=_name)

    @_check_parent_has_not_changed
    def min(self, axis: Literal[0, 1] = 0) -> 'vdata.VData':
        """
        Return the minimum of the values over the requested axis.

        :param axis: compute minimum over columns (0: default) or over rows (1).
        :return: a TemporalDataFrame with minimum values.
        """
        _data, _time_list, _index = self.__mean_min_max_func('min', axis)

        _name = f"Minimum of {self.name}" if self.name != 'No_Name' else None
        return vdata.VData(data=_data, obs=pd.DataFrame(index=_index), time_list=_time_list, name=_name)

    @_check_parent_has_not_changed
    def max(self, axis: Literal[0, 1] = 0) -> 'vdata.VData':
        """
        Return the maximum of the values over the requested axis.

        :param axis: compute maximum over columns (0: default) or over rows (1).
        :return: a TemporalDataFrame with maximum values.
        """
        _data, _time_list, _index = self.__mean_min_max_func('max', axis)

        _name = f"Maximum of {self.name}" if self.name != 'No_Name' else None
        return vdata.VData(data=_data, obs=pd.DataFrame(index=_index), time_list=_time_list, name=_name)

    # writing ------------------------------------------------------------
    @_check_parent_has_not_changed
    def write(self,
              file: Union[str, Path]) -> None:
        """
        Save this VData object in HDF5 file format.

        Args:
            file: path to save the VData
        """
        write_vdata(self, file)

    @_check_parent_has_not_changed
    def write_to_csv(self,
                     directory: Union[str, Path],
                     sep: str = ",",
                     na_rep: str = "",
                     index: bool = True,
                     header: bool = True) -> None:
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
    def copy(self) -> 'vdata.VData':
        """
        Build an actual VData object from this view.
        """
        return vdata.VData(data=self.layers.dict_copy(),
                           obs=self.obs.copy(),
                           obsm=self.obsm.dict_copy(), obsp=self.obsp.dict_copy(),
                           var=self.var.copy(),
                           varm=self.varm.dict_copy(), varp=self.varp.dict_copy(),
                           timepoints=self.timepoints.copy(),
                           uns=self.uns,
                           dtype=self._parent.dtype,
                           name=f"{self.name}_copy",
                           no_check=True)

    # endregion
