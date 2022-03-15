# coding: utf-8
# Created on 15/01/2021 12:58
# Author : matteo

# ====================================================
# imports
import numpy as np
import pandas as pd
from typing import Union, Iterator, Literal

import vdata
from .arrays import ViewVTDFArrayContainer, ViewVObspArrayContainer, ViewVVarmArrayContainer, ViewVVarpArrayContainer
from ...utils import reformat_index, repr_index
from ...TDF import TemporalDataFrame, ViewTemporalDataFrame
from ...name_utils import PreSlicer
from vdata.utils import repr_array
from ....time_point import TimePoint
from ....IO import generalLogger, VTypeError, IncoherenceError, ShapeError, VValueError


# ====================================================
# code
class ViewVData:
    """
    A view of a VData object.
    """

    def __init__(self,
                 parent: 'vdata.VData',
                 time_points_slicer: np.ndarray,
                 obs_slicer: np.ndarray,
                 var_slicer: np.ndarray):
        """
        Args:
            parent: a VData object to build a view of
            obs_slicer: the list of observations to view
            var_slicer: the list of variables to view
            time_points_slicer: the list of time points to view
        """
        self.name = f"{parent.name}_view"
        generalLogger.debug(u'\u23BE ViewVData creation : start ----------------------------------------------------- ')

        self._parent = parent

        # DataFrame slicers
        # time points -------------------------
        self._time_points_slicer = time_points_slicer

        # obs -------------------------
        self._obs_slicer = obs_slicer

        # var -------------------------
        self._var_slicer = var_slicer

        # first store obs : we get a sub-set of the parent's obs TemporalDataFrame
        # this is needed here because obs will be needed to recompute the time points and obs slicers
        self._obs = self._parent.obs[self._time_points_slicer, self._obs_slicer]

        # recompute time points and obs slicers since there could be empty subsets
        self._time_points_slicer = np.array([e for e in self._time_points_slicer if e in self._obs.time_points])

        generalLogger.debug(f"  1'. Recomputed time points slicer to : {repr_array(self._time_points_slicer)} "
                            f"({len(self._time_points_slicer)} value{'' if len(self._time_points_slicer) == 1 else 's'}"
                            f" selected)")

        self._obs_slicer = np.array(self._obs_slicer)[np.isin(self._obs_slicer, self._obs.index)]

        generalLogger.debug(f"  2'. Recomputed obs slicer to : {repr_array(self._obs_slicer)} "
                            f"({len(self._obs_slicer)} value{'' if len(self._obs_slicer) == 1 else 's'}"
                            f" selected)")

        # subset and store arrays
        self._layers = ViewVTDFArrayContainer(self._parent.layers,
                                              self._time_points_slicer, self._obs_slicer, self._var_slicer)
        self._time_points = self._parent.time_points[self._parent.time_points.value.isin(self._time_points_slicer)]
        self._var = self._parent.var.loc[self._var_slicer]

        self._obsm = ViewVTDFArrayContainer(self._parent.obsm, self._time_points_slicer, self._obs_slicer, slice(None))
        self._obsp = ViewVObspArrayContainer(self._parent.obsp, np.array(self._obs.index))
        self._varm = ViewVVarmArrayContainer(self._parent.varm, self._var_slicer)
        self._varp = ViewVVarpArrayContainer(self._parent.varp, self._var_slicer)
        self._uns = self._parent.uns

        generalLogger.debug(f"Guessed dimensions are : {self.shape}")

        generalLogger.debug(u'\u23BF ViewVData creation : end ------------------------------------------------------- ')

    def __repr__(self) -> str:
        """
        Description for this view of a Vdata object to print.
        :return: a description of this view
        """
        _n_obs = self.n_obs if len(self.n_obs) > 1 else self.n_obs[0]

        if self.is_empty:
            repr_str = f"Empty view of VData '{self._parent.name}' ({_n_obs} obs x {self.n_var} vars over " \
                       f"{self.n_time_points} time point{'' if self.n_time_points == 1 else 's'})."

        else:
            repr_str = f"View of VData '{self._parent.name}' with n_obs x n_var = {_n_obs} x {self.n_var} over " \
                       f"{self.n_time_points} time point{'' if self.n_time_points == 1 else 's'}"

        for attr_name in ["layers", "obs", "var", "time_points", "obsm", "varm", "obsp", "varp"]:
            attr = getattr(self, attr_name)
            keys = attr.keys()

            if len(keys) > 0:
                repr_str += f"\n\t{attr_name}: {str(list(keys))[1:-1]}"

        if len(self.uns):
            repr_str += f"\n\tuns: {str(list(self.uns.keys()))[1:-1]}"

        return repr_str

    def __getitem__(self, index: Union['PreSlicer',
                                       tuple['PreSlicer', 'PreSlicer'],
                                       tuple['PreSlicer', 'PreSlicer', 'PreSlicer']])\
            -> 'ViewVData':
        """
        Get a subset of a view of a VData object.
        :param index: A sub-setting index. It can be a single index, a 2-tuple or a 3-tuple of indexes.
        """
        generalLogger.debug('ViewVData sub-setting - - - - - - - - - - - - - - ')
        generalLogger.debug(f'  Got index \n{repr_index(index)}')

        # convert to a 3-tuple
        index = reformat_index(index, self._time_points_slicer, self._obs_slicer, self._var_slicer)

        generalLogger.debug(f"  1. Refactored index to \n{repr_index(index)}")

        return ViewVData(self._parent, index[0], index[1], index[2])

    # Shapes -------------------------------------------------------------
    @property
    def is_empty(self) -> bool:
        """
        Is this view of a Vdata object empty ? (no obs or no vars)
        :return: is view empty ?
        """
        if not len(self.layers) or not self.n_time_points or not self.n_obs_total or not self.n_var:
            return True
        return False

    @property
    def n_time_points(self) -> int:
        """
        Number of time points in this view of a VData object.
        :return: number of time points in this view
        """
        return self.layers.shape[1]

    @property
    def n_obs(self) -> list[int]:
        """
        Number of observations in this view of a VData object.
        :return: number of observations in this view
        """
        return self.layers.shape[2]

    @property
    def n_obs_total(self) -> int:
        """
        Get the total number of observations across all time points.
        :return: the total number of observations across all time points.
        """
        return sum(self.n_obs)

    @property
    def n_var(self) -> int:
        """
        Number of variables in this view of a VData object.
        :return: number of variables in this view
        """
        return self.layers.shape[3]

    @property
    def shape(self) -> tuple[int, int, list[int], int]:
        """
        Shape of this view of a VData object.
        :return: view's shape
        """
        return self.layers.shape

    # DataFrames ---------------------------------------------------------
    @property
    def time_points(self) -> pd.DataFrame:
        """
        Get a view on the time points DataFrame in this ViewVData.
        :return: a view on the time points DataFrame.
        """
        return self._time_points

    @property
    def time_points_values(self) -> list['TimePoint']:
        """
        Get the list of time points values (with the unit if possible).

        :return: the list of time points values (with the unit if possible).
        """
        return self.time_points.value.values

    @property
    def time_points_strings(self) -> Iterator[str]:
        """
        Get the list of time points as strings.

        :return: the list of time points as strings.
        """
        return map(str, self.time_points.value.values)

    @property
    def time_points_numerical(self) -> list[float]:
        """
        Get the list of bare values from the time points.

        :return: the list of bare values from the time points.
        """
        return [tp.value for tp in self.time_points.value]

    # @time_points.setter
    # def time_points(self, df: pd.DataFrame) -> None:
    #     if not isinstance(df, pd.DataFrame):
    #         raise VTypeError("'time_points' must be a pandas DataFrame.")
    #
    #     elif df.columns != self._parent.time_points.columns:
    #         raise IncoherenceError("'time_points' must have the same column names as the original 'time_points' "
    #                                "it replaces.")
    #
    #     elif df.shape[0] != self.n_time_points:
    #         raise ShapeError(f"'time_points' has {df.shape[0]} lines, it should have {self.n_time_points}.")
    #
    #     else:
    #         df.index = self._parent.time_points[self._time_points_slicer].index
    #         self._parent.time_points[self._time_points_slicer] = df

    @property
    def obs(self) -> ViewTemporalDataFrame:
        """
        Get a view on the obs in this ViewVData.
        :return: a view on the obs.
        """
        return self._obs

    @obs.setter
    def obs(self, df: Union['TemporalDataFrame', 'ViewTemporalDataFrame']) -> None:
        if not isinstance(df, (TemporalDataFrame, ViewTemporalDataFrame)):
            raise VTypeError("'obs' must be a TemporalDataFrame.")

        elif df.columns != self._parent.obs.columns:
            raise IncoherenceError("'obs' must have the same column names as the original 'obs' it replaces.")

        elif df.shape[0] != self.n_obs:
            raise ShapeError(f"'obs' has {df.shape[0]} lines, it should have {self.n_obs}.")

        else:
            df.index = self._parent.obs[self._obs_slicer].index
            self._parent.obs[self._obs_slicer] = df

    @property
    def var(self) -> pd.DataFrame:
        """
        Get a view on the var DataFrame in this ViewVData.
        :return: a view on the var DataFrame.
        """
        return self._var

    @var.setter
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

    @property
    def uns(self) -> dict:
        """
        Get a view on the uns dictionary in this ViewVData.
        :return: a view on the uns dictionary in this ViewVData.
        """
        return self._uns

    # Array containers ---------------------------------------------------
    @property
    def layers(self) -> ViewVTDFArrayContainer:
        """
        Get a view on the layers in this ViewVData.
        :return: a view on the layers.
        """
        return self._layers

    @property
    def obsm(self) -> ViewVTDFArrayContainer:
        """
        Get a view on the obsm in this ViewVData.
        :return: a view on the obsm.
        """
        return self._obsm

    @property
    def obsp(self) -> ViewVObspArrayContainer:
        """
        Get a view on the obsp in this ViewVData.
        :return: a view on the obsp.
        """
        return self._obsp

    @property
    def varm(self) -> ViewVVarmArrayContainer:
        """
        Get a view on the varm in this ViewVData.
        :return: a view on the varm.
        """
        return self._varm

    @property
    def varp(self) -> ViewVVarpArrayContainer:
        """
        Get a view on the varp in this ViewVData.
        :return: a view on the varp.
        """
        return self._varp

    # Aliases ------------------------------------------------------------
    @property
    def cells(self) -> ViewTemporalDataFrame:
        """
        Alias for the obs attribute.
        :return: a view on the obs TemporalDataFrame.
        """
        return self.obs

    @cells.setter
    def cells(self, df: Union['TemporalDataFrame', 'ViewTemporalDataFrame']) -> None:
        self.obs = df

    @property
    def genes(self) -> pd.DataFrame:
        """
        Alias for the var attribute.
        :return: a view on the var DataFrame.
        """
        return self.var

    @genes.setter
    def genes(self, df: pd.DataFrame) -> None:
        self.var = df

    # functions ----------------------------------------------------------
    def __mean_min_max_func(self, func: Literal['mean', 'min', 'max'], axis) \
            -> tuple[dict[str, TemporalDataFrame], list[TimePoint], pd.Index]:
        """
        Compute mean, min or max of the values over the requested axis.
        """
        if axis == 0:
            _data = {layer: getattr(self.layers[layer], func)(axis=axis).T for layer in self.layers}
            _time_list = self.time_points_values
            _index = pd.Index(['mean' for _ in range(self.n_time_points)])

        elif axis == 1:
            _data = {layer: getattr(self.layers[layer], func)(axis=axis) for layer in self.layers}
            _time_list = self.obs.time_points_column
            _index = self.obs.index

        else:
            raise VValueError(f"Invalid axis '{axis}', should be 0 (on columns) or 1 (on rows).")

        return _data, _time_list, _index

    def mean(self, axis: Literal[0, 1] = 0) -> 'vdata.VData':
        """
        Return the mean of the values over the requested axis.

        :param axis: compute mean over columns (0: default) or over rows (1).
        :return: a TemporalDataFrame with mean values.
        """
        _data, _time_list, _index = self.__mean_min_max_func('mean', axis)

        _name = f"Mean of {self.name}" if self.name != 'No_Name' else None
        return vdata.VData(data=_data, obs=pd.DataFrame(index=_index), time_list=_time_list, name=_name)

    def min(self, axis: Literal[0, 1] = 0) -> 'vdata.VData':
        """
        Return the minimum of the values over the requested axis.

        :param axis: compute minimum over columns (0: default) or over rows (1).
        :return: a TemporalDataFrame with minimum values.
        """
        _data, _time_list, _index = self.__mean_min_max_func('min', axis)

        _name = f"Minimum of {self.name}" if self.name != 'No_Name' else None
        return vdata.VData(data=_data, obs=pd.DataFrame(index=_index), time_list=_time_list, name=_name)

    def max(self, axis: Literal[0, 1] = 0) -> 'vdata.VData':
        """
        Return the maximum of the values over the requested axis.

        :param axis: compute maximum over columns (0: default) or over rows (1).
        :return: a TemporalDataFrame with maximum values.
        """
        _data, _time_list, _index = self.__mean_min_max_func('max', axis)

        _name = f"Maximum of {self.name}" if self.name != 'No_Name' else None
        return vdata.VData(data=_data, obs=pd.DataFrame(index=_index), time_list=_time_list, name=_name)

    # copy ---------------------------------------------------------------
    def copy(self) -> 'vdata.VData':
        """
        Build an actual VData object from this view.
        """
        return vdata.VData(data=self.layers.dict_copy(),
                           obs=self.obs.copy(),
                           obsm=self.obsm.dict_copy(), obsp=self.obsp.dict_copy(),
                           var=self.var,
                           varm=self.varm.dict_copy(), varp=self.varp.dict_copy(),
                           time_points=self.time_points,
                           uns=self.uns,
                           dtype=self._parent.dtype,
                           name=f"{self.name}_copy")
