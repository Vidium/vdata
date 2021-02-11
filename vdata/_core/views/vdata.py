# coding: utf-8
# Created on 15/01/2021 12:58
# Author : matteo

# ====================================================
# imports
import numpy as np
import pandas as pd
from typing import Union, Tuple, List, Dict, Any, NoReturn

import vdata
from vdata.NameUtils import PreSlicer
from .arrays import ViewVTDFArrayContainer, ViewVObspArrayContainer, ViewVVarmArrayContainer, ViewVVarpArrayContainer
from .. import utils
from ...utils import repr_array, repr_index, reformat_index
from ..._TDF.views.dataframe import ViewTemporalDataFrame
from ..._IO import generalLogger
from ..._IO.errors import VValueError, VTypeError, IncoherenceError, ShapeError


# ====================================================
# code
class ViewVData:
    """
    A view of a VData object.
    """

    def __init__(self, parent: 'vdata.VData', time_points_slicer: np.ndarray, obs_slicer: np.ndarray,
                 var_slicer: np.ndarray):
        """
        :param parent: a VData object to build a view of
        :param obs_slicer: the list of observations to view
        :param var_slicer: the list of variables to view
        :param time_points_slicer: the list of time points to view
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
        self._layers = ViewVTDFArrayContainer(self._parent.layers, self._time_points_slicer,
                                              self._obs_slicer, self._var_slicer)
        self._time_points = self._parent.time_points[self._parent.time_points.value.isin(self._time_points_slicer)]
        self._var = self._parent.var.loc[self._var_slicer]

        self._obsm = ViewVTDFArrayContainer(self._parent.obsm, self._time_points_slicer,
                                            self._obs_slicer, self._var_slicer)
        self._obsp = ViewVObspArrayContainer(self._parent.obsp, self._time_points_slicer, self._obs_slicer)
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
            repr_str = f"Empty view of a Vdata object ({_n_obs} obs x {self.n_var} vars over " \
                       f"{self.n_time_points} time point{'' if self.n_time_points == 1 else 's'})."

        else:
            repr_str = f"View of a Vdata object with n_obs x n_var = {_n_obs} x {self.n_var} over " \
                       f"{self.n_time_points} time point{'' if self.n_time_points == 1 else 's'}"

        for attr_name in ["layers", "obs", "var", "time_points", "obsm", "varm", "obsp", "varp"]:
            attr = getattr(self, attr_name)
            keys = attr.keys() if attr is not None else ()

            if len(keys) > 0:
                repr_str += f"\n\t{attr_name}: {str(list(keys))[1:-1]}"

        if len(self.uns):
            repr_str += f"\n\tuns: {str(list(self.uns.keys()))[1:-1]}"

        return repr_str

    def __getitem__(self, index: Union[PreSlicer, Tuple[PreSlicer, PreSlicer], Tuple[PreSlicer, PreSlicer, PreSlicer]])\
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
    def n_obs(self) -> List[int]:
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
    def shape(self) -> Tuple[int, int, List[int], int]:
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
    def obs(self, df: Union['vdata.TemporalDataFrame', ViewTemporalDataFrame]) -> None:
        if not isinstance(df, (vdata.TemporalDataFrame, ViewTemporalDataFrame)):
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
    def uns(self) -> Dict:
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

    @layers.setter
    def layers(self, *_: Any) -> NoReturn:
        raise VValueError("Cannot set layers in a view. Use the original VData object.")

    @property
    def obsm(self) -> ViewVTDFArrayContainer:
        """
        Get a view on the obsm in this ViewVData.
        :return: a view on the obsm.
        """
        return self._obsm

    @obsm.setter
    def obsm(self, *_: Any) -> NoReturn:
        raise VValueError("Cannot set obsm in a view. Use the original VData object.")

    @property
    def obsp(self) -> ViewVObspArrayContainer:
        """
        Get a view on the obsp in this ViewVData.
        :return: a view on the obsp.
        """
        return self._obsp

    @obsp.setter
    def obsp(self, *_: Any) -> NoReturn:
        raise VValueError("Cannot set obsp in a view. Use the original VData object.")

    @property
    def varm(self) -> ViewVVarmArrayContainer:
        """
        Get a view on the varm in this ViewVData.
        :return: a view on the varm.
        """
        return self._varm

    @varm.setter
    def varm(self, *_: Any) -> NoReturn:
        raise VValueError("Cannot set varm in a view. Use the original VData object.")

    @property
    def varp(self) -> ViewVVarpArrayContainer:
        """
        Get a view on the varp in this ViewVData.
        :return: a view on the varp.
        """
        return self._varp

    @varp.setter
    def varp(self, *_: Any) -> NoReturn:
        raise VValueError("Cannot set varp in a view. Use the original VData object.")

    # Aliases ------------------------------------------------------------
    @property
    def cells(self) -> ViewTemporalDataFrame:
        """
        Alias for the obs attribute.
        :return: a view on the obs TemporalDataFrame.
        """
        return self.obs

    @cells.setter
    def cells(self, df: Union['vdata.TemporalDataFrame', ViewTemporalDataFrame]) -> None:
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

    # copy ---------------------------------------------------------------
    def copy(self) -> 'vdata.VData':
        """
        Build an actual VData object from this view.
        """
        _obsp = utils.compact_obsp(self.obsp, self.obs.index)

        return vdata.VData(data=self.layers.dict_copy(),
                           obs=self.obs.copy(),
                           obsm=self.obsm.dict_copy(), obsp=_obsp,
                           var=self.var,
                           varm=self.varm.dict_copy(), varp=self.varp.dict_copy(),
                           time_points=self.time_points,
                           uns=self.uns,
                           dtype=self._parent.dtype,
                           name=f"{self.name}_copy")
