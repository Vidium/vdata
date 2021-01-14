# coding: utf-8
# Created on 11/25/20 4:25 PM
# Author : matteo

# ====================================================
# imports
import pandas as pd
import numpy as np
from typing import Tuple, Dict, Union, KeysView, ValuesView, ItemsView, Optional, List

from . import vdata
from .utils import format_index, repr_array
from .arrays import VLayerArrayContainer, VAxisArrayContainer, VPairwiseArrayContainer, VBaseArrayContainer, \
    VPairwiseArray
from .dataframe import ViewTemporalDataFrame
from ..NameUtils import PreSlicer, Slicer, ArrayLike_3D, ArrayLike_2D, DataFrame
from ..utils import slice_to_range, slice_to_list
from .._IO.errors import VTypeError, ShapeError
from .._IO.logger import generalLogger


# ====================================================
# code
class ViewVBaseArrayContainer:
    """
    A base view of a VBaseArrayContainer.
    This class is used to create views on VLayerArrayContainer, VAxisArrays and VPairwiseArrays.
    """

    def __init__(self, array_container: VBaseArrayContainer):
        """
        :param array_container: a VBaseArrayContainer object to build a view on.
        """
        self._array_container = array_container

    def __repr__(self) -> str:
        """
        Description for this view of a VBaseArrayContainer object to print.
        :return: a description of this view
        """
        return f"View of {self._array_container}"

    def keys(self) -> Union[Tuple[()], KeysView]:
        """
        Get keys of the VBaseArrayContainer.
        """
        return self._array_container.keys()

    def values(self) -> Union[Tuple[()], ValuesView]:
        """
        Get values of the VBaseArrayContainer.
        """
        return self._array_container.values()

    def items(self) -> Union[Tuple[()], ItemsView]:
        """
        Get items of the VBaseArrayContainer.
        """
        return self._array_container.items()

    def dict_copy(self) -> Dict[str, Union[DataFrame, VPairwiseArray]]:
        """
        Build an actual copy of this Array view in dict format.
        :return: Dictionary of (keys, ArrayLike) in this Array view.
        """
        return dict([(arr, self._array_container[arr]) for arr in self._array_container.keys()])


class ViewVLayerArrayContainer(ViewVBaseArrayContainer):
    """
    A view of a VLayerArrayContainer object.
    """

    def __init__(self, array_container: VLayerArrayContainer, time_points_slicer: np.ndarray,
                 obs_slicer: np.ndarray, var_slicer: np.ndarray):
        """
        :param array_container: a VLayerArrayContainer object to build a view on
        :param obs_slicer: the list of observations to view
        :param var_slicer: the list of variables to view
        :param time_points_slicer: the list of time points to view
        """
        generalLogger.debug(u'\u23BE ViewVLayersArraysContainer creation : begin '
                            '--------------------------------------------- ')
        super().__init__(array_container)

        self._time_points_slicer = time_points_slicer
        generalLogger.debug(f"  1. Time points slicer is : {repr_array(self._time_points_slicer)}.")

        # self._obs_slicer = self.__correct_obs_slicer(obs_slicer)
        self._obs_slicer = obs_slicer
        generalLogger.debug(f"  2. Obs slicer is : {repr_array(self._obs_slicer)}.")

        self._var_slicer = var_slicer
        generalLogger.debug(f"  3. Var slicer is : {repr_array(self._var_slicer)}.")

        generalLogger.debug(u'\u23BF ViewVLayersArraysContainer creation : end '
                            '----------------------------------------------- ')

    def __getitem__(self, array_name: str) -> ViewTemporalDataFrame:
        """
        Get a specific Array in this view.
        :param array_name: the name of the Array to get
        """
        return self._array_container[array_name][self._time_points_slicer, self._obs_slicer, self._var_slicer]

    def __setitem__(self, array_name: str, values: ArrayLike_3D) -> None:
        """
        Set values for a specific Array in this view with an array-like object.
        Shapes of the view and of the array-like object must match.
        :param array_name: the name of the Array for which to set values
        :param values: an array-like object of values with shape matching the view's.
        """
        # TODO : update
        if not isinstance(values, np.ndarray):
            raise VTypeError("Values must be a 3D array-like object (numpy arrays)")

        elif not values.shape == (len(self._time_points_slicer), len(self._obs_slicer), len(self._var_slicer)):
            raise ShapeError(f"Cannot set values, array-like object {values.shape} should have shape "
                             f"({len(self._time_points_slicer)}, {len(self._obs_slicer)}, {len(self._var_slicer)})")

        else:
            self._array_container[array_name][np.ix_(self._time_points_slicer, self._obs_slicer, self._var_slicer)] = values


class ViewVAxisArrayContainer(ViewVBaseArrayContainer):
    """
    A view of a VAxisArrayContainer object.
    """

    def __init__(self, arrays: VAxisArrayContainer, time_points_slicer: np.ndarray, axis_slicer: np.ndarray):
        """
        :param arrays: a VAxisArrayContainer object to build a view on
        :param axis_slicer: the list of observations/variables to view
        :param time_points_slicer: the list of time points to view
        """
        super().__init__(arrays)
        self._axis_slicer = axis_slicer
        self._time_points_slicer = time_points_slicer

    def __getitem__(self, array_name: str) -> ArrayLike_3D:
        """
        Get a specific Array in this view.
        :param array_name: the name of the Array to get
        """
        return self._arrays[array_name][np.ix_(self._time_points_slicer, self._axis_slicer)]

    def __setitem__(self, array_name: str, values: ArrayLike_3D) -> None:
        """
        Set values for a specific Array in this view with an array-like object.
        Shapes of the view and of the array-like object must match.
        :param array_name: the name of the Array for which to set values
        :param values: an array-like object of values with shape matching the view's.
        """
        array_shape = (len(self._time_points_slicer), len(self._axis_slicer), self._arrays[array_name].shape[2])

        if not isinstance(values, np.ndarray):
            raise VTypeError("Values must be a 3D array-like object (numpy arrays)")

        elif not values.shape == array_shape:
            raise ShapeError(f"Cannot set values, array-like object {values.shape} should have shape {array_shape}")

        else:
            self._arrays[array_name][np.ix_(self._time_points_slicer, self._axis_slicer)] = values


class ViewVPairwiseArrayContainer(ViewVBaseArrayContainer):
    """
    A view of a VPairwiseArrayContainer object.
    """

    def __init__(self, arrays: VPairwiseArrayContainer, axis_slicer: np.ndarray):
        """
        :param arrays: a VPairwiseArrayContainer object to build a view on
        :param axis_slicer: the list of observations/variables to view
        """
        super().__init__(arrays)
        self._axis_slicer = axis_slicer

    def __getitem__(self, array_name: str) -> ArrayLike_2D:
        """
        Get a specific Array in this view.
        :param array_name: the name of the Array to get
        """
        return self._arrays[array_name][np.ix_(self._axis_slicer, self._axis_slicer)]

    def __setitem__(self, array_name: str, values: ArrayLike_2D) -> None:
        """
        Set values for a specific Array in this view with an array-like object.
        Shapes of the view and of the array-like object must match.
        :param array_name: the name of the Array for which to set values
        :param values: an array-like object of values with shape matching the view's.
        """
        array_shape = (len(self._axis_slicer), len(self._axis_slicer))

        if not isinstance(values, np.ndarray):
            raise VTypeError("Values must be a 2D array-like object "
                             "(pandas DataFrame or numpy array)")

        elif not values.shape == array_shape:
            raise ShapeError(f"Cannot set values, array-like object {values.shape} should have shape {array_shape}")

        else:
            self._arrays[array_name][np.ix_(self._axis_slicer, self._axis_slicer)] = values


class ViewVData:
    """
    A view of a VData object.
    """

    def __init__(self, parent: 'vdata.VData', time_points_slicer: Slicer, obs_slicer: Slicer, var_slicer: Slicer):
        """
        :param parent: a VData object to build a view of
        :param obs_slicer: the list of observations to view
        :param var_slicer: the list of variables to view
        :param time_points_slicer: the list of time points to view
        """
        generalLogger.debug(u'\u23BE ViewVData creation : start ----------------------------------------------------- ')

        self._parent = parent

        # DataFrame slicers
        # time points -------------------------
        if not isinstance(time_points_slicer, slice):
            # boolean array : extract time point values from parent's time_points dataframe
            if isinstance(time_points_slicer, np.ndarray) and time_points_slicer.dtype == np.bool:
                self._time_points_slicer = self._parent.time_points[time_points_slicer].value.values

            # array of time point values : store values as is
            else:
                self._time_points_slicer = time_points_slicer

        elif time_points_slicer == slice(None, None, None):
            # slice from start to end : take all time points
            self._time_points_slicer = self._parent.time_points.value.values

        else:
            # slice from specific start to end time points : get list of sliced time points
            self._time_points_slicer = slice_to_list(time_points_slicer, self._parent.time_points.value.values)

        generalLogger.debug(f"  1. Time points slicer is : {repr_array(self._time_points_slicer)} "
                            f"({len(self._time_points_slicer)} value{'' if len(self._time_points_slicer) == 1 else 's'}"
                            f" selected)")

        # obs -------------------------
        if not isinstance(obs_slicer, slice):
            if isinstance(obs_slicer, np.ndarray) and obs_slicer.dtype == np.bool:
                # boolean array : extract obs index values from parent's obs TemporalDataFrame
                self._obs_slicer = self._parent.obs.index[obs_slicer]

            else:
                # array of obs index values : store values as is
                self._obs_slicer = obs_slicer

        elif obs_slicer == slice(None, None, None):
            # slice from start to end : take all obs index
            self._obs_slicer = self._parent.obs.index

        else:
            # slice from specific start to end obs index : get list of sliced obs index
            self._obs_slicer = slice_to_list(obs_slicer, self._parent.obs.index)

        generalLogger.debug(f"  2. Obs slicer is : {repr_array(self._obs_slicer)} "
                            f"({len(self._obs_slicer)} value{'' if len(self._obs_slicer) == 1 else 's'}"
                            f" selected)")

        # var -------------------------
        if not isinstance(var_slicer, slice):
            if isinstance(var_slicer, np.ndarray) and var_slicer.dtype == np.bool:
                # boolean array : extract var index values from parent's var DataFrame
                self._var_slicer = self._parent.var.index[var_slicer]

            else:
                # array of var index values : store values as is
                self._var_slicer = var_slicer

        elif var_slicer == slice(None, None, None):
            # slice from start to end : take all var index
            self._var_slicer = self._parent.var.index

        else:
            # slice from specific start to end var index : get list of sliced var index
            self._var_slicer = slice_to_list(var_slicer, self._parent.var.index)

        generalLogger.debug(f"  3. Var slicer is : {repr_array(self._var_slicer)} "
                            f"({len(self._var_slicer)} value{'' if len(self._var_slicer) == 1 else 's'}"
                            f" selected)")

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
        self._layers = ViewVLayerArrayContainer(self._parent.layers, self._time_points_slicer,
                                                  self._obs_slicer, self._var_slicer)
        self._time_points = self._parent.time_points[self._parent.time_points.value.isin(self._time_points_slicer)]

        # TODO
        self._obsm = None
        self._obsp = None
        self._var = self._parent.var.loc[self._var_slicer]
        # TODO
        self._varm = None
        self._varp = None
        self._uns = None

        generalLogger.debug(f"Guessed dimensions are : {self.shape}")

        generalLogger.debug(u'\u23BF ViewVData creation : end ------------------------------------------------------- ')

    def __repr__(self) -> str:
        """
        Description for this view of a Vdata object to print.
        :return: a description of this view
        """
        generalLogger.debug(u'\u23BE ViewVData repr : start --------------------------------------------------------- ')

        if self.is_empty:
            generalLogger.debug('ViewVData is empty.')
            repr_str = f"Empty view of a Vdata object ({self.n_obs} obs x {self.n_var} vars over " \
                       f"{self.n_time_points} time point{'s' if self.n_time_points > 1 else ''})."
        else:
            generalLogger.debug('ViewVData is not empty.')
            repr_str = f"View of a Vdata object with n_obs x n_var = {self.n_obs} x {self.n_var} over " \
                       f"{self.n_time_points} time point{'s' if self.n_time_points > 1 else ''}"

        for attr_name in ["layers", "obs", "var", "time_points", "obsm", "varm", "obsp", "varp", "uns"]:
            attr = getattr(self, attr_name)
            keys = attr.keys() if attr is not None else ()

            if len(keys) > 0:
                repr_str += f"\n\t{attr_name}: {str(list(keys))[1:-1]}"

        generalLogger.debug(u'\u23BF ViewVData repr : end ----------------------------------------------------------- ')

        return repr_str

    def __getitem__(self, index: Union[PreSlicer, Tuple[PreSlicer, PreSlicer], Tuple[PreSlicer, PreSlicer, PreSlicer]])\
            -> 'ViewVData':
        """
        Get a subset of a view of a VData object.
        :param index: A sub-setting index. It can be a single index, a 2-tuple or a 3-tuple of indexes.
        """
        # convert to a 3-tuple
        time_points_slicer, obs_slicer, var_slicer = format_index(index)

        # check time points slicer --------------------------------------------------------------------------
        if isinstance(time_points_slicer, type(Ellipsis)) or time_points_slicer == slice(None, None, None):
            time_points_slicer = self._time_points_slicer

        elif isinstance(time_points_slicer, (int, float, str)):
            if time_points_slicer in self._parent.time_points.value:
                time_points_slicer = np.array([time_points_slicer],
                                              dtype=self._parent.time_points.value.dtype) \
                    if self._time_points_slicer[list(self._parent.time_points.value).index(time_points_slicer)] else []
            else:
                time_points_slicer = []

        else:
            # convert slice to range for following steps
            if isinstance(time_points_slicer, slice):
                time_points_slicer = slice_to_range(time_points_slicer, len(self._time_points_slicer))

            # convert slicer to index's type
            time_points_slicer = np.array(time_points_slicer, dtype=self._parent.time_points.value.dtype)

            # restrict time_points_slicer to elements already selected in this view
            time_points_slicer = np.isin(self._parent.time_points.value, time_points_slicer) & self._time_points_slicer

        # check obs slicer ----------------------------------------------------------------------------------
        if isinstance(obs_slicer, type(Ellipsis)) or obs_slicer == slice(None, None, None):
            obs_slicer = self._obs_slicer

        elif isinstance(obs_slicer, (int, float, str)):
            if obs_slicer in self._parent.obs.index:
                obs_slicer = np.array([obs_slicer], dtype=self._parent.obs.index.dtype) \
                    if self._obs_slicer[list(self._parent.obs.index).index(obs_slicer)] else []
            else:
                obs_slicer = []

        else:
            # convert slice to range for following steps
            if isinstance(obs_slicer, slice):
                obs_slicer = slice_to_range(obs_slicer, len(self._obs_slicer))

            # convert slicer to index's type
            obs_slicer = np.array(obs_slicer, dtype=self._parent.obs.index.dtype)

            # restrict obs_slicer to elements already selected in this view
            obs_slicer = np.isin(self._parent.obs.index, obs_slicer) & self._obs_slicer

        # check var slicer ----------------------------------------------------------------------------------
        if isinstance(var_slicer, type(Ellipsis)) or var_slicer == slice(None, None, None):
            var_slicer = self._var_slicer

        elif isinstance(var_slicer, (int, float, str)):
            if var_slicer in self._parent.var.index:
                var_slicer = np.array([var_slicer], dtype=self._parent.var.index.dtype) \
                    if self._var_slicer[list(self._parent.var.index).index(var_slicer)] else []
            else:
                var_slicer = []

        else:
            # convert slice to range for following steps
            if isinstance(var_slicer, slice):
                var_slicer = slice_to_range(var_slicer, len(self._var_slicer))

            # convert slicer to index's type
            var_slicer = np.array(var_slicer, dtype=self._parent.var.index.dtype)

            # restrict var_slicer to elements already selected in this view
            var_slicer = np.isin(self._parent.var.index, var_slicer) & self._var_slicer

        return ViewVData(self._parent, time_points_slicer, obs_slicer, var_slicer)

    # Shapes -------------------------------------------------------------
    @property
    def is_empty(self) -> bool:
        """
        Is this view of a Vdata object empty ? (no obs or no vars)
        :return: is view empty ?
        """
        return True if self.n_obs == 0 or self.n_var == 0 or self.n_time_points == 0 else False

    @property
    def n_time_points(self) -> int:
        """
        Number of time points in this view of a VData object.
        :return: number of time points in this view
        """
        return len(self._time_points_slicer)

    @property
    def n_obs(self) -> List[int]:
        """
        Number of observations in this view of a VData object.
        :return: number of observations in this view
        """
        return [self.obs.len_index(TP) for TP in self._time_points_slicer]

    @property
    def n_var(self) -> int:
        """
        Number of variables in this view of a VData object.
        :return: number of variables in this view
        """
        return len(self._var_slicer)

    @property
    def shape(self) -> Tuple[int, List[int], int]:
        """
        Shape of this view of a VData object.
        :return: view's shape
        """
        return self.n_time_points, self.n_obs, self.n_var

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
    def var(self) -> pd.DataFrame:
        """
        Get a view on the var DataFrame in this ViewVData.
        :return: a view on the var DataFrame.
        """
        return self._var

    # @var.setter
    # def var(self, df: pd.DataFrame) -> None:
    #     if not isinstance(df, pd.DataFrame):
    #         raise VTypeError("'var' must be a pandas DataFrame.")
    #
    #     elif df.columns != self._parent.var.columns:
    #         raise IncoherenceError("'var' must have the same column names as the original 'var' it replaces.")
    #
    #     elif df.shape[0] != self.n_var:
    #         raise ShapeError(f"'var' has {df.shape[0]} lines, it should have {self.n_var}.")
    #
    #     else:
    #         df.index = self._parent.var[self._var_slicer].index
    #         self._parent.var[self._var_slicer] = df

    @property
    def uns(self) -> Optional[Dict]:
        """
        TODO
        """
        return self._uns

    # TemporalDataFrames -------------------------------------------------
    @property
    def obs(self) -> ViewTemporalDataFrame:
        """
        Get a view on the obs in this ViewVData.
        :return: a view on the obs.
        """
        return self._obs

    # @obs.setter
    # def obs(self, df: Union[TemporalDataFrame, ViewTemporalDataFrame]) -> None:
    #     if not isinstance(df, (TemporalDataFrame, ViewTemporalDataFrame)):
    #         raise VTypeError("'obs' must be a TemporalDataFrame.")
    #
    #     elif df.columns != self._parent.obs.columns:
    #         raise IncoherenceError("'obs' must have the same column names as the original 'obs' it replaces.")
    #
    #     elif df.shape[0] != self.n_obs:
    #         raise ShapeError(f"'obs' has {df.shape[0]} lines, it should have {self.n_obs}.")
    #
    #     else:
    #         df.index = self._parent.obs[self._obs_slicer].index
    #         self._parent.obs[self._obs_slicer] = df

    # Arrays -------------------------------------------------------------
    @property
    def layers(self) -> ViewVLayerArrayContainer:
        """
        Get a view on the layers in this ViewVData.
        :return: a view on the layers.
        """
        return self._layers

    # @layers.setter
    # def layers(self, *_: Any) -> NoReturn:
    #     raise VValueError("Cannot set layers in a view. Use the original VData object.")

    @property
    def obsm(self) -> ViewVAxisArrayContainer:
        """
        Get a view on the obsm in this ViewVData.
        :return: a view on the obsm.
        """
        return self._obsm
        # return ViewVAxisArrayContainer(self._parent.obsm, self._time_points_array_slicer, self._obs_array_slicer)

    # @obsm.setter
    # def obsm(self, *_: Any) -> NoReturn:
    #     raise VValueError("Cannot set obsm in a view. Use the original VData object.")

    @property
    def obsp(self) -> ViewVPairwiseArrayContainer:
        """
        Get a view on the obsp in this ViewVData.
        :return: a view on the obsp.
        """
        return self._obsp
        # return ViewVPairwiseArrayContainer(self._parent.obsp, self._obs_array_slicer)

    # @obsp.setter
    # def obsp(self, *_: Any) -> NoReturn:
    #     raise VValueError("Cannot set obsp in a view. Use the original VData object.")

    @property
    def varm(self) -> ViewVAxisArrayContainer:
        """
        Get a view on the varm in this ViewVData.
        :return: a view on the varm.
        """
        return self._varm
        # return ViewVAxisArrayContainer(self._parent.varm, self._time_points_array_slicer, self._var_array_slicer)

    # @varm.setter
    # def varm(self, *_: Any) -> NoReturn:
    #     raise VValueError("Cannot set varm in a view. Use the original VData object.")

    @property
    def varp(self) -> ViewVPairwiseArrayContainer:
        """
        Get a view on the varp in this ViewVData.
        :return: a view on the varp.
        """
        return self._varp
        # return ViewVPairwiseArrayContainer(self._parent.varp, self._var_array_slicer)

    # @varp.setter
    # def varp(self, *_: Any) -> NoReturn:
    #     raise VValueError("Cannot set varp in a view. Use the original VData object.")

    # aliases ------------------------------------------------------------
    @property
    def cells(self) -> ViewTemporalDataFrame:
        """
        Alias for the obs attribute.
        :return: a view on the obs.
        """
        return self.obs

    # @cells.setter
    # def cells(self, df: Union[TemporalDataFrame, ViewTemporalDataFrame]) -> None:
    #     self.obs = df

    @property
    def genes(self) -> pd.DataFrame:
        """
        Alias for the var attribute.
        :return: a view on the var DataFrame.
        """
        return self.var

    # @genes.setter
    # def genes(self, df: pd.DataFrame) -> None:
    #     self.var = df

    # copy ---------------------------------------------------------------
    def copy(self) -> 'vdata.VData':
        """
        Build an actual VData object from this view.
        """
        return vdata.VData(self.layers.dict_copy(),
                           self.obs.copy(), self.obsm.dict_copy(), self.obsp.dict_copy(),
                           self.var, self.varm.dict_copy(), self.varp.dict_copy(),
                           self.time_points,
                           self.uns,
                           self._parent.dtype)
