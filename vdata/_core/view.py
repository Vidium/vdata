# coding: utf-8
# Created on 11/25/20 4:25 PM
# Author : matteo

# ====================================================
# imports
import pandas as pd
import numpy as np
import abc
from abc import ABC
from scipy import sparse
from typing import Tuple, Dict, Union, KeysView, ValuesView, ItemsView, NoReturn, Any, Optional

from . import vdata
from .arrays import VLayersArrays, VAxisArray, VPairwiseArray, VBaseArrayContainer
from ..NameUtils import PreSlicer, Slicer, ArrayLike_3D, ArrayLike_2D, ArrayLike
from ..utils import slice_to_range
from .._IO.errors import VTypeError, IncoherenceError, VValueError, ShapeError


# ====================================================
# code
class ViewBaseArray:
    """
    A base view of a VBaseArrayContainer.
    This class is used to create views on VLayersArrays, VAxisArrays and VPairwiseArrays.
    """

    def __init__(self, arrays: VBaseArrayContainer):
        """
        :param arrays: a VBaseArrayContainer object to build a view on
        """
        self._arrays = arrays

    def __repr__(self) -> str:
        """
        Description for this view of a VBaseArrayContainer object to print.
        :return: a description of this view
        """
        return f"View of {self._arrays}"

    @abc.abstractmethod
    def __getitem__(self, array_name: str) -> ArrayLike:
        """
        Get a specific Array in this view.
        :param array_name: the name of the Array to get
        """
        pass

    @abc.abstractmethod
    def __setitem__(self, array_name: str, values: ArrayLike) -> None:
        """
        Set values for a specific Array in this view with an array-like object.
        Shapes of the view and of the array-like object must match.
        :param array_name: the name of the Array for which to set values
        :param values: an array-like object of values with shape matching the view's.
        """
        pass

    def keys(self) -> Union[Tuple[()], KeysView]:
        """
        Get keys of the VBaseArrayContainer.
        """
        return self._arrays.keys()

    def values(self) -> Union[Tuple[()], ValuesView]:
        """
        Get values of the VBaseArrayContainer.
        """
        return self._arrays.values()

    def items(self) -> Union[Tuple[()], ItemsView]:
        """
        Get items of the VBaseArrayContainer.
        """
        return self._arrays.items()

    def dict_copy(self) -> Dict[str, ArrayLike]:
        """
        Build an actual copy of this Array view in dict format.
        :return: Dictionary of (keys, ArrayLike) in this Array view.
        """
        return dict([(arr, self[arr]) for arr in self._arrays.keys()])


class ViewVLayersArrays(ViewBaseArray):
    """
    A view of a VLayersArrays object.
    """

    def __init__(self, arrays: VLayersArrays, time_points_slicer: np.ndarray,
                 obs_slicer: np.ndarray, var_slicer: np.ndarray):
        """
        :param arrays: a VLayersArrays object to build a view on
        :param obs_slicer: the list of observations to view
        :param var_slicer: the list of variables to view
        :param time_points_slicer: the list of time points to view
        """
        super().__init__(arrays)
        self._obs_slicer = obs_slicer
        self._var_slicer = var_slicer
        self._time_points_slicer = time_points_slicer

    def __getitem__(self, array_name: str) -> ArrayLike_3D:
        """
        Get a specific Array in this view.
        :param array_name: the name of the Array to get
        """
        return self._arrays[array_name][np.ix_(self._time_points_slicer, self._obs_slicer, self._var_slicer)]

    def __setitem__(self, array_name: str, values: ArrayLike_3D) -> None:
        """
        Set values for a specific Array in this view with an array-like object.
        Shapes of the view and of the array-like object must match.
        :param array_name: the name of the Array for which to set values
        :param values: an array-like object of values with shape matching the view's.
        """
        if not isinstance(values, (np.ndarray, sparse.spmatrix)):
            raise VTypeError("Values must be a 3D array-like object (numpy array or scipy sparse matrix)")

        elif not values.shape == (len(self._time_points_slicer), len(self._obs_slicer), len(self._var_slicer)):
            raise ShapeError(f"Cannot set values, array-like object {values.shape} should have shape "
                             f"({len(self._time_points_slicer)}, {len(self._obs_slicer)}, {len(self._var_slicer)})")

        else:
            self._arrays[array_name][np.ix_(self._time_points_slicer, self._obs_slicer, self._var_slicer)] = values


class ViewVAxisArray(ViewBaseArray):
    """
    A view of a VAxisArray object.
    """

    def __init__(self, arrays: VAxisArray, time_points_slicer: np.ndarray, axis_slicer: np.ndarray):
        """
        :param arrays: a VAxisArray object to build a view on
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

        if not isinstance(values, (np.ndarray, sparse.spmatrix)):
            raise VTypeError("Values must be a 3D array-like object (numpy array or scipy sparse matrix)")

        elif not values.shape == array_shape:
            raise ShapeError(f"Cannot set values, array-like object {values.shape} should have shape {array_shape}")

        else:
            self._arrays[array_name][np.ix_(self._time_points_slicer, self._axis_slicer)] = values


class ViewVPairwiseArray(ViewBaseArray):
    """
    A view of a VPairwiseArray object.
    """

    def __init__(self, arrays: VPairwiseArray, axis_slicer: np.ndarray):
        """
        :param arrays: a VPairwiseArray object to build a view on
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

        if not isinstance(values, (np.ndarray, sparse.spmatrix)):
            raise VTypeError("Values must be a 2D array-like object "
                             "(pandas DataFrame, numpy array or scipy sparse matrix)")

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
        self._parent = parent
        # DataFrame slicers
        # time points -------------------------
        if not isinstance(time_points_slicer, slice):
            if isinstance(time_points_slicer, np.ndarray) and time_points_slicer.dtype == np.bool:
                self._time_points_slicer = time_points_slicer
            else:
                time_points_slicer = np.array(time_points_slicer, dtype=self._parent.time_points.index.dtype)
                self._time_points_slicer = np.isin(self._parent.time_points.index, time_points_slicer)
        elif time_points_slicer == slice(None, None, None):
            self._time_points_slicer = np.array([True] * self._parent.n_time_points)
        else:
            time_points_slicer = np.array(slice_to_range(time_points_slicer, len(self._parent.time_points)),
                                          dtype=self._parent.time_points.index.dtype)
            self._time_points_slicer = np.isin(self._parent.time_points.index, time_points_slicer)

        # obs -------------------------
        if not isinstance(obs_slicer, slice):
            if isinstance(obs_slicer, np.ndarray) and obs_slicer.dtype == np.bool:
                self._obs_slicer = obs_slicer
            else:
                obs_slicer = np.array(obs_slicer, dtype=self._parent.obs.index.dtype)
                self._obs_slicer = np.isin(self._parent.obs.index, obs_slicer)
        elif obs_slicer == slice(None, None, None):
            self._obs_slicer = np.array([True] * self._parent.n_obs)
        else:
            obs_slicer = np.array(slice_to_range(obs_slicer, len(self._parent.obs)), dtype=self._parent.obs.index.dtype)
            self._obs_slicer = np.isin(self._parent.obs.index, obs_slicer)

        # var -------------------------
        if not isinstance(var_slicer, slice):
            if isinstance(var_slicer, np.ndarray) and var_slicer.dtype == np.bool:
                self._var_slicer = var_slicer
            else:
                var_slicer = np.array(var_slicer, dtype=self._parent.var.index.dtype)
                self._var_slicer = np.isin(self._parent.var.index, var_slicer)
        elif var_slicer == slice(None, None, None):
            self._var_slicer = np.array([True] * self._parent.n_var)
        else:
            var_slicer = np.array(slice_to_range(var_slicer, len(self._parent.var)), dtype=self._parent.var.index.dtype)
            self._var_slicer = np.isin(self._parent.var.index, var_slicer)

        # array slicers
        self._time_points_array_slicer = np.where(self._time_points_slicer)[0]
        self._obs_array_slicer = np.where(self._obs_slicer)[0]
        self._var_array_slicer = np.where(self._var_slicer)[0]

    def __repr__(self) -> str:
        """
        Description for this view of a Vdata object to print.
        :return: a description of this view
        """
        if self.is_empty:
            repr_str = f"Empty view of a Vdata object ({self.n_obs} obs x {self.n_var} vars over " \
                       f"{self.n_time_points} time point{'s' if self.n_time_points > 1 else ''})."
        else:
            repr_str = f"View of a Vdata object with n_obs x n_var = {self.n_obs} x {self.n_var} over " \
                       f"{self.n_time_points} time point{'s' if self.n_time_points > 1 else ''}"

        for attr in ["layers", "obs", "var", "time_points", "obsm", "varm", "obsp", "varp", "uns"]:
            keys = getattr(self, attr).keys() if getattr(self, attr) is not None else ()
            if len(keys) > 0:
                repr_str += f"\n    {attr}: {str(list(keys))[1:-1]}"

        return repr_str


    def __getitem__(self, index: Union[PreSlicer, Tuple[PreSlicer, PreSlicer], Tuple[PreSlicer, PreSlicer, PreSlicer]])\
            -> 'ViewVData':
        """
        Get a subset of a view of a VData object.
        :param index: A sub-setting index. It can be a single index, a 2-tuple or a 3-tuple of indexes.
        """
        # convert to a 3-tuple
        if not isinstance(index, tuple):
            index = (index, ..., ...)

        elif len(index) == 2:
            index = (index[0], index[1], ...)

        # get slicers
        time_points_slicer = index[0]
        obs_slicer = index[1]
        var_slicer = index[2]

        # check time points slicer --------------------------------------------------------------------------
        if isinstance(time_points_slicer, type(Ellipsis)) or time_points_slicer == slice(None, None, None):
            time_points_slicer = self._time_points_slicer

        elif isinstance(time_points_slicer, (int, float, str)):
            if time_points_slicer in self._parent.time_points.index:
                time_points_slicer = np.array([time_points_slicer],
                                              dtype=self._parent.time_points.index.dtype) \
                    if self._time_points_slicer[list(self._parent.time_points.index).index(time_points_slicer)] else []
            else:
                time_points_slicer = []

        else:
            # convert slice to range for following steps
            if isinstance(time_points_slicer, slice):
                time_points_slicer = slice_to_range(time_points_slicer, len(self._time_points_slicer))

            # convert slicer to index's type
            time_points_slicer = np.array(time_points_slicer, dtype=self._parent.time_points.index.dtype)

            # restrict time_points_slicer to elements already selected in this view
            time_points_slicer = np.isin(self._parent.time_points.index, time_points_slicer) & self._time_points_slicer

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
        return int(np.sum(self._time_points_slicer))

    @property
    def n_obs(self) -> int:
        """
        Number of observations in this view of a VData object.
        :return: number of observations in this view
        """
        return int(np.sum(self._obs_slicer))

    @property
    def n_var(self) -> int:
        """
        Number of variables in this view of a VData object.
        :return: number of variables in this view
        """
        return int(np.sum(self._var_slicer))

    def shape(self) -> Tuple[int, int, int]:
        """
        Shape of this view of a VData object.
        :return: view's shape
        """
        return self.n_time_points, self.n_obs, self.n_var

    # DataFrames ---------------------------------------------------------
    @property
    def time_points(self) -> pd.DataFrame:
        return self._parent.time_points[self._time_points_slicer]

    @time_points.setter
    def time_points(self, df: pd.DataFrame) -> None:
        if not isinstance(df, pd.DataFrame):
            raise VTypeError("'time_points' must be a pandas DataFrame.")

        elif df.columns != self._parent.time_points.columns:
            raise IncoherenceError("'time_points' must have the same column names as the original 'time_points' "
                                   "it replaces.")

        elif df.shape[0] != self.n_time_points:
            raise ShapeError(f"'time_points' has {df.shape[0]} lines, it should have {self.n_time_points}.")

        else:
            df.index = self._parent.time_points[self._time_points_slicer].index
            self._parent.time_points[self._time_points_slicer] = df

    @property
    def obs(self) -> pd.DataFrame:

        return self._parent.obs[self._obs_slicer]

    @obs.setter
    def obs(self, df: pd.DataFrame) -> None:
        if not isinstance(df, pd.DataFrame):
            raise VTypeError("'obs' must be a pandas DataFrame.")

        elif df.columns != self._parent.obs.columns:
            raise IncoherenceError("'obs' must have the same column names as the original 'obs' it replaces.")

        elif df.shape[0] != self.n_obs:
            raise ShapeError(f"'obs' has {df.shape[0]} lines, it should have {self.n_obs}.")

        else:
            df.index = self._parent.obs[self._obs_slicer].index
            self._parent.obs[self._obs_slicer] = df

    @property
    def var(self) -> pd.DataFrame:
        return self._parent.var[self._var_slicer]

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
    def uns(self) -> Optional[Dict]:
        return self._parent.uns

    # Arrays -------------------------------------------------------------
    @property
    def layers(self) -> ViewVLayersArrays:
        return ViewVLayersArrays(self._parent.layers, self._time_points_array_slicer,
                                 self._obs_array_slicer, self._var_array_slicer)

    @layers.setter
    def layers(self, *_: Any) -> NoReturn:
        raise VValueError("Cannot set layers in a view. Use the original VData object.")

    @property
    def obsm(self) -> ViewVAxisArray:
        return ViewVAxisArray(self._parent.obsm, self._time_points_array_slicer, self._obs_array_slicer)

    @obsm.setter
    def obsm(self, *_: Any) -> NoReturn:
        raise VValueError("Cannot set obsm in a view. Use the original VData object.")

    @property
    def obsp(self) -> ViewVPairwiseArray:
        return ViewVPairwiseArray(self._parent.obsp, self._obs_array_slicer)

    @obsp.setter
    def obsp(self, *_: Any) -> NoReturn:
        raise VValueError("Cannot set obsp in a view. Use the original VData object.")

    @property
    def varm(self) -> ViewVAxisArray:
        return ViewVAxisArray(self._parent.varm, self._time_points_array_slicer, self._var_array_slicer)

    @varm.setter
    def varm(self, *_: Any) -> NoReturn:
        raise VValueError("Cannot set varm in a view. Use the original VData object.")

    @property
    def varp(self) -> ViewVPairwiseArray:
        return ViewVPairwiseArray(self._parent.varp, self._var_array_slicer)

    @varp.setter
    def varp(self, *_: Any) -> NoReturn:
        raise VValueError("Cannot set varp in a view. Use the original VData object.")

    # aliases ------------------------------------------------------------
    @property
    def cells(self) -> pd.DataFrame:
        return self.obs

    @cells.setter
    def cells(self, df: pd.DataFrame) -> None:
        self.obs = df

    @property
    def genes(self) -> pd.DataFrame:
        return self.var

    @genes.setter
    def genes(self, df: pd.DataFrame) -> None:
        self.var = df

    # copy ---------------------------------------------------------------
    def copy(self) -> 'vdata.VData':
        """
        Build an actual VData object from this view.
        """
        return vdata.VData(self.layers.dict_copy(),
                           self.obs, self.obsm.dict_copy(), self.obsp.dict_copy(),
                           self.var, self.varm.dict_copy(), self.varp.dict_copy(),
                           self.time_points,
                           self.uns,
                           self._parent.dtype, self._parent.log_level)
