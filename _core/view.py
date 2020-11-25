# coding: utf-8
# Created on 11/25/20 4:25 PM
# Author : matteo

# ====================================================
# imports
import pandas as pd
import numpy as np
from typing import Tuple

from . import vdata
from .arrays import VLayersArrays, VAxisArray, VPairwiseArray
from ..NameUtils import Slicer
from .._IO.errors import VTypeError, IncoherenceError


# ====================================================
# code
class ViewVLayersArrays:
    """
    TODO + subsetting of VLayersArrays
    """

    def __init__(self, array: VLayersArrays, obs_slicer: np.ndarray, var_slicer: np.ndarray, time_points_slicer: np.ndarray):
        self._array = array

    def __repr__(self) -> str:
        """
        Description for this view of a VLayersArrays object to print.
        :return: a description of this view
        """
        return f"View of {self._array}"


class ViewVAxisArray:
    """
    TODO + subsetting of VAxisArray
    """

    def __init__(self, array: VAxisArray, axis_slicer: np.ndarray, time_points_slicer: np.ndarray):
        self._array = array

    def __repr__(self) -> str:
        """
        Description for this view of a VAxisArray object to print.
        :return: a description of this view
        """
        return f"View of {self._array}"


class ViewVPairwiseArray:
    """
    TODO + subsetting of VPairwiseArray
    """

    def __init__(self, array: VPairwiseArray, slicer: np.ndarray):
        self._array = array

    def __repr__(self) -> str:
        """
        Description for this view of a VPairwiseArray object to print.
        :return: a description of this view
        """
        return f"View of {self._array}"


class ViewVData:
    """
    TODO
    """

    def __init__(self, parent: 'vdata.VData', obs_slicer: Slicer, var_slicer: Slicer, time_points_slicer: Slicer):
        self._parent = parent
        # DataFrame slicers
        self._obs_slicer = np.array(obs_slicer, dtype=self._parent.obs.index.dtype) if self._parent.obs is not None else obs_slicer
        self._var_slicer = np.array(var_slicer, dtype=self._parent.var.index.dtype) if self._parent.var is not None else var_slicer
        self._time_points_slicer = np.array(time_points_slicer, dtype=self._parent.time_points.index.dtype) if self._parent.time_points is not None else time_points_slicer
        # array slicers
        self._obs_array_slicer = np.where(np.isin(self.obs.index, self._obs_slicer))[0]
        self._var_array_slicer = np.where(np.isin(self.var.index, self._var_slicer))[0]
        self._time_points_array_slicer = np.where(np.isin(self.time_points.index, self.time_points))[0]

        # compute size
        self._n_obs = int(np.sum(self._parent.obs.index.isin(self._obs_slicer)))
        self._n_var = int(np.sum(self._parent.var.index.isin(self._var_slicer)))
        self._n_time_points = int(np.sum(self._parent.time_points.index.isin(self._time_points_slicer)))

    def __repr__(self) -> str:
        """
        Description for this view of a Vdata object to print.
        :return: a description of this view
        """
        return f"View of {self._parent}"

    def __getitem__(self, item) -> 'ViewVData':
        """
        TODO
        """
        pass

    def __setitem__(self, key, value) -> None:
        """
        TODO
        """
        pass

    @property
    def is_empty(self) -> bool:
        """
        Is this view of a Vdata object empty ? (no obs or no vars)
        :return: is view empty ?
        """
        return True if self.n_obs == 0 or self.n_var == 0 or self.n_time_points == 0 else False

    @property
    def n_obs(self) -> int:
        """
        Number of observations in this view of a VData object.
        :return: number of observations in this view
        """
        return self._n_obs

    @property
    def n_var(self) -> int:
        """
        Number of variables in this view of a VData object.
        :return: number of variables in this view
        """
        return self._n_var

    @property
    def n_time_points(self) -> int:
        """
        Number of time points in this view of a VData object.
        :return: number of time points in this view
        """
        return self._n_time_points

    def shape(self) -> Tuple[int, int, int]:
        """
        Shape of this view of a VData object.
        :return: view's shape
        """
        return self.n_time_points, self.n_obs, self.n_var

    # DataFrames ------------------------------------------------------------
    @property
    def obs(self) -> pd.DataFrame:
        return self._parent.obs[self._parent.obs.index.isin(self._obs_slicer)]

    @obs.setter
    def obs(self, df: pd.DataFrame) -> None:
        if not isinstance(df, pd.DataFrame):
            raise VTypeError("'obs' must be a pandas DataFrame.")

        elif df.columns != self._parent.obs.columns:
            raise IncoherenceError("'obs' must have the same column names as the original 'obs' it replaces.")

        else:
            self._parent.obs = df

    @property
    def var(self) -> pd.DataFrame:
        return self._parent.var[self._parent.var.index.isin(self._var_slicer)]

    @var.setter
    def var(self, df: pd.DataFrame) -> None:
        if not isinstance(df, pd.DataFrame):
            raise VTypeError("'var' must be a pandas DataFrame.")

        elif df.columns != self._parent.var.columns:
            raise IncoherenceError("'var' must have the same column names as the original 'var' it replaces.")

        else:
            self._parent.var = df

    @property
    def time_points(self) -> pd.DataFrame:
        return self._parent.time_points[self._parent.time_points.index.isin(self._time_points_slicer)]

    @time_points.setter
    def time_points(self, df: pd.DataFrame) -> None:
        if not isinstance(df, pd.DataFrame):
            raise VTypeError("'time_points' must be a pandas DataFrame.")

        elif df.columns != self._parent.time_points.columns:
            raise IncoherenceError("'time_points' must have the same column names as the original 'time_points' it replaces.")

        else:
            self._parent.time_points = df

    # Arrays ------------------------------------------------------------
    @property
    def layers(self) -> ViewVLayersArrays:
        return ViewVLayersArrays(self._parent.layers, self._obs_array_slicer, self._var_array_slicer, self._time_points_array_slicer)

    # @layers.setter
    # def layers(self, data: Optional[Union[ArrayLike, Dict[Any, ArrayLike]]]) -> None:
    #     if data is None:
    #         self._layers = VLayersArrays(self, None)
    #
    #     else:
    #         if isinstance(data, (np.ndarray, sparse.spmatrix, pd.DataFrame)):
    #             data = {"data": data}
    #
    #         elif not isinstance(data, dict):
    #             raise VTypeError("'layers' should be set with a 3D array-like object (numpy array, scipy sparse matrix) or with a dictionary of them.")
    #
    #         for arr_index, arr in data.items():
    #             if self.n_time_points > 1:
    #                 if not isinstance(arr, (np.ndarray, sparse.spmatrix)):
    #                     raise VTypeError(f"'{arr_index}' array for layers should be a 3D array-like object (numpy array, scipy sparse matrix).")
    #
    #                 elif arr.ndim != 3:
    #                     raise VTypeError(f"'{arr_index}' array for layers should be a 3D array-like object (numpy array, scipy sparse matrix).")
    #
    #             else:
    #                 if not isinstance(arr, (np.ndarray, sparse.spmatrix, pd.DataFrame)):
    #                     raise VTypeError(f"'{arr_index}' array for layers should be a 2D or 3D array-like object (numpy array, scipy sparse matrix, pandas DataFrame).")
    #
    #                 elif arr.ndim not in (2, 3):
    #                     raise VTypeError(f"'{arr_index}' array for layers should be a 2D or 3D array-like object (numpy array, scipy sparse matrix, pandas DataFrame).")
    #
    #                 elif arr.ndim == 2:
    #                     data[arr_index] = self._reshape_to_3D(arr)
    #
    #         self._layers = VLayersArrays(self, data)

    @property
    def obsm(self) -> ViewVAxisArray:
        return ViewVAxisArray(self._parent.obsm, self._obs_array_slicer, self._time_points_array_slicer)

    # @obsm.setter
    # def obsm(self, data: Optional[Dict[Any, ArrayLike_3D]]) -> None:
    #     if data is None:
    #         self._obsm = VAxisArray(self, 'obs', None)
    #
    #     else:
    #         if not isinstance(data, dict):
    #             raise VTypeError("'obsm' should be set with a dictionary of 3D array-like objects (numpy array, scipy sparse matrix).")
    #
    #         for arr_index, arr in data.items():
    #             if self.n_time_points > 1:
    #                 if not isinstance(arr, (np.ndarray, sparse.spmatrix)):
    #                     raise VTypeError(f"'{arr_index}' array for obsm should be a 3D array-like object (numpy array, scipy sparse matrix).")
    #
    #                 elif arr.ndim != 3:
    #                     raise VTypeError(f"'{arr_index}' array for obsm should be a 3D array-like object (numpy array, scipy sparse matrix).")
    #
    #             else:
    #                 if not isinstance(arr, (np.ndarray, sparse.spmatrix, pd.DataFrame)):
    #                     raise VTypeError(f"'{arr_index}' array for obsm should be a 2D or 3D array-like object (numpy array, scipy sparse matrix, pandas DataFrame).")
    #
    #                 elif arr.ndim not in (2, 3):
    #                     raise VTypeError(f"'{arr_index}' array for obsm should be a 2D or 3D array-like object (numpy array, scipy sparse matrix, pandas DataFrame).")
    #
    #                 elif arr.ndim == 2:
    #                     data[arr_index] = self._reshape_to_3D(arr)
    #
    #         self._obsm = VAxisArray(self, 'obs', data)

    @property
    def obsp(self) -> ViewVPairwiseArray:
        return ViewVPairwiseArray(self._parent.obsp, self._obs_array_slicer)

    # @obsp.setter
    # def obsp(self, data: Optional[Dict[Any, ArrayLike_2D]]) -> None:
    #     if data is None:
    #         self._obsp = VPairwiseArray(self, 'obs', None)
    #
    #     else:
    #         if not isinstance(data, dict):
    #             raise VTypeError("'obsp' should be set with a dictionary of 2D array-like objects (numpy array, scipy sparse matrix).")
    #
    #         for arr_index, arr in data.items():
    #             if not isinstance(arr, (np.ndarray, sparse.spmatrix, pd.DataFrame)):
    #                 raise VTypeError(f"'{arr_index}' array for obsp should be a 2D array-like object (numpy array, scipy sparse matrix, pandas DataFrame).")
    #
    #             elif arr.ndim != 2:
    #                 raise VTypeError(f"'{arr_index}' array for obsm should be a 2D array-like object (numpy array, scipy sparse matrix, pandas DataFrame).")
    #
    #         self._obsp = VPairwiseArray(self, 'obs', data)

    @property
    def varm(self) -> ViewVAxisArray:
        return ViewVAxisArray(self._parent.varm, self._var_array_slicer, self._time_points_array_slicer)

    # @varm.setter
    # def varm(self, data: Optional[Dict[Any, ArrayLike_3D]]) -> None:
    #     if data is None:
    #         self._varm = VAxisArray(self, 'var', None)
    #
    #     else:
    #         if not isinstance(data, dict):
    #             raise VTypeError("'varm' should be set with a dictionary of 3D array-like objects (numpy array, scipy sparse matrix).")
    #
    #         for arr_index, arr in data.items():
    #             if self.n_time_points > 1:
    #                 if not isinstance(arr, (np.ndarray, sparse.spmatrix)):
    #                     raise VTypeError(f"'{arr_index}' array for varm should be a 3D array-like object (numpy array, scipy sparse matrix).")
    #
    #                 elif arr.ndim != 3:
    #                     raise VTypeError(f"'{arr_index}' array for varm should be a 3D array-like object (numpy array, scipy sparse matrix).")
    #
    #             else:
    #                 if not isinstance(arr, (np.ndarray, sparse.spmatrix, pd.DataFrame)):
    #                     raise VTypeError(
    #                         f"'{arr_index}' array for varm should be a{' 2D or' if self.n_time_points == 1 else ''} 3D array-like object (numpy array, "
    #                         f"scipy sparse matrix{', pandas DataFrame' if self.n_time_points == 1 else ''}).")
    #
    #                 elif arr.ndim not in (2, 3):
    #                     raise VTypeError(
    #                         f"'{arr_index}' array for varm should be a{' 2D or' if self.n_time_points == 1 else ''} 3D array-like object (numpy array, "
    #                         f"scipy sparse matrix{', pandas DataFrame' if self.n_time_points == 1 else ''}).")
    #
    #                 elif arr.ndim == 2:
    #                     data[arr_index] = self._reshape_to_3D(arr)
    #
    #         self._varm = VAxisArray(self, 'var', data)

    @property
    def varp(self) -> ViewVPairwiseArray:
        return ViewVPairwiseArray(self._parent.varp, self._var_array_slicer)

    # @varp.setter
    # def varp(self, data: Optional[Dict[Any, ArrayLike_2D]]) -> None:
    #     if data is None:
    #         self._varp = VPairwiseArray(self, 'var', None)
    #
    #     else:
    #         if not isinstance(data, dict):
    #             raise VTypeError("'varp' should be set with a dictionary of 2D array-like objects (numpy array, scipy sparse matrix).")
    #
    #         for arr_index, arr in data.items():
    #             if not isinstance(arr, (np.ndarray, sparse.spmatrix, pd.DataFrame)):
    #                 raise VTypeError(f"'{arr_index}' array for varp should be a 2D array-like object (numpy array, scipy sparse matrix, pandas DataFrame).")
    #
    #             elif arr.ndim != 2:
    #                 raise VTypeError(f"'{arr_index}' array for varp should be a 2D array-like object (numpy array, scipy sparse matrix, pandas DataFrame).")
    #
    #         self._varp = VPairwiseArray(self, 'var', data)

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
