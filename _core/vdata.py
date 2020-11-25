# coding: utf-8
# Created on 11/4/20 10:03 AM
# Author : matteo

# ====================================================
# imports
import sys
import os
import h5py
import pandas as pd
import numpy as np
from anndata import AnnData
from scipy import sparse
from pathlib import Path
from typing import Optional, Union, Dict, Tuple, Any

from .arrays import VAxisArray, VPairwiseArray, VLayersArrays
from ..utils import is_in
from ..NameUtils import ArrayLike_3D, ArrayLike_2D, ArrayLike, DTypes, DType, LoggingLevel, LoggingLevels
from .._IO.errors import VTypeError, IncoherenceError, VValueError, ShapeError, VBaseError, VPathError
from .._IO.logger import generalLogger, Tb, original_excepthook
from .._IO.write import write_data


# ====================================================
# code
class VData:
    """
    A VData object stores data points in matrices of observations x variables in the same way as the AnnData object,
    but also accounts for the time information. The 2D matrices in AnnData are replaced by 3D matrices here.
    """
    # TODO : add support for backed data on .h5 file
    def __init__(self,
                 data: Optional[Union[ArrayLike, Dict[Any, ArrayLike], AnnData]] = None,
                 obs: Optional[pd.DataFrame] = None,
                 obsm: Optional[Dict[Any, ArrayLike]] = None,
                 obsp: Optional[Dict[Any, ArrayLike]] = None,
                 var: Optional[pd.DataFrame] = None,
                 varm: Optional[Dict[Any, ArrayLike]] = None,
                 varp: Optional[Dict[Any, ArrayLike]] = None,
                 time_points: Optional[pd.DataFrame] = None,
                 uns: Optional[Dict] = None,
                 dtype: DType = np.float32,
                 log_level: LoggingLevel = "WARNING"):
        # disable traceback messages, except if the loggingLevel is set to DEBUG
        def exception_handler(exception_type, exception, traceback, debug_hook=original_excepthook):
            Tb.trace = traceback
            Tb.exception = exception_type

            if log_level == 'DEBUG':
                if not issubclass(exception_type, VBaseError):
                    self.logger.uncaught_error(exception)
                debug_hook(exception_type, exception, traceback)
            else:
                if not issubclass(exception_type, VBaseError):
                    self.logger.uncaught_error(exception)
                else:
                    print(exception)

        sys.excepthook = exception_handler

        # get logger
        if log_level not in LoggingLevels:
            raise VTypeError(f"Incorrect logging level '{log_level}', should be in {LoggingLevels}")

        else:
            self.logger = generalLogger
            self.logger.set_level(log_level)

        self._dtype = dtype
        self._log_level = log_level

        self._obs = None
        self._var = None
        self._time_points = None
        self._uns: Optional[Dict] = None

        # check formats of arguments
        _layers, _obsm, _obsp, _varm, _varp, df_obs, df_var = self._check_formats(data, obs, obsm, obsp, var, varm, varp, time_points, uns)

        # set number of obs and vars from available data
        self._n_obs, self._n_vars, self._n_time_points = 0, 0, 0

        # if 'layers' is set, get all sizes from there
        if _layers is not None:
            self._n_time_points, self._n_obs, self._n_vars = list(_layers.values())[0].shape

        # otherwise, check other arrays to get the sizes
        else:
            # set obs size
            if self._obs is not None:
                self._n_obs = self._obs.shape[0]
            elif _obsm is not None:
                self._n_obs = list(_obsm.values())[0].shape[0]
            elif _obsp is not None:
                self._n_obs = list(_obsp.values())[0].shape[0]
            elif df_obs is not None:
                self._n_obs = df_obs.shape[0]

            # set var size
            if self._var is not None:
                self._n_vars = self._var.shape[0]
            elif _varm is not None:
                self._n_vars = list(_varm.values())[0].shape[0]
            elif _varp is not None:
                self._n_vars = list(_varp.values())[0].shape[0]
            elif df_var is not None:
                self._n_vars = df_var.shape[0]

            # set time_points size
            if self._time_points is not None:
                self._n_time_points = len(self._time_points)

        # create arrays linked to VData
        self._layers = VLayersArrays(self, data=_layers)
        self._obsm = VAxisArray(self, 'obs', data=_obsm)
        self._obsp = VPairwiseArray(self, 'obs', data=_obsp)
        self._varm = VAxisArray(self, 'var', data=_varm)
        self._varp = VPairwiseArray(self, 'var', data=_varp)

        # finish initializing VData
        self._init_data(df_obs, df_var)

    def __repr__(self) -> str:
        """
        Description for this Vdata object to print.
        :return: a description of this Vdata object
        """
        if self.is_empty:
            repr_str = f"Empty Vdata object ({self.n_obs} obs x {self.n_var} vars over {self.n_time_points} time point{'s' if self.n_time_points > 1 else ''})."
        else:
            repr_str = f"Vdata object with n_obs x n_var = {self.n_obs} x {self.n_var} over {self.n_time_points} time point{'s' if self.n_time_points > 1 else ''}"

        for attr in ["layers", "obs", "var", "time_points", "obsm", "varm", "obsp", "varp", "uns"]:
            keys = getattr(self, attr).keys() if getattr(self, attr) is not None else ()
            if len(keys) > 0:
                repr_str += f"\n    {attr}: {str(list(keys))[1:-1]}"

        return repr_str

    @property
    def is_empty(self) -> bool:
        """
        Is this Vdata object empty ? (no obs or no vars)
        :return: Vdata empty ?
        """
        return True if self.n_obs == 0 or self.n_var == 0 or self.n_time_points == 0 else False

    @property
    def n_obs(self) -> int:
        """
        Number of observations in this VData object. n_obs can be extracted directly from self.obs or from parameters supplied
        during this VData object's creation :
            - nb of observations in the layers
            - nb of observations in obsm
            - nb of observations in obsp
        :return: VData's number of observations
        """
        return self._n_obs

    @property
    def n_var(self) -> int:
        """
        Number of variables in this VData object. n_var can be extracted directly from self.var or from parameters supplied
        during this VData object's creation :
            - nb of variables in the layers
            - nb of variables in varm
            - nb of variables in varp
        :return: VData's number of variables
        """
        return self._n_vars

    @property
    def n_time_points(self) -> int:
        """
        Number of time points in this VData object. n_time_points can be extracted directly from self.time_points or from the
        nb of time points in the layers. If no data was given, a default list of time points was created with integer values.
        :return: VData's number of time points
        """
        return self._n_time_points

    @staticmethod
    def _reshape_to_3D(arr: ArrayLike_2D):
        """
        Reshape a 2D array-like object into a 3D array-like. Pandas DataFrames are first converted into numpy arrays.
        """
        if isinstance(arr, np.ndarray):
            return np.reshape(arr, (1, arr.shape[0], arr.shape[1]))
        elif isinstance(arr, pd.DataFrame):
            return np.reshape(np.array(arr), (1, arr.shape[0], arr.shape[1]))
        else:
            return arr.reshape((1, arr.shape[0], arr.shape[1]))

    @property
    def obs(self) -> pd.DataFrame:
        return self._obs

    @obs.setter
    def obs(self, df: pd.DataFrame) -> None:
        if not isinstance(df, pd.DataFrame):
            raise VTypeError("'obs' must be a pandas DataFrame.")

        elif df.shape[0] != self.n_obs:
            raise ShapeError(f"'obs' has {df.shape[0]} lines, it should have {self.n_obs}.")

        else:
            self._obs = df

    @property
    def obsm(self) -> VAxisArray:
        return self._obsm

    @obsm.setter
    def obsm(self, data: Optional[Dict[Any, ArrayLike_3D]]) -> None:
        if data is None:
            self._obsm = VAxisArray(self, 'obs', None)

        else:
            if not isinstance(data, dict):
                raise VTypeError("'obsm' should be set with a dictionary of 3D array-like objects (numpy array, scipy sparse matrix).")

            for arr_index, arr in data.items():
                if self.n_time_points > 1:
                    if not isinstance(arr, (np.ndarray, sparse.spmatrix)):
                        raise VTypeError(f"'{arr_index}' array for obsm should be a 3D array-like object (numpy array, scipy sparse matrix).")

                    elif arr.ndim != 3:
                        raise VTypeError(f"'{arr_index}' array for obsm should be a 3D array-like object (numpy array, scipy sparse matrix).")

                else:
                    if not isinstance(arr, (np.ndarray, sparse.spmatrix, pd.DataFrame)):
                        raise VTypeError(f"'{arr_index}' array for obsm should be a 2D or 3D array-like object (numpy array, scipy sparse matrix, pandas DataFrame).")

                    elif arr.ndim not in (2, 3):
                        raise VTypeError(f"'{arr_index}' array for obsm should be a 2D or 3D array-like object (numpy array, scipy sparse matrix, pandas DataFrame).")

                    elif arr.ndim == 2:
                        data[arr_index] = self._reshape_to_3D(arr)

            self._obsm = VAxisArray(self, 'obs', data)

    @property
    def obsp(self) -> VPairwiseArray:
        return self._obsp

    @obsp.setter
    def obsp(self, data: Optional[Dict[Any, ArrayLike_2D]]) -> None:
        if data is None:
            self._obsp = VPairwiseArray(self, 'obs', None)

        else:
            if not isinstance(data, dict):
                raise VTypeError("'obsp' should be set with a dictionary of 2D array-like objects (numpy array, scipy sparse matrix).")

            for arr_index, arr in data.items():
                if not isinstance(arr, (np.ndarray, sparse.spmatrix, pd.DataFrame)):
                    raise VTypeError(f"'{arr_index}' array for obsp should be a 2D array-like object (numpy array, scipy sparse matrix, pandas DataFrame).")

                elif arr.ndim != 2:
                    raise VTypeError(f"'{arr_index}' array for obsm should be a 2D array-like object (numpy array, scipy sparse matrix, pandas DataFrame).")

            self._obsp = VPairwiseArray(self, 'obs', data)

    @property
    def var(self) -> pd.DataFrame:
        return self._var

    @var.setter
    def var(self, df: pd.DataFrame) -> None:
        if not isinstance(df, pd.DataFrame):
            raise VTypeError("'var' must be a pandas DataFrame.")

        elif df.shape[0] != self.n_var:
            raise ShapeError(f"'var' has {df.shape[0]} lines, it should have {self.n_var}.")

        else:
            self._var = df

    @property
    def varm(self) -> VAxisArray:
        return self._varm

    @varm.setter
    def varm(self, data: Optional[Dict[Any, ArrayLike_3D]]) -> None:
        if data is None:
            self._varm = VAxisArray(self, 'var', None)

        else:
            if not isinstance(data, dict):
                raise VTypeError("'varm' should be set with a dictionary of 3D array-like objects (numpy array, scipy sparse matrix).")

            for arr_index, arr in data.items():
                if self.n_time_points > 1:
                    if not isinstance(arr, (np.ndarray, sparse.spmatrix)):
                        raise VTypeError(f"'{arr_index}' array for varm should be a 3D array-like object (numpy array, scipy sparse matrix).")

                    elif arr.ndim != 3:
                        raise VTypeError(f"'{arr_index}' array for varm should be a 3D array-like object (numpy array, scipy sparse matrix).")

                else:
                    if not isinstance(arr, (np.ndarray, sparse.spmatrix, pd.DataFrame)):
                        raise VTypeError(
                            f"'{arr_index}' array for varm should be a{' 2D or' if self.n_time_points == 1 else ''} 3D array-like object (numpy array, "
                            f"scipy sparse matrix{', pandas DataFrame' if self.n_time_points == 1 else ''}).")

                    elif arr.ndim not in (2, 3):
                        raise VTypeError(
                            f"'{arr_index}' array for varm should be a{' 2D or' if self.n_time_points == 1 else ''} 3D array-like object (numpy array, "
                            f"scipy sparse matrix{', pandas DataFrame' if self.n_time_points == 1 else ''}).")

                    elif arr.ndim == 2:
                        data[arr_index] = self._reshape_to_3D(arr)

            self._varm = VAxisArray(self, 'var', data)

    @property
    def varp(self) -> VPairwiseArray:
        return self._varp

    @varp.setter
    def varp(self, data: Optional[Dict[Any, ArrayLike_2D]]) -> None:
        if data is None:
            self._varp = VPairwiseArray(self, 'var', None)

        else:
            if not isinstance(data, dict):
                raise VTypeError("'varp' should be set with a dictionary of 2D array-like objects (numpy array, scipy sparse matrix).")

            for arr_index, arr in data.items():
                if not isinstance(arr, (np.ndarray, sparse.spmatrix, pd.DataFrame)):
                    raise VTypeError(f"'{arr_index}' array for varp should be a 2D array-like object (numpy array, scipy sparse matrix, pandas DataFrame).")

                elif arr.ndim != 2:
                    raise VTypeError(f"'{arr_index}' array for varp should be a 2D array-like object (numpy array, scipy sparse matrix, pandas DataFrame).")

            self._varp = VPairwiseArray(self, 'var', data)

    @property
    def time_points(self) -> pd.DataFrame:
        return self._time_points

    @time_points.setter
    def time_points(self, df: pd.DataFrame) -> None:
        if not isinstance(df, pd.DataFrame):
            raise VTypeError("'time points' must be a pandas DataFrame.")

        elif df.shape[0] != self.n_time_points:
            raise ShapeError(f"'time points' has {df.shape[0]} lines, it should have {self.n_time_points}.")

        else:
            self._time_points = df

    @property
    def uns(self) -> Optional[Dict]:
        return self._uns

    @uns.setter
    def uns(self, data: Optional[Dict]) -> None:
        if not isinstance(data, dict):
            raise VTypeError("'uns' must be a dictionary.")

        else:
            self._uns = dict(zip([str(k) for k in data.keys()], data.values()))

    @property
    def layers(self) -> VLayersArrays:
        return self._layers

    @layers.setter
    def layers(self, data: Optional[Union[ArrayLike, Dict[Any, ArrayLike]]]) -> None:
        if data is None:
            self._layers = VLayersArrays(self, None)

        else:
            if isinstance(data, (np.ndarray, sparse.spmatrix, pd.DataFrame)):
                data = {"data": data}

            elif not isinstance(data, dict):
                raise VTypeError("'layers' should be set with a 3D array-like object (numpy array, scipy sparse matrix) or with a dictionary of them.")

            for arr_index, arr in data.items():
                if self.n_time_points > 1:
                    if not isinstance(arr, (np.ndarray, sparse.spmatrix)):
                        raise VTypeError(f"'{arr_index}' array for layers should be a 3D array-like object (numpy array, scipy sparse matrix).")

                    elif arr.ndim != 3:
                        raise VTypeError(f"'{arr_index}' array for layers should be a 3D array-like object (numpy array, scipy sparse matrix).")

                else:
                    if not isinstance(arr, (np.ndarray, sparse.spmatrix, pd.DataFrame)):
                        raise VTypeError(f"'{arr_index}' array for layers should be a 2D or 3D array-like object (numpy array, scipy sparse matrix, pandas DataFrame).")

                    elif arr.ndim not in (2, 3):
                        raise VTypeError(f"'{arr_index}' array for layers should be a 2D or 3D array-like object (numpy array, scipy sparse matrix, pandas DataFrame).")

                    elif arr.ndim == 2:
                        data[arr_index] = self._reshape_to_3D(arr)

            self._layers = VLayersArrays(self, data)

    # aliases ------------------------------------------------------------
    @property
    def cells(self) -> pd.DataFrame:
        return self._obs

    @cells.setter
    def cells(self, df: pd.DataFrame) -> None:
        self.obs = df

    @property
    def genes(self) -> pd.DataFrame:
        return self._var

    @genes.setter
    def genes(self, df: pd.DataFrame) -> None:
        self.var = df
    # --------------------------------------------------------------------

    def shape(self) -> Tuple[int, int, int]:
        """
        Shape of this VData object.
        :return: VData's shape
        """
        return self.n_time_points, self.n_obs, self.n_var

    def _check_formats(self, data: Optional[Union[ArrayLike, Dict[Any, ArrayLike], AnnData]],
                       obs: Optional[pd.DataFrame], obsm: Optional[Dict[Any, ArrayLike]], obsp: Optional[Dict[Any, ArrayLike]],
                       var: Optional[pd.DataFrame], varm: Optional[Dict[Any, ArrayLike]], varp: Optional[Dict[Any, ArrayLike]],
                       time_points: Optional[pd.DataFrame],
                       uns: Optional[Dict]) -> \
            Tuple[Optional[Dict[str, ArrayLike_3D]],
                  Optional[Dict[str, ArrayLike_3D]], Optional[Dict[str, ArrayLike_2D]],
                  Optional[Dict[str, ArrayLike_3D]], Optional[Dict[str, ArrayLike_2D]],
                  Optional[pd.Index], Optional[pd.Index]]:
        """
        Function for checking the types and formats of the parameters supplied to the VData object at creation.
        If the types are not accepted, an error is raised. obsm, obsp, varm, varp and layers are prepared for
        being converted into custom arrays for maintaining coherence with this VData object.
        :return: Arrays in correct format.
        """
        df_obs, df_var = None, None
        layers = None

        # TODO : propagate dtype to obsm, obsp, layers, varm, varp
        # first, check dtype is correct because it will be needed right away
        if self._dtype not in DTypes.keys():
            raise VTypeError(f"Incorrect data type '{self._dtype}', should be in {list(DTypes.keys())}")
        else:
            self._dtype = DTypes[self._dtype]

        # time_points
        if time_points is not None:
            if not isinstance(time_points, pd.DataFrame):
                raise VTypeError("'time points' must be a pandas DataFrame.")
            else:
                time_points = self._check_df_types(time_points)

        nb_time_points = 1 if time_points is None else len(time_points)

        # if an AnnData is being imported, obs, obsm, obsp, var, varm, varp and uns should be None because they will be set from the AnnData
        if isinstance(data, AnnData):
            for attr in ('obs', 'obsm', 'obsp', 'var', 'varm', 'varp', 'uns'):
                if eval(f"{attr} is not None"):
                    raise VValueError(f"'{attr}' should be set to None when importing data from an AnnData.")

            if nb_time_points > 1:
                raise VValueError("Only one time point must be provided when importing data from an AnnData.")

            # import data from AnnData
            if is_in(data.X, list(data.layers.values())):
                layers = dict((key, self._reshape_to_3D(arr)) for key, arr in data.layers.items())

            else:
                layers = dict({"data":  self._reshape_to_3D(data.X)}, **dict((key, self._reshape_to_3D(arr)) for key, arr in data.layers.items()))

            obs, obsm, obsp = data.obs, dict(data.obsm), dict(data.obsp)
            var, varm, varp = data.var, dict(data.varm), dict(data.varp)
            uns = dict(data.uns)

        else:
            # check formats
            # layers
            if data is not None:
                layers = {}

                # data is a pandas DataFrame
                if isinstance(data, pd.DataFrame):
                    if nb_time_points > 1:
                        raise VTypeError("'data' is a 2D pandas DataFrame but more than 1 time point was provided.")

                    df_obs = data.index
                    df_var = data.columns

                    layers = {"data": self._reshape_to_3D(np.array(data, dtype=self._dtype))}

                # data is an array
                elif isinstance(data, (np.ndarray, sparse.spmatrix)):
                    if data.ndim == 2:
                        reshaped_data = self._reshape_to_3D(data)

                        layers = {"data": reshaped_data}

                    elif data.ndim == 3:
                        layers = {"data": data}

                    else:
                        raise ShapeError("'data' must be a 2D or 3D array-like object (numpy array, scipy sparse matrix, pandas DataFrame).")

                else:
                    for key, value in data.items():
                        if not isinstance(value, (np.ndarray, sparse.spmatrix)):
                            raise VTypeError(f"Layer '{key}' must be a 2D or 3D array-like object (numpy array, scipy sparse matrix, pandas DataFrame).")
                        elif value.ndim not in (2, 3):
                            raise VTypeError(f"Layer '{key}' must contain 2D or 3D array-like object (numpy array, scipy sparse matrix, pandas DataFrame).")
                        elif value.ndim == 2:
                            if nb_time_points > 1:
                                raise VTypeError(f"Layer '{key}' must be a 3D array-like object (numpy array, scipy sparse matrix) if providing more than 1 time point.")

                            value = self._reshape_to_3D(value)

                        layers[str(key)] = value

            # obs
            if obs is not None:
                if not isinstance(obs, pd.DataFrame):
                    raise VTypeError("obs must be a pandas DataFrame.")
                else:
                    obs = self._check_df_types(obs)

            # obsm
            if obsm is not None:
                if not isinstance(obsm, dict):
                    raise VTypeError("obsm must be a dictionary.")
                else:
                    for key, value in obsm.items():
                        if not isinstance(value, (np.ndarray, sparse.spmatrix)):
                            raise VTypeError(f"obsm '{key}' must be a 2D or 3D array-like object (numpy array, scipy sparse matrix, pandas DataFrame).")
                        elif value.ndim not in (2, 3):
                            raise VTypeError(f"obsm '{key}' must be a 2D or 3D array-like object (numpy array, scipy sparse matrix, pandas DataFrame).")
                        elif value.ndim == 2:
                            if nb_time_points > 1:
                                raise VTypeError(f"obsm '{key}' must be a 3D array-like object (numpy array, scipy sparse matrix) if providing more than 1 time point.")

                            value = self._reshape_to_3D(value)

                        obsm[str(key)] = value

            # obsp
            if obsp is not None:
                if isinstance(obsp, dict):
                    for key, value in obsp.items():
                        if not isinstance(value, (np.ndarray, sparse.spmatrix, pd.DataFrame)) and value.ndim != 2:
                            raise VTypeError(f"'{key}' object in obsp is not a 2D array-like object (numpy array, scipy sparse matrix, pandas DataFrame).")

                    obsp = dict((str(k), v) for k, v in obsp.items())

                else:
                    raise VTypeError("obsp must be a dictionary of 2D array-like object (numpy array, scipy sparse matrix, pandas DataFrame).")

            # var
            if var is not None:
                if not isinstance(var, pd.DataFrame):
                    raise VTypeError("var must be a pandas DataFrame.")
                else:
                    var = self._check_df_types(var)

            # varm
            if varm is not None:
                if not isinstance(varm, dict):
                    raise VTypeError("varm must be a dictionary.")
                else:
                    for key, value in varm.items():
                        if not isinstance(value, (np.ndarray, sparse.spmatrix)):
                            raise VTypeError(f"varm '{key}' must be a 2D or 3D array-like object (numpy array, scipy sparse matrix, pandas DataFrame).")
                        elif value.ndim not in (2, 3):
                            raise VTypeError(f"varm '{key}' must be a 2D or 3D array-like object (numpy array, scipy sparse matrix, pandas DataFrame).")
                        elif value.ndim == 2:
                            if nb_time_points > 1:
                                raise VTypeError(f"varm '{key}' must be a 3D array-like object (numpy array, scipy sparse matrix) if providing more than 1 time point.")

                            value = self._reshape_to_3D(value)

                        varm[str(key)] = value

            # varp
            if varp is not None:
                if isinstance(varp, dict):
                    for key, value in varp.items():
                        if not isinstance(value, (np.ndarray, sparse.spmatrix, pd.DataFrame)) and value.ndim != 2:
                            raise VTypeError(f"'{key}' object in varp is not a 2D array-like object (numpy array, scipy sparse matrix, pandas DataFrame).")

                    varp = dict((str(k), v) for k, v in varp.items())

                else:
                    raise VTypeError("'varp' must be a dictionary of 2D array-like object (numpy array, scipy sparse matrix, pandas DataFrame).")

            # uns
            if uns is not None and not isinstance(uns, dict):
                raise VTypeError("'uns' must be a dictionary.")

        # if time points are not given, assign default values 0, 1, 2, ...
        if time_points is None:
            if layers is not None:
                time_points = pd.DataFrame(range(list(layers.values())[0].shape[0]))
            elif obsm is not None:
                time_points = pd.DataFrame(range(list(obsm.values())[0].shape[0]))
            elif varm is not None:
                time_points = pd.DataFrame(range(list(varm.values())[0].shape[0]))

        self._obs = obs
        self._var = var
        self._uns = dict(zip([str(k) for k in uns.keys()], uns.values())) if uns is not None else None
        self._time_points = time_points

        return layers, obsm, obsp, varm, varp, df_obs, df_var

    def _check_df_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Function for coercing data types of the columns and of the index in a pandas DataFrame.
        """
        # check index : convert to correct dtype if it is not a string type
        try:
            df.index.astype(self._dtype)
        except TypeError:
            df.index.astype(np.dtype('O'))

        # check columns : convert to correct dtype if it is not a string type
        for col_name in df.columns:
            try:
                df[col_name].astype(self._dtype)
            except ValueError:
                df[col_name].astype(np.dtype('O'))

        return df

    def _init_data(self, df_obs: Optional[pd.Index], df_var: Optional[pd.Index]) -> None:
        """
        Function for finishing the initialization of the VData object. It checks for incoherence in the user-supplied
        arrays and raises an error in case something is wrong.
        :param df_obs: If X was supplied as a pandas DataFrame, index of observations
        :param df_var: If X was supplied as a pandas DataFrame, index of variables
        """
        # check coherence with number of time points in VData
        if self._time_points is not None:
            for attr in ('layers', 'obsm', 'varm'):
                dataset = getattr(self, attr)
                if len(dataset):
                    if len(self._time_points) != dataset.shape[0]:
                        raise IncoherenceError(f"{attr} has {dataset.shape[0]} time points but {len(self._time_points)} {'was' if len(self._time_points) == 1 else 'were'} given.")

        # if data was given as a dataframe, check that obs and data match in row names
        if self._obs is None and df_obs is not None:
            self._obs = pd.DataFrame(index=df_obs)
        elif self._obs is not None and df_obs is not None:
            if not self._obs.index.equals(df_obs):
                raise VValueError("Indexes in dataFrames 'data' and 'obs' do not match.")

        # if data was given as a dataframe, check that var row names match data col names
        if self._var is None and df_var is not None:
            self._var = pd.DataFrame(index=df_var)
        elif self._var is not None and df_var is not None:
            if not self._var.index.equals(df_var):
                raise VValueError("Columns in dataFrame 'data' does not match 'var''s index.")

        # check coherence between layers, obs, var and time points
        if self._layers is not None:
            for layer_name, layer in self._layers.items():
                if layer.shape != (self.n_time_points, self.n_obs, self.n_var):
                    if layer.shape[0] != self._time_points:
                        raise IncoherenceError(f"layer '{layer_name}' has incoherent number of time points {layer.shape[0]}, should be {self.n_time_points}.")
                    elif layer.shape[1] != self.n_obs:
                        raise IncoherenceError(f"layer '{layer_name}' has incoherent number of observations {layer.shape[1]}, should be {self.n_obs}.")
                    else:
                        raise IncoherenceError(f"layer '{layer_name}' has incoherent number of variables {layer.shape[2]}, should be {self.n_var}.")

        # check coherence between obs, obsm and obsp shapes
        if self._obs is not None:
            if self._obsm is not None and self.n_obs != self._obsm.shape[1]:
                raise IncoherenceError(f"obs and obsm have different lengths ({self.n_obs} vs {self._obsm.shape[1]})")

            if self._obsp is not None and self.n_obs != self._obsp.shape[0]:
                raise IncoherenceError(f"obs and obsp have different lengths ({self.n_obs} vs {self._obsp.shape[0]})")

        # check coherence between var, varm, varp shapes
        if self._var is not None:
            if self._varm is not None and self.n_var != self._varm.shape[1]:
                raise IncoherenceError(f"var and varm have different lengths ({self.n_var} vs {self._varm.shape[1]})")

            if self._varp is not None and self.n_var != self._varp.shape[0]:
                raise IncoherenceError(f"var and varp have different lengths ({self.n_var} vs {self._varp.shape[0]})")
    # --------------------------------------------------------------------

    def write(self, file: Union[str, Path]) -> None:
        """
        Save this VData object in HDF5 file format.

        :param file: path to save the VData
        """
        # make sure file is a path
        if not isinstance(file, Path):
            file = Path(file)

        # make sure the path exists
        if not os.path.exists(os.path.dirname(file)):
            os.makedirs(os.path.dirname(file))

        with h5py.File(file, 'w') as save_file:
            # save layers
            write_data(self.layers.data, save_file, 'layers')
            # save obs
            write_data(self.obs, save_file, 'obs')
            write_data(self.obsm.data, save_file, 'obsm')
            write_data(self.obsp.data, save_file, 'obsp')
            # save var
            write_data(self.var, save_file, 'var')
            write_data(self.varm.data, save_file, 'varm')
            write_data(self.varp.data, save_file, 'varp')
            # save time points
            write_data(self.time_points, save_file, 'time_points')
            # save uns
            write_data(self.uns, save_file, 'uns')
            # save descriptive data about the VData object
            write_data(self._dtype, save_file, 'dtype')
            write_data(self._log_level, save_file, 'log_level')

    def write_to_csv(self, directory: Union[str, Path], sep: str = ",", na_rep: str = "",
                     index: bool = True, header: bool = True) -> None:
        """
        Save layers, time_points, obs, obsm, obsp, var, varm and varp to csv files in a directory.
        3D matrices are converted to 2D matrices with an added "time point" column to keep track of time.

        :param directory: path to a directory for saving the matrices
        :param sep: delimiter character
        :param na_rep: string to replace NAs
        :param index: write row names ?
        :param header: Write col names ?
        """
        # make sure directory is a path
        if not isinstance(directory, Path):
            directory = Path(directory)

        # make sure the directory exists and is empty
        if not os.path.exists(directory):
            os.makedirs(directory)
        if len(os.listdir(directory)):
            raise VPathError("The directory is not empty.")

        # save matrices
        self.obs.to_csv(directory / "obs.csv", sep, na_rep, index=index, header=header)
        self.var.to_csv(directory / "var.csv", sep, na_rep, index=index, header=header)
        self.time_points.to_csv(directory / "time_points.csv", sep, na_rep, index=index, header=header)

        for dataset in (self.layers, self.obsm, self.obsp, self.varm, self.varp):
            dataset.to_csv(directory, sep, na_rep, index, header)
