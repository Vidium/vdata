# coding: utf-8
# Created on 11/4/20 10:03 AM
# Author : matteo

# ====================================================
# imports
import sys
import os
import pickle
import pandas as pd
import numpy as np
from scipy import sparse
from pathlib import Path
from typing import Optional, Union, Dict, List, Tuple, Any

from .arrays import VAxisArray, VPairwiseArray, VLayersArrays
from ..NameUtils import ArrayLike_3D, ArrayLike_2D, ArrayLike, DTypes, DType, LoggingLevel, LoggingLevels
from ..IO.errors import VTypeError, NotEnoughDataError, IncoherenceError, VValueError
from ..IO.logger import generalLogger


# ====================================================
# code
class VData:
    """
    A VData object stores data points in matrices of observations x variables in the same way as the AnnData object,
    but also accounts for the time information. The 2D matrices in AnnData are replaced by 3D matrices here.
    """

    def __init__(self,
                 X: Optional[ArrayLike_2D] = None,
                 obs: Optional[pd.DataFrame] = None,
                 obsm: Optional[Dict[Any, ArrayLike]] = None,
                 obsp: Optional[Dict[str, ArrayLike]] = None,
                 var: Optional[pd.DataFrame] = None,
                 varm: Optional[Dict[Any, ArrayLike]] = None,
                 varp: Optional[Dict[str, ArrayLike]] = None,
                 layers: Optional[Dict[Any, ArrayLike]] = None,
                 uns: Optional[Dict] = None,
                 time_points: Optional[Union[List[Union[str, DType]], np.ndarray]] = None,
                 dtype: DType = np.float32,
                 log_level: LoggingLevel = "INFO"):
        # disable traceback messages, except if the loggingLevel is set to DEBUG
        def exception_handler(exception_type, exception, traceback, debug_hook=sys.excepthook):
            if log_level == 'DEBUG':
                debug_hook(exception_type, exception, traceback)
            else:
                print(exception)

        sys.excepthook = exception_handler

        # get logger
        if log_level not in LoggingLevels:
            raise VTypeError(f"Incorrect logging level '{log_level}', should be in {LoggingLevels}")

        else:
            self.logger = generalLogger
            self.logger.set_level(log_level)

        self.dtype = dtype

        self.X = None
        self.obs = None
        self.var = None
        self.uns: Optional[Dict] = None
        self.time_points: Optional[Union[List[DType], np.ndarray]] = None

        # check formats of arguments
        _obsm, _obsp, _varm, _varp, _layers, df_obs, df_var = self._check_formats(X, obs, obsm, obsp, var, varm, varp, layers, uns, time_points)

        # set number of obs and vars from available data
        self._n_obs, self._n_vars = 0, 0

        if self.X is not None:
            self._n_obs, self._n_vars = self.X.shape[1:]
        elif _layers is not None:
            self._n_obs, self._n_vars = list(_layers.values())[0].shape[1:]
        elif self.obs is not None:
            self._n_obs = self.obs.shape[0]
        elif _obsm is not None:
            self._n_obs = list(_obsm.values())[0].shape[0]
        elif _obsp is not None:
            self._n_obs = list(_obsp.values())[0].shape[0]
        elif df_obs is not None:
            self._n_obs = df_obs.shape[0]
        elif self.var is not None:
            self._n_vars = self.var.shape[0]
        elif _varm is not None:
            self._n_vars = list(_varm.values())[0].shape[0]
        elif _varp is not None:
            self._n_vars = list(_varp.values())[0].shape[0]
        elif df_var is not None:
            self._n_vars = df_var.shape[0]

        # create arrays linked to VData
        self.obsm = VAxisArray(self, 'obs', data=_obsm) if _obsm is not None else None
        self.obsp = VPairwiseArray(self, 'obs', data=_obsp) if _obsp is not None else None
        self.varm = VAxisArray(self, 'var', data=_varm) if _varm is not None else None
        self.varp = VPairwiseArray(self, 'var', data=_varp) if _varp is not None else None
        self.layers = VLayersArrays(self, data=_layers) if _layers is not None else None

        # finish initializing VData
        self._init_data(df_obs, df_var)

    def __repr__(self) -> str:
        """
        Description for this Vdata object to print.
        :return: a description of this Vdata object
        """
        if self.is_empty:
            repr_str = f"Empty Vdata object ({self.n_obs} obs x {self.n_var} vars)."
        else:
            repr_str = f"Vdata object with n_obs x n_var = {self.n_obs} x {self.n_var} over {self.n_time_points} time point{'s' if self.n_time_points > 1 else ''}"

            for attr in ["obs", "var", "uns", "obsm", "varm", "layers", "obsp", "varp"]:
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
        return True if self.n_obs == 0 or self.n_var == 0 else False

    @property
    def n_obs(self) -> int:
        """
        Number of observations in this VData object. n_obs can be extracted directly from self.obs or from parameters supplied
        during this VData object's creation :
            - nb of observations in X
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
            - nb of variables in X
            - nb of variables in the layers
            - nb of variables in varm
            - nb of variables in varp
        :return: VData's number of variables
        """
        return self._n_vars

    @property
    def n_time_points(self) -> int:
        """
        Number of time points in this VData object. This is recovered from the supplied list as the time_points argument
        during this VData object's creation. If no list was given, a default list of time points was created.
        :return: VData's number of time points
        """
        return len(self.time_points) if self.time_points is not None else 0

    def shape(self) -> Tuple[int, int, int]:
        """
        Shape of this VData object.
        :return: VData's shape
        """
        return self.n_time_points, self.n_obs, self.n_var

    def _check_formats(self, X: Optional[ArrayLike],
                       obs: Optional[pd.DataFrame], obsm: Optional[Dict[Any, ArrayLike]], obsp: Optional[Dict[Any, ArrayLike]],
                       var: Optional[pd.DataFrame], varm: Optional[Dict[Any, ArrayLike]], varp: Optional[Dict[Any, ArrayLike]],
                       layers: Optional[Dict[Any, ArrayLike]],
                       uns: Optional[Dict],
                       time_points: Optional[Union[List[DType], np.ndarray]]) -> \
            Tuple[Optional[Dict[str, ArrayLike_3D]], Optional[Dict[str, ArrayLike_2D]],
                  Optional[Dict[str, ArrayLike_3D]], Optional[Dict[str, ArrayLike_2D]],
                  Optional[Dict[str, ArrayLike_3D]],
                  Optional[pd.Index], Optional[pd.Index]]:
        """
        Function for checking the types and formats of the parameters supplied to the VData object at creation.
        If the types are not accepted, an error is raised. obsm, obsp, varm, varp and layers are prepared for
        being converted into custom arrays for maintaining coherence with this VData object.
        :return: Arrays in correct format.
        """
        df_obs, df_var = None, None

        # TODO : propagate dtype to obsm, obsp, layers, X, varm, varp
        # first, check dtype is correct because it will be needed right away
        if self.dtype not in DTypes.keys():
            raise VTypeError(f"Incorrect data type '{self.dtype}', should be in {list(DTypes.keys())}")
        else:
            self.dtype = DTypes[self.dtype]

        # time_points
        if time_points is not None:
            if not isinstance(time_points, (list, np.ndarray)) or (isinstance(time_points, np.ndarray) and time_points.ndim > 1):
                raise VTypeError("Time points must be a 1D list or numpy array.")

        nb_time_points = 1 if time_points is None else len(time_points)

        # check formats
        # X
        if X is not None:
            if not isinstance(X, (np.ndarray, sparse.spmatrix, pd.DataFrame)) and X.ndim not in (2, 3):
                raise VTypeError("X must be a 2D or 3D array-like object (numpy array, scipy sparse matrix, pandas DataFrame).")
            elif isinstance(X, pd.DataFrame):
                if nb_time_points > 1:
                    raise VTypeError("X must be a 3D array-like object (numpy array, scipy sparse matrix) if providing more than 1 time point.")

                df_obs = X.index
                df_var = X.columns

                # convert X to array :
                X = np.reshape(np.array(X, dtype=self.dtype), (1, X.shape[0], X.shape[1]))
            elif X.ndim == 2:
                if nb_time_points > 1:
                    raise VTypeError("X must be a 3D array-like object (numpy array, scipy sparse matrix) if providing more than 1 time point.")

                # TODO : reshape as sparse matrix if X is sparse
                X = np.reshape(np.array(X, dtype=self.dtype), (1, X.shape[0], X.shape[1]))

        # obs
        if obs is not None and not isinstance(obs, pd.DataFrame):
            raise VTypeError("obs must be a pandas DataFrame.")

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

                        value = np.reshape(value, (1, value.shape[0], value.shape[1]))

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
        if var is not None and not isinstance(var, pd.DataFrame):
            raise VTypeError("var must be a pandas DataFrame.")

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

                        value = np.reshape(value, (1, value.shape[0], value.shape[1]))

                    varm[str(key)] = value

        # varp
        if varp is not None:
            if isinstance(varp, dict):
                for key, value in varp.items():
                    if not isinstance(value, (np.ndarray, sparse.spmatrix, pd.DataFrame)) and value.ndim != 2:
                        raise VTypeError(f"'{key}' object in varp is not a 2D array-like object (numpy array, scipy sparse matrix, pandas DataFrame).")

                varp = dict((str(k), v) for k, v in varp.items())

            else:
                raise VTypeError("varp must be a dictionary of 2D array-like object (numpy array, scipy sparse matrix, pandas DataFrame).")

        # layers
        if layers is not None:
            if not isinstance(layers, dict):
                raise VTypeError("layers must be a dictionary.")
            else:
                for key, value in layers.items():
                    if not isinstance(value, (np.ndarray, sparse.spmatrix)):
                        raise VTypeError(f"Layer '{key}' must be a 2D or 3D array-like object (numpy array, scipy sparse matrix, pandas DataFrame).")
                    elif value.ndim not in (2, 3):
                        raise VTypeError(f"Layer '{key}' must contain 2D or 3D array-like object (numpy array, scipy sparse matrix, pandas DataFrame).")
                    elif value.ndim == 2:
                        if nb_time_points > 1:
                            raise VTypeError(f"Layer '{key}' must be a 3D array-like object (numpy array, scipy sparse matrix) if providing more than 1 time point.")

                        value = np.reshape(value, (1, value.shape[0], value.shape[1]))

                    layers[str(key)] = value

        # uns
        if uns is not None and not isinstance(uns, dict):
            raise VTypeError("uns must be a dictionary.")

        # if time points are not given, assign default values 0, 1, 2, ...
        if time_points is None:
            if X is not None:
                time_points = range(X.shape[0])
            elif layers is not None:
                time_points = range(list(layers.values())[0].shape[0])
            elif obsm is not None:
                time_points = range(list(obsm.values())[0].shape[0])
            elif varm is not None:
                time_points = range(list(varm.values())[0].shape[0])

        self.X = X
        self.obs = obs
        self.var = var
        self.uns = uns
        self.time_points = time_points

        return obsm, obsp, varm, varp, layers, df_obs, df_var

    def _init_data(self, df_obs: Optional[pd.Index], df_var: Optional[pd.Index]) -> None:
        """
        Function for finishing the initialization of the VData object. It checks for incoherence in the user-supplied
        arrays and raises an error in case something is wrong.
        :param df_obs: If X was supplied as a pandas DataFrame, index of observations
        :param df_var: If X was supplied as a pandas DataFrame, index of variables
        """
        # check coherence with number of time points in VData
        if self.time_points is not None:
            for attr in ('X', 'layers', 'obsm', 'varm'):
                dataset = getattr(self, attr)
                if dataset is not None:
                    if len(self.time_points) != dataset.shape[0]:
                        raise IncoherenceError(f"{attr} has {dataset.shape[0]} time points but only {len(self.time_points)} {'was' if len(self.time_points) == 1 else 'were'} given.")

        # if X was given as a dataframe, check that obs and X match in row names
        if self.obs is None and df_obs is not None:
            self.obs = pd.DataFrame(index=df_obs)
        elif self.obs is not None and df_obs is not None:
            if not self.obs.index.equals(df_obs):
                raise VValueError("Indexes in dataFrames X and obs do not match.")

        # if X was given as a dataframe, check that var row names match X col names
        if self.var is None and df_var is not None:
            self.var = pd.DataFrame(index=df_var)
        elif self.var is not None and df_var is not None:
            if not self.var.index.equals(df_var):
                raise VValueError("Columns in dataFrame X does not match var's index.")

        # if no X but layers were supplied, take first layer as X
        if self.X is None and self.layers is not None:
            self.X = list(self.layers.values())[0]

        # check necessary data was supplied
        if self.X is not None:
            if self.obs is None:
                raise NotEnoughDataError("Missing obs data.")

            elif self.var is None:
                raise NotEnoughDataError("Missing var data.")

            elif self.time_points is None:
                raise NotEnoughDataError("Missing time points data.")

        # todo : could this be removed ? do we really need obs for setting obsm and obsp ?
        if self.obs is None and (self.obsm is not None or self.obsp is not None):
            raise NotEnoughDataError("obs data was not supplied but obsm or obsp were.")

        # todo : could this be removed ? same for var
        if self.var is None and (self.varm is not None or self.varp is not None):
            raise NotEnoughDataError("var data was not supplied but varm or varp were.")

        # check coherence between X, obs, vars and time points
        if self.X is not None:
            if self.X.shape[1] != self.n_obs:
                raise IncoherenceError(f"X has different number of lines than obs ({self.X.shape[1]} vs {self.n_obs}).")

            if self.X.shape[2] != self.n_var:
                raise IncoherenceError(f"X has different number of columns than var ({self.X.shape[2]} vs {self.n_var}).")

        # check coherence between X and layers
        if self.X is not None and self.layers is not None:
            for layer_name, layer in self.layers.items():
                if layer.shape[1:] != (self.n_obs, self.n_var):
                    raise IncoherenceError(f"X and layer '{layer_name}' have different shapes ({layer.shape}) vs ({self.n_obs}, {self.n_var}).")

        # check coherence between obs, obsm and obsp shapes
        if self.obs is not None:
            if self.obsm is not None and self.n_obs != self.obsm.shape[1]:
                raise IncoherenceError(f"obs and obsm have different lengths ({self.n_obs} vs {self.obsm.shape[1]})")

            if self.obsp is not None and self.n_obs != self.obsp.shape[0]:
                raise IncoherenceError(f"obs and obsp have different lengths ({self.n_obs} vs {self.obsp.shape[0]})")

        # check coherence between var, varm, varp shapes
        if self.var is not None:
            if self.varm is not None and self.n_var != self.varm.shape[1]:
                raise IncoherenceError(f"var and varm have different lengths ({self.n_var} vs {self.varm.shape[1]})")

            if self.varp is not None and self.n_var != self.varp.shape[0]:
                raise IncoherenceError(f"var and varp have different lengths ({self.n_var} vs {self.varp.shape[0]})")

    def write(self, file: Union[str, Path]) -> None:
        """
        Save this VData object as pickle object.
        :param file: path to save the VData
        """
        # make sure file is a path
        if not isinstance(file, Path):
            file = Path(file)

        # make sure the path exists
        if not os.path.exists(os.path.dirname(file)):
            raise VValueError(f"The path {os.path.dirname(file)} does not exist.")

        with open(file, 'wb') as save_file:
            pickle.dump(self, save_file, protocol=pickle.HIGHEST_PROTOCOL)
