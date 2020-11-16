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
from anndata import AnnData
from scipy import sparse
from pathlib import Path
from typing import Optional, Union, Dict, List, Tuple, Any

from .arrays import VAxisArray, VPairwiseArray, VLayersArrays
from ..utils import is_in
from ..NameUtils import ArrayLike_3D, ArrayLike_2D, ArrayLike, DTypes, DType, LoggingLevel, LoggingLevels
from ..IO.errors import VTypeError, IncoherenceError, VValueError, ShapeError
from ..IO.logger import generalLogger


# ====================================================
# code
class VData:
    """
    A VData object stores data points in matrices of observations x variables in the same way as the AnnData object,
    but also accounts for the time information. The 2D matrices in AnnData are replaced by 3D matrices here.
    """

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

        self.obs = None
        self.var = None
        self.time_points = None
        self.uns: Optional[Dict] = None

        # check formats of arguments
        _layers, _obsm, _obsp, _varm, _varp, df_obs, df_var = self._check_formats(data, obs, obsm, obsp, var, varm, varp, time_points, uns)

        # set number of obs and vars from available data
        self._n_obs, self._n_vars, self._n_time_points = 0, 0, 0

        if _layers is not None:
            self._n_time_points, self._n_obs, self._n_vars = list(_layers.values())[0].shape
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
        elif self.time_points is not None:
            self._n_time_points = len(self.time_points)

        # create arrays linked to VData
        self.layers = VLayersArrays(self, data=_layers)
        self.obsm = VAxisArray(self, 'obs', data=_obsm)
        self.obsp = VPairwiseArray(self, 'obs', data=_obsp)
        self.varm = VAxisArray(self, 'var', data=_varm)
        self.varp = VPairwiseArray(self, 'var', data=_varp)

        # finish initializing VData
        self._init_data(df_obs, df_var)

        # create aliases
        self.cells = self.obs
        self.genes = self.var

    def __repr__(self) -> str:
        """
        Description for this Vdata object to print.
        :return: a description of this Vdata object
        """
        if self.is_empty:
            repr_str = f"Empty Vdata object ({self.n_obs} obs x {self.n_var} vars)."
        else:
            repr_str = f"Vdata object with n_obs x n_var = {self.n_obs} x {self.n_var} over {self.n_time_points} time point{'s' if self.n_time_points > 1 else ''}"

            for attr in ["layers", "obs", "var", "time_points", "uns", "obsm", "varm", "obsp", "varp"]:
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
        if self.dtype not in DTypes.keys():
            raise VTypeError(f"Incorrect data type '{self.dtype}', should be in {list(DTypes.keys())}")
        else:
            self.dtype = DTypes[self.dtype]

        # time_points
        if time_points is not None and not isinstance(time_points, pd.DataFrame):
            raise VTypeError("'time points' must be a pandas DataFrame.")

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
                layers = dict(data.layers.values())

            else:
                layers = dict({"data": data.X}, **dict(data.layers.values()))

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

                    layers = {"data": np.reshape(np.array(data, dtype=self.dtype), (1, data.shape[0], data.shape[1]))}

                # data is an array
                elif isinstance(data, (np.ndarray, sparse.spmatrix)):
                    if data.ndim == 2:
                        if isinstance(data, np.ndarray):
                            reshaped_data = np.reshape(data, (1, data.shape[0], data.shape[1]))
                        else:
                            reshaped_data = data.reshape((1, data.shape[0], data.shape[1]))

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

                            if isinstance(data, np.ndarray):
                                value = np.reshape(value, (1, value.shape[0], value.shape[1]))
                            else:
                                value = value.reshape((1, value.shape[0], value.shape[1]))

                        layers[str(key)] = value

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

            # uns
            if uns is not None and not isinstance(uns, dict):
                raise VTypeError("uns must be a dictionary.")

        # if time points are not given, assign default values 0, 1, 2, ...
        if time_points is None:
            if layers is not None:
                time_points = range(list(layers.values())[0].shape[0])
            elif obsm is not None:
                time_points = range(list(obsm.values())[0].shape[0])
            elif varm is not None:
                time_points = range(list(varm.values())[0].shape[0])

        self.obs = obs
        self.var = var
        self.uns = uns
        self.time_points = time_points

        return layers, obsm, obsp, varm, varp, df_obs, df_var

    def _init_data(self, df_obs: Optional[pd.Index], df_var: Optional[pd.Index]) -> None:
        """
        Function for finishing the initialization of the VData object. It checks for incoherence in the user-supplied
        arrays and raises an error in case something is wrong.
        :param df_obs: If X was supplied as a pandas DataFrame, index of observations
        :param df_var: If X was supplied as a pandas DataFrame, index of variables
        """
        # check coherence with number of time points in VData
        if self.time_points is not None:
            for attr in ('layers', 'obsm', 'varm'):
                dataset = getattr(self, attr)
                if dataset is not None:
                    if len(self.time_points) != dataset.shape[0]:
                        raise IncoherenceError(f"{attr} has {dataset.shape[0]} time points but only {len(self.time_points)} {'was' if len(self.time_points) == 1 else 'were'} given.")

        # if data was given as a dataframe, check that obs and data match in row names
        if self.obs is None and df_obs is not None:
            self.obs = pd.DataFrame(index=df_obs)
        elif self.obs is not None and df_obs is not None:
            if not self.obs.index.equals(df_obs):
                raise VValueError("Indexes in dataFrames 'data' and 'obs' do not match.")

        # if data was given as a dataframe, check that var row names match data col names
        if self.var is None and df_var is not None:
            self.var = pd.DataFrame(index=df_var)
        elif self.var is not None and df_var is not None:
            if not self.var.index.equals(df_var):
                raise VValueError("Columns in dataFrame 'data' does not match 'var''s index.")

        # TODO : do we ALWAYS want to have obs/var for setting obsm/varm and obsp/varp ?
        # if self.obs is None and (self.obsm is not None or self.obsp is not None):
        #     raise NotEnoughDataError("obs data was not supplied but obsm or obsp were.")
        #
        # if self.var is None and (self.varm is not None or self.varp is not None):
        #     raise NotEnoughDataError("var data was not supplied but varm or varp were.")

        # check coherence between layers, obs, var and time points
        if self.layers is not None:
            for layer_name, layer in self.layers.items():
                if layer.shape != (self.n_time_points, self.n_obs, self.n_var):
                    if layer.shape[0] != self.time_points:
                        raise IncoherenceError(f"layer '{layer_name}' has incoherent number of time points {layer.shape[0]}, should be {self.n_time_points}.")
                    elif layer.shape[1] != self.n_obs:
                        raise IncoherenceError(f"layer '{layer_name}' has incoherent number of observations {layer.shape[1]}, should be {self.n_obs}.")
                    else:
                        raise IncoherenceError(f"layer '{layer_name}' has incoherent number of variables {layer.shape[2]}, should be {self.n_var}.")

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
