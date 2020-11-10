# coding: utf-8
# Created on 11/4/20 10:03 AM
# Author : matteo

# ====================================================
# imports
import pandas as pd
import numpy as np
from scipy import sparse
from typing import Optional, Union, Dict, List, Tuple, Any

from .arrays import VAxisArray, VPairwiseArray, VLayersArrays
from .NameUtils import ArrayLike_3D, ArrayLike_2D, DTypes, DType, LoggingLevel
from .IO.errors import VTypeError, NotEnoughDataError, IncoherenceError, VValueError
from .IO.logger import generalLogger


# ====================================================
# code
class VData:
    """"""

    def __init__(self,
                 X: Optional[ArrayLike_2D] = None,
                 obs: Optional[pd.DataFrame] = None,
                 obsm: Optional[Dict[Any, ArrayLike_2D]] = None,
                 obsp: Optional[ArrayLike_2D] = None,
                 var: Optional[pd.DataFrame] = None,
                 varm: Optional[Dict[Any, ArrayLike_2D]] = None,
                 varp: Optional[ArrayLike_2D] = None,
                 layers: Optional[Dict[Any, ArrayLike_2D]] = None,
                 uns: Optional[Dict] = None,
                 time_points: Optional[Union[List[DType], np.ndarray]] = None,
                 dtype: DType = np.float32,
                 log_level: LoggingLevel = "INFO"):
        # get logger
        if log_level not in LoggingLevel.__args__:
            raise VTypeError(f"Incorrect logging level '{log_level}', should be in {LoggingLevel.__args__}")

        else:
            self.logger = generalLogger.re_init(log_level)

        self.dtype = dtype

        self._n_obs, self._n_vars = 0, 0

        # check formats of arguments
        self.X, self.obs, self.obsm, self.obsp, self.var, self.varm, self.varp, \
            self.layers, self.uns, self.time_points, df_obs, df_var = self._check_formats(X, obs, obsm, obsp, var, varm, varp, layers, uns, time_points)

        # initialize the VData object
        self._init_data(df_obs, df_var)

    def __repr__(self):
        """
        Description for this Vdata object to print.
        :return: a description of this Vdata object
        """
        if self.is_empty:
            repr_str = f"Empty Vdata object ({self.n_obs} obs x {self.n_vars} vars)."
        else:
            repr_str = f"Vdata object with n_obs x n_var = {self.n_obs} x {self.n_vars} over {self.n_time_points} time points"

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
        return True if self.n_obs == 0 or self.n_vars == 0 else False

    @property
    def n_obs(self) -> int:
        return self._n_obs

    @property
    def n_vars(self) -> int:
        return self._n_vars

    @property
    def n_time_points(self) -> int:
        return len(self.time_points) if self.time_points is not None else 0

    def _check_formats(self, X: Optional[ArrayLike_2D],
                       obs: Optional[pd.DataFrame], obsm: Optional[Dict[Any, ArrayLike_2D]], obsp: Optional[ArrayLike_2D],
                       var: Optional[pd.DataFrame], varm: Optional[Dict[Any, ArrayLike_2D]], varp: Optional[ArrayLike_2D],
                       layers: Optional[Dict[Any, ArrayLike_2D]],
                       uns: Optional[Dict],
                       time_points: Optional[Union[List[DType], np.ndarray]]) -> \
            Tuple[Optional[ArrayLike_3D],
                  Optional[pd.DataFrame], Optional[VAxisArray], Optional[VPairwiseArray],
                  Optional[pd.DataFrame], Optional[VAxisArray], Optional[VPairwiseArray],
                  Optional[VLayersArrays],
                  Optional[Dict],
                  Optional[Union[List[DType], np.ndarray]],
                  Optional[pd.Index], Optional[pd.Index]]:
        """
        Function for checking the types and formats of the parameters supplied to the VData object at creation.
        If the types are not accepted, an error is raised. obsm, obsp, varm, varp and layers are converted into
        custom arrays for maintaining coherence with this VData object.
        :return: Arrays in correct format.
        """
        df_obs, df_var = None, None
        _obsm, _obsp = None, None
        _varm, _varp = None, None
        _layers = None

        # first, check dtype is correct because it will be needed right away
        if self.dtype not in DTypes.keys():
            raise VTypeError(f"Incorrect data type '{self.dtype}', should be in {list(DTypes.keys())}")
        else:
            self.dtype = DTypes[self.dtype]

        # check formats
        # X
        if X is not None:
            if not isinstance(X, (np.ndarray, sparse.spmatrix, pd.DataFrame)) and X.ndim not in (2, 3):
                raise VTypeError("X must be a 2D or 3D array-like object (numpy array, scipy sparse matrix, pandas DataFrame).")
            elif isinstance(X, pd.DataFrame):
                df_obs = X.index
                df_var = X.columns

                # convert X to array :
                X = np.reshape(np.array(X), (1, X.shape[0], X.shape[1]))
            elif X.ndim == 2:
                X = np.reshape(X, (X.shape[0], X.shape[1], 1))

        # obs
        if obs is not None and not isinstance(obs, pd.DataFrame):
            raise VTypeError("obs must be a pandas DataFrame.")

        # obsm
        if obsm is not None:
            if not isinstance(obsm, dict):
                raise VTypeError("obsm must be a dictionary.")
            else:
                _obsm_data = {}
                for key, value in obsm.items():
                    if not isinstance(value, (np.ndarray, sparse.spmatrix)):
                        raise VTypeError("obsm must contain 2D or 3D array-like object (numpy array, scipy sparse matrix, pandas DataFrame).")
                    elif value.ndim not in (2, 3):
                        raise VTypeError("obsm must contain 2D or 3D array-like object (numpy array, scipy sparse matrix, pandas DataFrame).")
                    elif value.ndim == 2:
                        value = np.reshape(value, (1, value.shape[0], value.shape[1]))

                    _obsm_data[str(key)] = value

                _obsm = VAxisArray(self, 'obs', data=_obsm_data)

        # obsp
        if obsp is not None:
            if not isinstance(obsp, (np.ndarray, sparse.spmatrix, pd.DataFrame)) and obsp.ndim != 2:
                raise VTypeError("obsp must be a 2D array-like object (numpy array, scipy sparse matrix, pandas DataFrame).")
            else:
                _obsp = VPairwiseArray(self, 'obs', obsp)

        # var
        if var is not None and not isinstance(var, pd.DataFrame):
            raise VTypeError("var must be a pandas DataFrame.")

        # varm
        if varm is not None:
            if not isinstance(varm, dict):
                raise VTypeError("varm must be a dictionary.")
            else:
                _varm_data = {}
                for key, value in varm.items():
                    if not isinstance(value, (np.ndarray, sparse.spmatrix)):
                        raise VTypeError("varm must contain 2D or 3D array-like object (numpy array, scipy sparse matrix, pandas DataFrame).")
                    elif value.ndim not in (2, 3):
                        raise VTypeError("varm must contain 2D or 3D array-like object (numpy array, scipy sparse matrix, pandas DataFrame).")
                    elif value.ndim == 2:
                        value = np.reshape(value, (1, value.shape[0], value.shape[1]))

                    _varm_data[str(key)] = value

                _varm = VAxisArray(self, 'var', data=_varm_data)

        # varp
        if varp is not None:
            if not isinstance(varp, (np.ndarray, sparse.spmatrix, pd.DataFrame)) and varp.ndim != 2:
                raise VTypeError("varp must be a 2D array-like object (numpy array, scipy sparse matrix, pandas DataFrame).")
            else:
                _varp = VPairwiseArray(self, 'var', data=varp)

        # layers
        if layers is not None:
            if not isinstance(layers, dict):
                raise VTypeError("layers must be a dictionary.")
            else:
                _layers_data = {}
                for key, value in layers.items():
                    if not isinstance(value, (np.ndarray, sparse.spmatrix)):
                        raise VTypeError("layers must contain 2D or 3D array-like object (numpy array, scipy sparse matrix, pandas DataFrame).")
                    elif value.ndim not in (2, 3):
                        raise VTypeError("layers must contain 2D or 3D array-like object (numpy array, scipy sparse matrix, pandas DataFrame).")
                    elif value.ndim == 2:
                        value = np.reshape(value, (1, value.shape[0], value.shape[1]))

                    _layers_data[str(key)] = value

                _layers = VLayersArrays(self, data=_layers_data)

        # uns
        if uns is not None and not isinstance(uns, dict):
            raise VTypeError("uns must be a dictionary.")

        # time_points
        if time_points is not None and not isinstance(time_points, (list, np.ndarray)):
            raise VTypeError("Time points must be a list or a numpy array.")

        return X, obs, _obsm, _obsp, var, _varm, _varp, _layers, uns, time_points, df_obs, df_var

    def _init_data(self, df_obs: Optional[pd.Index], df_var: Optional[pd.Index]) -> None:
        """
        Function for finishing the initialization of the VData object.
        It creates all needed arrays for maintaining coherence. It also checks for incoherence in the user-supplied arrays and raises an error in case something
        is wrong.
        :param df_obs: If X was supplied as a pandas DataFrame, index of observations
        :param df_var: If X was supplied as a pandas DataFrame, index of variables
        """
        # if time points are not given, assign default values 0, 1, 2, ...
        last_attr = 'time points'
        for attr in ('X', 'layers', 'obsm', 'varm'):
            dataset = getattr(self, attr)
            if dataset is not None:
                if self.time_points is None:
                    self.time_points = range(dataset.shape[0])
                    last_attr = attr

                elif len(self.time_points) != dataset.shape[0]:
                    raise IncoherenceError(f"{last_attr} and {attr} have different numbers of time points.")

        # check consistent number of time points everywhere
        else:
            if self.X is not None:
                if self.n_time_points > self.X.shape[0]:
                    raise IncoherenceError(f"Too many time points for X with depth {self.X.shape[0]}.")
                elif self.n_time_points < self.X.shape[0]:
                    raise IncoherenceError(f"Not enough time points for X with depth {self.X.shape[0]}.")

            if self.layers is not None:
                if self.n_time_points > self.layers.shape[0]:
                    raise IncoherenceError(f"Too many time points for layers with depth {self.layers.shape[0]}.")
                elif self.n_time_points < self.layers.shape[0]:
                    raise IncoherenceError(f"Not enough time points for layers with depth {self.layers.shape[0]}.")

            if self.obsm is not None:
                if self.n_time_points > self.obsm.shape[0]:
                    raise IncoherenceError(f"Too many time points for obsm with depth {self.obsm.shape[0]}.")
                elif self.n_time_points < self.obsm.shape[0]:
                    raise IncoherenceError(f"Not enough time points for obsm with depth {self.obsm.shape[0]}.")

            if self.varm is not None:
                if self.n_time_points > self.varm.shape[0]:
                    raise IncoherenceError(f"Too many time points for varm with depth {self.varm.shape[0]}.")
                elif self.n_time_points < self.varm.shape[0]:
                    raise IncoherenceError(f"Not enough time points for varm with depth {self.varm.shape[0]}.")

        # if X was given as a dataframe, check that obs and X match in row names
        if self.obs is None and df_obs is not None:
            self.obs = pd.DataFrame(index=df_obs)
        elif self.obs is not None and df_obs is not None:
            if self.obs.index != df_obs:
                raise VValueError("Indexes in dataFrames X and obs do not match.")

        # if X was given as a dataframe, check that var row names match X col names
        if self.var is None and df_var is not None:
            self.var = pd.DataFrame(index=df_var)
        elif self.var is not None and df_var is not None:
            if self.var.index != df_var:
                raise VValueError("Columns in dataFrame X does not match var's index.")

        # if no X but layers were supplied, take first layer as X
        if self.X is None and self.layers is not None:
            self.X = self.layers.values[0]

        # check necessary data was supplied
        if self.X is not None:
            if self.obs is None:
                raise NotEnoughDataError("Missing obs data.")

            elif self.var is None:
                raise NotEnoughDataError("Missing var data.")

            elif self.time_points is None:
                raise NotEnoughDataError("Missing time points data.")

            self._n_obs = self.X.shape[0]
            self._n_vars = self.X.shape[1]

        else:
            if self.obs is not None:
                self._n_obs = self.obs.shape[0]

            if self.var is not None:
                self._n_vars = self.var.shape[0]

        if self.obs is None and (self.obsm is not None or self.obsp is not None):
            raise NotEnoughDataError("obs data was not supplied but obsm or obsp were.")

        if self.var is None and (self.varm is not None or self.varp is not None):
            raise NotEnoughDataError("var data was not supplied but varm or varp were.")

        # check coherence between X, obs, vars and time points
        if self.X is not None and self.n_obs:
            if self.X.shape[0] != self.n_obs:
                raise IncoherenceError(f"X has different number of lines than obs ({self.X.shape[0]} vs {self.n_obs}).")

            if self.X.shape[1] != self.n_vars:
                raise IncoherenceError(f"X has different number of columns than var ({self.X.shape[1]} vs {self.n_vars}).")

        # check coherence between X and layers
        if self.X is not None and self.layers is not None:
            for layer_name, layer in self.layers.items():
                if layer.shape != (self.n_obs, self.n_vars):
                    raise IncoherenceError(f"X and layer '{layer_name}' have different shapes.")

        # check coherence between obs, obsm and obsp shapes
        if self.obs is not None and self.n_obs:
            if self.obsm is not None and self.n_obs != self.obsm.shape[0]:
                raise IncoherenceError(f"obs and obsm have different lengths ({self.n_obs} vs {self.obsm.shape[0]})")

            if self.obsp is not None and self.n_obs != self.obsp.shape[0]:
                raise IncoherenceError(f"obs and obsp have different lengths ({self.n_obs} vs {self.obsp.shape[0]})")

        # check coherence between var, varm, varp shapes
        if self.var is not None and self.n_vars:
            if self.varm is not None and self.n_vars != self.varm.shape[0]:
                raise IncoherenceError(f"var and varm have different lengths ({self.n_vars} vs {self.varm.shape[0]})")

            if self.varp is not None and self.n_vars != self.varp.shape[0]:
                raise IncoherenceError(f"var and varp have different lengths ({self.n_vars} vs {self.varp.shape[0]})")
