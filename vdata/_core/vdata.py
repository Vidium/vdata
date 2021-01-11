# coding: utf-8
# Created on 11/4/20 10:03 AM
# Author : matteo

# ====================================================
# imports
import os
import h5py
import pandas as pd
import numpy as np
from anndata import AnnData
from pathlib import Path
from builtins import Ellipsis
from typing import Optional, Union, Dict, Tuple, Any, List, TypeVar

from . import view
from .utils import format_index, reshape_to_3D
from .arrays import VAxisArray, VPairwiseArray, VLayersArrays
from .dataframe import TemporalDataFrame
from ..utils import is_in
from ..NameUtils import ArrayLike_3D, ArrayLike_2D, ArrayLike, DTypes, DType, PreSlicer, \
    Slicer
from .._IO.errors import VTypeError, IncoherenceError, VValueError, ShapeError, VPathError, VAttributeError
from .._IO.logger import generalLogger
from .._IO.write import write_data


DF = TypeVar('DF', pd.DataFrame, TemporalDataFrame)


# ====================================================
# code
class VData:
    """
    A VData object stores data points in matrices of observations x variables in the same way as the AnnData object,
    but also accounts for the time information. The 2D matrices in AnnData are replaced by 3D matrices here.
    """

    # TODO : add support for backed data on .h5 file
    def __init__(self,
                 data: Optional[Union[AnnData, ArrayLike, Dict[Any, ArrayLike]]] = None,
                 obs: Optional[Union[pd.DataFrame, TemporalDataFrame]] = None,
                 obsm: Optional[Dict[Any, ArrayLike]] = None,
                 obsp: Optional[Dict[Any, ArrayLike]] = None,
                 var: Optional[pd.DataFrame] = None,
                 varm: Optional[Dict[Any, ArrayLike]] = None,
                 varp: Optional[Dict[Any, ArrayLike]] = None,
                 time_points: Optional[pd.DataFrame] = None,
                 uns: Optional[Dict] = None,
                 time_col: Optional[str] = None,
                 time_list: Optional[List[str]] = None,
                 dtype: DType = np.float32):
        """
        :param data: a single array-like object or a dictionary of them for storing data for each observation/cell
            and for each variable/gene.
            'data' can also be an AnnData to be converted to the VData format.
        :param obs: a pandas DataFrame or a TemporalDataFrame describing the observations/cells
        :param obsm: a dictionary of array-like objects describing measurements on the observations/cells
        :param obsp: a dictionary of array-like objects describing pairwise comparisons on the observations/cells
        :param var: a pandas DataFrame describing the variables/genes
        :param varm: a dictionary of array-like objects describing measurements on the variables/genes
        :param varp: a dictionary of array-like objects describing pairwise comparisons on the variables/genes
        :param time_points: a pandas DataFrame describing the times points
        :param uns: a dictionary of unstructured data
        :param time_col: if obs is a pandas DataFrame (or the VData is created from an AnnData), the column name in
            obs that contains time information.
        :param time_list: if obs is a pandas DataFrame (or the VData is created from an AnnData), a list containing
            time information of the same length as the number of rows in obs.
        :param dtype: a data type to impose on datasets stored in this VData
        """
        generalLogger.debug(u'\u23BE VData creation : begin -------------------------------------------------------- ')

        # first, check dtype is correct because it will be needed right away
        if dtype not in DTypes.keys():
            raise VTypeError(f"Incorrect data-type '{dtype}', should be in {list(DTypes.keys())}")
        else:
            self._dtype = DTypes[dtype]
        generalLogger.debug(f'Set data-type to {dtype}')

        self._obs = None
        self._var = None
        self._time_points = None
        self._uns: Optional[Dict] = None

        # check formats of arguments
        _layers, _obsm, _obsp, _varm, _varp, df_obs, df_var = self._check_formats(data,
                                                                                  obs, obsm, obsp,
                                                                                  var, varm, varp,
                                                                                  time_points, uns,
                                                                                  time_col, time_list)

        # set number of obs and vars from available data
        self._n_time_points, self._n_obs, self._n_vars = 0, [0], 0

        # if 'layers' is set, get all sizes from there
        if _layers is not None:
            ref_array = list(_layers.values())[0]
            self._n_time_points = ref_array.shape[0]
            self._n_obs = [ref_array[i].shape[0] for i in range(len(ref_array))]
            self._n_vars = ref_array[0].shape[1]

        # otherwise, check other arrays to get the sizes
        else:
            # set obs size
            if self._obs is not None:
                self._n_obs = self._obs.shape[1]
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

        generalLogger.debug(f"Guessed dimensions are : ({self._n_time_points}, {self._n_obs}, {self._n_vars})")

        # make sure a TemporalDataFrame is set to .obs, even if not data was supplied
        if self._obs is None:
            generalLogger.debug("Default empty TemporalDataFrame for obs.")
            self._obs = TemporalDataFrame(index=range(self.n_obs_total) if df_obs is None else df_obs)

        # make sure a pandas DataFrame is set to .var and .time_points, even if no data was supplied
        if self._var is None:
            generalLogger.debug("Default empty DataFrame for vars.")
            self._var = pd.DataFrame(index=range(self.n_var) if df_var is None else df_var)

        if self._time_points is None:
            generalLogger.debug("Default empty DataFrame for time points.")
            self._time_points = pd.DataFrame(index=range(self._n_time_points))

        # create arrays linked to VData
        self._layers = VLayersArrays(self, data=_layers)
        self._obsm = VAxisArray(self, 'obs', data=_obsm)
        self._obsp = VPairwiseArray(self, 'obs', data=_obsp)
        self._varm = VAxisArray(self, 'var', data=_varm)
        self._varp = VPairwiseArray(self, 'var', data=_varp)

        # finish initializing VData
        self._init_data(df_obs, df_var)

        generalLogger.debug(u'\u23BF VData creation : end ---------------------------------------------------------- ')

    def __repr__(self) -> str:
        """
        Description for this Vdata object to print.
        :return: a description of this Vdata object
        """
        _n_obs = self.n_obs if len(self.n_obs) > 1 else self.n_obs[0]

        if self.is_empty:
            repr_str = f"Empty Vdata object ({_n_obs} obs x {self.n_var} vars over {self.n_time_points} " \
                       f"time point{'s' if self.n_time_points > 1 else ''})."
        else:
            repr_str = f"Vdata object with n_obs x n_var = {_n_obs} x {self.n_var} over {self.n_time_points} " \
                       f"time point{'s' if self.n_time_points > 1 else ''}"

        for attr in ["layers", "obs", "var", "time_points", "obsm", "varm", "obsp", "varp", "uns"]:
            keys = getattr(self, attr).keys() if getattr(self, attr) is not None else ()
            if len(keys) > 0:
                repr_str += f"\n\t{attr}: {str(list(keys))[1:-1]}"

        return repr_str

    def __getitem__(self, index: Union[PreSlicer, Tuple[PreSlicer, PreSlicer], Tuple[PreSlicer, PreSlicer, PreSlicer]])\
            -> 'view.ViewVData':
        """
        Get a view of this VData object with the usual sub-setting mechanics.
        :param index: A sub-setting index. It can be a single index, a 2-tuple or a 3-tuple of indexes.
            An index can be a string, an int, a float, a sequence of those, a range, a slice or an ellipsis ('...').
            Single indexes and 2-tuples of indexes are converted to a 3-tuple :
                * single index --> (index, ..., ...)
                * 2-tuple      --> (index[0], index[1], ...)

            The first element in the 3-tuple is the list of time points to view, the second element is the list of
            observations to view and the third element is the list of variables to view.

            The values ':' or '...' are shortcuts for 'take all values in the axis'.

            Example:
                * VData[:] or VData[...]                            --> view all
                * VData[:, 'cell_1'] or VData[:, 'cell_1', :]       --> view all time points and variables for
                                                                        observation 'cell_1'
                * VData[0, ('cell_1', 'cell_9'), range(0, 10)]      --> view observations 'cell_1' and 'cell_2'
                                                                        with variables 0 to 9 on time point 0
        :return: a view on this VData
        """
        generalLogger.debug('VData sub-setting - - - - - - - - - - - - - - ')
        generalLogger.debug(f'  Got index : {index}')

        time_points_slicer, obs_slicer, var_slicer = format_index(index)

        def check_slicer(slicer: PreSlicer) -> Slicer:
            if isinstance(slicer, type(Ellipsis)):
                return slice(None, None, None)

            elif isinstance(slicer, (int, float, str)):
                return [slicer]
            # I have to ignore the type for mypy here because technically type(Ellipsis) is not the same as ellipsis
            # but 'ellipsis' causes an error
            return slicer  # type: ignore

        time_points_slicer = check_slicer(time_points_slicer)
        generalLogger.debug(f"  1. Time points slicer is : {time_points_slicer}")
        obs_slicer = check_slicer(obs_slicer)
        generalLogger.debug(f"  2. Obs slicer is : {obs_slicer}")
        var_slicer = check_slicer(var_slicer)
        generalLogger.debug(f"  3. Var slicer is : {var_slicer}")

        return view.ViewVData(self, time_points_slicer, obs_slicer, var_slicer)

    # Shapes -------------------------------------------------------------
    @property
    def is_empty(self) -> bool:
        """
        Is this VData object empty ? (no obs or no vars)
        :return: VData empty ?
        """
        return self.layers.empty
        # return True if self.n_obs == 0 or self.n_var == 0 or self.n_time_points == 0 else False

    @property
    def n_time_points(self) -> int:
        """
        Number of time points in this VData object. n_time_points can be extracted directly from self.time_points or
        from the nb of time points in the layers. If no data was given, a default list of time points was created
        with integer values.
        :return: VData's number of time points
        """
        return self._n_time_points

    @property
    def n_obs(self) -> List[int]:
        """
        Number of observations in this VData object. n_obs can be extracted directly from self.obs or from parameters
        supplied during this VData object's creation :
            - nb of observations in the layers
            - nb of observations in obsm
            - nb of observations in obsp
        :return: VData's number of observations
        """
        return self._n_obs

    @property
    def n_obs_total(self) -> int:
        """
        Get the total number of observations across all time points.
        :return: the total number of observations across all time points.
        """
        return sum(self._n_obs)

    @property
    def n_var(self) -> int:
        """
        Number of variables in this VData object. n_var can be extracted directly from self.var or from parameters
        supplied during this VData object's creation :
            - nb of variables in the layers
            - nb of variables in varm
            - nb of variables in varp
        :return: VData's number of variables
        """
        return self._n_vars

    @property
    def shape(self) -> Tuple[int, List[int], int]:
        """
        Shape of this VData object.
        :return: VData's shape
        """
        return self.n_time_points, self.n_obs, self.n_var

    # DataFrames ---------------------------------------------------------
    @property
    def time_points(self) -> pd.DataFrame:
        """
        Get time points data.
        :return: the time points DataFrame.
        """
        return self._time_points

    @time_points.setter
    def time_points(self, df: pd.DataFrame) -> None:
        """
        Set the time points data.
        :param df: a pandas DataFrame with at least the 'value' column.
        """
        if not isinstance(df, pd.DataFrame):
            raise VTypeError("'time points' must be a pandas DataFrame.")

        elif df.shape[0] != self.n_time_points:
            raise ShapeError(f"'time points' has {df.shape[0]} lines, it should have {self.n_time_points}.")

        elif 'value' not in df.columns:
            raise VValueError(f"Time points DataFrame should contain a 'value' column.")

        else:
            self._time_points = df

    @property
    def obs(self) -> TemporalDataFrame:
        """
        Get the obs data.
        :return: the obs TemporalDataFrame.
        """
        return self._obs

    @obs.setter
    def obs(self, df: Union[pd.DataFrame, TemporalDataFrame]) -> None:
        """
        Set the obs data.
        :param df: a pandas DataFrame or a TemporalDataFrame.
        """
        # TODO : is every thing checked here ?
        if not isinstance(df, (pd.DataFrame, TemporalDataFrame)):
            raise VTypeError("'obs' must be a pandas DataFrame or a TemporalDataFrame.")

        elif df.shape[0] != self.n_obs:
            raise ShapeError(f"'obs' has {df.shape[0]} lines, it should have {self.n_obs}.")

        # cast to TemporalDataFrame
        if isinstance(df, pd.DataFrame):
            df = TemporalDataFrame(df, index=self.obs.index,
                                   time_list=self.obs.time_points_column,
                                   time_col=self.obs.time_points_column_name)

        self._obs = df

    @property
    def var(self) -> pd.DataFrame:
        """
        Get the var data.
        :return: the var DataFrame.
        """
        return self._var

    @var.setter
    def var(self, df: pd.DataFrame) -> None:
        """
        Set the var data.
        :param df: a pandas DataFrame.
        """
        if not isinstance(df, pd.DataFrame):
            raise VTypeError("'var' must be a pandas DataFrame.")

        elif df.shape[0] != self.n_var:
            raise ShapeError(f"'var' has {df.shape[0]} lines, it should have {self.n_var}.")

        else:
            self._var = df

    # Arrays -------------------------------------------------------------
    # TODO : docstrings
    @property
    def obsm(self) -> VAxisArray:
        return self._obsm

    @obsm.setter
    def obsm(self, data: Optional[Dict[Any, ArrayLike_3D]]) -> None:
        if data is None:
            self._obsm = VAxisArray(self, 'obs', None)

        else:
            if not isinstance(data, dict):
                raise VTypeError("'obsm' should be set with a dictionary of 3D array-like objects (numpy array).")

            for arr_index, arr in data.items():
                if self.n_time_points > 1:
                    if not isinstance(arr, np.ndarray):
                        raise VTypeError(
                            f"'{arr_index}' array for obsm should be a 3D array-like object (numpy array).")

                    elif arr.ndim != 3:
                        raise VTypeError(
                            f"'{arr_index}' array for obsm should be a 3D array-like object (numpy array).")

                else:
                    if not isinstance(arr, (np.ndarray, pd.DataFrame)):
                        raise VTypeError(
                            f"'{arr_index}' array for obsm should be a 2D or 3D array-like object "
                            f"(numpy array, pandas DataFrame).")

                    elif arr.ndim not in (2, 3):
                        raise VTypeError(
                            f"'{arr_index}' array for obsm should be a 2D or 3D array-like object "
                            f"(numpy array, pandas DataFrame).")

                    elif arr.ndim == 2:
                        data[arr_index] = reshape_to_3D(arr, np.zeros(len(arr)))

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
                raise VTypeError(
                    "'obsp' should be set with a dictionary of 2D array-like objects (numpy array).")

            for arr_index, arr in data.items():
                if not isinstance(arr, (np.ndarray, pd.DataFrame)):
                    raise VTypeError(
                        f"'{arr_index}' array for obsp should be a 2D array-like object "
                        f"(numpy array, pandas DataFrame).")

                elif arr.ndim != 2:
                    raise VTypeError(
                        f"'{arr_index}' array for obsm should be a 2D array-like object "
                        f"(numpy array, pandas DataFrame).")

            self._obsp = VPairwiseArray(self, 'obs', data)

    @property
    def varm(self) -> VAxisArray:
        return self._varm

    @varm.setter
    def varm(self, data: Optional[Dict[Any, ArrayLike_3D]]) -> None:
        if data is None:
            self._varm = VAxisArray(self, 'var', None)

        else:
            if not isinstance(data, dict):
                raise VTypeError("'varm' should be set with a dictionary of 3D array-like objects (numpy array).")

            for arr_index, arr in data.items():
                if self.n_time_points > 1:
                    if not isinstance(arr, np.ndarray):
                        raise VTypeError(f"'{arr_index}' array for varm should be a 3D array-like object "
                                         f"(numpy array).")

                    elif arr.ndim != 3:
                        raise VTypeError(f"'{arr_index}' array for varm should be a 3D array-like object "
                                         f"(numpy array).")

                else:
                    if not isinstance(arr, (np.ndarray, pd.DataFrame)):
                        raise VTypeError(
                            f"'{arr_index}' array for varm should be a{' 2D or' if self.n_time_points == 1 else ''} "
                            f"3D array-like object "
                            f"(numpy array{', pandas DataFrame' if self.n_time_points == 1 else ''}).")

                    elif arr.ndim not in (2, 3):
                        raise VTypeError(
                            f"'{arr_index}' array for varm should be a{' 2D or' if self.n_time_points == 1 else ''} "
                            f"3D array-like object "
                            f"(numpy array{', pandas DataFrame' if self.n_time_points == 1 else ''}).")

                    elif arr.ndim == 2:
                        data[arr_index] = reshape_to_3D(arr, np.zeros(len(arr)))

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
                raise VTypeError("'varp' should be set with a dictionary of 2D array-like objects (numpy array).")

            for arr_index, arr in data.items():
                if not isinstance(arr, (np.ndarray, pd.DataFrame)):
                    raise VTypeError(f"'{arr_index}' array for varp should be a 2D array-like object "
                                     f"(numpy array, pandas DataFrame).")

                elif arr.ndim != 2:
                    raise VTypeError(f"'{arr_index}' array for varp should be a 2D array-like object "
                                     f"(numpy array, pandas DataFrame).")

            self._varp = VPairwiseArray(self, 'var', data)

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
            if isinstance(data, (np.ndarray, pd.DataFrame)):
                data = {"data": data}

            elif not isinstance(data, dict):
                raise VTypeError("'layers' should be set with a 3D array-like object (numpy array) "
                                 "or with a dictionary of them.")

            for arr_index, arr in data.items():
                if self.n_time_points > 1:
                    if not isinstance(arr, np.ndarray):
                        raise VTypeError(f"'{arr_index}' array for layers should be a 3D array-like object "
                                         f"(numpy array).")

                    elif arr.ndim != 3:
                        raise VTypeError(f"'{arr_index}' array for layers should be a 3D array-like object "
                                         f"(numpy array).")

                else:
                    if not isinstance(arr, (np.ndarray, pd.DataFrame)):
                        raise VTypeError(f"'{arr_index}' array for layers should be a 2D or 3D array-like object "
                                         f"(numpy array, pandas DataFrame).")

                    elif arr.ndim not in (2, 3):
                        raise VTypeError(f"'{arr_index}' array for layers should be a 2D or 3D array-like object "
                                         f"(numpy array, pandas DataFrame).")

                    elif arr.ndim == 2:
                        data[arr_index] = reshape_to_3D(arr, np.zeros(len(arr)))

            self._layers = VLayersArrays(self, data)

    # special ------------------------------------------------------------
    @property
    def dtype(self) -> DType:
        return self._dtype

    @dtype.setter
    def dtype(self, type_: DType) -> None:
        self._dtype = type_
        # update dtype of linked Arrays
        self.layers.update_dtype(type_)
        self.obsm.update_dtype(type_)
        self.obsp.update_dtype(type_)
        self.varm.update_dtype(type_)
        self.varp.update_dtype(type_)

        generalLogger.info(f"Set type {type_} for VData object.")

    # aliases ------------------------------------------------------------
    @property
    def cells(self) -> TemporalDataFrame:
        """
        Get cells (= obs) data.
        :return: the cells TemporalDataFrame.
        """
        return self._obs

    @cells.setter
    def cells(self, df: Union[pd.DataFrame, TemporalDataFrame]) -> None:
        """
        Set cells (= obs) data.
        :param df: a pandas DataFrame or a TemporalDataFrame.
        """
        self.obs = df

    @property
    def genes(self) -> pd.DataFrame:
        """
        Get the genes (= var) data.
        :return: the gene (= var) data.
        """
        return self._var

    @genes.setter
    def genes(self, df: pd.DataFrame) -> None:
        """
        Set the var (= genes) data.
        :param df: a pandas DataFrame.
        """
        self.var = df

    # init functions -----------------------------------------------------

    def _check_formats(self, data: Optional[Union[ArrayLike, Dict[Any, ArrayLike], AnnData]],
                       obs: Optional[Union[pd.DataFrame, TemporalDataFrame]], obsm: Optional[Dict[Any, ArrayLike]],
                       obsp: Optional[Dict[Any, ArrayLike]],
                       var: Optional[pd.DataFrame], varm: Optional[Dict[Any, ArrayLike]],
                       varp: Optional[Dict[Any, ArrayLike]],
                       time_points: Optional[pd.DataFrame],
                       uns: Optional[Dict],
                       time_col: Optional[str] = None,
                       time_list: Optional[List[str]] = None) -> Tuple[
        Optional[Dict[str, ArrayLike_3D]], Optional[Dict[str, ArrayLike_3D]],
        Optional[Dict[str, ArrayLike_2D]], Optional[Dict[str, ArrayLike_3D]],
        Optional[Dict[str, ArrayLike_2D]], Optional[pd.Index], Optional[pd.Index]
    ]:
        """
        Function for checking the types and formats of the parameters supplied to the VData object at creation.
        If the types are not accepted, an error is raised. obsm, obsp, varm, varp and layers are prepared for
        being converted into custom arrays for maintaining coherence with this VData object.
        :param data: a single array-like object or a dictionary of them for storing data for each observation/cell
            and for each variable/gene.
            'data' can also be an AnnData to be converted to the VData format.
        :param obs: a pandas DataFrame or a TemporalDataFrame describing the observations/cells
        :param obsm: a dictionary of array-like objects describing measurements on the observations/cells
        :param obsp: a dictionary of array-like objects describing pairwise comparisons on the observations/cells
        :param var: a pandas DataFrame describing the variables/genes
        :param varm: a dictionary of array-like objects describing measurements on the variables/genes
        :param varp: a dictionary of array-like objects describing pairwise comparisons on the variables/genes
        :param time_points: a pandas DataFrame describing the times points
        :param uns: a dictionary of unstructured data
        :param time_col: if obs is a pandas DataFrame (or the VData is created from an AnnData), the column name in
            obs that contains time information.
        :param time_list: if obs is a pandas DataFrame (or the VData is created from an AnnData), a list containing
            time information of the same length as the number of rows in obs.
        :return: Arrays in correct format.
        """
        def check_time_match(_time_points: Optional[pd.DataFrame],
                             _time_list: Optional[List[str]],
                             _time_col: Optional[str],
                             _obs: TemporalDataFrame) -> Tuple[Optional[pd.DataFrame], int]:
            """
            Build time_points DataFrame if it was not given by the user but 'time_list' or 'time_col' were given.
            Otherwise, if both time_points and 'time_list' or 'time_col' were given, check that they match.
            :param _time_points: a pandas DataFrame with time points data.
            :param _time_list: a list of time points of the same length as the number of rows in obs.
            :param _time_col: a column name which contains time points information in obs.
            :param _obs: the obs TemporalDataFrame.
            :return: a time points DataFrame if possible and the number of found time points.
            """
            if _time_points is None:
                # build time_points DataFrame from time_list or time_col
                if _time_list is not None or _time_col is not None:
                    if _time_list is not None:
                        unique_time_points = np.unique(_time_list)

                    else:
                        unique_time_points = _obs.time_points

                    return pd.DataFrame({'value': unique_time_points}), len(unique_time_points)

                # time_points cannot be guessed
                else:
                    return None, 1

            # check that time_points and _time_list and _time_col match
            else:
                if _time_list is not None:
                    if not all(np.isin(_time_list, _time_points['value'])):
                        raise VValueError("There are values in 'time_list' unknown in 'time_points'.")

                elif _time_col is not None:
                    if not all(np.isin(_obs.time_points, _time_points['value'])):
                        raise VValueError("There are values in obs['time_col'] unknown in 'time_points'.")

                return _time_points, len(_time_points)

        generalLogger.debug(u"  \u23BE Check arrays' formats. -- -- -- -- -- -- -- -- -- -- ")

        df_obs, df_var = None, None
        layers = None

        # time_points
        if time_points is not None:
            generalLogger.debug(f"  'time points' DataFrame is a {type(time_points).__name__}.")
            if not isinstance(time_points, pd.DataFrame):
                raise VTypeError("'time points' must be a pandas DataFrame.")
            else:
                if 'value' not in time_points.columns:
                    raise VValueError("'time points' must have at least a column 'value' to store time points value.")

                time_points = self._check_df_types(time_points)

        else:
            generalLogger.debug("  'time points' DataFrame was not found.")

        nb_time_points = 1 if time_points is None else len(time_points)
        generalLogger.debug(f"  {nb_time_points} time point{' was' if nb_time_points == 1 else 's were'} found so far.")
        generalLogger.debug(f"    \u21B3 Time point{' is' if nb_time_points == 1 else 's are'} : "
                            f"{[0] if nb_time_points == 1 else time_points.value.values}")

        # if an AnnData is being imported, obs, obsm, obsp, var, varm, varp and uns should be None because
        # they will be set from the AnnData
        if isinstance(data, AnnData):
            generalLogger.debug('  VData creation from an AnnData.')

            for attr in ('obs', 'obsm', 'obsp', 'var', 'varm', 'varp', 'uns'):
                if eval(f"{attr} is not None"):
                    raise VValueError(f"'{attr}' should be set to None when importing data from an AnnData.")

            # import and cast obs to a TemporalDataFrame
            obs = TemporalDataFrame(data.obs, time_list=time_list, time_col=time_col)

            # find time points list
            time_points, nb_time_points = check_time_match(time_points, time_list, time_col, obs)

            generalLogger.debug(f"  {nb_time_points} time point{' was' if nb_time_points == 1 else 's were'} "
                                f"found after data extraction from the AnnData.")
            generalLogger.debug(f"    \u21B3 Time point{' is' if nb_time_points == 1 else 's are'} : "
                                f"{[0] if nb_time_points == 1 else time_points.value.values}")

            # import data from AnnData
            if time_list is not None:
                ref_time_list = time_list

            elif time_col is not None:
                ref_time_list = data.obs[time_col].values

            else:
                ref_time_list = np.zeros(data.n_obs)

            if is_in(data.X, list(data.layers.values())):
                layers = dict((key, reshape_to_3D(arr, time_points.value.values, ref_time_list))
                              for key, arr in data.layers.items())

            else:
                layers = dict({"data": reshape_to_3D(data.X, time_points.value.values, ref_time_list)},
                              **dict((key, reshape_to_3D(arr, time_points.value.values, ref_time_list))
                                     for key, arr in data.layers.items()))

            # import other arrays
            obsm, obsp = dict(data.obsm), dict(data.obsp)
            var, varm, varp = data.var, dict(data.varm), dict(data.varp)
            uns = dict(data.uns)

        else:
            # TODO : need to work on time_col and time_list params here
            generalLogger.debug('  VData creation from scratch.')

            # obs
            if obs is not None:
                generalLogger.debug(f"    2. \u2713 'obs' is a {type(obs).__name__}.")

                if not isinstance(obs, (pd.DataFrame, TemporalDataFrame)):
                    raise VTypeError("obs must be a pandas DataFrame or a TemporalDataFrame.")
                elif isinstance(obs, pd.DataFrame):
                    obs = self._check_df_types(TemporalDataFrame(obs, time_list=time_list, time_col=time_col))
                else:
                    obs = self._check_df_types(obs)
                    if time_list is not None:
                        generalLogger.warning("'time_list' parameter cannot be used since 'obs' is already a "
                                              "TemporalDataFrame.")
                    if time_col is not None:
                        generalLogger.warning("'time_col' parameter cannot be used since 'obs' is already a "
                                              "TemporalDataFrame.")

                # find time points list
                time_points, nb_time_points = check_time_match(time_points, time_list, time_col, obs)

                generalLogger.debug(
                    f"  {nb_time_points} time point{' was' if nb_time_points == 1 else 's were'} "
                    f"found from the provided data.")
                generalLogger.debug(f"    \u21B3 Time point{' is' if nb_time_points == 1 else 's are'} : "
                                    f"{[0] if nb_time_points == 1 else time_points.value.values}")

            else:
                generalLogger.debug(f"    2. \u2717 'obs' was not found.")
                if time_list is not None:
                    generalLogger.warning("'time_list' parameter cannot be used since 'obs' was not found.")
                if time_col is not None:
                    generalLogger.warning("'time_col' parameter cannot be used since 'obs' was not found.")

            # check formats
            # layers
            if data is not None:
                layers = {}

                # data is a pandas DataFrame
                if isinstance(data, pd.DataFrame):
                    generalLogger.debug(f"    1. \u2713 'data' is a pandas DataFrame.")

                    if nb_time_points > 1:
                        raise VTypeError("'data' is a 2D pandas DataFrame but more than 1 time points were provided.")

                    df_obs = data.index
                    df_var = data.columns

                    if time_list is not None:
                        ref_time_list = time_list

                    elif time_col is not None:
                        ref_time_list = data[time_col].values

                    else:
                        ref_time_list = np.zeros(len(data))

                    layers = {"data": reshape_to_3D(np.array(data, dtype=self._dtype),
                                                    time_points.value.values if time_points is not None else ['0'],
                                                    ref_time_list)}

                # data is an array
                elif isinstance(data, np.ndarray):
                    generalLogger.debug(f"    1. \u2713 'data' is a numpy array.")

                    if data.ndim == 1:
                        if all([isinstance(arr, (np.ndarray, pd.DataFrame)) and arr.ndim == 2 for arr in data]):
                            layers = {"data": data}

                        else:
                            raise VTypeError("When supplying arrays of different shapes to 'data', those arrays must "
                                             "be 2D array-like objects (numpy array, pandas DataFrame)")

                    elif data.ndim == 2:
                        reshaped_data = reshape_to_3D(data, None, None)

                        layers = {"data": reshaped_data}

                    elif data.ndim == 3:
                        layers = {"data": data}

                    else:
                        raise ShapeError("'data' must be a 2D or 3D array-like object "
                                         "(numpy array, pandas DataFrame).")

                elif isinstance(data, dict):
                    generalLogger.debug(f"    1. \u2713 'data' is a dictionary.")

                    for key, value in data.items():
                        if not isinstance(value, (np.ndarray, pd.DataFrame)):
                            raise VTypeError(f"Layer '{key}' must be a 2D or 3D array-like object "
                                             f"(numpy array, pandas DataFrame).")
                        elif value.ndim not in (1, 2, 3):
                            raise VTypeError(f"Layer '{key}' must contain 2D or 3D array-like object "
                                             f"(numpy array, pandas DataFrame).")
                        elif value.ndim == 1:
                            for arr in value:
                                if arr.ndim != 2:
                                    raise VTypeError(f"Layer '{key}' must contain 2D or 3D array-like object "
                                                     f"(numpy array, pandas DataFrame).")
                        elif value.ndim == 2:
                            if nb_time_points > 1:
                                raise VTypeError(f"Layer '{key}' must be a 3D array-like object "
                                                 f"(numpy arrays) if providing more than 1 time point.")

                            if isinstance(value, pd.DataFrame):
                                if time_list is not None:
                                    ref_time_list = time_list

                                elif time_col is not None:
                                    ref_time_list = value[time_col].values

                                else:
                                    ref_time_list = np.zeros(len(value))

                                value = reshape_to_3D(value,
                                                      time_points.value.values if time_points is not None else ['0'],
                                                      ref_time_list)

                            else:
                                value = reshape_to_3D(value)

                        layers[str(key)] = value

                else:
                    raise VTypeError(f"Type '{type(data)}' is not allowed for 'data' parameter, should be a dict,"
                                     f"a pandas DataFrame, a numpy array or an AnnData object.")

            else:
                generalLogger.debug(f"    1. \u2717 'data' was not found.")

            # obsm
            if obsm is not None:
                generalLogger.debug(f"    3. \u2713 'obsm' is a {type(obsm).__name__}.")

                if not isinstance(obsm, dict):
                    raise VTypeError("obsm must be a dictionary.")
                else:
                    for key, value in obsm.items():
                        if not isinstance(value, np.ndarray):
                            raise VTypeError(f"obsm '{key}' must be a 2D or 3D array-like object "
                                             f"(numpy array, pandas DataFrame).")
                        elif value.ndim not in (2, 3):
                            raise VTypeError(f"obsm '{key}' must be a 2D or 3D array-like object "
                                             f"(numpy array, pandas DataFrame).")
                        elif value.ndim == 2:
                            if nb_time_points > 1:
                                raise VTypeError(f"obsm '{key}' must be a 3D array-like object (numpy array) "
                                                 f"if providing more than 1 time point.")

                            value = reshape_to_3D(value)

                        obsm[str(key)] = value

            else:
                generalLogger.debug(f"    3. \u2717 'obsm' was not found.")

            # obsp
            if obsp is not None:
                generalLogger.debug(f"    4. \u2713 'obsp' is a {type(obsp).__name__}.")

                if isinstance(obsp, dict):
                    for key, value in obsp.items():
                        if not isinstance(value, (np.ndarray, pd.DataFrame)) and value.ndim != 2:
                            raise VTypeError(f"'{key}' object in obsp is not a 2D array-like object "
                                             f"(numpy array, pandas DataFrame).")

                    obsp = dict((str(k), v) for k, v in obsp.items())

                else:
                    raise VTypeError("obsp must be a dictionary of 2D array-like object "
                                     "(numpy array, pandas DataFrame).")

            else:
                generalLogger.debug(f"    4. \u2717 'obsp' was not found.")

            # var
            if var is not None:
                generalLogger.debug(f"    5. \u2713 'var' is a {type(var).__name__}.")

                if not isinstance(var, pd.DataFrame):
                    raise VTypeError("var must be a pandas DataFrame.")
                else:
                    var = self._check_df_types(var)

            else:
                generalLogger.debug(f"    5. \u2717 'var' was not found.")

            # varm
            if varm is not None:
                generalLogger.debug(f"    6. \u2713 'varm' is a {type(varm).__name__}.")

                if not isinstance(varm, dict):
                    raise VTypeError("varm must be a dictionary.")
                else:
                    for key, value in varm.items():
                        if not isinstance(value, np.ndarray):
                            raise VTypeError(f"varm '{key}' must be a 2D or 3D array-like object "
                                             f"(numpy arrays).")
                        elif value.ndim not in (2, 3):
                            raise VTypeError(f"varm '{key}' must be a 2D or 3D array-like object "
                                             f"(numpy array, pandas DataFrame).")
                        elif value.ndim == 2:
                            if nb_time_points > 1:
                                raise VTypeError(f"varm '{key}' must be a 3D array-like object (numpy array) "
                                                 f"if providing more than 1 time point.")

                            value = reshape_to_3D(value)

                        varm[str(key)] = value

            else:
                generalLogger.debug(f"    6. \u2717 'varm' was not found.")

            # varp
            if varp is not None:
                generalLogger.debug(f"    7. \u2713 'varp' is a {type(varp).__name__}.")

                if isinstance(varp, dict):
                    for key, value in varp.items():
                        if not isinstance(value, (np.ndarray, pd.DataFrame)) and value.ndim != 2:
                            raise VTypeError(f"'{key}' object in varp is not a 2D array-like object "
                                             f"(numpy array, pandas DataFrame).")

                    varp = dict((str(k), v) for k, v in varp.items())

                else:
                    raise VTypeError("'varp' must be a dictionary of 2D array-like object "
                                     "(numpy array, pandas DataFrame).")

            else:
                generalLogger.debug(f"    7. \u2717 'varp' was not found.")

            # uns
            if uns is not None:
                if not isinstance(uns, dict):
                    raise VTypeError("'uns' must be a dictionary.")
                generalLogger.debug(f"    8. \u2713 'uns' is a dictionary.")

            else:
                generalLogger.debug(f"    8. \u2717 'uns' was not found.")

        # if time points are not given, assign default values 0, 1, 2, ...
        if time_points is None:
            if layers is not None:
                time_points = pd.DataFrame({'value': range(list(layers.values())[0].shape[0])})
            elif obsm is not None:
                time_points = pd.DataFrame({'value': range(list(obsm.values())[0].shape[0])})
            elif varm is not None:
                time_points = pd.DataFrame({'value': range(list(varm.values())[0].shape[0])})

        if time_points is not None:
            generalLogger.debug(f"  {len(time_points)} time point{' was' if len(time_points) == 1 else 's were'} "
                                f"found finally.")
            generalLogger.debug(f"    \u21B3 Time point{' is' if nb_time_points == 1 else 's are'} : "
                                f"{time_points.value.values}")

        else:
            generalLogger.debug(f"  Could not find time points.")

        self._obs = obs
        self._var = var
        self._uns = dict(zip([str(k) for k in uns.keys()], uns.values())) if uns is not None else None
        self._time_points = time_points

        generalLogger.debug(u"  \u23BF Arrays' formats were OK.  -- -- -- -- -- -- -- -- -- ")

        return layers, obsm, obsp, varm, varp, df_obs, df_var

    def _check_df_types(self, df: DF) -> DF:
        """
        Function for coercing data types of the columns and of the index in a pandas DataFrame.
        :param df: a pandas DataFrame
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
            except VAttributeError:
                try:
                    df.asColType(col_name, self._dtype)
                except ValueError:
                    df.asColType(col_name, np.dtype('O'))

        return df

    def _init_data(self, df_obs: Optional[pd.Index], df_var: Optional[pd.Index]) -> None:
        """
        Function for finishing the initialization of the VData object. It checks for incoherence in the user-supplied
        arrays and raises an error in case something is wrong.
        :param df_obs: If X was supplied as a pandas DataFrame, index of observations
        :param df_var: If X was supplied as a pandas DataFrame, index of variables
        """
        generalLogger.debug("Initialize the VData.")

        # check coherence with number of time points in VData
        if self._time_points is not None:
            for attr in ('layers', 'obsm', 'varm'):
                dataset = getattr(self, attr)
                if len(dataset):
                    if len(self._time_points) != dataset.shape[0]:
                        raise IncoherenceError(f"{attr} has {dataset.shape[0]} time points but {len(self._time_points)}"
                                               f" {'was' if len(self._time_points) == 1 else 'were'} given.")

        generalLogger.debug("Time points were coherent across arrays.")

        # if data was given as a dataframe, check that obs and data match in row names
        if self.obs.empty and df_obs is not None:
            self.obs = pd.DataFrame(index=df_obs)
        elif df_obs is not None:
            if not self.obs.index.equals(df_obs):
                raise VValueError(f"Indexes in dataFrames 'data' ({df_obs}) and 'obs' ({self.obs.index}) do not match.")

        # if data was given as a dataframe, check that var row names match data col names
        if self.var.empty and df_var is not None:
            self.var = pd.DataFrame(index=df_var)
        elif df_var is not None:
            if not self.var.index.equals(df_var):
                raise VValueError("Columns in dataFrame 'data' does not match 'var''s index.")

        # check coherence between layers, obs, var and time points
        if self._layers is not None:
            for layer_name, layer in self._layers.items():
                layer_shape = (layer.shape[0], [layer[i].shape[0] for i in range(len(layer))], layer[0].shape[1])
                if layer_shape != (self.n_time_points, self.n_obs, self.n_var):
                    if layer.shape[0] != self.n_time_points:
                        raise IncoherenceError(f"layer '{layer_name}' has incoherent number of time points "
                                               f"{layer.shape[0]}, should be {self.n_time_points}.")
                    elif [layer[i].shape[0] for i in range(len(layer))] != self.n_obs:
                        for i in range(len(layer)):
                            if layer[i].shape[0] != self.n_obs[i]:
                                raise IncoherenceError(f"layer '{layer_name}' has incoherent number of observations "
                                                       f"{layer[i].shape[0]}, should be {self.n_obs[i]}.")
                    else:
                        # TODO : need to check that layer has all arrays with same nb of genes
                        raise IncoherenceError(f"layer '{layer_name}' has incoherent number of variables "
                                               f"{layer[0].shape[1]}, should be {self.n_var}.")

        # check coherence between obs, obsm and obsp shapes
        if self._obsm is not None and self.n_obs != self._obsm.shape[1]:
            raise IncoherenceError(f"obs and obsm have different lengths ({self.n_obs} vs {self._obsm.shape[1]})")

        if self._obsp is not None and self.n_obs != self._obsp.shape[0]:
            raise IncoherenceError(f"obs and obsp have different lengths ({self.n_obs} vs {self._obsp.shape[0]})")

        # check coherence between var, varm, varp shapes
        if self._varm is not None and self.n_var != self._varm.shape[1]:
            raise IncoherenceError(f"var and varm have different lengths ({self.n_var} vs {self._varm.shape[1]})")

        if self._varp is not None and self.n_var != self._varp.shape[0]:
            raise IncoherenceError(f"var and varp have different lengths ({self.n_var} vs {self._varp.shape[0]})")

    # writing ------------------------------------------------------------

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

    # copy ---------------------------------------------------------------
    def copy(self) -> 'VData':
        """
        Build an actual copy of this VData object and not a view.
        """
        return VData(self.layers.dict_copy(),
                     self.obs, self.obsm.dict_copy(), self.obsp.dict_copy(),
                     self.var, self.varm.dict_copy(), self.varp.dict_copy(),
                     self.time_points,
                     self.uns,
                     self.dtype)
