# coding: utf-8
# Created on 11/4/20 10:03 AM
# Author : matteo

# ====================================================
# imports
import pandas as pd
import numpy as np
from anndata import AnnData
from pathlib import Path
from typing import Optional, Union, Dict, Tuple, Any, List, TypeVar, Collection

from vdata.NameUtils import DTypes, DType, str_DType, PreSlicer, DataFrame
from .utils import array_isin
from .arrays import VLayerArrayContainer, VObsmArrayContainer, VObspArrayContainer, VVarmArrayContainer, \
    VVarpArrayContainer
from .views import ViewVData
from ..utils import TimePoint, reformat_index, repr_index, repr_array, match_time_points, to_tp_list
from .._TDF.dataframe import TemporalDataFrame
from .._IO.write import generalLogger, write_vdata, write_vdata_to_csv
from .._IO.errors import VTypeError, IncoherenceError, VValueError, ShapeError

DF = TypeVar('DF', bound=DataFrame)
Array2D = Union[pd.DataFrame, np.ndarray]


# ====================================================
# code
class VData:
    """
    A VData object stores data points in matrices of observations x variables in the same way as the AnnData object,
    but also accounts for the time information. The 2D matrices in AnnData are replaced by 3D matrices here.
    """

    # TODO : add support for backed data on .h5 file
    def __init__(self,
                 data: Optional[Union[AnnData, DataFrame, Dict[Any, DataFrame]]] = None,
                 obs: Optional[DataFrame] = None,
                 obsm: Optional[Dict[Any, DataFrame]] = None,
                 obsp: Optional[Dict[Any, Array2D]] = None,
                 var: Optional[pd.DataFrame] = None,
                 varm: Optional[Dict[Any, pd.DataFrame]] = None,
                 varp: Optional[Dict[Any, Array2D]] = None,
                 time_points: Optional[pd.DataFrame] = None,
                 uns: Optional[Dict] = None,
                 time_col: Optional[str] = None,
                 time_list: Optional[List[str]] = None,
                 dtype: Optional[Union[DType, str_DType]] = None,
                 name: Optional[Any] = None):
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
        :param dtype: a data type to impose on datasets stored in this VData.
        """
        self.name = str(name) if name is not None else 'No_Name'

        generalLogger.debug(f"\u23BE VData '{self.name}' creation : begin "
                            f"-------------------------------------------------------- ")

        # first, check dtype is correct because it will be needed right away
        if dtype is not None:
            if dtype not in DTypes.keys():
                raise VTypeError(f"Incorrect data-type '{dtype}', should be in {list(DTypes.keys())}")
            else:
                self._dtype: DType = DTypes[dtype]
        else:
            self._dtype = None
        generalLogger.debug(f'Set data-type to {self._dtype}')

        self._obs = None
        self._var = None
        self._time_points = None
        self._uns: Dict = {}

        # check formats of arguments
        _layers, _obsm, _obsp, _varm, _varp, obs_index, var_index = self._check_formats(data,
                                                                                        obs, obsm, obsp,
                                                                                        var, varm, varp,
                                                                                        time_points, uns,
                                                                                        time_col, time_list)

        ref_TDF = list(_layers.values())[0] if _layers is not None else None

        # make sure a TemporalDataFrame is set to .obs, even if not data was supplied
        if self._obs is None:
            generalLogger.debug("Default empty TemporalDataFrame for obs.")

            time_list = ref_TDF.time_points_column if ref_TDF is not None else None

            self._obs = TemporalDataFrame(time_list=time_list,
                                          index=(ref_TDF.index if ref_TDF is not None else None) if obs_index is None
                                          else obs_index,
                                          name='obs')

        # make sure a pandas DataFrame is set to .var and .time_points, even if no data was supplied
        if self._var is None:
            generalLogger.debug("Default empty DataFrame for vars.")
            self._var = pd.DataFrame(index=(range(ref_TDF.shape[2]) if ref_TDF is not None else None) if
                                     var_index is None else var_index)

        if self._time_points is None:
            generalLogger.debug("Default empty DataFrame for time points.")
            self._time_points = pd.DataFrame({"value": self.obs.time_points})

        # create arrays linked to VData
        self._layers = VLayerArrayContainer(self, data=_layers)

        generalLogger.debug(f"Guessed dimensions are : ({self.n_time_points}, {self.n_obs}, {self.n_var})")

        self._obsm = VObsmArrayContainer(self, data=_obsm)
        self._obsp = VObspArrayContainer(self, data=_obsp)
        self._varm = VVarmArrayContainer(self, data=_varm)
        self._varp = VVarpArrayContainer(self, data=_varp)

        # finish initializing VData
        self._init_data(obs_index, var_index)

        generalLogger.debug(f"\u23BF VData '{self.name}' creation : end "
                            f"---------------------------------------------------------- ")

    def __repr__(self) -> str:
        """
        Description for this Vdata object to print.
        :return: a description of this Vdata object
        """
        _n_obs = self.n_obs if len(self.n_obs) > 1 else self.n_obs[0] if len(self.n_obs) else 0

        if self.empty:
            repr_str = f"Empty Vdata object ({_n_obs} obs x {self.n_var} vars over {self.n_time_points} " \
                       f"time point{'' if self.n_time_points == 1 else 's'})."

        else:
            repr_str = f"Vdata object with n_obs x n_var = {_n_obs} x {self.n_var} over {self.n_time_points} " \
                       f"time point{'' if self.n_time_points == 1 else 's'}."

        for attr in ["layers", "obs", "var", "time_points", "obsm", "varm", "obsp", "varp"]:
            obj = getattr(self, attr)
            if obj is not None and not obj.empty:
                repr_str += f"\n\t{attr}: {str(list(obj.keys()))[1:-1]}"

        if len(self.uns):
            repr_str += f"\n\tuns: {str(list(self.uns.keys()))[1:-1]}"

        return repr_str

    def __getitem__(self, index: Union[PreSlicer, Tuple[PreSlicer, PreSlicer], Tuple[PreSlicer, PreSlicer, PreSlicer]])\
            -> ViewVData:
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
        generalLogger.debug(f'  Got index \n{repr_index(index)}')

        index = reformat_index(index, self.time_points.value.values, self.obs.index, self.var.index)

        generalLogger.debug(f"  Refactored index to \n{repr_index(index)}")

        if not len(index[0]):
            raise VValueError("Time points not found in this VData.")

        return ViewVData(self, index[0], index[1], index[2])

    # Shapes -------------------------------------------------------------
    @property
    def empty(self) -> bool:
        """
        Is this VData object empty ? (no time points or no obs or no vars)
        :return: VData empty ?
        """
        if not len(self.layers) or not self.n_time_points or not self.n_obs_total or not self.n_var:
            return True
        return False

    @property
    def n_time_points(self) -> int:
        """
        Number of time points in this VData object. n_time_points can be extracted directly from self.time_points or
        from the nb of time points in the layers. If no data was given, a default list of time points was created
        with integer values.
        :return: VData's number of time points
        """
        return self.layers.shape[1]

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
        Number of variables in this VData object. n_var can be extracted directly from self.var or from parameters
        supplied during this VData object's creation :
            - nb of variables in the layers
            - nb of variables in varm
            - nb of variables in varp
        :return: VData's number of variables
        """
        return self.layers.shape[3]

    @property
    def shape(self) -> Tuple[int, int, List[int], int]:
        """
        Shape of this VData object (# layers, # time points, # observations, # variables).
        :return: VData's shape.
        """
        return self.layers.shape

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
            raise VValueError("Time points DataFrame should contain a 'value' column.")

        else:
            # cast time points to TimePoint objects
            df['value'] = to_tp_list(df['value'])
            self._time_points = df

    @property
    def time_points_values(self) -> List[TimePoint]:
        """
        Get the list of time points values (with the unit if possible).

        :return: the list of time points values (with the unit if possible).
        """
        return self.time_points.value.values

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

        if isinstance(df, pd.DataFrame):
            if df.shape[0] != self.n_obs_total:
                raise ShapeError(f"'obs' has {df.shape[0]} rows, it should have {self.n_obs_total}.")

        else:
            if df.shape[1] != self.n_obs:
                raise ShapeError(f"'obs' has {df.shape[0]} rows, it should have {self.n_obs}.")

        # cast to TemporalDataFrame
        if isinstance(df, pd.DataFrame):
            df = TemporalDataFrame(df,
                                   time_list=self.obs.time_points_column,
                                   time_col=self.obs.time_points_column_name,
                                   index=self.obs.index,
                                   name='obs')

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

    @property
    def uns(self) -> Dict:
        """
        Get the uns dictionary in this VData.
        :return: the uns dictionary in this VData.
        """
        return self._uns

    @uns.setter
    def uns(self, data: Dict) -> None:
        if not isinstance(data, dict):
            raise VTypeError("'uns' must be a dictionary.")

        else:
            self._uns = dict(zip([str(k) for k in data.keys()], data.values()))

    # Array containers ---------------------------------------------------
    # TODO : docstrings
    @property
    def layers(self) -> VLayerArrayContainer:
        """
        Get the layers in this VData.
        :return: the layers.
        """
        return self._layers

    # @layers.setter
    # def layers(self, data: Optional[Union[ArrayLike, Dict[Any, ArrayLike]]]) -> None:
    #     if data is None:
    #         self._layers = VLayerArrayContainer(self, None)
    #
    #     else:
    #         if isinstance(data, (np.ndarray, pd.DataFrame)):
    #             data = {"data": data}
    #
    #         elif not isinstance(data, dict):
    #             raise VTypeError("'layers' should be set with a 3D array-like object (numpy array) "
    #                              "or with a dictionary of them.")
    #
    #         for arr_index, arr in data.items():
    #             if self.n_time_points > 1:
    #                 if not isinstance(arr, np.ndarray):
    #                     raise VTypeError(f"'{arr_index}' array for layers should be a 3D array-like object "
    #                                      f"(numpy array).")
    #
    #                 elif arr.ndim != 3:
    #                     raise VTypeError(f"'{arr_index}' array for layers should be a 3D array-like object "
    #                                      f"(numpy array).")
    #
    #             else:
    #                 if not isinstance(arr, (np.ndarray, pd.DataFrame)):
    #                     raise VTypeError(f"'{arr_index}' array for layers should be a 2D or 3D array-like object "
    #                                      f"(numpy array, pandas DataFrame).")
    #
    #                 elif arr.ndim not in (2, 3):
    #                     raise VTypeError(f"'{arr_index}' array for layers should be a 2D or 3D array-like object "
    #                                      f"(numpy array, pandas DataFrame).")
    #
    #                 elif arr.ndim == 2:
    #                     data[arr_index] = reshape_to_3D(arr, np.zeros(len(arr)))
    #
    #         self._layers = VLayerArrayContainer(self, data)

    @property
    def obsm(self) -> VObsmArrayContainer:
        """
        Get the obsm in this VData.
        :return: the obsm.
        """
        return self._obsm

    # @obsm.setter
    # def obsm(self, data: Optional[Dict[Any, ArrayLike_3D]]) -> None:
    #     if data is None:
    #         self._obsm = VAxisArrayContainer(self, 'obs', None)
    #
    #     else:
    #         if not isinstance(data, dict):
    #             raise VTypeError("'obsm' should be set with a dictionary of 3D array-like objects (numpy array).")
    #
    #         for arr_index, arr in data.items():
    #             if self.n_time_points > 1:
    #                 if not isinstance(arr, np.ndarray):
    #                     raise VTypeError(
    #                         f"'{arr_index}' array for obsm should be a 3D array-like object (numpy array).")
    #
    #                 elif arr.ndim != 3:
    #                     raise VTypeError(
    #                         f"'{arr_index}' array for obsm should be a 3D array-like object (numpy array).")
    #
    #             else:
    #                 if not isinstance(arr, (np.ndarray, pd.DataFrame)):
    #                     raise VTypeError(
    #                         f"'{arr_index}' array for obsm should be a 2D or 3D array-like object "
    #                         f"(numpy array, pandas DataFrame).")
    #
    #                 elif arr.ndim not in (2, 3):
    #                     raise VTypeError(
    #                         f"'{arr_index}' array for obsm should be a 2D or 3D array-like object "
    #                         f"(numpy array, pandas DataFrame).")
    #
    #                 elif arr.ndim == 2:
    #                     data[arr_index] = reshape_to_3D(arr, np.zeros(len(arr)))
    #
    #         self._obsm = VAxisArrayContainer(self, 'obs', data)

    @property
    def obsp(self) -> VObspArrayContainer:
        """
        Get obsp in this VData.
        :return: the obsp.
        """
        return self._obsp

    # @obsp.setter
    # def obsp(self, data: Optional[Dict[Any, ArrayLike_2D]]) -> None:
    #     if data is None:
    #         self._obsp = VPairwiseArrayContainer(self, 'obs', None)
    #
    #     else:
    #         if not isinstance(data, dict):
    #             raise VTypeError(
    #                 "'obsp' should be set with a dictionary of 2D array-like objects (numpy array).")
    #
    #         for arr_index, arr in data.items():
    #             if not isinstance(arr, (np.ndarray, pd.DataFrame)):
    #                 raise VTypeError(
    #                     f"'{arr_index}' array for obsp should be a 2D array-like object "
    #                     f"(numpy array, pandas DataFrame).")
    #
    #             elif arr.ndim != 2:
    #                 raise VTypeError(
    #                     f"'{arr_index}' array for obsm should be a 2D array-like object "
    #                     f"(numpy array, pandas DataFrame).")
    #
    #         self._obsp = VPairwiseArrayContainer(self, 'obs', data)

    @property
    def varm(self) -> VVarmArrayContainer:
        """
        Get the varm in this VData.
        :return: the varm.
        """
        return self._varm

    # @varm.setter
    # def varm(self, data: Optional[Dict[Any, ArrayLike_3D]]) -> None:
    #     if data is None:
    #         self._varm = VAxisArrayContainer(self, 'var', None)
    #
    #     else:
    #         if not isinstance(data, dict):
    #             raise VTypeError("'varm' should be set with a dictionary of 3D array-like objects (numpy array).")
    #
    #         for arr_index, arr in data.items():
    #             if self.n_time_points > 1:
    #                 if not isinstance(arr, np.ndarray):
    #                     raise VTypeError(f"'{arr_index}' array for varm should be a 3D array-like object "
    #                                      f"(numpy array).")
    #
    #                 elif arr.ndim != 3:
    #                     raise VTypeError(f"'{arr_index}' array for varm should be a 3D array-like object "
    #                                      f"(numpy array).")
    #
    #             else:
    #                 if not isinstance(arr, (np.ndarray, pd.DataFrame)):
    #                     raise VTypeError(
    #                         f"'{arr_index}' array for varm should be a{' 2D or' if self.n_time_points == 1 else ''} "
    #                         f"3D array-like object "
    #                         f"(numpy array{', pandas DataFrame' if self.n_time_points == 1 else ''}).")
    #
    #                 elif arr.ndim not in (2, 3):
    #                     raise VTypeError(
    #                         f"'{arr_index}' array for varm should be a{' 2D or' if self.n_time_points == 1 else ''} "
    #                         f"3D array-like object "
    #                         f"(numpy array{', pandas DataFrame' if self.n_time_points == 1 else ''}).")
    #
    #                 elif arr.ndim == 2:
    #                     data[arr_index] = reshape_to_3D(arr, np.zeros(len(arr)))
    #
    #         self._varm = VAxisArrayContainer(self, 'var', data)

    @property
    def varp(self) -> VVarpArrayContainer:
        """
        Get the varp in this VData.
        :return: the varp.
        """
        return self._varp

    # @varp.setter
    # def varp(self, data: Optional[Dict[Any, ArrayLike_2D]]) -> None:
    #     if data is None:
    #         self._varp = VPairwiseArrayContainer(self, 'var', None)
    #
    #     else:
    #         if not isinstance(data, dict):
    #             raise VTypeError("'varp' should be set with a dictionary of 2D array-like objects (numpy array).")
    #
    #         for arr_index, arr in data.items():
    #             if not isinstance(arr, (np.ndarray, pd.DataFrame)):
    #                 raise VTypeError(f"'{arr_index}' array for varp should be a 2D array-like object "
    #                                  f"(numpy array, pandas DataFrame).")
    #
    #             elif arr.ndim != 2:
    #                 raise VTypeError(f"'{arr_index}' array for varp should be a 2D array-like object "
    #                                  f"(numpy array, pandas DataFrame).")
    #
    #         self._varp = VPairwiseArrayContainer(self, 'var', data)

    # Special ------------------------------------------------------------
    @property
    def dtype(self) -> DType:
        """
        Get the data type of this VData object.
        :return: the data type of this VData object.
        """
        return self._dtype

    @dtype.setter
    def dtype(self, type_: Union[DType, str_DType]) -> None:
        """
        Set the data type of this VData object.
        :param type_: a data type.
        """
        if type_ not in DTypes.keys():
            raise VTypeError(f"Incorrect data-type '{type_}', should be in {list(DTypes.keys())}")
        else:
            self._dtype: DType = DTypes[type_]

        # update dtype of linked Arrays
        self.layers.update_dtype(type_)

        self.obsm.update_dtype(type_)
        self.obsp.update_dtype(type_)
        self.varm.update_dtype(type_)
        self.varp.update_dtype(type_)

        generalLogger.info(f"Set type {type_} for VData object.")

    # Aliases ------------------------------------------------------------
    @property
    def cells(self) -> TemporalDataFrame:
        """
        Alias for the obs attribute.
        :return: the obs TemporalDataFrame.
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
        Alias for the var attribute.
        :return: the var DataFrame.
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

    def _check_formats(self, data: Optional[Union[AnnData, DataFrame, Dict[Any, DataFrame]]],
                       obs: Optional[DataFrame],
                       obsm: Optional[Dict[Any, DataFrame]],
                       obsp: Optional[Dict[Any, Array2D]],
                       var: Optional[pd.DataFrame],
                       varm: Optional[Dict[Any, pd.DataFrame]],
                       varp: Optional[Dict[Any, Array2D]],
                       time_points: Optional[pd.DataFrame],
                       uns: Optional[Dict],
                       time_col: Optional[str] = None,
                       time_list: Optional[List[str]] = None) -> Tuple[
        Optional[Dict[str, TemporalDataFrame]],
        Optional[Dict[str, TemporalDataFrame]], Optional[Dict[str, pd.DataFrame]],
        Optional[Dict[str, pd.DataFrame]], Optional[Dict[str, pd.DataFrame]],
        Optional[pd.Index], Optional[pd.Index]
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

        :return: Arrays in correct format (layers, obsm, obsp, varm, varp, obs index, var index).
        """
        def check_time_match(_time_points: Optional[pd.DataFrame],
                             _time_list: Optional[List[TimePoint]],
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
                    if not all(match_time_points(_time_list, _time_points['value'])):
                        raise VValueError("There are values in 'time_list' unknown in 'time_points'.")

                elif _time_col is not None:
                    if not all(match_time_points(_obs.time_points, _time_points['value'])):
                        raise VValueError("There are values in obs['time_col'] unknown in 'time_points'.")

                return _time_points, len(_time_points)

        generalLogger.debug("  \u23BE Check arrays' formats. -- -- -- -- -- -- -- -- -- -- ")

        obs_index, var_index = None, None
        layers = None

        verified_time_list = to_tp_list(time_list) if time_list is not None else None

        # time_points
        if time_points is not None:
            generalLogger.debug(f"  'time points' DataFrame is a {type(time_points).__name__}.")
            if not isinstance(time_points, pd.DataFrame):
                raise VTypeError("'time points' must be a pandas DataFrame.")

            else:
                if 'value' not in time_points.columns:
                    raise VValueError("'time points' must have at least a column 'value' to store time points value.")

                time_points["value"] = sorted(to_tp_list(time_points["value"]))

                if len(time_points.columns) > 1:
                    time_points[time_points.columns[1:]] = self._check_df_types(time_points[time_points.columns[1:]])

        else:
            generalLogger.debug("  'time points' DataFrame was not found.")

        nb_time_points = 1 if time_points is None else len(time_points)
        generalLogger.debug(f"  {nb_time_points} time point{' was' if nb_time_points == 1 else 's were'} found so far.")
        generalLogger.debug(f"    \u21B3 Time point{' is' if nb_time_points == 1 else 's are'} : "
                            f"{[0] if nb_time_points == 1 else repr_array(time_points.value.values)}")

        # =========================================================================================
        if isinstance(data, AnnData):
            generalLogger.debug('  VData creation from an AnnData.')

            # if an AnnData is being imported, obs, obsm, obsp, var, varm, varp and uns should be None because
            # they will be set from the AnnData
            for attr in ('obs', 'obsm', 'obsp', 'var', 'varm', 'varp', 'uns'):
                if eval(f"{attr} is not None"):
                    raise VValueError(f"'{attr}' should be set to None when importing data from an AnnData.")

            # import and cast obs to a TemporalDataFrame
            obs = TemporalDataFrame(data.obs, time_list=verified_time_list, time_col=time_col, name='obs',
                                    dtype=self.dtype)
            reordering_index = obs.index

            # find time points list
            time_points, nb_time_points = check_time_match(time_points, verified_time_list, time_col, obs)

            generalLogger.debug(f"  {nb_time_points} time point{' was' if nb_time_points == 1 else 's were'} "
                                f"found after data extraction from the AnnData.")
            generalLogger.debug(f"    \u21B3 Time point{' is' if nb_time_points == 1 else 's are'} : "
                                f"{[0] if nb_time_points == 1 else time_points.value.values}")

            if array_isin(data.X, data.layers.values()):
                layers = dict((key, TemporalDataFrame(
                    pd.DataFrame(arr, index=data.obs.index, columns=data.var.index).reindex(reordering_index),
                    time_list=obs.time_points_column, name=key, dtype=self.dtype)
                               ) for key, arr in data.layers.items())

            else:
                layers = dict({"data": TemporalDataFrame(
                    pd.DataFrame(data.X, index=data.obs.index, columns=data.var.index).reindex(reordering_index),
                    time_list=obs.time_points_column, name='data', dtype=self.dtype)
                },
                              **dict((key, TemporalDataFrame(
                                  pd.DataFrame(arr, index=data.obs.index, columns=data.var.index).reindex(
                                      reordering_index),
                                  time_list=obs.time_points_column, name=key, dtype=self.dtype)
                                      ) for key, arr in data.layers.items()))

            # import other arrays
            obsm, obsp = dict(data.obsm), dict(data.obsp)
            var, varm, varp = data.var, dict(data.varm), dict(data.varp)
            uns = dict(data.uns)

        # =========================================================================================
        else:
            generalLogger.debug('  VData creation from scratch.')

            # check formats

            # -----------------------------------------------------------------
            # layers
            if data is not None:
                layers = {}
                _time_points = time_points.value.values if time_points is not None else None

                # data is a unique pandas DataFrame or a TemporalDataFrame
                if isinstance(data, pd.DataFrame):
                    generalLogger.debug("    1. \u2713 'data' is a pandas DataFrame.")

                    if nb_time_points > 1:
                        raise VTypeError("'data' is a 2D pandas DataFrame but more than 1 time points were provided.")

                    obs_index = data.index
                    var_index = data.columns

                    layers = {'data': TemporalDataFrame(data, time_list=verified_time_list, time_points=_time_points,
                                                        dtype=self._dtype, name='data')}

                    if obs is not None and not isinstance(obs, TemporalDataFrame) and verified_time_list is None:
                        verified_time_list = layers['data'].time_points_column

                elif isinstance(data, TemporalDataFrame):
                    generalLogger.debug("    1. \u2713 'data' is a TemporalDataFrame.")

                    if time_points is not None:
                        if not time_points.value.equals(pd.Series(data.time_points)):
                            raise VValueError("'time points' found in DataFrame do not match 'layers' time points.")

                    else:
                        time_points = pd.DataFrame({'value': data.time_points})
                        nb_time_points = data.n_time_points

                    obs_index = data.index
                    var_index = data.columns

                    if obs is not None and not isinstance(obs, TemporalDataFrame) and verified_time_list is None:
                        verified_time_list = data.time_points_column

                    layers = {'data': self._check_df_types(data)}

                elif isinstance(data, dict):
                    generalLogger.debug("    1. \u2713 'data' is a dictionary.")

                    for key, value in data.items():
                        if not isinstance(value, (pd.DataFrame, TemporalDataFrame)):
                            raise VTypeError(f"Layer '{key}' must be a TemporalDataFrame or a pandas DataFrame, "
                                             f"it is a {type(value)}.")

                        elif isinstance(value, pd.DataFrame):
                            if obs_index is None:
                                obs_index = value.index
                                var_index = value.columns

                            layers[str(key)] = TemporalDataFrame(value, time_list=verified_time_list,
                                                                 time_points=_time_points,
                                                                 dtype=self._dtype, name=str(key))

                            if obs is not None and not isinstance(obs, TemporalDataFrame) \
                                    and verified_time_list is None:
                                verified_time_list = layers[str(key)].time_points_column

                        else:
                            if obs_index is None:
                                obs_index = value.index
                                var_index = value.columns

                                if time_points is not None:
                                    if not time_points.value.equals(pd.Series(value.time_points)):
                                        raise VValueError(
                                            f"'time points' found in DataFrame ({repr_array(time_points.value)}) do "
                                            f"not match 'layers' time points ({repr_array(value.time_points)}).")

                                else:
                                    time_points = pd.DataFrame({'value': value.time_points})
                                    nb_time_points = value.n_time_points

                            if obs is not None and not isinstance(obs, TemporalDataFrame) \
                                    and verified_time_list is None:
                                verified_time_list = value.time_points_column

                            value.name = f"{value.name if value.name != 'No_Name' else ''}" \
                                         f"{'_' if value.name != 'No_Name' else ''}" \
                                         f"{str(key)}"
                            layers[str(key)] = self._check_df_types(value)

                else:
                    raise VTypeError(f"Type '{type(data)}' is not allowed for 'data' parameter, should be a dict,"
                                     f"a pandas DataFrame, a TemporalDataFrame or an AnnData object.")

            else:
                generalLogger.debug("    1. \u2717 'data' was not found.")

            # -----------------------------------------------------------------
            # obs
            if obs is not None:
                generalLogger.debug(f"    2. \u2713 'obs' is a {type(obs).__name__}.")

                if not isinstance(obs, (pd.DataFrame, TemporalDataFrame)):
                    raise VTypeError("'obs' must be a pandas DataFrame or a TemporalDataFrame.")

                elif isinstance(obs, pd.DataFrame):
                    _time_points = time_points.value.values if time_points is not None else None
                    obs = TemporalDataFrame(obs, time_list=verified_time_list, time_col=time_col,
                                            time_points=_time_points, dtype=self._dtype,
                                            name='obs')

                else:
                    obs = self._check_df_types(obs)
                    obs.name = f"{obs.name if obs.name != 'No_Name' else ''}" \
                               f"{'_' if obs.name != 'No_Name' else ''}" \
                               f"obs"

                    if verified_time_list is not None:
                        generalLogger.warning("'time_list' parameter cannot be used since 'obs' is already a "
                                              "TemporalDataFrame.")
                    if time_col is not None:
                        generalLogger.warning("'time_col' parameter cannot be used since 'obs' is already a "
                                              "TemporalDataFrame.")

                if obs_index is not None and all(obs.index.isin(obs_index)):
                    obs.reindex(obs_index)

                else:
                    obs_index = obs.index

                # find time points list
                time_points, nb_time_points = check_time_match(time_points, verified_time_list, time_col, obs)

                generalLogger.debug(
                    f"  {nb_time_points} time point{' was' if nb_time_points == 1 else 's were'} "
                    f"found from the provided data.")
                generalLogger.debug(f"    \u21B3 Time point{' is' if nb_time_points == 1 else 's are'} : "
                                    f"{[0] if nb_time_points == 1 else repr_array(time_points.value.values)}")

            else:
                generalLogger.debug("    2. \u2717 'obs' was not found.")
                if verified_time_list is not None:
                    generalLogger.warning("'time_list' parameter cannot be used since 'obs' was not found.")
                if time_col is not None:
                    generalLogger.warning("'time_col' parameter cannot be used since 'obs' was not found.")

            # -----------------------------------------------------------------
            # obsm
            if obsm is not None:
                generalLogger.debug(f"    3. \u2713 'obsm' is a {type(obsm).__name__}.")

                if obs is None and layers is None:
                    raise VValueError("'obsm' parameter cannot be set unless either 'data' or 'obs' are set.")

                valid_obsm = {}
                _time_points = time_points.value.values if time_points is not None else None

                if not isinstance(obsm, dict):
                    raise VTypeError("'obsm' must be a dictionary of DataFrames.")

                else:
                    for key, value in obsm.items():
                        if not isinstance(value, (pd.DataFrame, TemporalDataFrame)):
                            raise VTypeError(f"'obsm' '{key}' must be a TemporalDataFrame or a pandas DataFrame.")

                        elif isinstance(value, pd.DataFrame):
                            valid_obsm[str(key)] = TemporalDataFrame(value, time_list=verified_time_list,
                                                                     time_points=_time_points,
                                                                     dtype=self._dtype, name=str(key))

                        else:
                            value.name = f"{value.name if value.name != 'No_Name' else ''}" \
                                         f"{'_' if value.name != 'No_Name' else ''}" \
                                         f"{str(key)}"
                            valid_obsm[str(key)] = self._check_df_types(value)

                            if verified_time_list is not None:
                                generalLogger.warning(f"'time_list' parameter cannot be used since 'obsm' '{key}' is "
                                                      "already a TemporalDataFrame.")
                            if time_col is not None:
                                generalLogger.warning(f"'time_col' parameter cannot be used since 'obsm' '{key}' is "
                                                      "already a TemporalDataFrame.")

                        if all(valid_obsm[str(key)].index.isin(obs_index)):
                            valid_obsm[str(key)].reindex(obs_index)

                        else:
                            raise VValueError("Index of 'obsm' does not match 'obs' and 'layers' indexes.")

            else:
                generalLogger.debug("    3. \u2717 'obsm' was not found.")

            # -----------------------------------------------------------------
            # obsp
            if obsp is not None:
                generalLogger.debug(f"    4. \u2713 'obsp' is a {type(obsp).__name__}.")

                if obs is None and layers is None:
                    raise VValueError("'obsp' parameter cannot be set unless either 'data' or 'obs' are set.")

                valid_obsp = {}

                if not isinstance(obsp, dict):
                    raise VTypeError("'obsp' must be a dictionary of 2D numpy arrays or pandas DataFrames.")

                else:
                    for key, value in obsp.items():
                        if not isinstance(value, (np.ndarray, pd.DataFrame)) and value.ndim != 2:
                            raise VTypeError(f"'obsp' '{key}' must be a 2D numpy array or pandas DataFrame.")

                        if isinstance(value, pd.DataFrame):
                            if all(value.index.isin(obs_index)):
                                value.reindex(obs_index)

                                if all(value.columns.isin(obs_index)):
                                    value = value[obs_index]

                                else:
                                    raise VValueError("Column names of 'obsp' do not match 'obs' and 'layers' indexes.")

                            else:
                                raise VValueError(f"Index of 'obsp' '{key}' does not match 'obs' and 'layers' indexes.")

                        else:
                            value = pd.DataFrame(value, index=obs_index, columns=obs_index)

                        valid_obsp[str(key)] = self._check_df_types(value)

            else:
                generalLogger.debug("    4. \u2717 'obsp' was not found.")

            # -----------------------------------------------------------------
            # var
            if var is not None:
                generalLogger.debug(f"    5. \u2713 'var' is a {type(var).__name__}.")

                if not isinstance(var, pd.DataFrame):
                    raise VTypeError("var must be a pandas DataFrame.")
                else:
                    var = self._check_df_types(var)

            else:
                generalLogger.debug("    5. \u2717 'var' was not found.")

            # -----------------------------------------------------------------
            # varm
            if varm is not None:
                generalLogger.debug(f"    6. \u2713 'varm' is a {type(varm).__name__}.")

                if var is None and layers is None:
                    raise VValueError("'obsm' parameter cannot be set unless either 'data' or 'var' are set.")

                valid_varm = {}

                if not isinstance(varm, dict):
                    raise VTypeError("'varm' must be a dictionary of DataFrames.")

                else:
                    for key, value in varm.items():
                        if not isinstance(value, pd.DataFrame):
                            raise VTypeError(f"'varm' '{key}' must be a pandas DataFrame.")

                        else:
                            valid_varm[str(key)] = self._check_df_types(value)

                            if all(valid_varm[str(key)].index.isin(var_index)):
                                valid_varm[str(key)].reindex(var_index)

                            else:
                                raise VValueError("Index of 'varm' does not match 'var' and 'layers' column names.")

            else:
                generalLogger.debug("    6. \u2717 'varm' was not found.")

            # -----------------------------------------------------------------
            # varp
            if varp is not None:
                generalLogger.debug(f"    7. \u2713 'varp' is a {type(varp).__name__}.")

                if var is None and layers is None:
                    raise VValueError("'varp' parameter cannot be set unless either 'data' or 'var' are set.")

                valid_varp = {}

                if not isinstance(varp, dict):
                    raise VTypeError("'varp' must be a dictionary of 2D numpy arrays or pandas DataFrames.")

                else:
                    for key, value in varp.items():
                        if not isinstance(value, (np.ndarray, pd.DataFrame)) and value.ndim != 2:
                            raise VTypeError(f"'varp' '{key}' must be 2D numpy array or pandas DataFrame.")

                        if isinstance(value, pd.DataFrame):
                            if all(value.index.isin(var_index)):
                                value.reindex(var_index)

                                if all(value.columns.isin(var_index)):
                                    value = value[var_index]

                                else:
                                    raise VValueError(
                                        f"Column names of 'varp' '{key}' do not match 'var' and 'layers' column names.")

                            else:
                                raise VValueError(f"Index of 'varp' '{key}' does not match 'var' and 'layers' column "
                                                  f"names.")

                        else:
                            value = pd.DataFrame(value, index=var_index, columns=var_index)

                        valid_varp[str(key)] = self._check_df_types(value)

            else:
                generalLogger.debug("    7. \u2717 'varp' was not found.")

            # # -----------------------------------------------------------------
            # uns
            if uns is not None:
                if not isinstance(uns, dict):
                    raise VTypeError("'uns' must be a dictionary.")
                generalLogger.debug("    8. \u2713 'uns' is a dictionary.")

            else:
                generalLogger.debug("    8. \u2717 'uns' was not found.")

        # if time points are not given, assign default values 0, 1, 2, ...
        if time_points is None:
            if layers is not None:
                time_points = pd.DataFrame({'value': to_tp_list(range(list(layers.values())[0].shape[0]))})
            elif obsm is not None:
                time_points = pd.DataFrame({'value': to_tp_list(range(list(obsm.values())[0].shape[0]))})
            elif varm is not None:
                time_points = pd.DataFrame({'value': to_tp_list(range(list(varm.values())[0].shape[0]))})

        if time_points is not None:
            generalLogger.debug(f"  {len(time_points)} time point{' was' if len(time_points) == 1 else 's were'} "
                                f"found finally.")
            generalLogger.debug(f"    \u21B3 Time point{' is' if nb_time_points == 1 else 's are'} : "
                                f"{repr_array(time_points.value.values)}")

        else:
            generalLogger.debug("  Could not find time points.")

        self._obs = obs
        self._var = var
        self._time_points = time_points

        if uns is not None:
            self._uns = dict(zip([str(k) for k in uns.keys()], uns.values()))

        generalLogger.debug(u"  \u23BF Arrays' formats are OK.  -- -- -- -- -- -- -- -- -- ")

        return layers, obsm, obsp, varm, varp, obs_index, var_index

    def _check_df_types(self, df: DataFrame) -> DataFrame:
        """
        Function for coercing data types of the columns and of the index in a pandas DataFrame.
        :param df: a pandas DataFrame or a TemporalDataFrame.
        """
        generalLogger.debug(u"  \u23BE Check DataFrame's column types.  -  -  -  -  -  -  -  -  -  -")
        # check index : convert to correct dtype if it is not a string type
        if self._dtype is not None:
            try:
                df.index.astype(self._dtype)
            except TypeError:
                df.index.astype(np.dtype('O'))

            # check columns : convert to correct dtype if it is not a string type
            if isinstance(df, pd.DataFrame):
                for col_name in df.columns:
                    try:
                        df[col_name].astype(self._dtype)
                        generalLogger.debug(f"Column '{col_name}' set to {self._dtype}.")

                    except (ValueError, TypeError):
                        if df[col_name].dtype.type in (np.datetime64, np.timedelta64, pd.CategoricalDtype.type):
                            generalLogger.debug(f"Column '{col_name}' kept to {df[col_name].dtype.type}.")

                        else:
                            df[col_name].astype(np.dtype('O'))
                            generalLogger.debug(f"Column '{col_name}' set to string or TimePoint.")

            elif isinstance(df, TemporalDataFrame):
                for col_name in df.columns:
                    try:
                        df.asColType(col_name, self._dtype)
                        generalLogger.debug(f"Column '{col_name}' set to {self._dtype}.")

                    except (ValueError, TypeError):
                        df.asColType(col_name, np.dtype('O'))
                        generalLogger.debug(f"Column '{col_name}' set to string.")

            else:
                raise VTypeError(f"Invalid type '{type(df)}' for function '_check_df_types()'.")

        generalLogger.debug(u"  \u23BF DataFrame's column types are OK.  -  -  -  -  -  -  -  -  -  -")

        return df

    def _init_data(self, obs_index: Optional[pd.Index], var_index: Optional[pd.Index]) -> None:
        """
        Function for finishing the initialization of the VData object. It checks for incoherence in the user-supplied
        arrays and raises an error in case something is wrong.
        :param obs_index: If X was supplied as a pandas DataFrame, index of observations
        :param var_index: If X was supplied as a pandas DataFrame, index of variables
        """
        generalLogger.debug("Initialize the VData.")

        # check coherence with number of time points in VData
        if self._time_points is not None:
            for attr in ('layers', 'obsm'):
                dataset = getattr(self, attr)
                if not dataset.empty and len(self._time_points) != dataset.shape[1]:
                    raise IncoherenceError(f"{attr} has {dataset.shape[0]} time point"
                                           f"{'' if dataset.shape[0] == 1 else 's'} but {len(self._time_points)}"
                                           f" {'was' if len(self._time_points) == 1 else 'were'} given.")

        generalLogger.debug("Time points were coherent across arrays.")

        # if data was given as a dataframe, check that obs and data match in row names
        if self.obs.empty and obs_index is not None:
            self.obs = pd.DataFrame(index=obs_index)

        elif obs_index is not None:
            if not self.obs.index.equals(obs_index):
                raise VValueError(f"Indexes in dataFrames 'data' ({obs_index}) and 'obs' ({self.obs.index}) "
                                  f"do not match.")

        # if data was given as a dataframe, check that var row names match data col names
        if self.var.empty and var_index is not None:
            self.var = pd.DataFrame(index=var_index)

        elif var_index is not None:
            if not self.var.index.equals(var_index):
                raise VValueError(f"Columns in dataFrame 'data' ({var_index}) do not match index of 'var' "
                                  f"({self.var.index}).")

        # check coherence between layers, obs, var and time points
        if self._layers is not None:
            for layer_name, layer in self._layers.items():
                if layer.shape != (self.n_time_points, self.n_obs, self.n_var):
                    if layer.shape[0] != self.n_time_points:
                        raise IncoherenceError(f"layer '{layer_name}' has incoherent number of time points "
                                               f"{layer.shape[0]}, should be {self.n_time_points}.")

                    elif [layer[i].shape[0] for i in range(len(layer))] != self.n_obs:
                        for i in range(len(layer)):
                            if layer[i].shape[0] != self.n_obs[i]:
                                raise IncoherenceError(f"layer '{layer_name}' has incoherent number of observations "
                                                       f"{layer[i].shape[0]}, should be {self.n_obs[i]}.")

                    else:
                        raise IncoherenceError(f"layer '{layer_name}' has incoherent number of variables "
                                               f"{layer[0].shape[1]}, should be {self.n_var}.")

        # check coherence between obs, obsm and obsp shapes
        for attr in ('obsm', 'obsp'):
            dataset = getattr(self, attr)
            if not dataset.empty and self.n_obs != dataset.shape[2]:
                raise IncoherenceError(f"'obs' and '{attr}' have different lengths ({self.n_obs} vs "
                                       f"{dataset.shape[2]})")

        # check coherence between var, varm, varp shapes
        for attr in ('varm', 'varp'):
            dataset = getattr(self, attr)
            if not dataset.empty and self.n_var != dataset.shape[1]:
                raise IncoherenceError(f"'var' and 'varm' have different lengths ({self.n_var} vs "
                                       f"{dataset.shape[1]})")

    # writing ------------------------------------------------------------
    def write(self, file: Union[str, Path]) -> None:
        """
        Save this VData object in HDF5 file format.

        :param file: path to save the VData
        """
        write_vdata(self, file)

    def write_to_csv(self, directory: Union[str, Path], sep: str = ",", na_rep: str = "",
                     index: bool = True, header: bool = True) -> None:
        """
        Save layers, time_points, obs, obsm, obsp, var, varm and varp to csv files in a directory.

        :param directory: path to a directory for saving the matrices
        :param sep: delimiter character
        :param na_rep: string to replace NAs
        :param index: write row names ?
        :param header: Write col names ?
        """
        write_vdata_to_csv(self, directory, sep, na_rep, index, header)

    # copy ---------------------------------------------------------------
    def copy(self) -> 'VData':
        """
        Build a deep copy of this VData object and not a view.
        :return: a new VData, which is a deep copy of this VData.
        """
        _obsp = {key: pd.DataFrame(index=self.obs.index, columns=self.obs.index) for key in self.obsp.keys()}

        index_cumul = 0
        for key in self.obsp.keys():
            for arr in self.obsp[key]:
                _obsp[key].iloc[index_cumul:index_cumul + len(arr), index_cumul:index_cumul + len(arr)] = arr
                index_cumul += len(arr)

        return VData(data=self.layers.dict_copy(),
                     obs=self.obs, obsm=self.obsm.dict_copy(), obsp=_obsp,
                     var=self.var, varm=self.varm.dict_copy(), varp=self.varp.dict_copy(),
                     time_points=self.time_points,
                     uns=self.uns,
                     dtype=self.dtype,
                     name=f"{self.name}_copy")

    # conversion ---------------------------------------------------------
    def to_AnnData(self, time_points_list: Optional[Union[str, TimePoint, Collection[Union[str, TimePoint]]]] = None,
                   into_one: bool = True, time_points_column: str = 'time_points') \
            -> Union[AnnData, List[AnnData]]:
        """
        Convert a VData object to an AnnData object.

        :param time_points_list: a list of time points for which to extract data to build the AnnData. If set to
            None, all time points are selected.
        :param into_one: Build one AnnData, concatenating the data for multiple time points (True), or build one
            AnnData for each time point (False) ?
        :param time_points_column: a column name for storing time points data in the obs DataFrame. This is only used
            when concatenating the data into a single AnnData (i.e. into_one=True). Set to 'time_points' by default.
        :return: an AnnData object with data for selected time points.
        """
        generalLogger.debug(u'\u23BE VData conversion to AnnData : begin '
                            u'---------------------------------------------------------- ')

        if time_points_list is None:
            _time_points_list = np.array(self.time_points_values)

        else:
            _time_points_list = to_tp_list(time_points_list)
            _time_points_list = np.array(_time_points_list)[np.where(match_time_points(_time_points_list,
                                                                                       self.time_points_values))]

        generalLogger.debug(f"Selected time points are : {repr_array(_time_points_list)}")

        if into_one:
            generalLogger.debug("Convert to one AnnData object.")

            generalLogger.debug('\u23BF VData conversion to AnnData : end '
                                '---------------------------------------------------------- ')

            view = self[_time_points_list]
            X_layer = list(view.layers.keys())[0]

            X = view.layers[X_layer].to_pandas()
            X.index = X.index.astype(str)
            X.columns = X.columns.astype(str)

            return AnnData(X=X,
                           layers={key: layer.to_pandas() for key, layer in view.layers.items()},
                           obs=view.obs.to_pandas(with_time_points=time_points_column),
                           obsm={key: arr.to_pandas() for key, arr in view.obsm.items()},
                           var=view.var,
                           varm={key: arr for key, arr in view.varm.items()},
                           varp={key: arr for key, arr in view.varp.items()},
                           uns=view.uns)

        else:
            generalLogger.debug("Convert to many AnnData objects.")

            result = []
            for time_point in _time_points_list:
                view = self[time_point]
                X_layer = list(view.layers.keys())[0]

                X = view.layers[X_layer].to_pandas()
                X.index = X.index.astype(str)
                X.columns = X.columns.astype(str)

                result.append(AnnData(X=X,
                                      layers={key: layer.to_pandas() for key, layer in view.layers.items()},
                                      obs=view.obs.to_pandas(),
                                      obsm={key: arr.to_pandas() for key, arr in view.obsm.items()},
                                      var=view.var,
                                      varm={key: arr for key, arr in view.varm.items()},
                                      varp={key: arr for key, arr in view.varp.items()},
                                      uns=view.uns))

            generalLogger.debug(u'\u23BF VData conversion to AnnData : end '
                                u'---------------------------------------------------------- ')

            return result
