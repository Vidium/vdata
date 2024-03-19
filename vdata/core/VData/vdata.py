# coding: utf-8
# Created on 11/4/20 10:03 AM
# Author : matteo

# ====================================================
# imports
from __future__ import annotations
import logging

import numpy as np
import pandas as pd
from pathlib import Path

# from sys import getsizeof
from anndata import AnnData
from scipy.sparse import spmatrix
from pympler.asizeof import asizeof

from typing import (
    Optional,
    Union,
    Any,
    TypeVar,
    Collection,
    Iterator,
    Sequence,
    cast,
    Literal,
)

from vdata.core.VData.name_utils import DataFrame
from vdata.core.VData.utils import array_isin
from vdata.core.VData.arrays import (
    VLayerArrayContainer,
    VObsmArrayContainer,
    VObspArrayContainer,
    VVarmArrayContainer,
    VVarpArrayContainer,
)
from vdata.core.VData.views import ViewVData
from vdata.core.backed_dict import BackedDict
from vdata.core.name_utils import PreSlicer
from vdata.core.tdf.base import BaseTemporalDataFrame
from vdata.core.utils import (
    reformat_index,
    to_tp_list,
    match_timepoints,
    repr_index,
    list_to_tp_list_strict,
)
from vdata.core.tdf import TemporalDataFrame
from vdata.core.tdf.name_utils import DEFAULT_TIME_POINTS_COL_NAME
from vdata.name_utils import DType, StrDType
from vdata.utils import repr_array, deep_dict_convert
from vdata.time_point import TimePoint
from vdata.IO import (
    VTypeError,
    IncoherenceError,
    VValueError,
    ShapeError,
    VClosedFileError,
    VReadOnlyError,
)
from vdata.read_write import write_vdata, write_vdata_to_csv, H5GroupReader
from vdata.vdataframe import VDataFrame

DF = TypeVar("DF", bound=DataFrame)
Array2D = Union[pd.DataFrame, np.ndarray, VDataFrame]


# ====================================================
# code
logger = logging.getLogger("vdata")


class VData:
    """
    A VData object stores data points in matrices of observations x variables in the same way as the AnnData object,
    but also accounts for the time information. The 2D matrices in AnnData are replaced by 3D matrices here.
    """

    __slots__ = (
        "name",
        "_file",
        "_dtype",
        "_obs",
        "_obsm",
        "_obsp",
        "_var",
        "_varm",
        "_varp",
        "_timepoints",
        "_uns",
        "_layers",
    )

    def __init__(
        self,
        data: Optional[Union[AnnData, "DataFrame", dict[Any, "DataFrame"]]] = None,
        obs: Optional["DataFrame"] = None,
        obsm: Optional[dict[Any, "DataFrame"]] = None,
        obsp: Optional[dict[Any, Array2D]] = None,
        var: Optional[Union[pd.DataFrame, VDataFrame]] = None,
        varm: Optional[dict[Any, Union[pd.DataFrame, VDataFrame]]] = None,
        varp: Optional[dict[Any, Array2D]] = None,
        timepoints: Optional[Union[pd.DataFrame, VDataFrame]] = None,
        uns: Optional[dict] = None,
        time_col_name: Optional[str] = None,
        time_list: Optional[Sequence[Union[str, TimePoint]]] = None,
        dtype: Optional[Union["DType", "StrDType"]] = None,
        name: Optional[Any] = None,
        file: Optional[H5GroupReader] = None,
        no_check: bool = False,
    ):
        """
        Args:
            data: a single array-like object or a dictionary of them for storing data for each observation/cell
                and for each variable/gene.
                'data' can also be an AnnData to be converted to the VData format.
            obs: a pandas DataFrame or a TemporalDataFrame describing the observations/cells.
            obsm: a dictionary of array-like objects describing measurements on the observations/cells.
            obsp: a dictionary of array-like objects describing pairwise comparisons on the observations/cells.
            var: a pandas DataFrame describing the variables/genes.
            varm: a dictionary of array-like objects describing measurements on the variables/genes.
            varp: a dictionary of array-like objects describing pairwise comparisons on the variables/genes.
            timepoints: a pandas DataFrame describing the times points.
            uns: a dictionary of unstructured data.
            time_col_name: if obs is a pandas DataFrame (or the VData is created from an AnnData), the column name in
                obs that contains time information.
            time_list: if obs is a pandas DataFrame (or the VData is created from an AnnData), a list containing
                time information of the same length as the number of rows in obs.
            dtype: a data type to impose on datasets stored in this VData.
            name: a name for this VData.
            file: an open h5 file from which this VData is read.
            no_check: skip checks (only use if you guaranty the data passed to create this VData is valid !)
        """
        self.name = str(name) if name is not None else "No_Name"

        logger.debug(
            f"\u23be VData '{self.name}' creation : begin "
            f"-------------------------------------------------------- "
        )

        self._file = file

        # first, check dtype is correct
        self._dtype = np.dtype(dtype)
        logger.debug(f"Set data-type to {self._dtype}")

        if no_check:
            (
                _obs,
                _var,
                _layers,
                _timepoints,
                _obsm,
                _obsp,
                _varm,
                _varp,
                obs_index,
                var_index,
            ) = obs, var, data, timepoints, obsm, obsp, varm, varp, obs.index, var.index

            # TODO : split class into regular VData and backed VData
            if not isinstance(_layers, (dict, BackedDict)):
                raise VTypeError(
                    "'data' parameter should be an optional dictionary of [str, TemporalDataFrame] when "
                    "using 'no_check' !"
                )

        else:
            # check formats of arguments
            (
                _obs,
                _var,
                _layers,
                _timepoints,
                _obsm,
                _obsp,
                _varm,
                _varp,
                obs_index,
                var_index,
                uns,
            ) = self._check_formats(
                data,
                obs,
                obsm,
                obsp,
                var,
                varm,
                varp,
                timepoints,
                uns,
                time_col_name,
                time_list,
            )

        if file is not None:
            self._uns = uns

        elif uns is not None:
            self._uns = dict(zip([str(k) for k in uns.keys()], uns.values()))

        else:
            self._uns = {}

        ref_TDF = list(_layers.values())[0] if _layers is not None else None

        # make sure a TemporalDataFrame is set to .obs, even if not data was supplied
        if _obs is None:
            logger.debug("Default empty TemporalDataFrame for obs.")

            time_list = ref_TDF.timepoints_column if ref_TDF is not None else None

            self._obs = TemporalDataFrame(
                time_list=time_list,
                index=(ref_TDF.index if ref_TDF is not None else None)
                if obs_index is None
                else obs_index,
                name="obs",
                lock=(True, False),
            )

        else:
            self._obs = _obs

        # make sure a pandas DataFrame is set to .var and .timepoints, even if no data was supplied
        if _var is None:
            logger.debug("Default empty DataFrame for vars.")
            self._var = VDataFrame(
                index=(range(ref_TDF.shape[2]) if ref_TDF is not None else None)
                if var_index is None
                else var_index,
                file=self._file.group["var"] if self._file is not None else None,
            )

        else:
            self._var = _var

        if _timepoints is None:
            logger.debug("Default empty DataFrame for time points.")
            self._timepoints = VDataFrame(
                {"value": self.obs.timepoints},
                file=self._file.group["timepoints"] if self._file is not None else None,
            )
        else:
            self._timepoints = _timepoints

        # create arrays linked to VData
        self._layers = VLayerArrayContainer(self, data=_layers)

        logger.debug(
            f"Guessed dimensions are : ({self.n_timepoints}, {self.n_obs}, {self.n_var})"
        )

        self._obsm = VObsmArrayContainer(self, data=_obsm)
        self._obsp: VObspArrayContainer = VObspArrayContainer(
            self,
            data=_obsp,
            file=self._file.group["obsp"] if self._file is not None else None,
        )
        self._varm = VVarmArrayContainer(
            self,
            data=_varm,
            file=self._file.group["varm"] if self._file is not None else None,
        )
        self._varp = VVarpArrayContainer(
            self,
            data=_varp,
            file=self._file.group["varp"] if self._file is not None else None,
        )

        # finish initializing VData
        self._init_data(obs_index, var_index)

        logger.debug(
            f"\u23bf VData '{self.name}' creation : end "
            f"---------------------------------------------------------- "
        )

    def __repr__(self) -> str:
        """
        Description for this Vdata object to print.
        :return: a description of this Vdata object
        """
        if not self.is_closed:
            _n_obs = (
                self.n_obs
                if len(self.n_obs) > 1
                else self.n_obs[0]
                if len(self.n_obs)
                else 0
            )

            if self.empty:
                repr_str = (
                    f"Empty {'backed ' if self.is_backed else ''}VData '{self.name}' ({_n_obs} obs x"
                    f" {self.n_var} vars over {self.n_timepoints} time point"
                    f"{'' if self.n_timepoints == 1 else 's'})."
                )

            else:
                repr_str = (
                    f"{'Backed ' if self.is_backed else ''}VData '{self.name}' with n_obs x n_var = {_n_obs} x"
                    f" {self.n_var} over {self.n_timepoints} time point{'' if self.n_timepoints == 1 else 's'}."
                )

            for attr in [
                "layers",
                "obs",
                "var",
                "timepoints",
                "obsm",
                "varm",
                "obsp",
                "varp",
            ]:
                obj = getattr(self, attr)
                if not obj.empty:
                    if isinstance(obj, BaseTemporalDataFrame):
                        repr_str += f"\n\t{attr}: {str(list(obj.columns))[1:-1]}"

                    else:
                        repr_str += f"\n\t{attr}: {str(list(obj.keys()))[1:-1]}"

            if len(self.uns):
                repr_str += f"\n\tuns: {str(list(self.uns.keys()))[1:-1]}"

            return repr_str

        else:
            return "Backed VData with closed file."

    def __del__(self) -> None:
        """
        Close file on object delete.
        """
        if self._file is not None and not self.is_closed:
            self._file.close()

    def __getitem__(
        self,
        index: Union[
            "PreSlicer",
            tuple["PreSlicer", "PreSlicer"],
            tuple["PreSlicer", "PreSlicer", "PreSlicer"],
        ],
    ) -> ViewVData:
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
        if self.is_closed:
            raise VClosedFileError("Cannot get data, file is closed.")

        logger.debug("VData sub-setting - - - - - - - - - - - - - - ")
        logger.debug(f"  Got index \n{repr_index(index)}")

        formatted_index = reformat_index(
            index, self.timepoints.value.values, self.obs.index, self.var.index
        )

        logger.debug(f"  Refactored index to \n{repr_index(formatted_index)}")

        if formatted_index[0] is not None and not len(formatted_index[0]):
            raise VValueError("Time points not found in this VData.")

        return ViewVData(
            self, formatted_index[0], formatted_index[1], formatted_index[2]
        )

    @property
    def is_backed(self) -> bool:
        """
        Is this VData object backed on an h5 file ?
        :return: Is this VData object backed on an h5 file ?
        """
        return self._file is not None

    @property
    def is_backed_w(self) -> bool:
        """
        Is this VData object backed on an h5 file and writable ?
        :return: Is this VData object backed on an h5 file and writable ?
        """
        return self._file is not None and self._file.mode == "r+"

    @property
    def is_closed(self) -> bool:
        """
        Is this VData's file closed ?

        Returns:
            Is this VData's file closed ?
        """
        return self._file is not None and not self._file.group.id.valid

    @property
    def is_read_only(self) -> bool:
        """
        Is this VData's file open in read only mode ?
        """
        return self._file is not None and self._file.mode == "r"

    @property
    def has_repeated_obs_index(self) -> bool:
        if not self.obs.empty:
            return self.obs.has_repeating_index

        elif not self.layers.empty:
            return self.layers.has_repeating_index

        return False

    @property
    def file(self) -> Optional[H5GroupReader]:
        """
        Get this VData's h5 file.
        :return: this VData's h5 file.
        """
        return self._file

    @file.setter
    def file(self, file_reader: H5GroupReader) -> None:
        """
        Set this VData's h5 file.

        Args:
            file_reader: a h5 file reader for this VData.
        """
        if not isinstance(file_reader, H5GroupReader):
            raise VTypeError(
                f"Cannot read h5 file from an object of type '{type(file_reader)}'."
            )

        self._file = file_reader

        # self.layers.set_file(file_reader.group['layers'])
        # self.obs.file = file_reader.group['obs']
        # self.obsm.set_file(file_reader.group['obsm'])
        self.obsp.set_file(file_reader.group["obsp"])
        self.var.file = file_reader.group["var"]
        self.varm.set_file(file_reader.group["varm"])
        self.varp.set_file(file_reader.group["varp"])
        # self.timepoints.file = file_reader.group['timepoints']

    @property
    def size_in_memory(self) -> int:
        """
        Get the in-memory size of this VData (in bytes).
        """
        return asizeof(self)

    @property
    def size_in_file(self) -> int:
        """"""
        # TODO
        return 1 if self.is_backed else 0

    @property
    def size(self) -> tuple[int, int]:
        """"""
        return self.size_in_memory, self.size_in_file

    # Shapes -------------------------------------------------------------
    @property
    def empty(self) -> bool:
        """
        Is this VData object empty ? (no time points or no obs or no vars)

        Returns:
            VData empty ?
        """
        if (
            not len(self.layers)
            or not self.n_timepoints
            or not self.n_obs_total
            or not self.n_var
        ):
            return True
        return False

    @property
    def n_timepoints(self) -> int:
        """
        Number of time points in this VData object. n_timepoints can be extracted directly from self.timepoints or
        from the nb of time points in the layers. If no data was given, a default list of time points was created
        with integer values.
        Returns:
            VData's number of time points
        """
        return self.timepoints.shape[0]

    @property
    def n_obs(self) -> list[int]:
        """
        Number of observations in this VData object per time point. n_obs can be extracted directly from self.obs
        or from parameters supplied during this VData object's creation :
            - nb of observations in the layers
            - nb of observations in obsm
            - nb of observations in obsp

        Returns:
            VData's number of observations
        """
        return self.layers.shape[2]

    @property
    def n_obs_total(self) -> int:
        """
        Get the total number of observations across all time points.

        Returns:
            The total number of observations across all time points.
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

        Returns:
            VData's number of variables
        """
        return self.layers.shape[3]

    @property
    def shape(self) -> tuple[int, int, list[int], int]:
        """
        Shape of this VData object (# layers, # time points, # observations, # variables).
        Returns:
            VData's shape.
        """
        return self.layers.shape

    # DataFrames ---------------------------------------------------------
    @property
    def timepoints(self) -> VDataFrame:
        """
        Get time points data.
        :return: the time points DataFrame.
        """
        return self._timepoints

    @timepoints.setter
    def timepoints(self, df: Union[pd.DataFrame, VDataFrame]) -> None:
        """
        Set the time points data.
        Args:
            df: a pandas DataFrame with at least the 'value' column.
        """
        if self.is_read_only:
            raise VReadOnlyError

        if not isinstance(df, (pd.DataFrame, VDataFrame)):
            raise VTypeError("'time points' must be a pandas DataFrame.")

        elif df.shape[0] != self.n_timepoints:
            raise ShapeError(
                f"'time points' has {df.shape[0]} lines, it should have {self.n_timepoints}."
            )

        elif "value" not in df.columns:
            raise VValueError("Time points DataFrame should contain a 'value' column.")

        else:
            # cast time points to TimePoint objects
            df["value"] = to_tp_list(df["value"])
            self._timepoints = df

    @property
    def timepoints_values(self) -> list["TimePoint"]:
        """
        Get the list of time points values (with the unit if possible).

        :return: the list of time points values (with the unit if possible).
        """
        return self.timepoints.value.values

    @property
    def timepoints_strings(self) -> Iterator[str]:
        """
        Get the list of time points as strings.

        :return: the list of time points as strings.
        """
        return map(str, self.timepoints.value.values)

    @property
    def timepoints_numerical(self) -> list[float]:
        """
        Get the list of bare values from the time points.

        :return: the list of bare values from the time points.
        """
        return [tp.value for tp in self.timepoints.value]

    @property
    def obs(self) -> TemporalDataFrame:
        """
        Get the obs data.
        :return: the obs TemporalDataFrame.
        """
        return self._obs

    @obs.setter
    def obs(self, df: Union[pd.DataFrame, VDataFrame, TemporalDataFrame]) -> None:
        """
        Set the obs data.

        Args:
            df: a pandas DataFrame or a TemporalDataFrame.
        """
        if self.is_read_only:
            raise VReadOnlyError

        if not isinstance(df, (pd.DataFrame, VDataFrame, TemporalDataFrame)):
            raise VTypeError("'obs' must be a pandas DataFrame or a TemporalDataFrame.")

        if not df.shape[0] == self.obs.n_index_total:
            raise ShapeError(
                f"'obs' has {df.shape[0]} rows, it should have {self.n_obs_total}."
            )

        if isinstance(df, (pd.DataFrame, VDataFrame)):
            # cast to TemporalDataFrame
            if (
                self.obs.timepoints_column_name is not None
                and self.obs.timepoints_column_name in df.columns
            ):
                _time_col_name: Optional[str] = self.obs.timepoints_column_name
            else:
                _time_col_name = None

            _time_list = self.obs.timepoints_column if _time_col_name is None else None

            df = TemporalDataFrame(
                df,
                time_list=_time_list,
                time_col_name=_time_col_name,
                index=self.obs.index,
                name="obs",
            )

        else:
            if df.timepoints != self.obs.timepoints:
                raise VValueError("'obs' time points do not match.")

            if not np.all(df.index == self.obs.index):
                raise VValueError("'obs' index does not match.")

            if not np.all(df.columns == self.obs.columns):
                raise VValueError("'obs' column names do not match.")

        self._obs = df
        self._obs.lock_indices()

    def set_obs_index(self, values: Collection, repeating_index: bool = False) -> None:
        """
        Set a new index for observations.

        Args:
            values: collection of new index values.
            repeating_index: does the index repeat itself at all time-points ? (default: False)
        """
        if self.is_read_only:
            raise VReadOnlyError

        values = np.array(values)

        for layer in self.layers.values():
            layer.unlock_indices()
            layer.set_index(values, repeating_index)
            layer.lock_indices()

        self.obs.unlock_indices()
        self.obs.set_index(values, repeating_index)
        self.obs.lock_indices()

        for TDF in self.obsm.values():
            TDF.unlock_indices()
            TDF.set_index(values, repeating_index)
            TDF.lock_indices()

        self.obsp.set_index(values)

    def make_unique_obs_index(self) -> None:
        """
        Concatenates the obs index with the time-point to make all index values unique.
        """
        self.set_obs_index(
            np.char.add(
                np.char.add(self.obs.index.astype(str), "_"),
                self.obs.timepoints_column_str,
            )
        )

    @property
    def var(self) -> VDataFrame:
        """
        Get the var data.
        :return: the var DataFrame.
        """
        return self._var

    @var.setter
    def var(self, df: Union[pd.DataFrame, VDataFrame]) -> None:
        """
        Set the var data.
        Args:
            df: a pandas DataFrame.
        """
        if self.is_read_only:
            raise VReadOnlyError

        if not isinstance(df, (pd.DataFrame, VDataFrame)):
            raise VTypeError("'var' must be a pandas DataFrame.")

        elif df.shape[0] != self.n_var:
            raise ShapeError(
                f"'var' has {df.shape[0]} lines, it should have {self.n_var}."
            )

        else:
            self._var = df

    def set_var_index(self, values: Collection) -> None:
        """
        Set a new index for variables.
        Args:
            values: collection of new index values.
        """
        if self.is_read_only:
            raise VReadOnlyError

        for layer in self.layers.values():
            layer.unlock_columns()
            layer.columns = values
            layer.lock_columns()

        self.var.index = values

        for df in self.varm.values():
            df.index = values

        self.varp.set_index(values)

    @property
    def uns(self) -> dict:
        """
        Get the uns dictionary in this VData.
        :return: the uns dictionary in this VData.
        """
        return self._uns

    @uns.setter
    def uns(self, data: dict | BackedDict) -> None:
        if self.is_read_only:
            raise VReadOnlyError

        if isinstance(data, BackedDict):
            self._uns = data

        elif isinstance(data, dict):
            self._uns = dict(zip([str(k) for k in data.keys()], data.values()))

        else:
            raise VTypeError("'uns' must be a dictionary.")

    # Array containers ---------------------------------------------------
    @property
    def layers(self) -> VLayerArrayContainer:
        """
        Get the layers in this VData.
        :return: the layers.
        """
        return self._layers

    @property
    def obsm(self) -> VObsmArrayContainer:
        """
        Get the obsm in this VData.
        :return: the obsm.
        """
        return self._obsm

    @property
    def obsp(self) -> VObspArrayContainer:
        """
        Get obsp in this VData.
        :return: the obsp.
        """
        return self._obsp

    @property
    def varm(self) -> VVarmArrayContainer:
        """
        Get the varm in this VData.
        :return: the varm.
        """
        return self._varm

    @property
    def varp(self) -> VVarpArrayContainer:
        """
        Get the varp in this VData.
        :return: the varp.
        """
        return self._varp

    # Special ------------------------------------------------------------
    @property
    def dtype(self) -> np.dtype:
        """
        Get the data type of this VData object.
        :return: the data type of this VData object.
        """
        return self._dtype

    @dtype.setter
    def dtype(self, dtype: Union["DType", "StrDType"]) -> None:
        """
        Set the data type of this VData object.
        Args:
            dtype: a data type.
        """
        if self.is_read_only:
            raise VReadOnlyError

        self._dtype = np.dtype(dtype)

        # update dtype of linked Arrays
        self.layers.update_dtype(dtype)

        self.obsm.update_dtype(dtype)
        self.obsp.update_dtype(dtype)
        self.varm.update_dtype(dtype)
        self.varp.update_dtype(dtype)

        logger.info(f"Set type {dtype} for VData object.")

    # Aliases ------------------------------------------------------------
    @property
    def cells(self) -> TemporalDataFrame:
        """
        Alias for the obs attribute.
        :return: the obs TemporalDataFrame.
        """
        return self._obs

    @cells.setter
    def cells(self, df: Union[pd.DataFrame, VDataFrame, TemporalDataFrame]) -> None:
        """
        Set cells (= obs) data.
        :param df: a pandas DataFrame or a TemporalDataFrame.
        """
        self.obs = df

    @property
    def genes(self) -> VDataFrame:
        """
        Alias for the var attribute.
        :return: the var DataFrame.
        """
        return self._var

    @genes.setter
    def genes(self, df: VDataFrame) -> None:
        """
        Set the var (= genes) data.
        :param df: a pandas DataFrame.
        """
        self.var = df

    @property
    def timepoints_names(self) -> pd.Index:
        """
        Alias for the time points index names.

        Returns:
            The time points index names.
        """
        return self.timepoints.index

    @property
    def obs_names(self) -> pd.Index:
        """
        Alias for the obs index names.

        Returns:
            The obs index names.
        """
        return pd.Index(self.obs.index)

    @property
    def var_names(self) -> pd.Index:
        """
        Alias for the var index names.

        Returns:
            The var index names.
        """
        return self.var.index

    # init functions -----------------------------------------------------

    def _check_formats(
        self,
        data: Optional[Union[AnnData, "DataFrame", dict[Any, "DataFrame"]]],
        obs: Optional["DataFrame"],
        obsm: Optional[dict[Any, "DataFrame"]],
        obsp: Optional[dict[Any, Array2D]],
        var: Optional[Union[pd.DataFrame, VDataFrame]],
        varm: Optional[dict[Any, Union[pd.DataFrame, VDataFrame]]],
        varp: Optional[dict[Any, Array2D]],
        timepoints: Optional[Union[pd.DataFrame, VDataFrame]],
        uns: Optional[dict],
        time_col_name: Optional[str] = None,
        time_list: Optional[Sequence[Union[str, TimePoint]]] = None,
    ) -> tuple[
        Optional[TemporalDataFrame],
        Optional[VDataFrame],
        Optional[dict[str, TemporalDataFrame]],
        Optional[VDataFrame],
        Optional[dict[str, TemporalDataFrame]],
        Optional[dict[str, VDataFrame]],
        Optional[dict[str, VDataFrame]],
        Optional[dict[str, VDataFrame]],
        Optional[np.ndarray],
        Optional[np.ndarray],
        Optional[dict[Any, Any]],
    ]:
        """
        Function for checking the types and formats of the parameters supplied to the VData object at creation.
        If the types are not accepted, an error is raised. obsm, obsp, varm, varp and layers are prepared for
        being converted into custom arrays for maintaining coherence with this VData object.

        Args:
            data: a single array-like object or a dictionary of them for storing data for each observation/cell
                and for each variable/gene.
                'data' can also be an AnnData to be converted to the VData format.
            obs: a pandas DataFrame or a TemporalDataFrame describing the observations/cells
            obsm: a dictionary of array-like objects describing measurements on the observations/cells
            obsp: a dictionary of array-like objects describing pairwise comparisons on the observations/cells
            var: a pandas DataFrame describing the variables/genes
            varm: a dictionary of array-like objects describing measurements on the variables/genes
            varp: a dictionary of array-like objects describing pairwise comparisons on the variables/genes
            timepoints: a DataFrame describing the times points
            uns: a dictionary of unstructured data
            time_col_name: if obs is a pandas DataFrame (or the VData is created from an AnnData), the column name
                in obs that contains time information.
            time_list: if obs is a pandas DataFrame (or the VData is created from an AnnData), a list containing
                time information of the same length as the number of rows in obs.

        Returns:
            Arrays in correct format (layers, obsm, obsp, varm, varp, obs index, var index).
        """

        def check_time_match(
            _timepoints: Optional[Union[pd.DataFrame, VDataFrame]],
            _time_list: Optional[list[TimePoint]],
            _time_col_name: Optional[str],
            _obs: TemporalDataFrame,
        ) -> tuple[Optional[VDataFrame], int]:
            """
            Build timepoints DataFrame if it was not given by the user but 'time_list' or 'time_col_name' were given.
            Otherwise, if both timepoints and 'time_list' or 'time_col_name' were given, check that they match.

            Args:
                _timepoints: a pandas DataFrame with time points data.
                _time_list: a list of time points of the same length as the number of rows in obs.
                _time_col_name: a column name which contains time points information in obs.
                _obs: the obs TemporalDataFrame.

            :return: a time points DataFrame if possible and the number of found time points.
            """
            if _timepoints is None:
                # build timepoints DataFrame from time_list or time_col_name
                if _time_list is not None or _time_col_name is not None:
                    if _time_list is not None:
                        unique_timepoints = np.unique(_time_list)

                    else:
                        unique_timepoints = _obs.timepoints

                    return VDataFrame({"value": unique_timepoints}), len(
                        unique_timepoints
                    )

                # timepoints cannot be guessed
                else:
                    return None, 1

            # check that timepoints and _time_list and _time_col_name match
            else:
                if _time_list is not None:
                    if not all(match_timepoints(_time_list, _timepoints["value"])):
                        raise VValueError(
                            "There are values in 'time_list' unknown in 'timepoints'."
                        )

                elif _time_col_name is not None:
                    if not all(match_timepoints(_obs.timepoints, _timepoints["value"])):
                        raise VValueError(
                            "There are values in obs['time_col_name'] unknown in 'timepoints'."
                        )

                return (
                    VDataFrame(
                        _timepoints,
                        file=self._file.group["timepoints"]
                        if self._file is not None
                        else None,
                    ),
                    len(_timepoints),
                )

        def no_dense_data(_data: Union[np.ndarray, spmatrix]) -> np.ndarray:
            """
            Convert sparse matrices to dense.
            """
            if isinstance(_data, spmatrix):
                return _data.todense()

            return _data

        logger.debug("  \u23be Check arrays' formats. -- -- -- -- -- -- -- -- -- -- ")

        _timepoints_VDF: Optional[VDataFrame] = None
        obs_index: Optional[np.ndarray] = None
        repeating_obs_index = False
        var_index: Optional[np.ndarray] = None
        layers: Optional[dict[str, TemporalDataFrame]] = None

        if time_list is not None:
            verified_time_list: Optional[Sequence[TimePoint]] = list_to_tp_list_strict(
                time_list
            )

        elif (
            isinstance(obs, (pd.DataFrame, TemporalDataFrame))
            and time_col_name is not None
        ):
            if time_col_name in obs.columns:
                verified_time_list = list_to_tp_list_strict(
                    cast(Sequence[Any], obs[time_col_name])
                )
                time_col_name = None

            else:
                raise VValueError(f"Could not find column '{time_col_name}' in obs.")

        else:
            verified_time_list = None

        # timepoints
        if timepoints is not None:
            logger.debug(f"  'time points' DataFrame is a {type(timepoints).__name__}.")
            if not isinstance(timepoints, (pd.DataFrame, VDataFrame)):
                raise VTypeError("'time points' must be a pandas DataFrame.")

            else:
                if "value" not in timepoints.columns:
                    raise VValueError(
                        "'time points' must have at least a column 'value' to store time points value."
                    )

                timepoints["value"] = sorted(to_tp_list(timepoints["value"]))

                if len(timepoints.columns) > 1:
                    timepoints[timepoints.columns[1:]] = self._check_df_types(
                        timepoints[timepoints.columns[1:]]
                    )

                _timepoints_VDF = VDataFrame(timepoints)

        else:
            logger.debug("  'time points' DataFrame was not found.")

        if _timepoints_VDF is None:
            nb_timepoints = 1
            logger.debug("  1 time point was found so far.")
            logger.debug("    \u21b3 Time point is : [0]")

        else:
            nb_timepoints = len(_timepoints_VDF)
            logger.debug(
                f"  {nb_timepoints} time point{' was' if nb_timepoints == 1 else 's were'} found so far."
            )
            logger.debug(
                f"    \u21b3 Time point{' is' if nb_timepoints == 1 else 's are'} : "
                f"{repr_array(_timepoints_VDF.value.values)}"
            )

        # =========================================================================================
        if isinstance(data, AnnData):
            logger.debug("  VData creation from an AnnData.")

            # TODO : better handling ?
            # convert sparse matrices to regular numpy matrices for conversion to VData
            if isinstance(data.X, spmatrix):
                data.X = data.X.toarray()

            for layer_name in data.layers:
                if isinstance(data.layers[layer_name], spmatrix):
                    data.layers[layer_name] = data.layers[layer_name].toarray()

            # if an AnnData is being imported, obs, obsm, obsp, var, varm, varp and uns should be None because
            # they will be set from the AnnData
            for attr in ("obs", "obsm", "obsp", "var", "varm", "varp", "uns"):
                if eval(f"{attr} is not None"):
                    raise VValueError(
                        f"'{attr}' should be set to None when importing data from an AnnData."
                    )

            # import and cast obs to a TemporalDataFrame
            obs = TemporalDataFrame(
                data.obs,
                time_list=verified_time_list,
                time_col_name=time_col_name,
                name="obs",
                lock=(True, False),
            )
            reordering_index = obs.index

            # find time points list
            _timepoints_VDF, nb_timepoints = check_time_match(
                _timepoints_VDF, verified_time_list, time_col_name, obs
            )

            logger.debug(
                f"  {nb_timepoints} time point{' was' if nb_timepoints == 1 else 's were'} "
                f"found after data extraction from the AnnData."
            )
            logger.debug(
                f"    \u21b3 Time point{' is' if nb_timepoints == 1 else 's are'} : "
                f"{[0] if nb_timepoints == 1 else _timepoints_VDF.value.values}"
            )

            if array_isin(data.X, data.layers.values()):
                layers = dict(
                    (
                        key,
                        TemporalDataFrame(
                            pd.DataFrame(
                                arr, index=data.obs.index, columns=data.var.index
                            ).reindex(reordering_index),
                            time_list=obs.timepoints_column,
                            name=key,
                        ),
                    )
                    for key, arr in data.layers.items()
                )

            else:
                layers = dict(
                    {
                        "data": TemporalDataFrame(
                            pd.DataFrame(
                                data.X, index=data.obs.index, columns=data.var.index
                            ).reindex(reordering_index),
                            time_list=obs.timepoints_column,
                            name="data",
                        )
                    },
                    **dict(
                        (
                            key,
                            TemporalDataFrame(
                                pd.DataFrame(
                                    arr, index=data.obs.index, columns=data.var.index
                                ).reindex(reordering_index),
                                time_list=obs.timepoints_column,
                                name=key,
                            ),
                        )
                        for key, arr in data.layers.items()
                    ),
                )

            # import other arrays
            obsm = {
                TDF_name: TemporalDataFrame(
                    pd.DataFrame(no_dense_data(TDF_data)),
                    time_list=obs.timepoints_column,
                    index=obs.index,
                    name=TDF_name,
                )
                for TDF_name, TDF_data in data.obsm.items()
            }
            obsp = {
                VDF_name: VDataFrame(
                    no_dense_data(VDF_data),
                    index=obs.index,
                    columns=obs.index,
                    file=self._file.group["obsp"][VDF_name]
                    if self._file is not None
                    else None,
                )
                for VDF_name, VDF_data in data.obsp.items()
            }
            var = VDataFrame(
                data.var,
                file=self._file.group["var"] if self._file is not None else None,
            )
            varm = {
                VDF_name: VDataFrame(
                    no_dense_data(VDF_data),
                    index=var.index,
                    file=self._file.group["varm"][VDF_name]
                    if self._file is not None
                    else None,
                )
                for VDF_name, VDF_data in data.varm.items()
            }
            varp = {
                VDF_name: VDataFrame(
                    no_dense_data(VDF_data),
                    index=var.index,
                    columns=var.index,
                    file=self._file.group["varp"][VDF_name]
                    if self._file is not None
                    else None,
                )
                for VDF_name, VDF_data in data.varp.items()
            }
            uns = deep_dict_convert(data.uns)

        # =========================================================================================
        else:
            logger.debug("  VData creation from scratch.")

            # check formats

            # -----------------------------------------------------------------
            # layers
            if data is not None:
                layers = {}

                # data is a unique pandas DataFrame or a TemporalDataFrame
                if isinstance(data, (pd.DataFrame, VDataFrame)):
                    logger.debug("    1. \u2713 'data' is a pandas DataFrame.")

                    if nb_timepoints > 1:
                        raise VTypeError(
                            "'data' is a 2D pandas DataFrame but more than 1 time points were provided."
                        )

                    obs_index = data.index.values
                    var_index = data.columns.values

                    layers = {
                        "data": TemporalDataFrame(
                            data,
                            time_list=verified_time_list,
                            # timepoints=_timepoints_VDF.value.values if
                            # _timepoints_VDF is not None else None,
                            name="data",
                        )
                    }

                    if (
                        obs is not None
                        and not isinstance(obs, TemporalDataFrame)
                        and verified_time_list is None
                    ):
                        verified_time_list = layers["data"].timepoints_column

                elif isinstance(data, TemporalDataFrame):
                    logger.debug("    1. \u2713 'data' is a TemporalDataFrame.")

                    data.unlock_indices()
                    data.unlock_columns()

                    if _timepoints_VDF is not None:
                        if not _timepoints_VDF.value.equals(pd.Series(data.timepoints)):
                            raise VValueError(
                                "'time points' found in DataFrame do not match 'layers' time points."
                            )

                    else:
                        _timepoints_VDF = VDataFrame(
                            {"value": data.timepoints},
                            file=self._file.group["timepoints"]
                            if self._file is not None
                            else None,
                        )
                        nb_timepoints = data.n_timepoints

                    obs_index = data.index
                    repeating_obs_index = data.has_repeating_index
                    var_index = data.columns

                    if (
                        obs is not None
                        and not isinstance(obs, TemporalDataFrame)
                        and verified_time_list is None
                    ):
                        verified_time_list = data.timepoints_column

                    layers = {"data": self._check_df_types(data.copy())}

                elif isinstance(data, dict):
                    logger.debug("    1. \u2713 'data' is a dictionary.")

                    for key, value in data.items():
                        if not isinstance(
                            value, (pd.DataFrame, VDataFrame, TemporalDataFrame)
                        ):
                            raise VTypeError(
                                f"Layer '{key}' must be a TemporalDataFrame or a pandas DataFrame, "
                                f"it is a {type(value)}."
                            )

                        if isinstance(value, (pd.DataFrame, VDataFrame)):
                            logger.debug(f"        \u2713 '{key}' is DataFrame.")

                            if obs_index is None:
                                obs_index = value.index.values
                                var_index = value.columns.values

                            layers[str(key)] = TemporalDataFrame(
                                value, time_list=verified_time_list, name=str(key)
                            )

                            if (
                                obs is not None
                                and not isinstance(obs, TemporalDataFrame)
                                and verified_time_list is None
                            ):
                                verified_time_list = layers[str(key)].timepoints_column

                        else:
                            logger.debug(
                                f"        \u2713 '{key}' is TemporalDataFrame."
                            )

                            value.unlock_indices()
                            value.unlock_columns()

                            value = value.copy() if not value.is_backed else value

                            if obs_index is None:
                                obs_index = value.index
                                repeating_obs_index = value.has_repeating_index
                                var_index = value.columns

                                if _timepoints_VDF is not None:
                                    if not np.all(
                                        _timepoints_VDF.value.values == value.timepoints
                                    ):
                                        raise VValueError(
                                            f"'time points' found in DataFrame ({repr_array(_timepoints_VDF.value)}) "
                                            f"do not match 'layers' time points ("
                                            f"{repr_array(value.timepoints)})."
                                        )

                                else:
                                    _timepoints_VDF = VDataFrame(
                                        {"value": value.timepoints},
                                        file=self._file.group["timepoints"]
                                        if self._file is not None
                                        else None,
                                    )
                                    nb_timepoints = value.n_timepoints

                            if (
                                obs is not None
                                and not isinstance(obs, TemporalDataFrame)
                                and verified_time_list is None
                            ):
                                verified_time_list = value.timepoints_column

                            if not value.name == str(key):
                                value.name = (
                                    f"{value.name if value.name != 'No_Name' else ''}"
                                    f"{'_' if value.name != 'No_Name' else ''}"
                                    f"{str(key)}"
                                )
                            layers[str(key)] = self._check_df_types(value)

                else:
                    raise VTypeError(
                        f"Type '{type(data)}' is not allowed for 'data' parameter, should be a dict,"
                        f"a pandas DataFrame, a TemporalDataFrame or an AnnData object."
                    )

            else:
                logger.debug("    1. \u2717 'data' was not found.")

            # -----------------------------------------------------------------
            # obs
            if obs is not None:
                logger.debug(f"    2. \u2713 'obs' is a {type(obs).__name__}.")

                if not isinstance(obs, (pd.DataFrame, VDataFrame, TemporalDataFrame)):
                    raise VTypeError(
                        "'obs' must be a pandas DataFrame or a TemporalDataFrame."
                    )

                elif isinstance(obs, (pd.DataFrame, VDataFrame)):
                    obs = TemporalDataFrame(
                        obs,
                        time_list=verified_time_list,
                        time_col_name=time_col_name,
                        name="obs",
                        index=obs.index,
                        repeating_index=repeating_obs_index,
                    )

                else:
                    obs.unlock_indices()
                    obs.unlock_columns()

                    obs = self._check_df_types(obs)
                    if not obs.name == "obs":
                        obs.name = (
                            f"{obs.name if obs.name != 'No_Name' else ''}"
                            f"{'_' if obs.name != 'No_Name' else ''}"
                            f"obs"
                        )

                    if verified_time_list is not None:
                        logger.warning(
                            "'time_list' parameter cannot be used since 'obs' is already a "
                            "TemporalDataFrame."
                        )

                    if time_col_name is not None:
                        logger.warning(
                            "'time_col_name' parameter cannot be used since 'obs' is already a "
                            "TemporalDataFrame."
                        )

                if obs_index is not None and all(np.isin(obs.index, obs_index)):
                    obs.reindex(obs_index, repeating_index=repeating_obs_index)

                else:
                    obs_index = obs.index

                # find time points list
                _timepoints_VDF, nb_timepoints = check_time_match(
                    _timepoints_VDF, verified_time_list, time_col_name, obs
                )

                logger.debug(
                    f"  {nb_timepoints} time point{' was' if nb_timepoints == 1 else 's were'} "
                    f"found from the provided data."
                )
                logger.debug(
                    f"    \u21b3 Time point{' is' if nb_timepoints == 1 else 's are'} : "
                    f"{[0] if nb_timepoints == 1 else repr_array(_timepoints_VDF.value.values)}"
                )

                obs.lock_indices()

            else:
                logger.debug("    2. \u2717 'obs' was not found.")
                if verified_time_list is not None:
                    logger.warning(
                        "'time_list' parameter cannot be used since 'obs' was not found."
                    )
                if time_col_name is not None:
                    logger.warning(
                        "'time_col_name' parameter cannot be used since 'obs' was not found."
                    )

            # -----------------------------------------------------------------
            # obsm
            if obsm is not None:
                logger.debug(f"    3. \u2713 'obsm' is a {type(obsm).__name__}.")

                if obs is None and layers is None:
                    raise VValueError(
                        "'obsm' parameter cannot be set unless either 'data' or 'obs' are set."
                    )

                valid_obsm = {}

                if not isinstance(obsm, dict):
                    raise VTypeError("'obsm' must be a dictionary of DataFrames.")

                else:
                    for key, value in obsm.items():
                        if not isinstance(
                            value, (pd.DataFrame, VDataFrame, TemporalDataFrame)
                        ):
                            raise VTypeError(
                                f"'obsm' '{key}' must be a TemporalDataFrame or a pandas DataFrame."
                            )

                        elif isinstance(value, (pd.DataFrame, VDataFrame)):
                            if verified_time_list is None:
                                if obs is not None:
                                    verified_time_list = obs.timepoints_column
                                else:
                                    verified_time_list = list(layers.values())[
                                        0
                                    ].timepoints_column

                            valid_obsm[str(key)] = TemporalDataFrame(
                                value,
                                time_list=verified_time_list,
                                # timepoints=_timepoints_VDF.value.values if _timepoints_VDF is not None else None,
                                name=str(key),
                            )

                        else:
                            value.unlock_indices()
                            value.unlock_columns()

                            if not value.name == str(key):
                                value.name = (
                                    f"{value.name if value.name != 'No_Name' else ''}"
                                    f"{'_' if value.name != 'No_Name' else ''}"
                                    f"{str(key)}"
                                )
                            valid_obsm[str(key)] = self._check_df_types(value)

                            if verified_time_list is not None:
                                logger.warning(
                                    f"'time_list' parameter cannot be used since 'obsm' '{key}' is "
                                    "already a TemporalDataFrame."
                                )
                            if time_col_name is not None:
                                logger.warning(
                                    f"'time_col_name' parameter cannot be used since 'obsm' '{key}' "
                                    f"is already a TemporalDataFrame."
                                )

                        if np.all(np.isin(valid_obsm[str(key)].index, obs_index)):
                            valid_obsm[str(key)].reindex(obs_index)

                        else:
                            raise VValueError(
                                f"Index of 'obsm' '{key}' does not match 'obs' and 'layers' indexes."
                            )

                obsm = valid_obsm

            else:
                logger.debug("    3. \u2717 'obsm' was not found.")

            # -----------------------------------------------------------------
            # obsp
            if obsp is not None:
                logger.debug(f"    4. \u2713 'obsp' is a {type(obsp).__name__}.")

                if obs is None and layers is None:
                    raise VValueError(
                        "'obsp' parameter cannot be set unless either 'data' or 'obs' are set."
                    )

                valid_obsp = {}

                if not isinstance(obsp, dict):
                    raise VTypeError(
                        "'obsp' must be a dictionary of 2D numpy arrays or pandas DataFrames."
                    )

                else:
                    for key, value in obsp.items():
                        if (
                            not isinstance(
                                value, (np.ndarray, pd.DataFrame, VDataFrame)
                            )
                            or value.ndim != 2
                        ):
                            raise VTypeError(
                                f"'obsp' '{key}' must be a 2D numpy array or pandas DataFrame."
                            )

                        if isinstance(value, (pd.DataFrame, VDataFrame)):
                            if all(value.index.isin(obs_index)):
                                value.reindex(obs_index)

                                if all(value.columns.isin(obs_index)):
                                    value = value[obs_index]

                                else:
                                    raise VValueError(
                                        "Column names of 'obsp' do not match 'obs' and 'layers' indexes."
                                    )

                            else:
                                raise VValueError(
                                    f"Index of 'obsp' '{key}' does not match 'obs' and 'layers' indexes."
                                )

                        else:
                            value = VDataFrame(
                                value, index=obs_index, columns=obs_index
                            )

                        valid_obsp[str(key)] = self._check_df_types(value)

                obsp = valid_obsp

            else:
                logger.debug("    4. \u2717 'obsp' was not found.")

            # -----------------------------------------------------------------
            # var
            if var is not None:
                logger.debug(f"    5. \u2713 'var' is a {type(var).__name__}.")

                if not isinstance(var, (pd.DataFrame, VDataFrame)):
                    raise VTypeError("var must be a pandas DataFrame.")
                else:
                    var = self._check_df_types(
                        VDataFrame(
                            var,
                            file=self._file.group["var"]
                            if self._file is not None
                            else None,
                        )
                    )

            else:
                logger.debug("    5. \u2717 'var' was not found.")

            # -----------------------------------------------------------------
            # varm
            if varm is not None:
                logger.debug(f"    6. \u2713 'varm' is a {type(varm).__name__}.")

                if var is None and layers is None:
                    raise VValueError(
                        "'obsm' parameter cannot be set unless either 'data' or 'var' are set."
                    )

                valid_varm = {}

                if not isinstance(varm, dict):
                    raise VTypeError("'varm' must be a dictionary of DataFrames.")

                else:
                    for key, value in varm.items():
                        if not isinstance(value, (pd.DataFrame, VDataFrame)):
                            raise VTypeError(
                                f"'varm' '{key}' must be a pandas DataFrame."
                            )

                        else:
                            valid_varm[str(key)] = self._check_df_types(value)

                            if np.all(np.isin(valid_varm[str(key)].index, var_index)):
                                valid_varm[str(key)].reindex(var_index)

                            else:
                                raise VValueError(
                                    "Index of 'varm' does not match 'var' and 'layers' column names."
                                )

                varm = valid_varm

            else:
                logger.debug("    6. \u2717 'varm' was not found.")

            # -----------------------------------------------------------------
            # varp
            if varp is not None:
                logger.debug(f"    7. \u2713 'varp' is a {type(varp).__name__}.")

                if var is None and layers is None:
                    raise VValueError(
                        "'varp' parameter cannot be set unless either 'data' or 'var' are set."
                    )

                valid_varp = {}

                if not isinstance(varp, dict):
                    raise VTypeError(
                        "'varp' must be a dictionary of 2D numpy arrays or pandas DataFrames."
                    )

                else:
                    for key, value in varp.items():
                        if (
                            not isinstance(
                                value, (np.ndarray, (pd.DataFrame, VDataFrame))
                            )
                            and value.ndim != 2
                        ):
                            raise VTypeError(
                                f"'varp' '{key}' must be 2D numpy array or pandas DataFrame."
                            )

                        if isinstance(value, (pd.DataFrame, VDataFrame)):
                            if all(value.index.isin(var_index)):
                                value.reindex(var_index)

                                if all(value.columns.isin(var_index)):
                                    value = value[var_index]

                                else:
                                    raise VValueError(
                                        f"Column names of 'varp' '{key}' do not match 'var' and 'layers' column names."
                                    )

                            else:
                                raise VValueError(
                                    f"Index of 'varp' '{key}' does not match 'var' and 'layers' column "
                                    f"names."
                                )

                        else:
                            value = VDataFrame(
                                value, index=var_index, columns=var_index
                            )

                        valid_varp[str(key)] = self._check_df_types(value)

                varp = valid_varp

            else:
                logger.debug("    7. \u2717 'varp' was not found.")

            # # -----------------------------------------------------------------
            # uns
            if uns is not None:
                if not isinstance(uns, dict):
                    raise VTypeError("'uns' must be a dictionary.")
                logger.debug("    8. \u2713 'uns' is a dictionary.")

            else:
                logger.debug("    8. \u2717 'uns' was not found.")

        # if time points are not given, assign default values 0, 1, 2, ...
        if _timepoints_VDF is None:
            if layers is not None:
                _timepoints_VDF = VDataFrame(
                    {"value": to_tp_list(range(list(layers.values())[0].shape[0]))},
                    file=self._file.group["timepoints"]
                    if self._file is not None
                    else None,
                )
            elif obsm is not None:
                _timepoints_VDF = VDataFrame(
                    {"value": to_tp_list(range(list(obsm.values())[0].shape[0]))},
                    file=self._file.group["timepoints"]
                    if self._file is not None
                    else None,
                )
            elif varm is not None:
                _timepoints_VDF = VDataFrame(
                    {"value": to_tp_list(range(list(varm.values())[0].shape[0]))},
                    file=self._file.group["timepoints"]
                    if self._file is not None
                    else None,
                )

        if _timepoints_VDF is not None:
            logger.debug(
                f"  {len(_timepoints_VDF)} time point"
                f"{' was' if len(_timepoints_VDF) == 1 else 's were'} found finally."
            )
            logger.debug(
                f"    \u21b3 Time point{' is' if nb_timepoints == 1 else 's are'} : "
                f"{repr_array(_timepoints_VDF.value.values)}"
            )

        else:
            logger.debug("  Could not find time points.")

        logger.debug("  \u23bf Arrays' formats are OK.  -- -- -- -- -- -- -- -- -- ")

        return (
            obs,
            var,
            layers,
            _timepoints_VDF,
            obsm,
            obsp,
            varm,
            varp,
            obs_index,
            var_index,
            uns,
        )

    def _check_df_types(self, df: "DataFrame") -> "DataFrame":
        """
        Function for coercing data types of the columns and of the index in a pandas DataFrame.
        :param df: a pandas DataFrame or a TemporalDataFrame.
        """
        logger.debug(
            "  \u23be Check DataFrame's column types.   -  -  -  -  -  -  -  -  -  -"
        )
        # check index : convert to correct dtype if it is not a string type
        if self._dtype is not None:
            try:
                df.index.astype(self._dtype)
            except (ValueError, TypeError):
                df.index.astype(np.dtype("O"))

            # check columns : convert to correct dtype if it is not a string type
            if isinstance(df, (pd.DataFrame, VDataFrame)):
                for col_name in df.columns:
                    try:
                        df[col_name].astype(self._dtype)
                        logger.debug(f"Column '{col_name}' set to {self._dtype}.")

                    except (ValueError, TypeError):
                        if df[col_name].dtype.type in (
                            np.datetime64,
                            np.timedelta64,
                            pd.CategoricalDtype.type,
                        ):
                            logger.debug(
                                f"Column '{col_name}' kept to {df[col_name].dtype.type}."
                            )

                        else:
                            df[col_name].astype(np.dtype("O"))
                            logger.debug(
                                f"Column '{col_name}' set to string or TimePoint."
                            )

            elif isinstance(df, TemporalDataFrame):
                pass
                # TODO : implement astype (type casting) in TDFs
                # try:
                #     df.astype(self._dtype)
                #
                # except ValueError:
                #     logger.warning(f"Cannot set TemporalDataFrame '{df.name}' to type '{self._dtype}'.")

            else:
                raise VTypeError(
                    f"Invalid type '{type(df)}' for function '_check_df_types()'."
                )

        logger.debug(
            "  \u23bf DataFrame's column types are OK.  -  -  -  -  -  -  -  -  -  -"
        )

        return df

    def _init_data(
        self, obs_index: Optional[np.ndarray], var_index: Optional[np.ndarray]
    ) -> None:
        """
        Function for finishing the initialization of the VData object. It checks for incoherence in the user-supplied
        arrays and raises an error in case something is wrong.

        Args:
            obs_index: If X was supplied as a pandas DataFrame, index of observations
            var_index: If X was supplied as a pandas DataFrame, index of variables
        """
        logger.debug("Initialize the VData.")

        # get shape once for performance
        n_timepoints, n_obs, n_var = self.n_timepoints, self.n_obs, self.n_var

        # check coherence with number of time points in VData
        for attr in ("layers", "obsm"):
            dataset = getattr(self, attr)
            if not dataset.empty and n_timepoints != dataset.shape[1]:
                raise IncoherenceError(
                    f"{attr} has {dataset.shape[0]} time point"
                    f"{'' if dataset.shape[0] == 1 else 's'} but {n_timepoints}"
                    f" {'was' if n_timepoints == 1 else 'were'} given."
                )

        logger.debug("Time points were coherent across arrays.")

        # if data was given as a dataframe, check that obs and data match in row names
        if obs_index is not None:
            if not np.all(self.obs.index == obs_index):
                raise VValueError(
                    f"Indexes in dataFrames 'data' ({obs_index}) and 'obs' ({self.obs.index}) "
                    f"do not match."
                )

        # if data was given as a dataframe, check that var row names match data col names
        if var_index is not None:
            if not np.all(self.var.index == var_index):
                raise VValueError(
                    f"Columns in dataFrame 'data' ({var_index}) do not match index of 'var' "
                    f"({self.var.index})."
                )

        # check coherence between layers, obs, var and time points
        if self._layers is not None:
            for layer_name, layer in self._layers.items():
                if layer.shape != (n_timepoints, n_obs, n_var):
                    if layer.shape[0] != n_timepoints:
                        raise IncoherenceError(
                            f"layer '{layer_name}' has incoherent number of time points "
                            f"{layer.shape[0]}, should be {n_timepoints}."
                        )

                    elif [layer[i].shape[0] for i in range(len(layer))] != n_obs:
                        for i in range(len(layer)):
                            if layer[i].shape[0] != n_obs[i]:
                                raise IncoherenceError(
                                    f"layer '{layer_name}' has incoherent number of observations "
                                    f"{layer[i].shape[0]}, should be {n_obs[i]}."
                                )

                    else:
                        raise IncoherenceError(
                            f"layer '{layer_name}' has incoherent number of variables "
                            f"{layer[0].shape[1]}, should be {n_var}."
                        )

        # check coherence between obs, obsm and obsp shapes
        if not self.obsm.empty and n_obs != self.obsm.shape[2]:
            raise IncoherenceError(
                f"'obs' and 'obsm' have different lengths ({n_obs} vs "
                f"{self.obsm.shape[2]})"
            )

        if not self.obsp.empty and self.n_obs_total != self.obsp.shape[1]:
            raise IncoherenceError(
                f"'obs' and 'obsp' have different lengths ({n_obs} vs "
                f"{self.obsp.shape[1]})"
            )

        # check coherence between var, varm, varp shapes
        for attr in ("varm", "varp"):
            dataset = getattr(self, attr)
            if not dataset.empty and n_var != dataset.shape[1]:
                raise IncoherenceError(
                    f"'var' and 'varm' have different lengths ({n_var} vs "
                    f"{dataset.shape[1]})"
                )

    # functions ----------------------------------------------------------
    def __mean_min_max_func(
        self, func: Literal["mean", "min", "max"], axis
    ) -> tuple[dict[str, TemporalDataFrame], Sequence[TimePoint], pd.Index]:
        """
        Compute mean, min or max of the values over the requested axis.
        """
        if axis == 0:
            _data = {
                layer: getattr(self.layers[layer], func)(axis=axis).T
                for layer in self.layers
            }
            _time_list = self.timepoints_values
            _index = pd.Index(["mean" for _ in range(self.n_timepoints)])

        elif axis == 1:
            _data = {
                layer: getattr(self.layers[layer], func)(axis=axis)
                for layer in self.layers
            }
            _time_list = self.obs.timepoints_column
            _index = self.obs.index

        else:
            raise VValueError(
                f"Invalid axis '{axis}', should be 0 (on columns) or 1 (on rows)."
            )

        return _data, _time_list, _index

    def mean(self, axis: Literal[0, 1] = 0) -> "VData":
        """
        Return the mean of the values over the requested axis.

        :param axis: compute mean over columns (0: default) or over rows (1).
        :return: a TemporalDataFrame with mean values.
        """
        _data, _time_list, _index = self.__mean_min_max_func("mean", axis)

        _name = f"Mean of {self.name}" if self.name != "No_Name" else None
        return VData(
            data=_data, obs=pd.DataFrame(index=_index), time_list=_time_list, name=_name
        )

    def min(self, axis: Literal[0, 1] = 0) -> "VData":
        """
        Return the minimum of the values over the requested axis.

        :param axis: compute minimum over columns (0: default) or over rows (1).
        :return: a TemporalDataFrame with minimum values.
        """
        _data, _time_list, _index = self.__mean_min_max_func("min", axis)

        _name = f"Minimum of {self.name}" if self.name != "No_Name" else None
        return VData(
            data=_data, obs=pd.DataFrame(index=_index), time_list=_time_list, name=_name
        )

    def max(self, axis: Literal[0, 1] = 0) -> "VData":
        """
        Return the maximum of the values over the requested axis.

        :param axis: compute maximum over columns (0: default) or over rows (1).
        :return: a TemporalDataFrame with maximum values.
        """
        _data, _time_list, _index = self.__mean_min_max_func("max", axis)

        _name = f"Maximum of {self.name}" if self.name != "No_Name" else None
        return VData(
            data=_data, obs=pd.DataFrame(index=_index), time_list=_time_list, name=_name
        )

    # writing ------------------------------------------------------------
    def write(
        self, file: Optional[Union[str, Path]] = None, verbose: bool = True
    ) -> None:
        """
        Save this VData object in HDF5 file format.

        Args:
            file: path to save the VData
            verbose: print a progress bar while saving objects in this VData ? (default: True)
        """
        if not self.is_backed and file is None:
            raise VValueError("No file path was provided for writing this VData.")

        write_vdata(self, file, show_progress=verbose)

    def write_to_csv(
        self,
        directory: Union[str, Path],
        sep: str = ",",
        na_rep: str = "",
        index: bool = True,
        header: bool = True,
    ) -> None:
        """
        Save layers, timepoints, obs, obsm, obsp, var, varm and varp to csv files in a directory.

        Args:
            directory: path to a directory for saving the matrices
            sep: delimiter character
            na_rep: string to replace NAs
            index: write row names ?
            header: Write col names ?
        """
        write_vdata_to_csv(self, directory, sep, na_rep, index, header)

    # copy ---------------------------------------------------------------
    def copy(self) -> "VData":
        """
        Build a deep copy of this VData object and not a view.

        Returns:
            A new VData, which is a deep copy of this VData.
        """
        return VData(
            data=self.layers.dict_copy(),
            obs=self.obs.copy(),
            obsm=self.obsm.dict_copy(),
            obsp=self._obsp.dict_copy(),
            var=self.var.copy(),
            varm=self.varm.dict_copy(),
            varp=self.varp.dict_copy(),
            timepoints=self.timepoints.copy(),
            uns=self.uns.copy(),
            dtype=self.dtype,
            name=f"{self.name}_copy",
        )

    # conversion ---------------------------------------------------------
    def to_AnnData(
        self,
        timepoints_list: Optional[
            Union[str, "TimePoint", Collection[Union[str, "TimePoint"]]]
        ] = None,
        into_one: bool = True,
        with_timepoints_column: bool = True,
        layer_as_X: Optional[str] = None,
        layers_to_export: Optional[list] = None,
    ) -> Union[AnnData, list[AnnData]]:
        """
        Convert a VData object to an AnnData object.

        Args:
            timepoints_list: a list of time points for which to extract data to build the AnnData. If set to
                None, all timepoints are selected.
            into_one: Build one AnnData, concatenating the data for multiple time points (True), or build one
                AnnData for each time point (False) ?
            with_timepoints_column: store time points data in the obs DataFrame. This is only used when
                concatenating the data into a single AnnData (i.e. into_one=True).
            layer_as_X: name of the layer to use as the X matrix. By default, the first layer is used.
            layers_to_export: if None export all layers

        Returns:
            An AnnData object with data for selected time points.
        """
        # TODO : obsp is not passed to AnnData

        logger.debug(
            "\u23be VData conversion to AnnData : begin "
            "---------------------------------------------------------- "
        )

        if timepoints_list is None:
            _timepoints_list = np.array(self.timepoints_values)

        else:
            _timepoints_list = to_tp_list(timepoints_list)
            _timepoints_list = np.array(_timepoints_list)[
                np.where(match_timepoints(_timepoints_list, self.timepoints_values))
            ]

        logger.debug(f"Selected time points are : {repr_array(_timepoints_list)}")

        if into_one:
            logger.debug("Convert to one AnnData object.")

            logger.debug(
                "\u23bf VData conversion to AnnData : end "
                "---------------------------------------------------------- "
            )

            if with_timepoints_column:
                tp_col_name = (
                    self.obs.timepoints_column_name
                    if self.obs.timepoints_column_name is not None
                    else DEFAULT_TIME_POINTS_COL_NAME
                )
            else:
                tp_col_name = None

            view = self[_timepoints_list]
            if layer_as_X is None:
                layer_as_X = list(view.layers.keys())[0]

            elif layer_as_X not in view.layers.keys():
                raise ValueError(f"Layer '{layer_as_X}' was not found.")

            X = view.layers[layer_as_X].to_pandas()
            X.index = X.index.astype(str)
            X.columns = X.columns.astype(str)
            if layers_to_export is None:
                layers_to_export = view.layers.keys()

            return AnnData(
                X=X,
                layers={
                    key: view.layers[key].to_pandas(str_index=True)
                    for key in layers_to_export
                },
                obs=view.obs.to_pandas(with_timepoints=tp_col_name, str_index=True),
                obsm={key: arr.values_num for key, arr in view.obsm.items()},
                obsp={key: arr.values for key, arr in view.obsp.items()},
                var=view.var.to_pandas(),
                varm={key: arr for key, arr in view.varm.items()},
                varp={key: arr for key, arr in view.varp.items()},
                uns=view.uns.copy(),
            )

        else:
            logger.debug("Convert to many AnnData objects.")

            result = []
            for time_point in _timepoints_list:
                view = self[time_point]
                if layer_as_X is None:
                    layer_as_X = list(view.layers.keys())[0]

                elif layer_as_X not in view.layers.keys():
                    raise ValueError(f"Layer '{layer_as_X}' was not found.")

                X = view.layers[layer_as_X].to_pandas()
                X.index = X.index.astype(str)
                X.columns = X.columns.astype(str)

                result.append(
                    AnnData(
                        X=X,
                        layers={
                            key: layer.to_pandas(str_index=True)
                            for key, layer in view.layers.items()
                        },
                        obs=view.obs.to_pandas(str_index=True),
                        obsm={
                            key: arr.to_pandas(str_index=True)
                            for key, arr in view.obsm.items()
                        },
                        var=view.var.to_pandas(),
                        varm={key: arr for key, arr in view.varm.items()},
                        varp={key: arr for key, arr in view.varp.items()},
                        uns=view.uns,
                    )
                )

            logger.debug(
                "\u23bf VData conversion to AnnData : end "
                "---------------------------------------------------------- "
            )

            return result
