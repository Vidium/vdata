from __future__ import annotations

from pathlib import Path
from types import TracebackType
from typing import Any, Collection, Iterator, Literal, Mapping, Sequence

import ch5mpy as ch
import numpy as np
import pandas as pd
from anndata import AnnData
from h5dataframe import H5DataFrame

import vdata.timepoint as tp
from vdata._meta import PrettyRepr
from vdata._typing import IFS, DictLike, NDArray_IFS, PreSlicer
from vdata.anndata_proxy import AnnDataProxy
from vdata.array_view import NDArrayView
from vdata.data._file import NoData
from vdata.data._indexing import reformat_index
from vdata.data._parse import ParsingDataIn, ParsingDataOut, parse_AnnData, parse_objects
from vdata.data.arrays import (
    VLayersArrayContainer,
    VObsmArrayContainer,
    VObspArrayContainer,
    VVarmArrayContainer,
    VVarpArrayContainer,
)
from vdata.data.convert import convert_anndata_to_vdata, convert_vdata_to_anndata
from vdata.data.read import read_from_csv
from vdata.data.view import VDataView
from vdata.data.write import write_vdata, write_vdata_in_h5dict, write_vdata_to_csv
from vdata.IO import (
    ShapeError,
    VClosedFileError,
    VReadOnlyError,
    generalLogger,
)
from vdata.names import NO_NAME
from vdata.tdf import Index, TemporalDataFrame, TemporalDataFrameBase
from vdata.update import CURRENT_VERSION, update_vdata
from vdata.utils import repr_array, repr_index


class VData(metaclass=PrettyRepr):
    """
    A VData object stores data points in matrices of observations x variables in the same way as the AnnData object,
    but also accounts for the time information. The 2D matrices in AnnData are replaced by 3D matrices here.
    """

    __slots__ = "_name", "_data", "_obs", "_obsm", "_obsp", "_var", "_varm", "_varp", "_timepoints", "_uns", "_layers"

    # region magic methods
    def __init__(
        self,
        data: AnnData
        | pd.DataFrame
        | H5DataFrame
        | TemporalDataFrameBase
        | Mapping[str, pd.DataFrame | H5DataFrame | TemporalDataFrameBase]
        | ch.H5Dict[H5DataFrame | TemporalDataFrame]
        | None = None,
        obs: pd.DataFrame | H5DataFrame | TemporalDataFrameBase | None = None,
        obsm: Mapping[str, pd.DataFrame | H5DataFrame | TemporalDataFrameBase] | None = None,
        obsp: Mapping[str, pd.DataFrame | H5DataFrame | NDArray_IFS] | None = None,
        var: pd.DataFrame | H5DataFrame | None = None,
        varm: Mapping[str, pd.DataFrame | H5DataFrame] | None = None,
        varp: Mapping[str, pd.DataFrame | H5DataFrame | NDArray_IFS] | None = None,
        timepoints: pd.DataFrame | H5DataFrame | None = None,
        uns: DictLike[Any] | None = None,
        time_col_name: str | None = None,
        timepoints_list: Sequence[str | tp.TimePoint] | tp.TimePointArray | None = None,
        name: str = "",
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
            timepoints_list: if obs is a pandas DataFrame (or the VData is created from an AnnData), a list containing
                time information of the same length as the number of rows in obs.
            name: a name for this VData.
        """
        generalLogger.debug(
            f"\u23BE VData '{name}' creation : begin " f"-------------------------------------------------------- "
        )

        self._data: ch.H5Dict[H5DataFrame | TemporalDataFrame] | None = None

        # create from h5 file
        if isinstance(data, ch.H5Dict):
            self._data = data
            _name = name or data.attributes.get("name", NO_NAME)
            _parsed_data = ParsingDataOut.from_h5(data)

        # create from AnnData
        elif isinstance(data, AnnData):
            _name = name or NO_NAME
            _parsed_data = parse_AnnData(
                data,
                ParsingDataIn.from_anndata(
                    data, obs, obsm, obsp, var, varm, varp, timepoints, time_col_name, timepoints_list, uns
                ),
            )

        # create from objects
        else:
            _name = name or NO_NAME
            _parsed_data = parse_objects(
                ParsingDataIn.from_objects(
                    data, obs, obsm, obsp, var, varm, varp, timepoints, time_col_name, timepoints_list, uns
                )
            )

        self._initialize(_parsed_data, _name)

        generalLogger.debug(lambda: f"Guessed dimensions are : ({self.n_timepoints}, {self.n_obs}, {self.n_var})")
        generalLogger.debug(
            f"\u23BF VData '{self._name}' creation : end "
            f"---------------------------------------------------------- "
        )

    def _initialize(self, data: ParsingDataOut, name: str) -> None:
        self._name = str(name)

        self._obs = data.obs
        self._var = data.var
        self._timepoints = data.timepoints
        self._layers = VLayersArrayContainer(data=data.layers, vdata=self)
        self._obsm = VObsmArrayContainer(data=data.obsm, vdata=self)
        self._obsp = VObspArrayContainer(data=data.obsp, vdata=self)
        self._varm = VVarmArrayContainer(data=data.varm, vdata=self)
        self._varp = VVarpArrayContainer(data=data.varp, vdata=self)
        self._uns = data.uns

    def __repr__(self) -> str:
        """
        Description for this Vdata object to print.
        :return: a description of this Vdata object
        """
        if self.is_closed:
            return "Backed VData with closed file."

        _n_obs = self.n_obs if len(self.n_obs) > 1 else self.n_obs[0] if len(self.n_obs) else 0

        if self.empty:
            repr_str = (
                f"Empty {'backed ' if self.is_backed else ''}VData '{self._name}' ({_n_obs} obs x"
                f" {self.n_var} vars over {self.n_timepoints} time point"
                f"{'' if self.n_timepoints == 1 else 's'})."
            )

        else:
            repr_str = (
                f"{'Backed ' if self.is_backed else ''}VData '{self._name}' ({_n_obs} obs x"
                f" {self.n_var} vars over {self.n_timepoints} time point"
                f"{'' if self.n_timepoints == 1 else 's'})."
            )

        for attr in ["layers", "obs", "var", "timepoints", "obsm", "varm", "obsp", "varp"]:
            obj = getattr(self, attr)
            if not obj.empty:
                if isinstance(obj, TemporalDataFrameBase):
                    repr_str += f"\n\t{attr}: {repr_array(obj.columns, n_max=100, print_length=False)[1:-1]}"

                else:
                    repr_str += f"\n\t{attr}: {repr_array(obj.keys(), n_max=100, print_length=False)[1:-1]}"

        if len(self.uns):
            repr_str += f"\n\tuns: {repr_array(self.uns.keys(), n_max=100, print_length=False)[1:-1]}"

        return repr_str

    def __del__(self) -> None:
        """
        Close file on object delete.
        """
        if self._data is not None and not self._data.is_closed:
            self._data.close()

    def __getitem__(
        self, index: PreSlicer | tuple[PreSlicer, PreSlicer] | tuple[PreSlicer, PreSlicer, PreSlicer]
    ) -> VData | VDataView:
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

        generalLogger.debug("VData sub-setting - - - - - - - - - - - - - - ")
        generalLogger.debug(lambda: f"  Got index \n{repr_index(index)}")

        formatted_index = reformat_index(
            index,
            tp.as_timepointarray(self.timepoints.value),
            self.obs.index,
            self.var.index.values,
        )

        generalLogger.debug(lambda: f"  Refactored index to \n{repr_index(formatted_index)}")

        if formatted_index is None:
            return self

        if formatted_index[0] is not None and not len(formatted_index[0]):
            raise ValueError("Time points not found in this VData.")

        return VDataView(self, formatted_index[0], formatted_index[1], formatted_index[2])

    def __enter__(self) -> VData:
        return self

    def __exit__(
        self, exc_type: type[BaseException] | None, exc_val: BaseException | None, exc_tb: TracebackType | None
    ) -> None:
        if self._data is not None and not self._data.is_closed:
            self._data._file.file.flush()
            self._data.close()

    def __h5_write__(self, values: ch.H5Dict[Any]) -> None:
        write_vdata_in_h5dict(self, values, verbose=False)

    @classmethod
    def __h5_read__(cls, values: ch.H5Dict[Any]) -> VData:
        with ch.options(error_mode="raise"):
            if values.attributes.get("__vdata_write_version__", -1) < CURRENT_VERSION:
                update_vdata(values)

            return VData(data=values)

    # endregion

    # region predicates
    @property
    def is_backed(self) -> bool:
        """
        Is this VData object backed on an h5 file ?
        :return: Is this VData object backed on an h5 file ?
        """
        return self._data is not None

    @property
    def is_backed_w(self) -> bool:
        """
        Is this VData object backed on an h5 file and writable ?
        :return: Is this VData object backed on an h5 file and writable ?
        """
        return self._data is not None and self._data.mode == ch.H5Mode.READ_WRITE

    @property
    def is_closed(self) -> bool:
        """
        Is this VData's file closed ?

        Returns:
            Is this VData's file closed ?
        """
        return self._data is not None and not self._data.file.id.valid

    @property
    def is_read_only(self) -> bool:
        """
        Is this VData's file open in read only mode ?
        """
        return self._data is not None and self._data.file.mode == "r"

    @property
    def is_view(self) -> Literal[False]:
        return False

    # endregion

    # region attributes
    @property
    def data(self) -> ch.H5Dict[H5DataFrame | TemporalDataFrame] | NoData:
        """Get this VData's h5 file."""
        return NoData._ if self._data is None else self._data

    @property
    def name(self) -> str:
        return self._name

    @property
    def empty(self) -> bool:
        """
        Is this VData object empty ? (no time points or no obs or no vars)

        Returns:
            VData empty ?
        """
        if not len(self._layers) or not self.n_timepoints or not self.n_obs_total or not self.n_var:
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
        return self._timepoints.shape[0]

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
        return self._obs.shape[1]

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
        return self._var.shape[0]

    @property
    def shape(self) -> tuple[int, int, list[int], list[int]]:
        """
        Shape of this VData object (# layers, # time points, # observations, # variables).
        Returns:
            VData's shape.
        """
        return self.layers.shape

    @property
    def timepoints(self) -> H5DataFrame:
        """
        Get time points data.
        :return: the time points DataFrame.
        """
        return self._timepoints

    @timepoints.setter
    def timepoints(self, df: pd.DataFrame | H5DataFrame) -> None:
        """
        Set the time points data.
        Args:
            df: a pandas DataFrame with at least the 'value' column.
        """
        if self.is_read_only:
            raise VReadOnlyError

        if not isinstance(df, (pd.DataFrame, H5DataFrame)):
            raise TypeError("'time points' must be a pandas DataFrame.")

        elif df.shape[0] != self.n_timepoints:
            raise ShapeError(f"'time points' has {df.shape[0]} lines, it should have {self.n_timepoints}.")

        elif "value" not in df.columns:
            raise ValueError("Time points DataFrame should contain a 'value' column.")

        df["value"] = tp.as_timepointarray(df["value"])
        self._timepoints = H5DataFrame(df)

    @property
    def timepoints_values(self) -> tp.TimePointArray | NDArrayView[tp.TimePoint]:
        """
        Get the list of time points values (with the unit if possible).

        :return: the list of time points values (with the unit if possible).
        """
        return tp.as_timepointarray(self.timepoints.value)

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
    def obs(self) -> TemporalDataFrameBase:
        """
        Get the obs data.
        :return: the obs TemporalDataFrame.
        """
        return self._obs

    @obs.setter
    def obs(self, df: pd.DataFrame | H5DataFrame | TemporalDataFrame) -> None:
        """
        Set the obs data.

        Args:
            df: a pandas DataFrame or a TemporalDataFrame.
        """
        if self.is_read_only:
            raise VReadOnlyError

        if not isinstance(df, (pd.DataFrame, H5DataFrame, TemporalDataFrame)):
            raise TypeError("'obs' must be a pandas DataFrame or a TemporalDataFrame.")

        if not df.shape[0] == self.obs.n_index_total:
            raise ShapeError(f"'obs' has {df.shape[0]} rows, it should have {self.n_obs_total}.")

        if isinstance(df, (pd.DataFrame, H5DataFrame)):
            # cast to TemporalDataFrame
            if self.obs.timepoints_column_name is not None and self.obs.timepoints_column_name in df.columns:
                _time_col_name: str | None = self.obs.timepoints_column_name
            else:
                _time_col_name = None

            _time_list = self.obs.timepoints_column if _time_col_name is None else None

            df = TemporalDataFrame(
                df, timepoints=_time_list, time_col_name=_time_col_name, index=self.obs.index, name="obs"
            )

        else:
            if df.timepoints != self.obs.timepoints:
                raise ValueError("'obs' time points do not match.")

            if not np.all(df.index == self.obs.index):
                raise ValueError("'obs' index does not match.")

            if not np.all(df.columns == self.obs.columns):
                raise ValueError("'obs' column names do not match.")

        self._obs = df
        self._obs.lock_indices()

    @property
    def var(self) -> H5DataFrame:
        """
        Get the var data.
        :return: the var DataFrame.
        """
        return self._var

    @var.setter
    def var(self, df: pd.DataFrame | H5DataFrame) -> None:
        """
        Set the var data.
        Args:
            df: a pandas DataFrame.
        """
        if self.is_read_only:
            raise VReadOnlyError

        if not isinstance(df, (pd.DataFrame, H5DataFrame)):
            raise TypeError("'var' must be a pandas DataFrame.")

        elif df.shape[0] != self.n_var:
            raise ShapeError(f"'var' has {df.shape[0]} lines, it should have {self.n_var}.")

        self._var = H5DataFrame(df)

    @property
    def uns(self) -> DictLike[Any]:
        """
        Get the uns dictionary in this VData.
        :return: the uns dictionary in this VData.
        """
        return self._uns

    @uns.setter
    def uns(self, data: DictLike[Any]) -> None:
        if self.is_read_only:
            raise VReadOnlyError

        if isinstance(data, ch.H5Dict):
            self._uns = data  # type: ignore[assignment]

        elif isinstance(data, dict):
            self._uns = dict(zip([str(k) for k in data.keys()], data.values()))

        else:
            raise TypeError("'uns' must be a dictionary.")

    @property
    def layers(self) -> VLayersArrayContainer:
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

    # endregion

    # region aliases
    cells = obs
    genes = var

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
        return pd.Index(self.obs.index.values)

    @property
    def var_names(self) -> pd.Index:
        """
        Alias for the var index names.

        Returns:
            The var index names.
        """
        return self.var.index

    # endregion

    # region methods
    def set_max_memory_usage(self, max_memory: int | str) -> None:
        """
        Set the maximum RAM amount that can be used by individual arrays backed on h5 files when
        performing computations.
        `max_memory` should either be a number of bytes or a string of the form "<x><U>", where:
            - <x> is an integer
            - <u> is a unit in ('B', 'K', 'M', 'G')
        """
        ch.set_options(max_memory=max_memory)

    def set_obs_index(self, values: Collection[IFS] | Index) -> None:
        """
        Set a new index for observations.

        Args:
            values: collection of new index values.
            repeating_index: does the index repeat itself at all time-points ? (default: False)
        """
        self.layers.set_index(values)

        self.obs.set_index(values, force=True)
        self.obsm.set_index(values)
        self.obsp.set_index(values)

    def make_unique_obs_index(self) -> None:
        """
        Concatenates the obs index with the time-point to make all index values unique.
        """
        self.set_obs_index(
            np.char.add(np.char.add(self.obs.index.values.astype(str), "_"), self.obs.timepoints_column_str)
        )

    def set_var_index(self, values: Collection[IFS]) -> None:
        """
        Set a new index for variables.
        Args:
            values: collection of new index values.
        """
        self.layers.set_columns(values)
        self.var.index = pd.Index(values)
        self.varm.set_index(values)
        self.varp.set_index(values)

    def _mean_min_max_func(
        self,
        func: Literal["mean", "min", "max"],
        axis: int,
    ) -> tuple[dict[str, TemporalDataFrame], tp.TimePointArray | NDArrayView[tp.TimePoint], Index]:
        """
        Compute mean, min or max of the values over the requested axis.
        """
        data: dict[str, TemporalDataFrame] = {
            layer: getattr(self.layers[layer], func)(axis=axis + 1) for layer in self.layers
        }

        if axis == 0:
            timepoints_list = self.timepoints_values
            index = Index(["mean"], repeats=self.n_timepoints)

            return data, timepoints_list, index

        elif axis == 1:
            timepoints_list = self._obs.timepoints_column
            index = self.obs.index

            return data, timepoints_list, index

        raise ValueError(f"Invalid axis '{axis}', should be 0 or 1.")

    def mean(self, axis: Literal[0, 1] = 0) -> VData:
        """
        Return the mean of the values over the requested axis.

        :param axis: compute mean over columns (0: default) or over rows (1).
        :return: a TemporalDataFrame with mean values.
        """
        data, timepoints_list, index = self._mean_min_max_func("mean", axis)

        name = f"Mean of {self._name}" if self._name != NO_NAME else "Mean"
        return VData(
            data=data,
            obs=TemporalDataFrame(index=index, timepoints=timepoints_list),
            timepoints_list=timepoints_list,
            name=name,
        )

    def min(self, axis: Literal[0, 1] = 0) -> VData:
        """
        Return the minimum of the values over the requested axis.

        :param axis: compute minimum over columns (0: default) or over rows (1).
        :return: a TemporalDataFrame with minimum values.
        """
        data, timepoints_list, index = self._mean_min_max_func("min", axis)

        name = f"Minimum of {self._name}" if self._name != NO_NAME else "Minimum"
        return VData(
            data=data,
            obs=pd.DataFrame(index=index.values),
            timepoints_list=timepoints_list,
            name=name,
        )

    def max(self, axis: Literal[0, 1] = 0) -> VData:
        """
        Return the maximum of the values over the requested axis.

        :param axis: compute maximum over columns (0: default) or over rows (1).
        :return: a TemporalDataFrame with maximum values.
        """
        data, timepoints_list, index = self._mean_min_max_func("max", axis)

        name = f"Maximum of {self._name}" if self._name != NO_NAME else "Maximum"
        return VData(
            data=data,
            obs=pd.DataFrame(index=index.values),
            timepoints_list=timepoints_list,
            name=name,
        )

    def write(self, file: str | Path | None = None, verbose: bool = True) -> None:
        """
        Save this VData object in HDF5 file format.

        Args:
            file: path to save the VData
            verbose: print a progress bar while saving objects in this VData ? (default: True)
        """
        h5_data = write_vdata(self, file, verbose)

        if not self.is_backed:
            self._data = h5_data
            self._initialize(ParsingDataOut.from_h5(h5_data), h5_data.attributes["name"])

    def write_to_csv(
        self, directory: str | Path, sep: str = ",", na_rep: str = "", index: bool = True, header: bool = True
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

    @classmethod
    def read(cls, path: str | Path, mode: Literal[ch.H5Mode.READ, ch.H5Mode.READ_WRITE] = ch.H5Mode.READ) -> VData:
        """
        Read VData from a h5 file.

        Args:
            path: path to a h5 file.
            mode: mode for opening the h5 file.
        """
        return VData.__h5_read__(ch.H5Dict.read(path, mode=ch.H5Mode(mode)))

    @classmethod
    def read_from_csv(
        cls,
        path: str | Path,
        time_list: Sequence[str | tp.TimePoint] | Literal["*"] | None = None,
        time_col_name: str | None = None,
        name: str = "",
    ) -> VData:
        """
        Read VData from a folder containing csv files (usually generated by first saving a VData with the
        .write_to_csv() method).

        Args:
            directory: a path to a directory containing csv datasets.
                The directory should have the format, for any combination of the following datasets :
                    ⊦ layers
                        ⊦ <...>.csv
                    ⊦ obsm
                        ⊦ <...>.csv
                    ⊦ obsp
                        ⊦ <...>.csv
                    ⊦ varm
                        ⊦ <...>.csv
                    ⊦ varp
                        ⊦ <...>.csv
                    ⊦ obs.csv
                    ⊦ timepoints.csv
                    ⊦ var.csv
            time_list: time points for the dataframe's rows. (see TemporalDataFrame's documentation for more details.)
            time_col: if time points are not given explicitly with the 'time_list' parameter, a column name can be
                given. This column will be used as the time data.
            name: an optional name for the loaded VData object.
        """
        return read_from_csv(path, time_list, time_col_name, name)

    @classmethod
    def read_from_anndata(
        cls,
        path: str | Path,
        timepoint: IFS | tp.TimePoint = tp.TimePoint("0h"),
        timepoints_column_name: str | None = None,
        inplace: bool = False,
    ) -> VData:
        """
        Args:
            path: path to the anndata h5 file to convert.
            time_point: a unique timepoint to set for the data in the anndata.
            time_column_name: the name of the column in anndata's obs to use as indicator of time point for the data.
            inplace: perform file conversion directly on the anndata h5 file ? (default False)
        """
        data = convert_anndata_to_vdata(path, timepoint, timepoints_column_name, inplace=inplace)
        return VData.__h5_read__(data)

    def copy(self) -> VData:
        """
        Build a deep copy of this VData object and not a view.

        Returns:
            A new VData, which is a deep copy of this VData.
        """
        return VData(
            data=self.layers.dict_copy(),
            obs=self._obs.copy(),
            obsm=self._obsm.dict_copy(),
            obsp=self._obsp.dict_copy(),
            var=self._var.copy(),
            varm=self._varm.dict_copy(),
            varp=self._varp.dict_copy(),
            timepoints=self._timepoints.copy(),
            uns=self._uns.copy(),
            name=f"{self._name}_copy",
        )

    def to_anndata(
        self,
        timepoints_list: str | tp.TimePoint | Collection[str | tp.TimePoint] | None = None,
        into_one: bool = True,
        with_timepoints_column: bool = True,
        layer_as_X: str | None = None,
        layers_to_export: list[str] | None = None,
    ) -> AnnData | list[AnnData]:
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
        return convert_vdata_to_anndata(
            self,
            timepoints_list=timepoints_list,
            into_one=into_one,
            with_timepoints_column=with_timepoints_column,
            layer_as_X=layer_as_X,
            layers_to_export=layers_to_export,
        )

    def as_anndata(self, X: str | None = None) -> AnnDataProxy:
        return AnnDataProxy(self, X=X)

    def close(self) -> None:
        if self._data is None:
            raise IOError("Cannot close VData since it is not backed on a file.")

        self._data.close()

    # endregion
