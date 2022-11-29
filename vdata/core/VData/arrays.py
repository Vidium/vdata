# coding: utf-8
# Created on 11/4/20 10:40 AM
# Author : matteo

# ====================================================
# imports
import os
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Union, KeysView, ValuesView, ItemsView, MutableMapping, Iterator, TypeVar, Collection, \
    Generic, Literal

import vdata
from .name_utils import DataFrame
from ..tdf import TemporalDataFrame
from vdata.vdataframe import VDataFrame
from vdata.name_utils import DType
from vdata.time_point import TimePoint
from vdata.IO import generalLogger, IncoherenceError, VAttributeError, ShapeError, VTypeError, VValueError, \
    VClosedFileError, VReadOnlyError
from vdata.h5pickle import File, Group
from vdata.read_write import read_TDF
from ...h5pickle.name_utils import H5Mode

# ====================================================
# code
D_VDF = TypeVar('D_VDF', bound=VDataFrame)
D_TDF = TypeVar('D_TDF', bound=TemporalDataFrame)

TD_K = tuple[Union[slice, TimePoint], str]
K_ = TypeVar('K_', str, TD_K)


# Containers ------------------------------------------------------------------
class TimedDict(dict, Generic[K_, D_VDF]):
    def __init__(self,
                 parent: 'vdata.VData',
                 **kwargs):
        dict.__init__(kwargs)

        self._parent = parent

    def __getitem__(self,
                    key: tuple[Union[slice, TimePoint], K_]) -> D_VDF:
        vdf = dict.__getitem__(self, key[1])

        if isinstance(key[0], slice):
            return vdf

        index = self._parent.obs.index_at(key[0])
        return vdf[index, index]

    def __setitem__(self,
                    key: K_,
                    value: D_VDF) -> None:
        dict.__setitem__(self, key, value)


D = TypeVar('D', DataFrame, TimedDict)
TD_ = TypeVar('TD_', bound=TimedDict)


# Base Containers -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
class VBaseArrayContainer(ABC, MutableMapping[str, D], Generic[K_, D]):
    """
    Base abstract class for ArrayContainers linked to a VData object (obsm, obsp, varm, varp, layers).
    All Arrays have a '_parent' attribute for linking them to a VData and a '_data' dictionary
    attribute for storing 2D/3D arrays.
    """

    def __init__(self, parent: 'vdata.VData', data: Optional[dict[K_, D]]):
        """
        Args:
            parent: the parent VData object this ArrayContainer is linked to.
            data: a dictionary of data items (pandas DataFrames, TemporalDataFrames or dictionaries of pandas
            DataFrames) to store in this ArrayContainer.
        """
        generalLogger.debug(f"== Creating {self.__class__.__name__}. ==========================")

        self._parent = parent
        self._data = self._check_init_data(data)

    @abstractmethod
    def _check_init_data(self, data: Optional[dict[K_, D]]) -> dict[K_, D]:
        """
        Function for checking, at ArrayContainer creation, that the supplied data has the correct format.

        Args:
            data: optional dictionary of data items.
        Returns:
            The data, if correct.
        """
        pass

    def __repr__(self) -> str:
        """
        Get a string representation of this ArrayContainer.
        :return: a string representation of this ArrayContainer.
        """
        if len(self):
            list_of_keys = "'" + "','".join(self.keys()) + "'"
            return f"{self.__class__.__name__} with keys : {list_of_keys}."
        else:
            return f"Empty {self.__class__.__name__}."

    def __getitem__(self, item: str) -> D:
        """
        Get a specific data item stored in this ArrayContainer.

        Args:
            item: key in _data linked to a data item.

        Returns:
            Data item stored in _data under the given key.
        """
        if self.is_closed:
            raise VClosedFileError

        if not len(self) or item not in self.keys():
            raise VAttributeError(f"{self.name} ArrayContainer has no attribute '{item}'")

        return self._data[item]

    @abstractmethod
    def __setitem__(self, key: K_, value: D) -> None:
        """
        Set a specific data item in _data. The given data item must have the correct shape.

        Args:
            key: key for storing a data item in this ArrayContainer.
            value: a data item to store.
        """
        pass

    def __delitem__(self, key: K_) -> None:
        """
        Delete a specific data item stored in this ArrayContainer.
        """
        if self.is_closed:
            raise VClosedFileError

        if self.is_read_only:
            raise VReadOnlyError

        del self._data[key]

    def __len__(self) -> int:
        """
        Length of this ArrayContainer : the number of data items in _data.
        :return: number of data items in _data.
        """
        return len(self.keys())

    def __iter__(self) -> Iterator[K_]:
        """
        Iterate on this ArrayContainer's keys.
        :return: an iterator over this ArrayContainer's keys.
        """
        if self.is_closed:
            raise VClosedFileError

        return iter(self.keys())

    @property
    def is_closed(self) -> bool:
        """
        Is the parent's file closed ?
        """
        return self._parent.is_closed

    @property
    def is_read_only(self) -> bool:
        """
        Is the parent's file open in read only mode ?
        """
        return self._parent.is_read_only

    @property
    @abstractmethod
    def empty(self) -> bool:
        """
        Whether this ArrayContainer is empty or not.
        :return: is this ArrayContainer empty ?
        """
        pass

    @abstractmethod
    def update_dtype(self, type_: 'DType') -> None:
        """
        Update the data type of Arrays stored in this ArrayContainer.

        Args:
            type_: the new data type.
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """
        Name for this ArrayContainer.
        :return: the name of this ArrayContainer.
        """
        pass

    @property
    @abstractmethod
    def shape(self) -> Union[
        tuple[int, int, int],
        tuple[int, int, list[int]],
        tuple[int, int, list[int], int]
    ]:
        """
        The shape of this ArrayContainer is computed from the shape of the Arrays it contains.
        See __len__ for getting the number of Arrays it contains.

        Returns:
            The shape of this ArrayContainer.
        """
        pass

    @property
    def data(self) -> dict[K_, D]:
        """
        Data of this ArrayContainer.

        Returns:
            The data of this ArrayContainer.
        """
        if self.is_closed:
            raise VClosedFileError

        return self._data

    def keys(self) -> KeysView[K_]:
        """
        KeysView of keys for getting the data items in this ArrayContainer.

        Returns:
            KeysView of this ArrayContainer.
        """
        if self.is_closed:
            raise VClosedFileError

        return self._data.keys()

    def values(self) -> ValuesView[D]:
        """
        ValuesView of data items in this ArrayContainer.

        Returns:
            ValuesView of this ArrayContainer.
        """
        if self.is_closed:
            raise VClosedFileError

        return self._data.values()

    def items(self) -> ItemsView[K_, D]:
        """
        ItemsView of pairs of keys and data items in this ArrayContainer.

        Returns:
            ItemsView of this ArrayContainer.
        """
        if self.is_closed:
            raise VClosedFileError

        return self._data.items()

    @abstractmethod
    def dict_copy(self) -> dict[K_, D]:
        """
        Dictionary of keys and data items in this ArrayContainer.

        Returns:
            Dictionary of this ArrayContainer.
        """
        pass

    @abstractmethod
    def to_csv(self, directory: Path, sep: str = ",", na_rep: str = "",
               index: bool = True, header: bool = True, spacer: str = '') -> None:
        """
        Save this ArrayContainer in CSV file format.

        Args:
            directory: path to a directory for saving the Array
            sep: delimiter character
            na_rep: string to replace NAs
            index: write row names ?
            header: Write col names ?
            spacer: for logging purposes, the recursion depth of calls to a read_h5 function.
        """
        pass

    @abstractmethod
    def set_file(self, file: Union[File, Group]) -> None:
        """
        Set the file to back the Arrays in this ArrayContainer.

        Args:
            file: a h5 file to back the Arrays on.
        """
        pass


# 3D Containers -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -
class VBase3DArrayContainer(VBaseArrayContainer, ABC, MutableMapping[str, D_TDF], Generic[K_, D_TDF]):
    """
    Base abstract class for ArrayContainers linked to a VData object that contain TemporalDataFrames (obsm and layers).
    It is based on VBaseArrayContainer and defines some functions shared by obsm and layers.
    """

    def __init__(self, parent: 'vdata.VData', data: Optional[dict[K_, D_TDF]]):
        """
        Args:
            parent: the parent VData object this ArrayContainer is linked to.
            data: a dictionary of TemporalDataFrames in this ArrayContainer.
        """
        super().__init__(parent, data)

    def __setitem__(self, key: K_, value: D_TDF) -> None:
        """
        Set a specific TemporalDataFrame in _data. The given TemporalDataFrame must have the correct shape.

        Args:
            key: key for storing a TemporalDataFrame in this VObsmArrayContainer.
            value: a TemporalDataFrame to store.
        """
        if self.is_closed:
            raise VClosedFileError

        if self.is_read_only:
            raise VReadOnlyError

        if not np.array_equal(self._parent.timepoints.value.values, value.timepoints):
            raise VValueError("Time points do not match.")

        if not np.array_equal(self._parent.obs.index, value.index):
            raise VValueError("Index does not match.")

        self._data[key] = value

    @property
    def empty(self) -> bool:
        """
        Whether this ArrayContainer is empty or not.
        :return: is this ArrayContainer empty ?
        """
        # return all([tdf.empty for tdf in self.values()])
        return len(self.keys()) == 0

    @property
    def has_repeating_index(self) -> bool:
        if self.empty:
            return False

        return list(self.values())[0].has_repeating_index

    def update_dtype(self, type_: 'DType') -> None:
        """
        Update the data type of TemporalDataFrames stored in this ArrayContainer.

        Args:
            type_: the new data type.
        """
        for arr in self.values():
            arr.astype(type_)

    @property
    def shape(self) -> tuple[int, int, list[int], int]:
        """
        The shape of this ArrayContainer is computed from the shape of the TemporalDataFrames it contains.
        See __len__ for getting the number of TemporalDataFrames it contains.
        :return: the shape of this ArrayContainer.
        """
        if len(self):
            _first_TDF = list(self.values())[0]
            _shape_TDF = _first_TDF.shape
            return len(self), _shape_TDF[0], _shape_TDF[1], _shape_TDF[2]

        else:
            return 0, 0, [], 0

    def dict_copy(self) -> dict[K_, D_TDF]:
        """
        Dictionary of keys and data items in this ArrayContainer.
        :return: Dictionary of this ArrayContainer.
        """
        return {k: v.copy() for k, v in self.items()}

    def to_csv(self,
               directory: Path,
               sep: str = ",",
               na_rep: str = "",
               index: bool = True,
               header: bool = True,
               spacer: str = '') -> None:
        """
        Save the ArrayContainer in CSV file format.

        Args:
            directory: path to a directory for saving the Array
            sep: delimiter character
            na_rep: string to replace NAs
            index: write row names ?
            header: Write col names ?
            spacer: for logging purposes, the recursion depth of calls to a read_h5 function.
        """
        if self.is_closed:
            raise VClosedFileError

        # create subdirectory for storing arrays
        os.makedirs(directory / self.name)

        for arr_name, arr in self.items():
            generalLogger.info(f"{spacer}Saving {arr_name}")

            # save array
            arr.to_csv(f"{directory / self.name / arr_name}.csv", sep, na_rep, index=index, header=header)

    def set_file(self, file: Union[File, Group]) -> None:
        """
        Set the file to back the TemporalDataFrames in this VBase3DArrayContainer.

        Args:
            file: a h5 file to back the TemporalDataFrames on.
        """
        if self.is_read_only:
            raise VReadOnlyError

        if not isinstance(file, (File, Group)):
            raise VTypeError(f"Cannot back TemporalDataFrames in this VBase3DArrayContainer with an object of type '"
                             f"{type(file)}'.")

        for arr_name, arr in self.items():
            arr.file = file[arr_name]


class VLayerArrayContainer(VBase3DArrayContainer):
    """
    Class for layers.
    This object contains any number of TemporalDataFrames, with shapes (n_timepoints, n_obs, n_var).
    The arrays-like objects can be accessed from the parent VData object by :
        VData.layers[<array_name>]
    """

    def __init__(self, parent: 'vdata.VData', data: Optional[dict[K_, D_TDF]]):
        """
        Args:
            parent: the parent VData object this VLayerArrayContainer is linked to.
            data: a dictionary of TemporalDataFrames in this VLayerArrayContainer.
        """
        super().__init__(parent, data)

    def _check_init_data(self, data: Optional[dict[K_, D_TDF]]) -> dict[K_, D_TDF]:
        """
        Function for checking, at VLayerArrayContainer creation, that the supplied data has the correct format :
            - the shape of the TemporalDataFrames in 'data' match the parent VData object's shape.
            - the index of the TemporalDataFrames in 'data' match the index of the parent VData's obs TemporalDataFrame.
            - the column names of the TemporalDataFrames in 'data' match the index of the parent VData's var DataFrame.
            - the time points of the TemporalDataFrames in 'data' match the index of the parent VData's time-points
            DataFrame.

        Args:
            data: optional dictionary of TemporalDataFrames.

        Returns:
            The data (dictionary of TemporalDataFrames), if correct.
        """
        if data is None or not len(data):
            generalLogger.debug("  No data was given.")
            return {'data': TemporalDataFrame(index=self._parent.obs.index, columns_numerical=self._parent.var.index,
                                              time_list=self._parent.obs.timepoints_column,
                                              # timepoints=self._parent.timepoints.value,
                                              name='data')}

        else:
            generalLogger.debug("  Data was found.")
            _data = {}
            _shape = (self._parent.timepoints.shape[0], self._parent.obs.shape[1], self._parent.var.shape[0])
            _index = self._parent.obs.index
            _columns = self._parent.var.index
            _timepoints: pd.Series = self._parent.timepoints['value']

            generalLogger.debug(f"  Reference shape is {_shape}.")

            for TDF_index, TDF in data.items():
                TDF_shape = TDF.shape

                generalLogger.debug(f"  Checking TemporalDataFrame '{TDF_index}' with shape {TDF_shape}.")

                if _shape != TDF_shape:

                    # check that shapes match
                    if _shape[0] != TDF_shape[0]:
                        raise IncoherenceError(f"Layer '{TDF_index}' has {TDF_shape[0]} "
                                               f"time point{'s' if TDF_shape[0] > 1 else ''}, "
                                               f"should have {_shape[0]}.")

                    elif _shape[1] != TDF_shape[1]:
                        for i in range(len(TDF.timepoints)):
                            if _shape[1][i] != TDF_shape[1][i]:
                                raise IncoherenceError(f"Layer '{TDF_index}' at time point {i} has"
                                                       f" {TDF_shape[1][i]} observations, "
                                                       f"should have {_shape[1][i]}.")

                    else:
                        raise IncoherenceError(f"Layer '{TDF_index}' has  {TDF_shape[2]} variables, "
                                               f"should have {_shape[2]}.")

                # check that indexes match
                if not np.all(_index == TDF.index):
                    raise IncoherenceError(f"Index of layer '{TDF_index}' ({TDF.index}) does not match obs' index. ("
                                           f"{_index})")

                if not np.all(_columns == TDF.columns):
                    raise IncoherenceError(f"Column names of layer '{TDF_index}' ({TDF.columns}) do not match var's "
                                           f"index. ({_columns})")

                if not np.all(_timepoints == TDF.timepoints):
                    raise IncoherenceError(f"Time points of layer '{TDF_index}' ({TDF.timepoints}) do not match "
                                           f"time_point's index. ({_timepoints})")

                # checks passed, store the TemporalDataFrame
                # TODO : change this once the VData object will be split into Backed and regular objects
                if not TDF.is_backed or TDF.h5_mode == 'r+':
                    TDF.lock_indices()
                    TDF.lock_columns()

                assert TDF.has_locked_indices and TDF.has_locked_columns

                _data[str(TDF_index)] = TDF

            generalLogger.debug("  Data was OK.")
            return _data

    def __setitem__(self, key: K_, value: D_TDF) -> None:
        """
        Set a specific TemporalDataFrame in _data. The given TemporalDataFrame must have the correct shape.

        Args:
            key: key for storing a TemporalDataFrame in this VObsmArrayContainer.
            value: a TemporalDataFrame to store.
        """
        if not isinstance(value, TemporalDataFrame):
            raise VTypeError(f"Cannot set {self.name} '{key}' from non TemporalDataFrame object.")

        if not self.shape[1:] == value.shape:
            raise ShapeError(f"Cannot set {self.name} '{key}' because of shape mismatch.")

        if not np.array_equal(self._parent.var.index, value.columns):
            raise VValueError("Column names do not match.")

        value_copy = value.copy()
        value_copy.name = key

        value_copy.lock_indices()
        value_copy.lock_columns()

        if self._parent.is_backed_w:
            if key not in self._parent.file['layers'].keys():
                self._parent.file['layers'].create_group(key)
            value_copy.write(self._parent.file['layers'][key].group)
            value_copy = read_TDF(self._parent.file['layers'][key].group, mode=H5Mode.READ_WRITE)

        super().__setitem__(key, value_copy)

    @property
    def name(self) -> Literal['layers']:
        """
        Name for this VLayerArrayContainer : layers.
        :return: name of this VLayerArrayContainer.
        """
        return 'layers'


class VObsmArrayContainer(VBase3DArrayContainer):
    """
    Class for obsm.
    This object contains any number of TemporalDataFrames, with shape (n_timepoints, n_obs, any).
    The TemporalDataFrames can be accessed from the parent VData object by :
        VData.obsm[<array_name>])
    """

    def __init__(self, parent: 'vdata.VData', data: Optional[dict[K_, D_TDF]] = None):
        """
        Args:
            parent: the parent VData object this VObsmArrayContainer is linked to.
            data: a dictionary of TemporalDataFrames in this VObsmArrayContainer.
        """
        super().__init__(parent, data)

    def _check_init_data(self, data: Optional[dict[K_, D_TDF]]) -> dict[K_, D_TDF]:
        """
        Function for checking, at VObsmArrayContainer creation, that the supplied data has the correct format :
            - the shape of the TemporalDataFrames in 'data' match the parent VData object's shape (except for the
            number of columns).
            - the index of the TemporalDataFrames in 'data' match the index of the parent VData's obs TemporalDataFrame.
            - the time points of the TemporalDataFrames in 'data' match the index of the parent VData's timepoints
            DataFrame.

        Args:
            data: optional dictionary of TemporalDataFrames.

        Returns:
            The data (dictionary of TemporalDataFrames), if correct.
        """
        if data is None or not len(data):
            generalLogger.debug("  No data was given.")
            return dict()

        else:
            generalLogger.debug("  Data was found.")
            _data = {}
            _shape = (self._parent.timepoints.shape[0],
                      self._parent.obs.shape[1],
                      'Any')
            _index = self._parent.obs.index
            _timepoints: pd.Series = self._parent.timepoints['value']

            generalLogger.debug(f"  Reference shape is {_shape}.")

            for TDF_index, TDF in data.items():
                TDF_shape = TDF.shape

                generalLogger.debug(f"  Checking TemporalDataFrame '{TDF_index}' with shape {TDF_shape}.")

                if _shape != TDF_shape:

                    # check that shapes match
                    if _shape[0] != TDF_shape[0]:
                        raise IncoherenceError(f"TemporalDataFrame '{TDF_index}' has {TDF_shape[0]} "
                                               f"time point{'s' if TDF_shape[0] > 1 else ''}, "
                                               f"should have {_shape[0]}.")

                    elif _shape[1] != TDF_shape[1]:
                        for i in range(len(TDF.timepoints)):
                            if _shape[1][i] != TDF_shape[1][i]:
                                raise IncoherenceError(f"TemporalDataFrame '{TDF_index}' at time point {i} has"
                                                       f" {TDF_shape[1][i]} rows, "
                                                       f"should have {_shape[1][i]}.")

                    else:
                        pass

                # check that indexes match
                if not np.all(_index == TDF.index):
                    raise IncoherenceError(f"Index of TemporalDataFrame '{TDF_index}' ({TDF.index}) does not match "
                                           f"obs' index. ({_index})")

                if not all(_timepoints == TDF.timepoints):
                    raise IncoherenceError(f"Time points of TemporalDataFrame '{TDF_index}' ({TDF.timepoints}) "
                                           f"do not match time_point's index. ({_timepoints})")

                # checks passed, store the TemporalDataFrame
                if not TDF.has_locked_indices:
                    TDF.lock_indices()
                _data[str(TDF_index)] = TDF

            generalLogger.debug("  Data was OK.")
            return _data

    def __setitem__(self, key: K_, value: D_TDF) -> None:
        """
        Set a specific TemporalDataFrame in _data. The given TemporalDataFrame must have the correct shape.

        Args:
            key: key for storing a TemporalDataFrame in this VObsmArrayContainer.
            value: a TemporalDataFrame to store.
        """
        if not isinstance(value, TemporalDataFrame):
            raise VTypeError(f"Cannot set {self.name} '{key}' from non TemporalDataFrame object.")

        if not self.shape[1:3] == value.shape[:2]:
            raise ShapeError(f"Cannot set {self.name} '{key}' because of shape mismatch.")

        value_copy = value.copy()
        value_copy.name = key

        value_copy.lock_indices()
        super().__setitem__(key, value_copy)

        if self._parent.is_backed_w:
            if key not in self._parent.file['obsm'].keys():
                self._parent.file['obsm'].create_group(key)

            value_copy.write(self._parent.file['obsm'][key].group)

    @property
    def shape(self) -> tuple[int, int, list[int], list[int]]:
        """
        The shape of this ArrayContainer is computed from the shape of the TemporalDataFrames it contains.
        See __len__ for getting the number of TemporalDataFrames it contains.

        Returns:
            The shape of this ArrayContainer.
        """
        n_timepoints = self._parent.shape[1]
        n_obs = self._parent.shape[2]

        return len(self._data), n_timepoints, n_obs, [d.shape[2] for d in self._data.values()]

    @property
    def name(self) -> Literal['obsm']:
        """
        Name for this VObsmArrayContainer : obsm.
        :return: name of this VObsmArrayContainer.
        """
        return 'obsm'


# Obsp Containers -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
class VObspArrayContainer(VBaseArrayContainer, Generic[K_, D_VDF]):
    """
    Class for obsp.
    This object contains sets of <nb time points> 2D square DataFrames of shapes (<n_obs>, <n_obs>) for each time point.
    The DataFrames can be accessed from the parent VData object by :
        VData.obsp[<array_name>][<time point>]
    """

    def __init__(self,
                 parent: 'vdata.VData',
                 data: Optional[dict[K_, pd.DataFrame]],
                 file: Optional[Union[File, Group]] = None):
        """
        Args:
            parent: the parent VData object this VObspArrayContainer is linked to.
            data: a dictionary of array-like objects to store in this VObspArrayContainer.
        """
        self._file = file

        super().__init__(parent, data)

    def _check_init_data(self,
                         data: Optional[dict[K_, pd.DataFrame]]) -> TimedDict[str, VDataFrame]:
        """
        Function for checking, at VObspArrayContainer creation, that the supplied data has the correct format :
            - the shape of the DataFrames in 'data' match the parent VData object's index length.
            - the index and columns names of the DataFrames in 'data' match the index of the parent VData's obs
            TemporalDataFrame.
            - the time points of the dictionaries of DataFrames in 'data' match the index of the parent VData's
            time-points DataFrame.

        Args:
            data: dictionary of dictionaries (TimePoint: DataFrame (n_obs x n_obs))

        Returns:
            The data (dictionary of dictionaries of DataFrames), if correct.
        """
        if data is None or not len(data):
            generalLogger.debug("  No data was given.")
            return TimedDict(parent=self._parent)

        else:
            generalLogger.debug("  Data was found.")
            _data = TimedDict(parent=self._parent)

            for key, df in data.items():
                generalLogger.debug(f"  Checking DataFrame at key '{key}' with shape {df.shape}.")

                _index = self._parent.obs.index
                file = self._file[key] if self._file is not None else None

                # check that square
                if df.shape[0] != df.shape[1]:
                    raise ShapeError(f"DataFrame at key '{key}' should be square.")

                # check that indexes match
                if not np.all(_index == df.index):
                    raise IncoherenceError(f"Index of DataFrame at key '{key}' ({df.index}) does not "
                                           f"match obs' index. ({_index})")

                if not np.all(_index == df.columns):
                    raise IncoherenceError(f"Column names of DataFrame at key '{key}' ({df.columns}) "
                                           f"do not match obs' index. ({_index})")

                # checks passed, store as VDataFrame
                _data[key] = VDataFrame(df, file=file)

            generalLogger.debug("  Data was OK.")
            return _data

    def __getitem__(self,
                    item: K_) -> D_VDF:
        """
        Get a specific set VDataFrame stored in this VObspArrayContainer.

        Args:
            item: key in _data linked to a set of DataFrames.

        Returns:
            A VDataFrame stored in _data under the given key.
        """
        if self.is_closed:
            raise VClosedFileError

        if len(self) and item in self.keys():
            return self._data[(slice(None), item)]

        else:
            raise VAttributeError(f"{self.name} ArrayContainer has no attribute '{item}'")

    def __setitem__(self,
                    key: K_,
                    value: Union[VDataFrame, pd.DataFrame]) -> None:
        """
        Set a specific DataFrame in _data. The given DataFrame must have the correct shape.

        Args:
            key: key for storing a set of DataFrames in this VObspArrayContainer.
            value: a set of DataFrames to store.
        """
        if self.is_closed:
            raise VClosedFileError

        if self.is_read_only:
            raise VReadOnlyError

        if not isinstance(value, (pd.DataFrame, VDataFrame)):
            raise VTypeError("The value should be a pandas DataFrame or a VDataFrame.")

        if isinstance(value, pd.DataFrame):
            value = VDataFrame(value)

        _index = self._parent.obs.index

        if not value.shape == (len(_index), len(_index)):
            raise ShapeError(f"DataFrame should have shape ({len(_index)}, {len(_index)}).")

        if not np.all(value.index == _index):
            raise VValueError("The index of the DataFrame does not match the index of the parent VData.")

        if not np.all(value.columns == _index):
            raise VValueError("The column names the DataFrame do not match the index of the parent VData.")

        self._data[key] = value

    @property
    def data(self) -> dict[K_, D_VDF]:
        """
        Data of this VObspArrayContainer.

        Returns:
            The data of this VObspArrayContainer.
        """
        if self.is_closed:
            raise VClosedFileError

        return self._data

    @property
    def empty(self) -> bool:
        """
        Whether this VObspArrayContainer is empty or not.

        Returns:
            Is this VObspArrayContainer empty ?
        """
        if not len(self) or all([vdf.empty for vdf in self.data.values()]):
            return True
        return False

    def update_dtype(self,
                     type_: 'DType') -> None:
        """
        Update the data type of VDataFrames stored in this VObspArrayContainer.

        Args:
            type_: the new data type.
        """
        for vdf_name in self.keys():
            self[vdf_name] = self[vdf_name].astype(type_)

    @property
    def name(self) -> Literal['obsp']:
        """
        Name for this VObspArrayContainer : obsp.
        :return: the name of this VObspArrayContainer.
        """
        return 'obsp'

    @property
    def shape(self) -> tuple[int, int, int]:
        """
        The shape of the VObspArrayContainer is computed from the shape of the Arrays it contains.
        See __len__ for getting the number of Arrays it contains.

        Returns:
            The shape of this VObspArrayContainer.
        """
        len_index = self._parent.n_obs_total

        if len(self):
            return len(self), len_index, len_index

        else:
            return 0, len_index, len_index

    def dict_copy(self) -> dict[K_, D_VDF]:
        """
        Dictionary of keys and copied data items in this ArrayContainer.

        Returns:
            A dictionary copy of this ArrayContainer.
        """
        return {key: vdf.copy() for key, vdf in self.items()}

    def to_csv(self,
               directory: Path,
               sep: str = ",",
               na_rep: str = "",
               index: bool = True,
               header: bool = True,
               spacer: str = '') -> None:
        """
        Save this VObspArrayContainer in CSV file format.

        Args:
            directory: path to a directory for saving the Array
            sep: delimiter character
            na_rep: string to replace NAs
            index: write row names ?
            header: Write col names ?
            spacer: for logging purposes, the recursion depth of calls to a read_h5 function.
        """
        if self.is_closed:
            raise VClosedFileError

        # create subdirectory for storing sets
        os.makedirs(directory / self.name)

        for vdf_name, vdf in self.items():
            generalLogger.info(f"{spacer}Saving {vdf_name}")

            # save array
            vdf.to_csv(f"{directory / self.name / vdf_name}.csv", sep, na_rep, index=index, header=header)

    def set_index(self,
                  values: Collection) -> None:
        """
        Set a new index for rows and columns.

        Args:
            values: collection of new index values.
        """
        for vdf_name in self.keys():

            self[vdf_name].index = values
            self[vdf_name].columns = values

    def set_file(self,
                 file: Union[File, Group]) -> None:
        """
        Set the file to back the VDataFrames in this VObspArrayContainer.

        Args:
            file: a h5 file to back the VDataFrames on.
        """
        if not isinstance(file, (File, Group)):
            raise VTypeError(f"Cannot back VDataFrames in this VObspArrayContainer with an object of type '"
                             f"{type(file)}'.")

        for vdf_name, vdf in self.items():
            vdf.file = file[vdf_name]


# 2D Containers -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -
class VBase2DArrayContainer(VBaseArrayContainer, ABC, MutableMapping[str, D_VDF], Generic[K_, D_VDF]):
    """
    Base abstract class for ArrayContainers linked to a VData object that contain DataFrames (varm and varp)
    It is based on VBaseArrayContainer and defines some functions shared by varm and varp.
    """

    def __init__(self, parent: 'vdata.VData', data: Optional[dict[K_, pd.DataFrame]],
                 file: Optional[Union[File, Group]] = None):
        """
        Args:
            parent: the parent VData object this ArrayContainer is linked to.
            data: a dictionary of DataFrames in this ArrayContainer.
        """
        self._file = file

        super().__init__(parent, data)

    @property
    def empty(self) -> bool:
        """
        Whether this ArrayContainer is empty or not.
        :return: is this ArrayContainer empty ?
        """
        return all([DF.empty for DF in self.values()])

    def update_dtype(self, type_: 'DType') -> None:
        """
        Update the data type of TemporalDataFrames stored in this ArrayContainer.

        Args:
            type_: the new data type.
        """
        for arr_name, arr in self.items():
            self[arr_name] = arr.astype(type_)

    def dict_copy(self) -> dict[K_, VDataFrame]:
        """
        Dictionary of keys and data items in this ArrayContainer.
        :return: Dictionary of this ArrayContainer.
        """
        return {k: v.copy() for k, v in self.items()}

    def to_csv(self,
               directory: Path,
               sep: str = ",",
               na_rep: str = "",
               index: bool = True,
               header: bool = True,
               spacer: str = '') -> None:
        """
        Save the ArrayContainer in CSV file format.

        Args:
            directory: path to a directory for saving the Array
            sep: delimiter character
            na_rep: string to replace NAs
            index: write row names ?
            header: Write col names ?
            spacer: for logging purposes, the recursion depth of calls to a read_h5 function.
        """
        if self.is_closed:
            raise VClosedFileError

        # create subdirectory for storing arrays
        os.makedirs(directory / self.name)

        for arr_name, arr in self.items():
            generalLogger.info(f"{spacer}Saving {arr_name}")

            # save array
            arr.to_csv(f"{directory / self.name / arr_name}.csv", sep, na_rep, index=index, header=header)

    def set_file(self, file: Union[File, Group]) -> None:
        """
        Set the file to back the VDataFrames in this VBase2DArrayContainer.

        Args:
            file: a h5 file to back the VDataFrames on.
        """
        if not isinstance(file, (File, Group)):
            raise VTypeError(f"Cannot back VDataFrames in this VBase2DArrayContainer with an object of type '"
                             f"{type(file)}'.")

        for arr_name, arr in self.items():
            arr.file = file[arr_name]


class VVarmArrayContainer(VBase2DArrayContainer):
    """
    Class for varm.
    This object contains any number of DataFrames, with shape (n_var, any).
    The DataFrames can be accessed from the parent VData object by :
        VData.varm[<array_name>])
    """

    def __init__(self, parent: 'vdata.VData', data: Optional[dict[K_, D_VDF]] = None,
                 file: Optional[Union[File, Group]] = None):
        """
        Args:
            parent: the parent VData object this VVarmArrayContainer is linked to.
            data: a dictionary of DataFrames in this VVarmArrayContainer.
        """
        super().__init__(parent, data, file)

    def _check_init_data(self, data: Optional[dict[K_, pd.DataFrame]]) -> dict[K_, D_VDF]:
        """
        Function for checking, at VVarmArrayContainer creation, that the supplied data has the correct format :
            - the index of the DataFrames in 'data' match the index of the parent VData's var DataFrame.
        :param data: optional dictionary of DataFrames.
        :return: the data (dictionary of D_DF), if correct.
        """
        if data is None or not len(data):
            generalLogger.debug("  No data was given.")
            return dict()

        else:
            generalLogger.debug("  Data was found.")
            _index = self._parent.var.index
            _data = {}

            for DF_index, DF in data.items():
                # check that indexes match
                if not _index.equals(DF.index):
                    raise IncoherenceError(f"Index of DataFrame '{DF_index}' does not  match var's index. ({_index})")

                _data[DF_index] = VDataFrame(DF, file=self._file[DF_index] if self._file is not None else None)

            generalLogger.debug("  Data was OK.")
            return _data

    def __getitem__(self, item: K_) -> D_VDF:
        """
        Get a specific DataFrame stored in this VVarmArrayContainer.
        :param item: key in _data linked to a DataFrame.
        :return: DataFrame stored in _data under the given key.
        """
        return super().__getitem__(item)

    def __setitem__(self, key: K_, value: Union[VDataFrame, pd.DataFrame]) -> None:
        """
        Set a specific DataFrame in _data. The given DataFrame must have the correct shape.

        Args:
            key: key for storing a DataFrame in this VVarmArrayContainer.
            value: a DataFrame to store.
        """
        if self.is_closed:
            raise VClosedFileError

        if self.is_read_only:
            raise VReadOnlyError

        if not isinstance(value, (pd.DataFrame, VDataFrame)):
            raise VTypeError(f"Cannot set varm '{key}' from non pandas DataFrame object.")

        if isinstance(value, pd.DataFrame):
            value = VDataFrame(value)

        if not self.shape[1] == value.shape[0]:
            raise ShapeError(f"Cannot set varm '{key}' because of shape mismatch.")

        if not self._parent.var.index.equals(value.index):
            raise VValueError("Index does not match.")

        self._data[key] = value

    @property
    def name(self) -> Literal['varm']:
        """
        Name for this VVarmArrayContainer : varm.
        :return: name of this VVarmArrayContainer.
        """
        return 'varm'

    @property
    def shape(self) -> tuple[int, int, list[int]]:
        """
        The shape of this VVarmArrayContainer is computed from the shape of the DataFrames it contains.
        See __len__ for getting the number of TemporalDataFrames it contains.
        :return: the shape of this VVarmArrayContainer.
        """
        if len(self):
            _first_DF = list(self.values())[0]
            return len(self), _first_DF.shape[0], [DF.shape[1] for DF in self.values()]

        else:
            return 0, self._parent.n_var, []


class VVarpArrayContainer(VBase2DArrayContainer):
    """
    Class for varp.
    This object contains any number of DataFrames, with shape (n_var, n_var).
    The DataFrames can be accessed from the parent VData object by :
        VData.varp[<array_name>])
    """

    def __init__(self, parent: 'vdata.VData', data: Optional[dict[K_, D_VDF]] = None,
                 file: Optional[Union[File, Group]] = None):
        """
        Args:
            parent: the parent VData object this VVarmArrayContainer is linked to.
            data: a dictionary of DataFrames in this VVarmArrayContainer.
        """
        super().__init__(parent, data, file)

    def _check_init_data(self, data: Optional[dict[K_, pd.DataFrame]]) -> dict[K_, D_VDF]:
        """
        Function for checking, at ArrayContainer creation, that the supplied data has the correct format :
            - the index and column names of the DataFrames in 'data' match the index of the parent VData's var
            DataFrame.
        :param data: optional dictionary of DataFrames.
        :return: the data (dictionary of D_DF), if correct.
        """
        if data is None or not len(data):
            generalLogger.debug("  No data was given.")
            return dict()

        else:
            generalLogger.debug("  Data was found.")
            _index = self._parent.var.index
            _data = {}

            for DF_index, DF in data.items():
                # check that indexes match
                if not _index.equals(DF.index):
                    raise IncoherenceError(f"Index of DataFrame '{DF_index}' does not  match var's index. ({_index})")

                # check that columns match
                if not _index.equals(DF.columns):
                    raise IncoherenceError(
                        f"Columns of DataFrame '{DF_index}' do not  match var's index. ({_index})")

                _data[DF_index] = VDataFrame(DF, file=self._file[DF_index] if self._file is not None else None)

            generalLogger.debug("  Data was OK.")
            return _data

    def __getitem__(self, item: K_) -> D_VDF:
        """
        Get a specific DataFrame stored in this VVarpArrayContainer.

        Args:
            item: key in _data linked to a DataFrame.

        Returns:
            DataFrame stored in _data under the given key.
        """
        return super().__getitem__(item)

    def __setitem__(self, key: K_, value: Union[VDataFrame, pd.DataFrame]) -> None:
        """
        Set a specific DataFrame in _data. The given DataFrame must have the correct shape.

        Args:
            key: key for storing a DataFrame in this VVarpArrayContainer.
            value: a DataFrame to store.
        """
        if self.is_closed:
            raise VClosedFileError

        if self.is_read_only:
            raise VReadOnlyError

        if not isinstance(value, (pd.DataFrame, VDataFrame)):
            raise VTypeError(f"Cannot set varp '{key}' from non pandas DataFrame object.")

        if isinstance(value, pd.DataFrame):
            value = VDataFrame(value)

        if not self.shape[1:] == value.shape:
            raise ShapeError(f"Cannot set varp '{key}' because of shape mismatch.")

        if not self._parent.var.index.equals(value.index):
            raise VValueError("Index does not match.")

        if not self._parent.var.index.equals(value.columns):
            raise VValueError("column names do not match.")

        self._data[key] = value

    @property
    def name(self) -> Literal['varp']:
        """
        Name for this VVarpArrayContainer : varp.
        :return: name of this VVarpArrayContainer.
        """
        return 'varp'

    @property
    def shape(self) -> tuple[int, int, int]:
        """
        The shape of this VVarpArrayContainer is computed from the shape of the DataFrames it contains.
        See __len__ for getting the number of TemporalDataFrames it contains.
        :return: the shape of this VVarpArrayContainer.
        """
        if len(self):
            _first_DF = self[list(self.keys())[0]]
            return len(self), _first_DF.shape[0], _first_DF.shape[1]

        else:
            return 0, self._parent.n_var, self._parent.n_var

    def set_index(self, values: Collection) -> None:
        """
        Set a new index for rows and columns.

        Args:
            values: collection of new index values.
        """
        for arr in self.values():

            arr.index = values
            arr.columns = values
