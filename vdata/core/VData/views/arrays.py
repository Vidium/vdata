# coding: utf-8
# Created on 15/01/2021 12:57
# Author : matteo

# ====================================================
# imports
import os
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Union, KeysView, ValuesView, ItemsView, Iterator, Mapping, TypeVar, Collection, Any

from vdata.core.VData.arrays import VBaseArrayContainer
from vdata.core.tdf import TemporalDataFrame, TemporalDataFrameView
from vdata.core.tdf.utils import parse_slicer       # TODO : move out of tdf package
from vdata.IO import generalLogger, VTypeError, VValueError, ShapeError
from vdata.time_point import TimePoint
from vdata.vdataframe import VDataFrame


# ====================================================
# code

D_V = TypeVar('D_V', TemporalDataFrameView, pd.DataFrame, dict[TimePoint, pd.DataFrame])
D_VTDF = TypeVar('D_VTDF', bound=TemporalDataFrameView)
D_VDF = TypeVar('D_VDF', bound=VDataFrame)


def _check_parent_has_not_changed(func):
    def wrapper(*args, **kwargs):
        self = args[0]

        if isinstance(self, ViewVLayerArrayContainer):
            if hash(tuple(self._array_container._parent.timepoints.value.values)) != self._parent_timepoints_hash or \
                    hash(tuple(self._array_container._parent.obs.index)) != self._parent_obs_hash or \
                    hash(tuple(self._array_container._parent.var.index)) != self._parent_var_hash:
                raise VValueError("View no longer valid since parent's VData has changed.")

        elif isinstance(self, ViewVObsmArrayContainer):
            if hash(tuple(self._array_container._parent.timepoints.value.values)) != self._parent_timepoints_hash or \
                    hash(tuple(self._array_container._parent.obs.index)) != self._parent_obs_hash:
                raise VValueError("View no longer valid since parent's VData has changed.")

        return func(*args, **kwargs)
    return wrapper


# Base Containers -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
class ViewVBaseArrayContainer(ABC, Mapping[str, D_V]):
    """
    A base abstract class for views of VBaseArrayContainers.
    This class is used to create views on VLayerArrayContainer, VAxisArrays and VPairwiseArrays.
    """

    def __init__(self, array_container: VBaseArrayContainer):
        """
        Args:
            array_container: a VBaseArrayContainer object to build a view on.
        """
        generalLogger.debug(f"== Creating {self.__class__.__name__}. ================================")

        self._array_container = array_container

    @_check_parent_has_not_changed
    def __repr__(self) -> str:
        """
        Description for this view  to print.
        :return: a description of this view.
        """
        return f"View of {self._array_container}"

    @abstractmethod
    def __getitem__(self, item: str) -> D_V:
        """
        Get a specific data item stored in this view.
        :param item: key in _data linked to a data item.
        :return: data item stored in _data under the given key.
        """
        pass

    @abstractmethod
    def __setitem__(self, key: str, value: D_V) -> None:
        """
        Set a specific data item in this view. The given data item must have the correct shape.
        :param key: key for storing a data item in this view.
        :param value: a data item to store.
        """
        pass

    @_check_parent_has_not_changed
    def __len__(self) -> int:
        """
        Length of this view : the number of data items in the VBaseArrayContainer.
        :return: number of data items in the VBaseArrayContainer.
        """
        return len(self.keys())

    @_check_parent_has_not_changed
    def __iter__(self) -> Iterator[str]:
        """
        Iterate on this view's keys.
        :return: an iterator over this view's keys.
        """
        return iter(self.keys())

    @property
    @abstractmethod
    def empty(self) -> bool:
        """
        Whether this view is empty or not.
        :return: is this view empty ?
        """
        pass

    @property
    def name(self) -> str:
        """
        Name for this view.
        :return: the name of this view.
        """
        return f"{self._array_container.name}_view"

    @property
    @abstractmethod
    def shape(self) -> Union[
        tuple[int, int, int],
        tuple[int, int, list[int]],
        tuple[int, int, list[int], int],
        tuple[int, int, list[int], list[int]]
    ]:
        """
        The shape of this view is computed from the shape of the Arrays it contains.
        See __len__ for getting the number of Arrays it contains.
        :return: the shape of this view.
        """
        pass

    @property
    @abstractmethod
    def data(self) -> dict[str, D_V]:
        """
        Data of this view.
        :return: the data of this view.
        """
        pass

    @_check_parent_has_not_changed
    def keys(self) -> KeysView[str]:
        """
        KeysView of keys for getting the data items in this view.
        :return: KeysView of this view.
        """
        return self._array_container.keys()

    @_check_parent_has_not_changed
    def values(self) -> ValuesView[D_V]:
        """
        ValuesView of data items in this view.
        :return: ValuesView of this view.
        """
        return self.data.values()

    @_check_parent_has_not_changed
    def items(self) -> ItemsView[str, D_V]:
        """
        ItemsView of pairs of keys and data items in this view.
        :return: ItemsView of this view.
        """
        return self.data.items()

    @abstractmethod
    def dict_copy(self) -> dict[str, D_V]:
        """
        Dictionary of keys and data items in this view.
        :return: Dictionary of this view.
        """
        pass

    @abstractmethod
    def to_csv(self, directory: Path, sep: str = ",", na_rep: str = "",
               index: bool = True, header: bool = True, spacer: str = '') -> None:
        """
        Save this view in CSV file format.
        :param directory: path to a directory for saving the Array
        :param sep: delimiter character
        :param na_rep: string to replace NAs
        :param index: write row names ?
        :param header: Write col names ?
        :param spacer: for logging purposes, the recursion depth of calls to a read_h5 function.
        """
        pass


# 3D Containers -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -
class ViewVTDFArrayContainer(ViewVBaseArrayContainer, Mapping[str, D_VTDF]):
    """
    Base abstract class for views of ArrayContainers that contain TemporalDataFrames (layers and obsm).
    It is based on VBaseArrayContainer.
    """

    def __init__(self,
                 array_container: VBaseArrayContainer,
                 timepoints_slicer: np.ndarray,
                 obs_slicer: np.ndarray,
                 var_slicer: Union[np.ndarray, slice]):
        """
        Args:
            array_container: a VBaseArrayContainer object to build a view on.
            obs_slicer: the list of observations to view.
            var_slicer: the list of variables to view.
            timepoints_slicer: the list of time points to view.
        """
        super().__init__(array_container)

        self._parent_timepoints_hash = hash(tuple(self._array_container._parent.timepoints.value.values))
        self._parent_obs_hash = hash(tuple(self._array_container._parent.obs.index))

    @_check_parent_has_not_changed
    def __getitem__(self, key: str) -> D_VTDF:
        """
        Get a specific data item stored in this view.

        Args:
            key: key in _data linked to a data item.

        Returns:
            The data item stored in _data under the given key.
        """
        return self._data[key]

    @_check_parent_has_not_changed
    def __setitem__(self, key: str, value: Any) -> None:
        """
        Set a specific data item in this view. The given data item must have the correct shape.

        Args:
            key: key for storing a data item in this view.
            value: a data item to store.
        """
        self._data[key] = value

    @property                                                                           # type: ignore
    @_check_parent_has_not_changed
    def empty(self) -> bool:
        """
        Whether this view is empty or not.

        Returns:
            Is this view empty ?
        """
        return all([VTDF.empty for VTDF in self.values()])

    @property                                                                           # type: ignore
    @_check_parent_has_not_changed
    def has_repeating_index(self) -> bool:
        if self.empty:
            return False

        return list(self.values())[0].has_repeating_index

    @property                                                                           # type: ignore
    @_check_parent_has_not_changed
    def shape(self) -> tuple[int, int, list[int], int]:
        """
        The shape of this view is computed from the shape of the Arrays it contains.
        See __len__ for getting the number of Arrays it contains.

        Returns:
            The shape of this view.
        """
        if len(self):
            _first_VTDF = list(self.values())[0]
            _shape_VTDF = _first_VTDF.shape
            return len(self), _shape_VTDF[0], _shape_VTDF[1], _shape_VTDF[2]

        else:
            return 0, 0, [], 0

    @property                                                                           # type: ignore
    @_check_parent_has_not_changed
    def data(self) -> dict[str, D_VTDF]:
        """
        Data of this view.

        Returns:
            The data of this view.
        """
        return self._data

    @_check_parent_has_not_changed
    def dict_copy(self) -> dict[str, 'TemporalDataFrame']:
        """
        Dictionary of keys and data items in this view.

        Returns:
            Dictionary of this view.
        """
        return {key: VTDF.copy() for key, VTDF in self.items()}

    @_check_parent_has_not_changed
    def to_csv(self,
               directory: Path,
               sep: str = ",",
               na_rep: str = "",
               index: bool = True,
               header: bool = True,
               spacer: str = '') -> None:
        """
        Save this view in CSV file format.

        Args:
            directory: path to a directory for saving the Array
            sep: delimiter character
            na_rep: string to replace NAs
            index: write row names ?
            header: Write col names ?
            spacer: for logging purposes, the recursion depth of calls to a read_h5 function.
        """
        # create sub directory for storing arrays
        os.makedirs(directory / self.name)

        for VTDF_name, VTDF in self.items():
            generalLogger.info(f"{spacer}Saving {VTDF_name}")

            # save view of TemporalDataFrame
            VTDF.to_csv(f"{directory / self.name / VTDF_name}.csv", sep, na_rep, index=index, header=header)


class ViewVLayerArrayContainer(ViewVTDFArrayContainer):
    """TODO"""

    def __init__(self,
                 array_container: VBaseArrayContainer,
                 timepoints_slicer: np.ndarray,
                 obs_slicer: np.ndarray,
                 var_slicer: Union[np.ndarray, slice]):
        """
        Args:
            array_container: a VBaseArrayContainer object to build a view on.
            obs_slicer: the list of observations to view.
            var_slicer: the list of variables to view.
            timepoints_slicer: the list of time points to view.
        """
        super().__init__(array_container, timepoints_slicer, obs_slicer, var_slicer)

        if len(array_container):
            # get slicers for each axis only once
            # TODO : fix this : does not work if some layers have num values and others have str values for the same
            #  column name
            # index_slicer, column_num_slicer, column_str_slicer, _ = \
            #     parse_slicer(list(array_container.values())[0], (timepoints_slicer, obs_slicer, var_slicer))

            # then create view directly
            self._data = {key: TemporalDataFrameView(TDF, *parse_slicer(TDF,
                                                                        (timepoints_slicer, obs_slicer, var_slicer)))
                          for key, TDF in array_container.items()}

        else:
            self._data = {}

        self._parent_var_hash = hash(tuple(self._array_container._parent.var.index))


class ViewVObsmArrayContainer(ViewVTDFArrayContainer):
    """TODO"""

    def __init__(self,
                 array_container: VBaseArrayContainer,
                 timepoints_slicer: np.ndarray,
                 obs_slicer: np.ndarray,
                 var_slicer: Union[np.ndarray, slice]):
        """
        Args:
            array_container: a VBaseArrayContainer object to build a view on.
            obs_slicer: the list of observations to view.
            var_slicer: the list of variables to view.
            timepoints_slicer: the list of time points to view.
        """
        super().__init__(array_container, timepoints_slicer, obs_slicer, var_slicer)

        self._data = {}

        for key, TDF in array_container.items():
            # get slicers for each axis only once
            index_slicer, column_num_slicer, column_str_slicer, _ = \
                parse_slicer(TDF, (timepoints_slicer, obs_slicer, var_slicer))

            # then create view directly
            self._data[key] = TemporalDataFrameView(TDF, index_slicer, column_num_slicer, column_str_slicer)


# Obsp Containers -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
class ViewVObspArrayContainer(ViewVBaseArrayContainer, Mapping[str, D_VDF]):
    """
    Class for views of obsp.
    """

    def __init__(self,
                 array_container: VBaseArrayContainer,
                 obs_slicer: np.ndarray):
        """
        Args:
            array_container: a VBaseArrayContainer object to build a view on.
            obs_slicer: the list of observations to view.
        """
        super().__init__(array_container)

        self._obs_slicer = obs_slicer

    def __getitem__(self,
                    item: str) -> D_VDF:
        """
        Get a specific data item stored in this view.

        Args:
            item: key in _data linked to a data item.

        Returns:
            Data item stored in _data under the given key.
        """
        return self._array_container[item].loc[self._obs_slicer, self._obs_slicer]

    def __setitem__(self, key: str, value: VDataFrame) -> None:
        """
        Set a specific data item in this view. The given data item must have the correct shape.

        Args:
            key: key for storing a data item in this view.
            value: a data item to store.
        """
        if not isinstance(value, (pd.DataFrame, VDataFrame)):
            raise VTypeError(f"Value at key '{key}' should be a pandas DataFrame or a VDataFrame.")

        _index = self._array_container[key].loc[self._obs_slicer, self._obs_slicer].index

        if not value.shape == (len(_index), len(_index)):
            raise ShapeError(f"DataFrame at key '{key}' should have shape ({len(_index)}, {len(_index)}).")

        if not value.index.equals(_index):
            raise VValueError(f"Index of DataFrame at key '{key}' does not match previous index.")

        if not value.columns.equals(_index):
            raise VValueError(f"Column names of DataFrame at key '{key}' do not match previous names.")

        self._array_container[key].loc[self._obs_slicer, self._obs_slicer] = value

    @property
    def empty(self) -> bool:
        """
        Whether this view is empty or not.

        Returns:
            Is this view empty ?
        """
        if not len(self) or all([vdf.empty for vdf in self.values()]):
            return True
        return False

    @property
    def shape(self) -> tuple[int, int, int]:
        """
        The shape of this view is computed from the shape of the Arrays it contains.
        See __len__ for getting the number of Arrays it contains.

        Returns:
            The shape of this view.
        """
        len_index = len(self._obs_slicer)

        if len(self):
            return len(self), len_index, len_index

        else:
            return 0, len_index, len_index

    @property
    def data(self) -> dict[str, D_VDF]:
        """
        Data of this view.

        Returns:
            The data of this view.
        """
        return {key: vdf.loc[self._obs_slicer, self._obs_slicer] for key, vdf in self._array_container.items()}

    def dict_copy(self) -> dict[str, VDataFrame]:
        """
        Dictionary of keys and data items in this view.

        Returns:
            Dictionary of this view.
        """
        return {key: vdf.loc[self._obs_slicer, self._obs_slicer].copy() for key, vdf in self.items()}

    def to_csv(self,
               directory: Path,
               sep: str = ",",
               na_rep: str = "",
               index: bool = True,
               header: bool = True,
               spacer: str = '') -> None:
        """
        Save this view in CSV file format.

        Args:
            directory: path to a directory for saving the Array
            sep: delimiter character
            na_rep: string to replace NAs
            index: write row names ?
            header: Write col names ?
            spacer: for logging purposes, the recursion depth of calls to a read_h5 function.
        """
        # create sub directory for storing sets
        os.makedirs(directory / self.name)

        for vdf_name, vdf in self.data:
            generalLogger.info(f"{spacer}Saving {vdf_name}")

            # save array
            vdf.to_csv(f"{directory / self.name / vdf_name}.csv", sep, na_rep, index=index, header=header)

    def set_index(self, values: Collection) -> None:
        """
        Set a new index for rows and columns.

        Args:
            values: collection of new index values.
        """
        for vdf_name, vdf in self.data:
            vdf.lock = (False, False)
            vdf.index = values
            vdf.columns = values
            vdf.lock = (True, True)


# 2D Containers -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -
class ViewVBase2DArrayContainer(ViewVBaseArrayContainer, ABC, Mapping[str, D_VDF]):
    """
    Base abstract class for views of ArrayContainers that contain DataFrames (varm and varp)
    It is based on VBaseArrayContainer.
    """

    def __init__(self, array_container: VBaseArrayContainer, var_slicer: np.ndarray):
        """
        :param array_container: a VBaseArrayContainer object to build a view on.
        :param var_slicer: the list of variables to view.
        """
        super().__init__(array_container)

        self._var_slicer = var_slicer

    def __getitem__(self, item: str) -> D_VDF:
        """
        Get a specific data item stored in this view.
        :param item: key in _data linked to a data item.
        :return: data item stored in _data under the given key.
        """
        return self._array_container[item].loc[self._var_slicer]

    def values(self) -> ValuesView[pd.DataFrame]:
        """
        ValuesView of data items in this view.
        :return: ValuesView of this view.
        """
        return super().values()

    @property
    def empty(self) -> bool:
        """
        Whether this view is empty or not.
        :return: is this view empty ?
        """
        return all([DF.empty for DF in self.values()])

    def dict_copy(self) -> dict[str, D_VDF]:
        """
        Dictionary of keys and data items in this view.
        :return: Dictionary of this view.
        """
        return {k: v.copy() for k, v in self.items()}

    def to_csv(self, directory: Path, sep: str = ",", na_rep: str = "", index: bool = True, header: bool = True,
               spacer: str = '') -> None:
        """
        Save this view in CSV file format.
        :param directory: path to a directory for saving the Array
        :param sep: delimiter character
        :param na_rep: string to replace NAs
        :param index: write row names ?
        :param header: Write col names ?
        :param spacer: for logging purposes, the recursion depth of calls to a read_h5 function.
        """
        # create sub directory for storing arrays
        os.makedirs(directory / self.name)

        for DF_name, DF in self.items():
            generalLogger.info(f"{spacer}Saving {DF_name}")

            # save array
            DF.to_csv(f"{directory / self.name / DF_name}.csv", sep, na_rep, index=index, header=header)


class ViewVVarmArrayContainer(ViewVBase2DArrayContainer):

    def __setitem__(self, key: str, value: pd.DataFrame) -> None:
        """
        Set a specific data item in this view. The given data item must have the correct shape.
        :param key: key for storing a data item in this view.
        :param value: a data item to store.
        """
        if not isinstance(value, pd.DataFrame):
            raise VTypeError(f"Cannot set varm view '{key}' from non pandas DataFrame object.")

        if not self.shape[1] == value.shape[0]:
            raise ShapeError(f"Cannot set varm '{key}' because of shape mismatch.")

        if not pd.Index(self._var_slicer).equals(value.index):
            raise VValueError("Index does not match.")

        self[key] = value

    @property
    def shape(self) -> tuple[int, int, list[int]]:
        """
        The shape of this view is computed from the shape of the Arrays it contains.
        See __len__ for getting the number of Arrays it contains.
        :return: the shape of this view.
        """
        if len(self):
            _first_DF = list(self.values())[0]
            return len(self), _first_DF.shape[0], [DF.shape[1] for DF in self.values()]

        else:
            return 0, 0, []

    @property
    def data(self) -> dict[str, D_VDF]:
        """
        Data of this view.
        :return: the data of this view.
        """
        return {key: DF.loc[self._var_slicer] for key, DF in self._array_container.items()}


class ViewVVarpArrayContainer(ViewVBase2DArrayContainer):

    def __getitem__(self, item: str) -> D_VDF:
        """
        Get a specific data item stored in this view.
        :param item: key in _data linked to a data item.
        :return: data item stored in _data under the given key.
        """
        return self._array_container[item].loc[self._var_slicer, self._var_slicer]

    def __setitem__(self, key: str, value: pd.DataFrame) -> None:
        """
        Set a specific data item in this view. The given data item must have the correct shape.
        :param key: key for storing a data item in this view.
        :param value: a data item to store.
        """
        if not isinstance(value, pd.DataFrame):
            raise VTypeError(f"Cannot set varp view '{key}' from non pandas DataFrame object.")

        if not self.shape[1:] == value.shape:
            raise ShapeError(f"Cannot set varp '{key}' because of shape mismatch.")

        _index = pd.Index(self._var_slicer)

        if not _index.equals(value.index):
            raise VValueError("Index does not match.")

        if not _index.equals(value.columns):
            raise VValueError("column names do not match.")

        self[key] = value

    @property
    def shape(self) -> tuple[int, int, int]:
        """
        The shape of this view is computed from the shape of the Arrays it contains.
        See __len__ for getting the number of Arrays it contains.
        :return: the shape of this view.
        """
        if len(self):
            _first_DF = list(self.values())[0]
            return len(self), _first_DF.shape[0], _first_DF.shape[1]

        else:
            return 0, 0, 0

    @property
    def data(self) -> dict[str, D_VDF]:
        """
        Data of this view.
        :return: the data of this view.
        """
        return {key: DF.loc[self._var_slicer, self._var_slicer] for key, DF in self._array_container.items()}
