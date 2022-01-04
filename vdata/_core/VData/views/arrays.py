# coding: utf-8
# Created on 15/01/2021 12:57
# Author : matteo

# ====================================================
# imports
import os
import numpy as np
import pandas as pd
import abc
from pathlib import Path
from typing import Union, KeysView, ValuesView, ItemsView, Iterator, Mapping, TypeVar, Collection, Any

from ..arrays import VBaseArrayContainer
from ...TDF import TemporalDataFrame, ViewTemporalDataFrame
from vdata.time_point import TimePoint
from ....IO import generalLogger, VTypeError, VValueError, ShapeError


# ====================================================
# code

D_V = TypeVar('D_V', ViewTemporalDataFrame, pd.DataFrame, dict['TimePoint', pd.DataFrame])
D_VTDF = TypeVar('D_VTDF', bound=ViewTemporalDataFrame)
D_VDF = TypeVar('D_VDF', bound=pd.DataFrame)


# Base Containers -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
class ViewVBaseArrayContainer(abc.ABC, Mapping[str, D_V]):
    """
    A base abstract class for views of VBaseArrayContainers.
    This class is used to create views on VLayerArrayContainer, VAxisArrays and VPairwiseArrays.
    """

    def __init__(self, array_container: VBaseArrayContainer):
        """
        :param array_container: a VBaseArrayContainer object to build a view on.
        """
        generalLogger.debug(f"== Creating {self.__class__.__name__}. ================================")

        self._array_container = array_container

    def __repr__(self) -> str:
        """
        Description for this view  to print.
        :return: a description of this view.
        """
        return f"View of {self._array_container}"

    @abc.abstractmethod
    def __getitem__(self, item: str) -> D_V:
        """
        Get a specific data item stored in this view.
        :param item: key in _data linked to a data item.
        :return: data item stored in _data under the given key.
        """
        pass

    @abc.abstractmethod
    def __setitem__(self, key: str, value: D_V) -> None:
        """
        Set a specific data item in this view. The given data item must have the correct shape.
        :param key: key for storing a data item in this view.
        :param value: a data item to store.
        """
        pass

    def __len__(self) -> int:
        """
        Length of this view : the number of data items in the VBaseArrayContainer.
        :return: number of data items in the VBaseArrayContainer.
        """
        return len(self.keys())

    def __iter__(self) -> Iterator[str]:
        """
        Iterate on this view's keys.
        :return: an iterator over this view's keys.
        """
        return iter(self.keys())

    @property
    @abc.abstractmethod
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
    @abc.abstractmethod
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
    @abc.abstractmethod
    def data(self) -> dict[str, D_V]:
        """
        Data of this view.
        :return: the data of this view.
        """
        pass

    def keys(self) -> KeysView[str]:
        """
        KeysView of keys for getting the data items in this view.
        :return: KeysView of this view.
        """
        return self._array_container.keys()

    def values(self) -> ValuesView[D_V]:
        """
        ValuesView of data items in this view.
        :return: ValuesView of this view.
        """
        return self.data.values()

    def items(self) -> ItemsView[str, D_V]:
        """
        ItemsView of pairs of keys and data items in this view.
        :return: ItemsView of this view.
        """
        return self.data.items()

    @abc.abstractmethod
    def dict_copy(self) -> dict[str, D_V]:
        """
        Dictionary of keys and data items in this view.
        :return: Dictionary of this view.
        """
        pass

    @abc.abstractmethod
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

    def __init__(self, array_container: VBaseArrayContainer, time_points_slicer: np.ndarray,
                 obs_slicer: np.ndarray, var_slicer: np.ndarray):
        """
        :param array_container: a VBaseArrayContainer object to build a view on.
        :param obs_slicer: the list of observations to view.
        :param var_slicer: the list of variables to view.
        :param time_points_slicer: the list of time points to view.
        """
        super().__init__(array_container)

        self._time_points_slicer = time_points_slicer
        self._obs_slicer = obs_slicer
        self._var_slicer = var_slicer

    def __getitem__(self, item: str) -> D_VTDF:
        """
        Get a specific data item stored in this view.
        :param item: key in _data linked to a data item.
        :return: data item stored in _data under the given key.
        """
        return self._array_container[item][self._time_points_slicer, self._obs_slicer, self._var_slicer]

    def __setitem__(self, key: str, value: Any) -> None:
        """
        Set a specific data item in this view. The given data item must have the correct shape.
        :param key: key for storing a data item in this view.
        :param value: a data item to store.
        """
        self[key] = value

    @property
    def empty(self) -> bool:
        """
        Whether this view is empty or not.
        :return: is this view empty ?
        """
        return all([VTDF.empty for VTDF in self.values()])

    @property
    def shape(self) -> tuple[int, int, list[int], int]:
        """
        The shape of this view is computed from the shape of the Arrays it contains.
        See __len__ for getting the number of Arrays it contains.
        :return: the shape of this view.
        """
        if len(self):
            _first_VTDF = list(self.values())[0]
            _shape_VTDF = _first_VTDF.shape
            return len(self), _shape_VTDF[0], _shape_VTDF[1], _shape_VTDF[2]

        else:
            return 0, 0, [], 0

    @property
    def data(self) -> dict[str, D_VTDF]:
        """
        Data of this view.
        :return: the data of this view.
        """
        return {key: TDF[self._time_points_slicer, self._obs_slicer, self._var_slicer]
                for key, TDF in self._array_container.items()}

    def dict_copy(self) -> dict[str, 'TemporalDataFrame']:
        """
        Dictionary of keys and data items in this view.
        :return: Dictionary of this view.
        """
        return {key: VTDF.copy() for key, VTDF in self.items()}

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

        for VTDF_name, VTDF in self.items():
            generalLogger.info(f"{spacer}Saving {VTDF_name}")

            # save view of TemporalDataFrame
            VTDF.to_csv(f"{directory / self.name / VTDF_name}.csv", sep, na_rep, index=index, header=header)


# Obsp Containers -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
class ViewVObspArrayContainer(ViewVBaseArrayContainer, Mapping[str, Mapping['TimePoint', D_VDF]]):
    """
    Class for views of obsp.
    """

    def __init__(self, array_container: VBaseArrayContainer, time_points_slicer: np.ndarray,
                 obs_slicer: np.ndarray):
        """
        :param array_container: a VBaseArrayContainer object to build a view on.
        :param obs_slicer: the list of observations to view.
        :param time_points_slicer: the list of time points to view.
        """
        super().__init__(array_container)

        self._time_points_slicer = time_points_slicer
        self._obs_slicer = obs_slicer

    def __getitem__(self, item: str) -> Mapping['TimePoint', D_VDF]:
        """
        Get a specific data item stored in this view.
        :param item: key in _data linked to a data item.
        :return: data item stored in _data under the given key.
        """
        return {tp: DF.loc[self._obs_slicer, self._obs_slicer]
                for tp, DF in self._array_container[item] if tp in self._time_points_slicer}

    def __setitem__(self, key: str, value: Mapping['TimePoint', pd.DataFrame]) -> None:
        """
        Set a specific data item in this view. The given data item must have the correct shape.
        :param key: key for storing a data item in this view.
        :param value: a data item to store.
        """
        if not isinstance(value, dict):
            raise VTypeError(f"Cannot set obsp view '{key}' from non dict object.")

        if not all([isinstance(tp, TimePoint) and tp in self._time_points_slicer
                    for tp in value.keys()]):
            raise VValueError("Time points do not match.")

        for tp, DF in value.items():
            if not isinstance(DF, pd.DataFrame):
                raise VTypeError(f"Value at time point '{tp}' should be a pandas DataFrame.")

            _index = self._array_container[key][tp].loc[self._obs_slicer]

            if not DF.shape == (len(_index), len(_index)):
                raise ShapeError(f"DataFrame at time point '{tp}' should have shape ({len(_index)}, {len(_index)}).")

            if not DF.index.equals(_index):
                raise VValueError(f"Index of DataFrame at time point '{tp}' does not match.")

            if not DF.columns.equals(_index):
                raise VValueError(f"Column names of DataFrame at time point '{tp}' do not match.")

            self._array_container[key][tp].loc[self._obs_slicer, self._obs_slicer] = DF

    @property
    def empty(self) -> bool:
        """
        Whether this view is empty or not.
        :return: is this view empty ?
        """
        if not len(self) or all([not len(_set) for _set in self.values()]) \
                or all([_set[tp].empty for _set in self.values() for tp in self._time_points_slicer]):
            return True
        return False

    @property
    def shape(self) -> tuple[int, int, list[int], list[int]]:
        """
        The shape of this view is computed from the shape of the Arrays it contains.
        See __len__ for getting the number of Arrays it contains.
        :return: the shape of this view.
        """
        if len(self):
            _first_set: dict['TimePoint', pd.DataFrame] = list(self.values())[0]
            len_index = [len(DF.loc[self._obs_slicer, self._obs_slicer].index) for DF in _first_set.values()]
            return len(self), len(self._time_points_slicer), len_index, len_index

        else:
            return 0, 0, [], []

    @property
    def data(self) -> dict[str, Mapping['TimePoint', D_VDF]]:
        """
        Data of this view.
        :return: the data of this view.
        """
        return {key: {tp: DF.loc[self._obs_slicer, self._obs_slicer]
                      for tp, DF in _set if tp in self._time_points_slicer}
                for key, _set in self.items()}

    def dict_copy(self) -> dict[str, Mapping['TimePoint', pd.DataFrame]]:
        """
        Dictionary of keys and data items in this view.
        :return: Dictionary of this view.
        """
        return {key: {TimePoint(tp): DF.copy() for tp, DF in _set.items()} for key, _set in self.items()}

    def compact(self) -> dict[str, pd.DataFrame]:
        """
        Transform this VObspArrayContainer into a dictionary of large square DataFrame per all TimePoints.

        :return: a dictionary of str:large concatenated square DataFrame.
        """
        _compact_obsp = {key: pd.DataFrame(index=self._obs_slicer,
                                           columns=self._obs_slicer) for key in self.keys()}

        for key in self.keys():
            index_cumul = 0

            for arr in self[key]:
                _compact_obsp[key].iloc[index_cumul:index_cumul + len(arr), index_cumul:index_cumul + len(arr)] = arr
                index_cumul += len(arr)

        return _compact_obsp

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
        # create sub directory for storing sets
        os.makedirs(directory / self.name)

        for set_name, set_dict in self.items():
            # create sub directory for storing arrays
            os.makedirs(directory / self.name / set_name)

            for arr_name, arr in set_dict.items():
                generalLogger.info(f"{spacer}Saving {set_name}:{arr_name}")

                # save array
                arr.to_csv(f"{directory / self.name / arr_name}.csv", sep, na_rep, index=index, header=header)

    def set_index(self, values: Collection) -> None:
        """
        Set a new index for rows and columns.
        :param values: collection of new index values.
        """
        for set_name in self.keys():

            index_cumul = 0
            for arr in self[set_name].values():

                arr.lock = (False, False)
                arr.index = values[index_cumul:index_cumul + len(arr)]
                arr.columns = values[index_cumul:index_cumul + len(arr)]
                arr.lock = (True, True)

                index_cumul += len(arr)


# 2D Containers -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -
class ViewVBase2DArrayContainer(ViewVBaseArrayContainer, abc.ABC, Mapping[str, D_VDF]):
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
