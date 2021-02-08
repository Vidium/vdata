# coding: utf-8
# Created on 11/4/20 10:40 AM
# Author : matteo

# ====================================================
# imports
import os
import abc
import pandas as pd
from abc import ABC
from pathlib import Path
from typing import Optional, Union, Dict, Tuple, KeysView, ValuesView, ItemsView, Any, Mapping, Iterator, TypeVar, List
from typing_extensions import Literal

from vdata.NameUtils import DType, DataFrame
from . import vdata
from .._TDF.dataframe import TemporalDataFrame
from .._IO import generalLogger, IncoherenceError, VAttributeError, ShapeError


# ====================================================
# code

D = TypeVar('D', DataFrame, Dict['vdata.TimePoint', pd.DataFrame])
D_DF = TypeVar('D_DF', bound=pd.DataFrame)
D_TDF = TypeVar('D_TDF', bound=TemporalDataFrame)


# Containers ------------------------------------------------------------------
# Base Containers -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
class VBaseArrayContainer(ABC, Mapping[str, D]):
    """
    Base abstract class for ArrayContainers linked to a VData object (obsm, obsp, varm, varp, layers).
    All Arrays have a '_parent' attribute for linking them to a VData and a '_data' dictionary
    attribute for storing 2D/3D arrays.
    """

    def __init__(self, parent: 'vdata.VData', data: Optional[Dict[str, D]]):
        """
        :param parent: the parent VData object this ArrayContainer is linked to.
        :param data: a dictionary of data items (pandas DataFrames, TemporalDataFrames or dictionaries of pandas
        DataFrames) to store in this ArrayContainer.
        """
        generalLogger.debug(f"== Creating {self.__class__.__name__}. ==========================")

        self._parent = parent
        self._data = self._check_init_data(data)

    @abc.abstractmethod
    def _check_init_data(self, data: Optional[Dict[str, D]]) -> Dict[str, D]:
        """
        Function for checking, at ArrayContainer creation, that the supplied data has the correct format.
        :param data: optional dictionary of data items.
        :return: the data, if correct.
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
        :param item: key in _data linked to a data item.
        :return: data item stored in _data under the given key.
        """
        if len(self) and item in self.keys():
            return self._data[item]

        else:
            raise VAttributeError(f"{self.name} ArrayContainer has no attribute '{item}'")

    @abc.abstractmethod
    def __setitem__(self, key: str, value: D) -> None:
        """
        Set a specific data item in _data. The given data item must have the correct shape.
        :param key: key for storing a data item in this ArrayContainer.
        :param value: a data item to store.
        """
        pass

    def __len__(self) -> int:
        """
        Length of this ArrayContainer : the number of data items in _data.
        :return: number of data items in _data.
        """
        return len(self.keys())

    def __iter__(self) -> Iterator[str]:
        """
        Iterate on this ArrayContainer's keys.
        :return: an iterator over this ArrayContainer's keys.
        """
        return iter(self.keys())

    @property
    @abc.abstractmethod
    def empty(self) -> bool:
        """
        Whether this ArrayContainer is empty or not.
        :return: is this ArrayContainer empty ?
        """
        pass

    @abc.abstractmethod
    def update_dtype(self, type_: DType) -> None:
        """
        Update the data type of Arrays stored in this ArrayContainer.
        :param type_: the new data type.
        """
        pass

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """
        Name for this ArrayContainer.
        :return: the name of this ArrayContainer.
        """
        pass

    @property
    @abc.abstractmethod
    def shape(self) -> Union[
        Tuple[int, int, int],
        Tuple[int, int, List[int]],
        Tuple[int, int, List[int], int],
        Tuple[int, int, List[int], List[int]]
    ]:
        """
        The shape of this ArrayContainer is computed from the shape of the Arrays it contains.
        See __len__ for getting the number of Arrays it contains.
        :return: the shape of this ArrayContainer.
        """
        pass

    @property
    def data(self) -> Dict[str, D]:
        """
        Data of this ArrayContainer.
        :return: the data of this ArrayContainer.
        """
        return self._data

    def keys(self) -> KeysView[str]:
        """
        KeysView of keys for getting the data items in this ArrayContainer.
        :return: KeysView of this ArrayContainer.
        """
        return self._data.keys()

    def values(self) -> ValuesView[D]:
        """
        ValuesView of data items in this ArrayContainer.
        :return: ValuesView of this ArrayContainer.
        """
        return self._data.values()

    def items(self) -> ItemsView[str, D]:
        """
        ItemsView of pairs of keys and data items in this ArrayContainer.
        :return: ItemsView of this ArrayContainer.
        """
        return self._data.items()

    @abc.abstractmethod
    def to_csv(self, directory: Path, sep: str = ",", na_rep: str = "",
               index: bool = True, header: bool = True, spacer: str = '') -> None:
        """
        Save this ArrayContainer in CSV file format.
        :param directory: path to a directory for saving the Array
        :param sep: delimiter character
        :param na_rep: string to replace NAs
        :param index: write row names ?
        :param header: Write col names ?
        :param spacer: for logging purposes, the recursion depth of calls to a read_h5 function.
        """
        pass


# 3D Containers -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -
class VBase3DArrayContainer(VBaseArrayContainer, ABC, Mapping[str, D_TDF]):
    """
    Base abstract class for ArrayContainers linked to a VData object that contain TemporalDataFrames (obsm and layers)
    It is based on VBaseArrayContainer and defines some functions shared by obsm and layers.
    """

    def __init__(self, parent: 'vdata.VData', data: Optional[Dict[str, D_TDF]]):
        """
        :param parent: the parent VData object this ArrayContainer is linked to.
        :param data: a dictionary of TemporalDataFrames in this ArrayContainer.
        """

        super().__init__(parent, data)

    @property
    def empty(self) -> bool:
        """
        Whether this ArrayContainer is empty or not.
        :return: is this ArrayContainer empty ?
        """
        return all([TDF.empty for TDF in self.values()])

    def update_dtype(self, type_: DType) -> None:
        """
        Update the data type of TemporalDataFrames stored in this ArrayContainer.
        :param type_: the new data type.
        """
        for arr in self.values():
            arr.astype(type_)

    @property
    def shape(self) -> Tuple[int, int, List[int], int]:
        """
        The shape of this ArrayContainer is computed from the shape of the TemporalDataFrames it contains.
        See __len__ for getting the number of TemporalDataFrames it contains.
        :return: the shape of this ArrayContainer.
        """
        _first_TDF = self[list(self.keys())[0]]
        _shape_TDF = _first_TDF.shape
        return len(self), _shape_TDF[0], _shape_TDF[1], _shape_TDF[2]

    def to_csv(self, directory: Path, sep: str = ",", na_rep: str = "",
               index: bool = True, header: bool = True, spacer: str = '') -> None:
        """
        Save the ArrayContainer in CSV file format.
        :param directory: path to a directory for saving the Array
        :param sep: delimiter character
        :param na_rep: string to replace NAs
        :param index: write row names ?
        :param header: Write col names ?
        :param spacer: for logging purposes, the recursion depth of calls to a read_h5 function.
        """
        # create sub directory for storing arrays
        os.makedirs(directory / self.name)

        for arr_name, arr in self.items():
            generalLogger.info(f"{spacer}Saving {arr_name}")

            # save array
            arr.to_csv(f"{directory / self.name / arr_name}.csv", sep, na_rep, index=index, header=header)


class VLayerArrayContainer(VBase3DArrayContainer):
    """
    Class for layers.
    This object contains any number of TemporalDataFrames, with shapes (n_time_points, n_obs, n_var).
    The arrays-like objects can be accessed from the parent VData object by :
        VData.layers[<array_name>]
    """

    def __init__(self, parent: 'vdata.VData', data: Optional[Dict[str, D_TDF]]):
        """
        :param parent: the parent VData object this VLayerArrayContainer is linked to.
        :param data: a dictionary of TemporalDataFrames in this VLayerArrayContainer.
        """
        super().__init__(parent, data)

    def _check_init_data(self, data: Optional[Dict[Any, D_TDF]]) \
            -> Dict[str, D_TDF]:
        """
        Function for checking, at VLayerArrayContainer creation, that the supplied data has the correct format :
            - the shape of the TemporalDataFrames in 'data' match the parent VData object's shape.
            - the index of the TemporalDataFrames in 'data' match the index of the parent VData's obs TemporalDataFrame.
            - the column names of the TemporalDataFrames in 'data' match the index of the parent VData's var DataFrame.
            - the time points of the TemporalDataFrames in 'data' match the index of the parent VData's time_points
            DataFrame.
        :param data: optional dictionary of TemporalDataFrames.
        :return: the data (dictionary of TemporalDataFrames), if correct.
        """
        if data is None or not len(data):
            generalLogger.debug("  No data was given.")
            return {'data': TemporalDataFrame(index=self._parent.obs.index, columns=self._parent.var.index,
                                              time_list=self._parent.obs.time_points_column,
                                              time_points=self._parent.time_points.value,
                                              name='data')}

        else:
            generalLogger.debug("  Data was found.")
            _data = {}
            _shape = (self._parent.time_points.shape[0], self._parent.obs.shape[1], self._parent.var.shape[0])
            _index = self._parent.obs.index
            _columns = self._parent.var.index
            _time_points = self._parent.time_points.index

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
                        for i in range(len(TDF.time_points)):
                            if _shape[1][i] != TDF_shape[1][i]:
                                raise IncoherenceError(f"Layer '{TDF_index}' at time point {i} has"
                                                       f" {TDF_shape[1][i]} observations, "
                                                       f"should have {_shape[1][i]}.")

                    else:
                        raise IncoherenceError(f"Layer '{TDF_index}' has  {TDF_shape[2]} variables, "
                                               f"should have {_shape[2]}.")

                # check that indexes match
                if not _index.equals(TDF.index):
                    raise IncoherenceError(f"Index of layer '{TDF_index}' ({TDF.index}) does not match obs' index. ("
                                           f"{_index})")

                if not _columns.equals(TDF.columns):
                    raise IncoherenceError(f"Column names of layer '{TDF_index}' ({TDF.columns}) do not match var's "
                                           f"index. ({_columns})")

                if not _time_points.equals(TDF.time_points):
                    raise IncoherenceError(f"Time points of layer '{TDF_index}' ({TDF.time_points}) do not match "
                                           f"time_point's index. ({_time_points})")

                # checks passed, store the TemporalDataFrame
                _data[str(TDF_index)] = TDF

            generalLogger.debug("  Data was OK.")
            return _data

    def __setitem__(self, key: str, value: D_TDF) -> None:
        """
        Set a specific TemporalDataFrame in _data. The given TemporalDataFrame must have the correct shape.
        :param key: key for storing a TemporalDataFrame in this VLayerArrayContainer.
        :param value: a TemporalDataFrame to store.
        """
        # TODO
        raise NotImplementedError

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
    This object contains any number of TemporalDataFrames, with shape (n_time_points, n_obs, any).
    The TemporalDataFrames can be accessed from the parent VData object by :
        VData.obsm[<array_name>])
    """

    def __init__(self, parent: 'vdata.VData', data: Optional[Dict[str, D_TDF]] = None):
        """
        :param parent: the parent VData object this VObsmArrayContainer is linked to.
        :param data: a dictionary of TemporalDataFrames in this VObsmArrayContainer.
        """
        super().__init__(parent, data)

    def _check_init_data(self, data: Optional[Dict[str, D_TDF]]) \
            -> Dict[str, D_TDF]:
        """
        Function for checking, at VObsmArrayContainer creation, that the supplied data has the correct format :
            - the shape of the TemporalDataFrames in 'data' match the parent VData object's shape (except for the
            number of columns).
            - the index of the TemporalDataFrames in 'data' match the index of the parent VData's obs TemporalDataFrame.
            - the time points of the TemporalDataFrames in 'data' match the index of the parent VData's time_points
            DataFrame.
        :param data: optional dictionary of TemporalDataFrames.
        :return: the data (dictionary of TemporalDataFrames), if correct.
        """
        if data is None or not len(data):
            generalLogger.debug("  No data was given.")
            return dict()

        else:
            generalLogger.debug("  Data was found.")
            _data = {}
            _shape = (self._parent.time_points.shape[0],
                      self._parent.obs.shape[1],
                      list(self.data.values())[0].shape[2])
            _index = self._parent.obs.index
            _time_points = self._parent.time_points.index

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
                        for i in range(len(TDF.time_points)):
                            if _shape[1][i] != TDF_shape[1][i]:
                                raise IncoherenceError(f"TemporalDataFrame '{TDF_index}' at time point {i} has"
                                                       f" {TDF_shape[1][i]} rows, "
                                                       f"should have {_shape[1][i]}.")

                    else:
                        raise IncoherenceError(f"TemporalDataFrame '{TDF_index}' has  {TDF_shape[2]} columns, "
                                               f"should have {_shape[2]}.")

                # check that indexes match
                if not _index.equals(TDF.index):
                    raise IncoherenceError(f"Index of TemporalDataFrame '{TDF_index}' ({TDF.index}) does not match "
                                           f"obs' index. ({_index})")

                if not _time_points.equals(TDF.time_points):
                    raise IncoherenceError(f"Time points of TemporalDataFrame '{TDF_index}' ({TDF.time_points}) "
                                           f"do not match time_point's index. ({_time_points})")

                # checks passed, store the TemporalDataFrame
                _data[str(TDF_index)] = TDF

            generalLogger.debug("  Data was OK.")
            return _data

    def __setitem__(self, key: str, value: D_TDF) -> None:
        """
        Set a specific TemporalDataFrame in _data. The given TemporalDataFrame must have the correct shape.
        :param key: key for storing a TemporalDataFrame in this VObsmArrayContainer.
        :param value: a TemporalDataFrame to store.
        """
        # TODO
        raise NotImplementedError

    @property
    def name(self) -> Literal['obsm']:
        """
        Name for this VObsmArrayContainer : obsm.
        :return: name of this VObsmArrayContainer.
        """
        return 'obsm'


# Obsp Containers -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
class VObspArrayContainer(VBaseArrayContainer, Mapping[str, Mapping['vdata.TimePoint', D_DF]]):
    """
    Class for obsp.
    This object contains sets of <nb time points> 2D square DataFrames of shapes (<n_obs>, <n_obs>) for each time point.
    The DataFrames can be accessed from the parent VData object by :
        VData.obsp[<array_name>][<time point>]
    """

    def __init__(self, parent: 'vdata.VData', data: Optional[Dict[str, Dict['vdata.TimePoint', D_DF]]]):
        """
        :param parent: the parent VData object this VObspArrayContainer is linked to.
        :param data: a dictionary of array-like objects to store in this VObspArrayContainer.
        """
        super().__init__(parent, data)

    def _check_init_data(self, data: Optional[Dict[str, Dict['vdata.TimePoint', D_DF]]]) \
            -> Dict[str, Dict['vdata.TimePoint', D_DF]]:
        """
        Function for checking, at VObspArrayContainer creation, that the supplied data has the correct format :
            - the shape of the DataFrames in 'data' match the parent VData object's index length.
            - the index and columns names of the DataFrames in 'data' match the index of the parent VData's obs
            TemporalDataFrame.
            - the time points of the dictionaries of DataFrames in 'data' match the index of the parent VData's
            time_points DataFrame.
        :param data: dictionary of dictionaries (TimePoint: DataFrame (n_obs x n_obs))
        :return: the data (dictionary of dictionaries of DataFrames), if correct.
        """
        if data is None or not len(data):
            generalLogger.debug("  No data was given.")
            return dict()

        else:
            generalLogger.debug("  Data was found.")
            _data = dict()
            _shape = self._parent.obs.shape[1]
            _index = self._parent.obs.index
            _time_points = self._parent.time_points.index

            generalLogger.debug(f"  Reference shape is {_shape}.")

            for DF_dict_index, DF_dict in data.items():
                data_time_points = list(DF_dict.keys())

                _data[DF_dict_index] = dict()

                if not _time_points.equals(data_time_points):
                    raise IncoherenceError(f"Time points of '{DF_dict_index}' ({data_time_points}) do not match "
                                           f"time_point's index. ({_time_points})")

                for time_point_index, (time_point, DF) in enumerate(DF_dict.items()):
                    DF_shape = DF.shape

                    generalLogger.debug(f"  Checking DataFrame at time point '{time_point}' with shape {DF_shape}.")

                    # check that square
                    if DF_shape[0] != DF_shape[1]:
                        raise ShapeError(f"DataFrame at time point '{time_point}' in '{DF_dict_index}' should be "
                                         f"square.")

                    # check that shapes match
                    if DF_shape[0] != _shape[time_point_index]:
                        raise IncoherenceError(f"DataFrame at time point '{time_point}' in '{DF_dict_index}' "
                                               f"has {DF_shape[0]} row{'s' if DF_shape[0] > 1 else ''}"
                                               f" and column{'s' if DF_shape[0] > 1 else ''}, should have"
                                               f" {_shape[time_point_index]}.")

                    # check that indexes match
                    if not _index.equals(DF.index):
                        raise IncoherenceError(f"Index of DataFrame at time point '{time_point}' ({DF.index}) does not "
                                               f"match obs' index. ({_index})")

                    if not _index.equals(DF.columns):
                        raise IncoherenceError(f"Column names of DataFrame at time point '{time_point}' ({DF.columns}) "
                                               f"do not match obs' index. ({_index})")

                    # checks passed, store the TemporalDataFrame
                    _data[DF_dict_index][time_point] = DF

            generalLogger.debug("  Data was OK.")
            return _data

    def __getitem__(self, item: str) -> D_DF:
        """
        Get a specific set of DataFrames stored in this VObspArrayContainer.
        :param item: key in _data linked to a set of DataFrames.
        :return: set of DataFrames stored in _data under the given key.
        """
        return super().__getitem__(item)

    def __setitem__(self, key: str, value: Dict['vdata.TimePoint', D_DF]) -> None:
        """
        Set a specific set of DataFrames in _data. The given set of DataFrames must have the correct shape.
        :param key: key for storing a set of DataFrames in this VObspArrayContainer.
        :param value: a set of DataFrames to store.
        """
        # TODO
        raise NotImplementedError

    @property
    def empty(self) -> bool:
        """
        Whether this VObspArrayContainer is empty or not.
        :return: is this VObspArrayContainer empty ?
        """
        if not len(self) or all([not len(self[set_name]) for set_name in self.keys()]) or \
                all([self[set_name][tp].empty for set_name in self.keys() for tp in self[set_name].keys()]):
            return True
        return False

    def update_dtype(self, type_: DType) -> None:
        """
        Update the data type of Arrays stored in this VObspArrayContainer.
        :param type_: the new data type.
        """
        for set_name in self.keys():
            for tp in self[set_name].keys():
                self[set_name][tp] = self[set_name][tp].astype(type_)

    @property
    def name(self) -> Literal['obsp']:
        """
        Name for this VObspArrayContainer : obsp.
        :return: the name of this VObspArrayContainer.
        """
        return 'obsp'

    @property
    def shape(self) -> Tuple[int, int, List[int], List[int]]:
        """
        The shape of the VObspArrayContainer is computed from the shape of the Arrays it contains.
        See __len__ for getting the number of Arrays it contains.
        :return: shape of this VObspArrayContainer.
        """
        _first_dict = self[list(self.keys())[0]]
        nb_time_points = len(_first_dict)
        len_index = [len(df.index) for df in _first_dict.values()]
        return len(self), nb_time_points, len_index, len_index

    def to_csv(self, directory: Path, sep: str = ",", na_rep: str = "",
               index: bool = True, header: bool = True, spacer: str = '') -> None:
        """
        Save this VObspArrayContainer in CSV file format.
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


# 2D Containers -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -
class VBase2DArrayContainer(VBaseArrayContainer, ABC, Mapping[str, D_DF]):
    """
    Base abstract class for ArrayContainers linked to a VData object that contain DataFrames (varm and varp)
    It is based on VBaseArrayContainer and defines some functions shared by varm and varp.
    """

    def __init__(self, parent: 'vdata.VData', data: Optional[Dict[str, D_DF]]):
        """
        :param parent: the parent VData object this ArrayContainer is linked to.
        :param data: a dictionary of DataFrames in this ArrayContainer.
        """

        super().__init__(parent, data)

    @property
    def empty(self) -> bool:
        """
        Whether this ArrayContainer is empty or not.
        :return: is this ArrayContainer empty ?
        """
        return all([DF.empty for DF in self.values()])

    def update_dtype(self, type_: DType) -> None:
        """
        Update the data type of TemporalDataFrames stored in this ArrayContainer.
        :param type_: the new data type.
        """
        for arr_name, arr in self.items():
            self[arr_name] = arr.astype(type_)

    def to_csv(self, directory: Path, sep: str = ",", na_rep: str = "",
               index: bool = True, header: bool = True, spacer: str = '') -> None:
        """
        Save the ArrayContainer in CSV file format.
        :param directory: path to a directory for saving the Array
        :param sep: delimiter character
        :param na_rep: string to replace NAs
        :param index: write row names ?
        :param header: Write col names ?
        :param spacer: for logging purposes, the recursion depth of calls to a read_h5 function.
        """
        # create sub directory for storing arrays
        os.makedirs(directory / self.name)

        for arr_name, arr in self.items():
            generalLogger.info(f"{spacer}Saving {arr_name}")

            # save array
            arr.to_csv(f"{directory / self.name / arr_name}.csv", sep, na_rep, index=index, header=header)


class VVarmArrayContainer(VBase2DArrayContainer):
    """
    Class for varm.
    This object contains any number of DataFrames, with shape (n_var, any).
    The DataFrames can be accessed from the parent VData object by :
        VData.varm[<array_name>])
    """

    def __init__(self, parent: 'vdata.VData', data: Optional[Dict[str, D_DF]] = None):
        """
        :param parent: the parent VData object this VVarmArrayContainer is linked to.
        :param data: a dictionary of DataFrames in this VVarmArrayContainer.
        """
        super().__init__(parent, data)

    def _check_init_data(self, data: Optional[Dict[str, D_DF]]) -> Dict[str, D_DF]:
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
            _data = dict()
            _index = self._parent.var.index

            for DF_index, DF in data.items():
                # check that indexes match
                if not _index.equals(DF.index):
                    raise IncoherenceError(f"Index of DataFrame '{DF_index}' does not  match var's index. ({_index})")

            generalLogger.debug("  Data was OK.")
            return _data

    def __getitem__(self, item: str) -> D_DF:
        """
        Get a specific DataFrame stored in this VVarmArrayContainer.
        :param item: key in _data linked to a DataFrame.
        :return: DataFrame stored in _data under the given key.
        """
        return super().__getitem__(item)

    def __setitem__(self, key: str, value: D_DF) -> None:
        """
        Set a specific DataFrame in _data. The given DataFrame must have the correct shape.
        :param key: key for storing a DataFrame in this VVarmArrayContainer.
        :param value: a DataFrame to store.
        """
        # TODO
        raise NotImplementedError

    @property
    def name(self) -> Literal['varm']:
        """
        Name for this VVarmArrayContainer : varm.
        :return: name of this VVarmArrayContainer.
        """
        return 'varm'

    @property
    def shape(self) -> Tuple[int, int, List[int]]:
        """
        The shape of this VVarmArrayContainer is computed from the shape of the DataFrames it contains.
        See __len__ for getting the number of TemporalDataFrames it contains.
        :return: the shape of this VVarmArrayContainer.
        """
        _first_DF = self[list(self.keys())[0]]
        return len(self), _first_DF.shape[0], [DF.shape[1] for DF in self.values()]


class VVarpArrayContainer(VBase2DArrayContainer):
    """
    Class for varp.
    This object contains any number of DataFrames, with shape (n_var, n_var).
    The DataFrames can be accessed from the parent VData object by :
        VData.varp[<array_name>])
    """

    def __init__(self, parent: 'vdata.VData', data: Optional[Dict[str, D_DF]] = None):
        """
        :param parent: the parent VData object this VVarmArrayContainer is linked to.
        :param data: a dictionary of DataFrames in this VVarmArrayContainer.
        """
        super().__init__(parent, data)

    def _check_init_data(self, data: Optional[Dict[str, D_DF]]) -> Dict[str, D_DF]:
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
            _data = dict()
            _index = self._parent.var.index

            for DF_index, DF in data.items():
                # check that indexes match
                if not _index.equals(DF.index):
                    raise IncoherenceError(f"Index of DataFrame '{DF_index}' does not  match var's index. ({_index})")

                # check that columns match
                if not _index.equals(DF.columns):
                    raise IncoherenceError(
                        f"Columns of DataFrame '{DF_index}' do not  match var's index. ({_index})")

            generalLogger.debug("  Data was OK.")
            return _data

    def __getitem__(self, item: str) -> D_DF:
        """
        Get a specific DataFrame stored in this VVarpArrayContainer.
        :param item: key in _data linked to a DataFrame.
        :return: DataFrame stored in _data under the given key.
        """
        return super().__getitem__(item)

    def __setitem__(self, key: str, value: D_DF) -> None:
        """
        Set a specific DataFrame in _data. The given DataFrame must have the correct shape.
        :param key: key for storing a DataFrame in this VVarpArrayContainer.
        :param value: a DataFrame to store.
        """
        # TODO
        raise NotImplementedError

    @property
    def name(self) -> Literal['varp']:
        """
        Name for this VVarpArrayContainer : varp.
        :return: name of this VVarpArrayContainer.
        """
        return 'varp'

    @property
    def shape(self) -> Tuple[int, int, int]:
        """
        The shape of this VVarpArrayContainer is computed from the shape of the DataFrames it contains.
        See __len__ for getting the number of TemporalDataFrames it contains.
        :return: the shape of this VVarpArrayContainer.
        """
        _first_DF = self[list(self.keys())[0]]
        return len(self), _first_DF.shape[0], _first_DF.shape[1]
