# coding: utf-8
# Created on 11/4/20 10:40 AM
# Author : matteo

# ====================================================
# imports
from __future__ import annotations

import os
import abc
import numpy as np
import pandas as pd
from abc import ABC
from pathlib import Path
from typing import Optional, Union, Dict, Tuple, KeysView, ValuesView, ItemsView, Any, Collection, Mapping, \
    Iterator, TypeVar, Type

from vdata.NameUtils import ArrayLike, DType, DataFrame
from .dataframe import TemporalDataFrame
from . import vdata
from .._IO import generalLogger
from .._IO.errors import ShapeError, IncoherenceError, VAttributeError


# ====================================================
# code

D = TypeVar('D', DataFrame, Type['VPairwiseArray'])


# Containers ------------------------------------------------------------------
class VBaseArrayContainer(ABC, Mapping[str, D]):
    """
    Base abstract class for Array containers linked to a VData object (obsm, obsp, varm, varp, layers).
    All Arrays have a '_parent' attribute for linking them to a VData and a '_data' dictionary
    attribute for storing 2D/3D arrays.
    """

    def __init__(self, parent: vdata.VData, data: Optional[Dict[str, D]]):
        """
        :param parent: the parent VData object this Array is linked to.
        :param data: a dictionary of pandas DataFrames to store in this Array container.
        """
        self._parent = parent
        self._data = self._check_init_data(data)

    @abc.abstractmethod
    def __repr__(self) -> str:
        pass

    def __getitem__(self, item: str) -> D:
        """
        Get a specific Array in _data.
        :param item: key in _data
        :return: Array stored in _data under the given key.
        """
        if self._data and item in self._data.keys():
            return self._data[item]

        else:
            raise VAttributeError(f"{self.name} array has no attribute '{item}'")

    @abc.abstractmethod
    def __setitem__(self, key: str, value: pd.DataFrame) -> None:
        """
        Set a specific Array in _data. The given Array must have the correct shape.
        :param key: key for storing Array in _data.
        :param value: an Array to store.
        """
        pass

    def __len__(self) -> int:
        """
        Length of the Array container : the number of Arrays in _data.
        :return: number of Arrays in _data.
        """
        return len(self._data.keys()) if self._data is not None else 0

    def __iter__(self) -> Iterator[str]:
        """
        Iterate on the Array container's keys.
        :return: an iterator over the Array container's keys.
        """
        return iter(self.keys())

    @property
    def empty(self) -> bool:
        """
        Whether this Array container is empty or not.
        :return: is this Array container empty ?
        """
        return True if not len(self) else False

    @abc.abstractmethod
    def _check_init_data(self, data: Optional[Dict[str, D]]) -> Optional[Dict[str, D]]:
        """
        Function for checking, upon creation of this Array container, whether the supplied data has the correct
        format.
        :param data: dictionary of pandas DataFrames.
        :return: dictionary of Arrays.
        """
        pass

    def update_dtype(self, type_: DType) -> None:
        """
        Function for updating the data type of Arrays in the Array container.
        :param type_: the new data type.
        """
        if self._data is not None:
            for arr_name, arr in self._data.items():
                self._data[arr_name] = arr.astype(type_)

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """
        Name for this Array container.
        :return: name of this Array container.
        """
        pass

    @property
    @abc.abstractmethod
    def shape(self) -> Union[Tuple[int, int, int], Tuple[int, int]]:
        """
        The shape of this Array container is computed from the shape of the Arrays it contains.
        See __len__ for getting the number of Arrays it contains.
        :return: shape of this Array container.
        """
        pass

    @property
    def data(self) -> Optional[Dict[str, D]]:
        """
        Data of this Array container.
        :return: the data of this Array container.
        """
        return self._data

    def keys(self) -> Union[Tuple[()], KeysView]:
        """
        KeysView of keys for getting the Arrays in this Array container.
        :return: KeysView of this Array container.
        """
        return self._data.keys() if self._data is not None else ()

    def values(self) -> Union[Tuple[()], ValuesView]:
        """
        ValuesView of Arrays in this Array container.
        :return: ValuesView of this Array container.
        """
        return self._data.values() if self._data is not None else ()

    def items(self) -> Union[Tuple[()], ItemsView]:
        """
        ItemsView of pairs of keys and Arrays in this Array container.
        :return: ItemsView of this Array container.
        """
        return self._data.items() if self._data is not None else ()

    def dict_copy(self) -> Dict[str, ArrayLike]:
        """
        Build a copy of this Array container in dict format.
        :return: Dictionary of (keys, Array) in this Array container.
        """
        return dict(self._data) if self._data is not None else dict()


class VBase3DArrayContainer(VBaseArrayContainer, ABC):
    """
    Base abstract class for Array containers linked to a VData object that contain pseudo 3D Arrays (obsm,
    varm, layers)
    It is based on VBaseArrayContainer and defines some functions shared by obsm, varm and layers.
    """

    @abc.abstractmethod
    def get_idx_names(self) -> Collection:
        """
        Get index of this Array container :
            - names of obs for layers and obsm.
            - names of var for varm.
        :return: index of this Array container.
        """
        pass

    @abc.abstractmethod
    def get_col_names(self, arr_name: Optional[str]) -> Collection:
        """
        Get columns of this Array container :
            - names of var for layers.
            - names of the columns for each Array in obsm and varm.
        :param arr_name: the name of the array in obsm or varm.
        :return: columns for the Array.
        """
        pass

    @property
    def shape(self) -> Tuple[int, int, int]:
        """
        The shape of the Array container is computed from the shape of the Arrays it contains.
        See __len__ for getting the number of Arrays it contains.
        :return: shape of the contained Arrays.
        """
        if self._data is not None and len(self):
            return self._data[list(self._data.keys())[0]].shape
        else:
            s1 = self._parent.n_obs if self.name in ("layers", "obsm") else self._parent.n_var
            s2 = self._parent.n_var if self.name == "layers" else 0
            return self._parent.n_time_points, s1, s2

    def to_csv(self, directory: Path, sep: str = ",", na_rep: str = "",
               index: bool = True, header: bool = True) -> None:
        """
        Save the Array in CSV file format.
        :param directory: path to a directory for saving the Array
        :param sep: delimiter character
        :param na_rep: string to replace NAs
        :param index: write row names ?
        :param header: Write col names ?
        """
        # create sub directory for storing arrays
        os.makedirs(directory / self.name)

        idx = self.get_idx_names()

        for arr_name, arr in self.items():
            col = self.get_col_names(arr_name)
            # cast array in 2D
            arr_2D = arr.reshape((arr.shape[1]*arr.shape[0], arr.shape[2]))
            # add time point information
            time_point_col = np.array(np.repeat(self._parent.time_points['value'], arr.shape[1]))
            arr_2D = np.concatenate((time_point_col[:, None], arr_2D), axis=1)
            # save array
            pd.DataFrame(arr_2D, index=pd.Series(np.repeat(idx, self._parent.n_time_points)),
                         columns=['Time_point'] + list(col)).to_csv(f"{directory / self.name / arr_name}.csv",
                                                                    sep, na_rep, index=index, header=header)


class VLayerArrayContainer(VBase3DArrayContainer):
    """
    Class for layers.
    This object contains any number of TemporalDataFrames, with shapes (n_time_points, n_obs, n_var).
    The arrays-like objects can be accessed from the parent VData object by :
        VData.layers['<array_name>']
    """

    def __init__(self, parent: vdata.VData, data: Optional[Dict[str, TemporalDataFrame]]):
        """
        :param parent: the parent VData object this Array is linked to
        :param data: a dictionary of array-like objects to store in this Array
        """
        generalLogger.debug(f"== Creating VLayerArrayContainer. ================================")

        super().__init__(parent, data)

    def _check_init_data(self, data: Optional[Dict[Any, TemporalDataFrame]]) \
            -> Optional[Dict[str, TemporalDataFrame]]:
        """
        Function for checking, at Array container creation, that the supplied data has the correct format :
            - the shape of the TemporalDataFrames in 'data' match the parent VData object's shape.
            - the index of the TemporalDataFrames in 'data' match the index of the parent VData's obs TemporalDataFrame.
        :return: the data (dictionary of Arrays), if correct
        """
        if data is None or not len(data):
            generalLogger.debug("  No data was given.")
            return None

        else:
            generalLogger.debug("  Data was found.")
            _data = {}
            _shape = self._parent.shape
            _index = self._parent.obs.index

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
                if _index.equals(TDF.index):
                    raise IncoherenceError(f"Index of layer '{TDF_index}' does not match obs' index.")

                # checks passed, store the TemporalDataFrame
                else:
                    _data[str(TDF_index)] = TDF

            generalLogger.debug("  Data was OK.")
            return _data

    def __repr__(self) -> str:
        if len(self):
            list_of_keys = "'" + "','".join(self.keys()) + "'"
            return f"VLayerArrayContainer with keys : {list_of_keys}."
        else:
            return "Empty VLayerArrayContainer of layers."

    def __setitem__(self, key: str, value: TemporalDataFrame) -> None:
        """
        Set specific Array in _data. The given Array must have the same shape as the parent.
        :param key: key for storing Array in _data
        :param value: an Array to store
        """
        # first check that value has the correct shape
        if value.shape == self._parent.shape:
            if self._data is None:
                self._data = {}
            self._data[key] = value

        else:
            raise ShapeError(f"The supplied array-like object has incorrect shape {value.shape}, "
                             f"expected {self._parent.shape}")

    @property
    def name(self) -> str:
        """
        Name for the Array : layers.
        :return: name of the array
        """
        return "layers"

    def get_idx_names(self) -> pd.Index:
        """
        Get index for the Array :
            - names of obs for layers and obsm
            - names of var for varm
        :return: index for the Array
        """
        return self._parent.obs.index

    def get_col_names(self, arr_name: Optional[str]) -> pd.Index:
        """
        Get columns for the Array :
            - names of var for layers
            - names of the columns for each array-like in obsm and varm
        :param arr_name: the name of the array in obsm or varm
        :return: columns for the Array
        """
        return self._parent.var.index


# class VAxisArrayContainer(VBase3DArrayContainer):
#     """
#     Class for obsm and varm.
#     These objects contain any number of 3D array-like objects, with shape (n_time_points, n_obs, any)
#         and (n_var, any) respectively.
#     The arrays-like objects can be accessed from the parent VData object by :
#         VData.obsm['<array_name>'])
#         VData.varm['<array_name>'])
#     """
#
#     def __init__(self, parent: "vdata.VData", axis: Literal['obs', 'var'],
#                  data: Optional[Dict[str, ArrayLike_3D]] = None,
#                  col_names: Optional[Dict[str, Collection]] = None):
#         """
#         :param parent: the parent VData object this Array is linked to
#         :param data: a dictionary of array-like objects to store in this Array
#         :col_names: a dictionary of collections of column names to describe array-like objects stored in the Array
#         """
#         generalLogger.debug(f"== Creating {axis}m VAxisArrayContainer. ==============================")
#         self._axis = axis
#         super().__init__(parent, data)
#         self._col_names = self._check_col_names(col_names)
#
#     def _check_init_data(self, data: Optional[Dict[Any, ArrayLike_3D]]) -> Optional[Dict[str, 'VBaseArray']]:
#         """
#         Function for checking, at Array container creation, that the supplied data has the correct format :
#             # TODO : this is not True for obsm and varm !
#             - all Arrays in 'data' have the same shape
#             - the shape of the Arrays in 'data' match the parent VData object's size
#         :return: the data (dictionary of Arrays), if correct
#         """
#         if data is None or not len(data):
#             generalLogger.debug("  No data was given.")
#             return None
#
#         else:
#             generalLogger.debug("  Data was found.")
#             _data = {}
#             _shape = self._parent.shape
#
#             generalLogger.debug(f"  Reference shape is {_shape}.")
#
#             for array_index, array in data.items():
#                 array_shape = (array.shape[0], [array[i].shape[0] for i in range(len(array))], array[0].shape[1])
#
#                 generalLogger.debug(f"  Checking array '{array_index}' with shape {array_shape}.")
#
#                 if not all([array[0].shape[1] == array[i].shape[1] for i in range(array.shape[0])]):
#                     raise IncoherenceError(f"{self.name} '{array_index}' has arrays of different third dimension, "
#                                            f"should all be the same.")
#
#                 if _shape != array_shape:
#
#                     if _shape[0] != array_shape[0]:
#                         raise IncoherenceError(f"{self.name} '{array_index}' has {array_shape[0]} "
#                                                f"time point{'s' if array_shape[0] > 1 else ''}, "
#                                                f"should have {_shape[0]}.")
#
#                     elif self.name in ("layers", "obsm") and _shape[1] != array_shape[1]:
#                         for i in range(len(array)):
#                             if _shape[1][i] != array_shape[1][i]:
#                                 raise IncoherenceError(f"{self.name} '{array_index}' at time point {i} has"
#                                                        f" {array_shape[1][i]} observations, "
#                                                        f"should have {_shape[1][i]}.")
#
#                     elif self.name in ("layers", "varm"):
#                         raise IncoherenceError(f"{self.name} '{array_index}' has  {array_shape[2]} variables, "
#                                                f"should have {_shape[2]}.")
#
#                     else:
#                         _data[str(array_index)] = np.array([arr.astype(self._parent.dtype) for arr in array],
#                                                            dtype=object)
#
#                 else:
#                     _data[str(array_index)] = np.array([arr.astype(self._parent.dtype) for arr in array],
#                     dtype=object)
#
#             generalLogger.debug("  Data was OK.")
#             return _data
#
#     def __repr__(self) -> str:
#         if len(self):
#             list_of_keys = "'" + "','".join(self.keys()) + "'"
#             return f"VAxisArrayContainer of {self.name} with keys : {list_of_keys}."
#         else:
#             return f"Empty VAxisArrayContainer of {self.name}."
#
#     def __setitem__(self, key: str, value: pd.DataFrame) -> None:
#         """
#         Set specific Array in _data. The given Array must have the same shape as the parent.
#         :param key: key for storing Array in _data
#         :param value: an Array to store
#         """
#         # first check that value has the correct shape
#         if value.shape == self._parent.shape:
#             if self._data is None:
#                 self._data = {}
#             self._data[key] = VAxisArray(value)
#
#         else:
#             raise ShapeError(f"The supplied array-like object has incorrect shape {value.shape}, "
#                              f"expected {self._parent.shape}")
#
#     def _check_col_names(self, col_names: Optional[Dict[str, Collection]]) -> Optional[Dict[str, Collection]]:
#         """
#         Function for checking that the supplied col names are of the right format :
#             - same keys in 'col_names' as in 'data'
#             - values in col_names are list of names for the columns of the array-like objects in 'data'.
#         :param col_names: dictionary of column names per array-like object in 'data'
#         :return: a properly formatted dictionary of column names per array-like object in 'data'
#         """
#         if self._data is None:
#             if col_names is not None:
#                 raise VValueError("Can't set col names if no data is supplied.")
#
#             else:
#                 return None
#
#         else:
#             _col_names: Dict[str, Collection] = {}
#
#             if col_names is None:
#                 for k, v in self._data.items():
#                     _col_names[k] = range(v.shape[2])
#                 return _col_names
#
#             else:
#                 if not isinstance(col_names, dict):
#                     raise VTypeError("'col_names' must be a dictionary with same keys as 'data' and values as lists "
#                                      "of column names for 'data'.")
#
#                 elif col_names.keys() != self._data.keys():
#                     raise VValueError("'col_names' must be the same as 'data' keys.")
#
#                 else:
#                     for k, v in col_names.items():
#                         _col_names[str(k)] = list(v)
#                     return _col_names
#
#     @property
#     def name(self):
#         """
#         Name for the Array, either obsm or varm.
#         :return: name of the array
#         """
#         return f"{self._axis}m"
#
#     def get_idx_names(self) -> pd.Index:
#         """
#         Get index for the Array :
#             - names of obs for layers and obsm
#             - names of var for varm
#         :return: index for the Array
#         """
#         return getattr(self._parent, self._axis).index
#
#     def get_col_names(self, arr_name: Optional[str]) -> Collection:
#         """
#         Get columns for the Array :
#             - names of var for layers
#             - names of the columns for each array-like in obsm and varm
#         :param arr_name: the name of the array in obsm or varm
#         :return: columns for the Array
#         """
#         if arr_name is None:
#             raise VValueError("No array-like name supplied.")
#         return self._col_names[arr_name] if self._col_names is not None else []
#
#
# class VPairwiseArrayContainer(VBaseArrayContainer):
#     """
#     Class for obsp and varp.
#     This object contains any number of 2D array-like objects, with shapes (n_time_points, n_obs, n_obs)
#         and (n_time_points, n_var, n_var)
#     respectively.
#     The arrays-like objects can be accessed from the parent VData object by :
#         VData.obsp['<array_name>']
#         VData.obsp['<array_name>']
#     """
#
#     def __init__(self, parent: "vdata.VData", axis: Literal['obs', 'var'], data: Optional[Dict[Any, ArrayLike_2D]]):
#         """
#         :param parent: the parent VData object this Array is linked to
#         :param axis: the axis this Array must conform to (obs or var)
#         :param data: a dictionary of array-like objects to store in this Array
#         """
#         generalLogger.debug(f"== Creating {axis}p VPairwiseArrayContainer. ==========================")
#         self._axis = axis
#         super().__init__(parent, data)
#
#     def __repr__(self) -> str:
#         if len(self):
#             list_of_keys = "'" + "','".join(self.keys()) + "'"
#             return f"VPairwiseArrayContainer of {self.name} with keys : {list_of_keys}."
#         else:
#             return f"Empty VPairwiseArrayContainer of {self.name}."
#
#     def __setitem__(self, key: str, value: ArrayLike_2D) -> None:
#         """
#         Set specific array-like in _data. The given array-like must have a square shape (n_obs, n_obs)
#         for obsm and (n_var, n_var) for varm.
#         :param key: key for storing array-like in _data
#         :param value: an array-like to store
#         """
#         # first check that value has the correct shape
#         shape_parent = getattr(self._parent, f"n_{self._axis}")
#
#         if value.shape[0] == value.shape[1]:
#             if value.shape[0] == shape_parent:
#                 if self._data is None:
#                     self._data = {}
#                 self._data[key] = value
#
#             else:
#                 raise IncoherenceError(f"The supplied array-like object has incorrect shape {value.shape}, "
#                                        f"expected ({shape_parent}, {shape_parent})")
#
#         else:
#             raise ShapeError("The supplied array-like object is not square.")
#
#     def _check_init_data(self, data: Optional[Dict[Any, ArrayLike_2D]]) -> Optional[Dict[str, ArrayLike_2D]]:
#         """
#         Function for checking, at Array creation, that the supplied data has the correct format:
#             - all array-like objects in 'data' are square
#             - their shape match the parent VData object's n_obs or n_var
#         :param data: dictionary of array-like objects.
#         :return: the data (dictionary of array-like objects), if correct
#         """
#         if data is None or not len(data):
#             generalLogger.debug("  No data was given.")
#             return None
#
#         else:
#             shape_parent = getattr(self._parent, f"n_{self._axis}")
#             _data = {}
#
#             for array_index, array in data.items():
#                 if array.shape[0] != array.shape[1]:
#                     raise ShapeError(f"The array-like object '{array_index}' supplied to {self.name} is not square.")
#
#                 elif array.shape[0] != shape_parent:
#                     raise IncoherenceError(f"The array-like object '{array_index}' supplied to {self.name} has shape "
#                                            f"{array.shape}, it should have shape ({shape_parent}, {shape_parent})")
#
#                 else:
#                     _data[str(array_index)] = array.astype(self._parent.dtype)
#
#             generalLogger.debug("  Data was OK.")
#             return _data
#
#     @property
#     def name(self) -> str:
#         """
#         Name for the Array, either obsp or varp.
#         :return: name of the array
#         """
#         return f"{self._axis}p"
#
#     @property
#     def shape(self) -> Tuple[int, int]:
#         """
#         The shape of the Array is computed from the shape of the array-like objects it contains.
#         See __len__ for getting the number of array-like objects it contains.
#         :return: shape of the contained array-like objects.
#         """
#         if self._data is not None:
#             return self._data[list(self._data.keys())[0]].shape
#
#         else:
#             return (self._parent.n_obs, self._parent.n_obs) \
#                 if self._axis == "obs" else (self._parent.n_var, self._parent.n_var)
#
#     def to_csv(self, directory: Path, sep: str = ",", na_rep: str = "",
#                index: bool = True, header: bool = True) -> None:
#         """
#         Save the Array in CSV file format.
#         :param directory: path to a directory for saving the Array
#         :param sep: delimiter character
#         :param na_rep: string to replace NAs
#         :param index: write row names ?
#         :param header: Write col names ?
#         """
#         # create sub directory for storing arrays
#         os.makedirs(directory / self.name)
#
#         idx = getattr(self._parent, self._axis).index
#
#         for arr_name, arr in self.items():
#             pd.DataFrame(arr, index=idx, columns=idx).to_csv(f"{directory / self.name / arr_name}.csv",
#                                                              sep, na_rep, index=index, header=header)
#
#
# # Arrays ----------------------------------------------------------------------
# class VPairwiseArray:
#     """"""
#
#     def __init__(self):
#         raise NotImplementedError
#
#     def __repr__(self) -> str:
#         """
#         Description for this VPairwiseArray object to print.
#         :return: a description of this VPairwiseArray object.
#         """
#         pass
#
#     def reorder(self, reference_index: Collection) -> None:
#         """
#         Reorder the Array to match the given reference_index.
#         The given reference_index must contain all elements in the index of this array's DataFrame exactly once but
#         can be in any desired order.
#         :param reference_index: a collection of elements matching the index of this array's DataFrame.
#         """
#         pass
#
#     def astype(self, dtype: DType):
#         """
#         Modify the data type of elements stored in this Array.
#         """
#         pass
#
#     def shape(self):
#         """TODO"""
#         pass
