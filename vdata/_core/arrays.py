# coding: utf-8
# Created on 11/4/20 10:40 AM
# Author : matteo

# ====================================================
# imports
import os
import abc
import numpy as np
import pandas as pd
from abc import ABC
from pathlib import Path
from typing import Optional, Union, Dict, Tuple, KeysView, ValuesView, ItemsView, Any, Collection
from typing_extensions import Literal

from . import vdata
from ..NameUtils import ArrayLike_2D, ArrayLike_3D, ArrayLike, DType
from .._IO.errors import ShapeError, IncoherenceError, VValueError, VTypeError, VAttributeError
from .._IO.logger import generalLogger


# ====================================================
# code
class VBaseArrayContainer(ABC):
    """
    Base abstract class for Arrays linked to a VData object (obsm, obsp, varm, varp, layers).
    All Arrays have a '_parent' attribute for linking them to a VData and a '_data' dictionary
    attribute for storing 2D/3D arrays.
    """

    def __init__(self, parent: "vdata.VData", data: Optional[Dict[str, ArrayLike]]):
        """
        :param parent: the parent VData object this Array is linked to
        :param data: a dictionary of array-like objects to store in this Array
        """
        self._parent = parent
        self._data = self._check_init_data(data)

    @abc.abstractmethod
    def __repr__(self) -> str:
        pass

    def __getitem__(self, item: str) -> ArrayLike:
        """
        Get specific array-like in _data
        :param item: key in _data
        :return: array-like stored in _data under given key
        """
        if self._data and item in self._data.keys():
            return self._data[item]

        else:
            raise VAttributeError(f"{self.name} array has no attribute '{item}'")

    @abc.abstractmethod
    def __setitem__(self, key: str, value: ArrayLike) -> None:
        """
        Set specific array-like in _data. The given array-like must have the correct shape.
        :param key: key for storing array-like in _data
        :param value: an array-like to store
        """
        pass

    def __len__(self) -> int:
        """
        Length of the Array : the number of array-like objects in _data
        :return: number of array-like objects in _data
        """
        return len(self._data.keys()) if self._data is not None else 0

    @property
    def empty(self) -> bool:
        """
        Whether this Array is empty or not.
        :return: is this array empty ?
        """
        return True if not len(self) else False

    @abc.abstractmethod
    def _check_init_data(self, data: Optional[Dict[str, ArrayLike]]) -> Optional[Dict[str, ArrayLike]]:
        """
        Function for checking, at Array creation, that the supplied data has the correct format.
        :param data: dictionary of array-like objects.
        :return: dictionary of array-like objects
        """
        pass

    def update_dtype(self, type_: DType) -> None:
        """
        Function for updating the data type of array-like objects contained in the Array.
        :param type_: the new data type
        """
        if self._data is not None:
            for arr_name, arr in self._data.items():
                self._data[arr_name] = arr.astype(type_)

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """
        Name for the Array.
        :return: name of the array
        """
        pass

    @property
    @abc.abstractmethod
    def shape(self) -> Union[Tuple[int, int, int], Tuple[int, int]]:
        """
        The shape of the Array is computed from the shape of the array-like objects it contains.
        See __len__ for getting the number of array-like objects it contains.
        :return: shape of the contained array-like objects.
        """
        pass

    @abc.abstractmethod
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
        pass

    @property
    def data(self) -> Optional[Dict[str, ArrayLike]]:
        """
        Data of the Array.
        :return: name of the array
        """
        return self._data

    def keys(self) -> Union[Tuple[()], KeysView]:
        """
        KeysView of keys for getting the array-like objects.
        :return: KeysView of this Array.
        """
        return self._data.keys() if self._data is not None else ()

    def values(self) -> Union[Tuple[()], ValuesView]:
        """
        ValuesView of array-like objects in this Array
        :return:ValuesView of this Array.
        """
        return self._data.values() if self._data is not None else ()

    def items(self) -> Union[Tuple[()], ItemsView]:
        """
        ItemsView of pairs of keys and array-like objects in this Array.
        :return: ItemsView of this Array.
        """
        return self._data.items() if self._data is not None else ()

    def dict_copy(self) -> Dict[str, ArrayLike]:
        """
        Build a copy of this Array in dict format.
        :return: Dictionary of (keys, ArrayLike) in this Array.
        """
        return dict(self._data) if self._data is not None else dict()


class VBase3DArrayContainer(VBaseArrayContainer, ABC):
    """
    Base abstract class for arrays linked to a VData object that contain 3D array-like objects (obsm, varm, layers)
    It is based on VBaseArrayContainer and defines some functions shared by obsm, varm and layers.
    """
    def __setitem__(self, key: str, value: ArrayLike_3D) -> None:
        """
        Set specific array-like in _data. The given array-like must have the same shape as the parent
        VData X array-like.
        :param key: key for storing array-like in _data
        :param value: an array-like to store
        """
        # first check that value has the correct shape
        if value.shape == self._parent.shape:
            if self._data is None:
                self._data = {}
            self._data[key] = value

        else:
            raise ShapeError(f"The supplied array-like object has incorrect shape {value.shape}, "
                             f"expected {self._parent.shape}")

    def _check_init_data(self, data: Optional[Dict[Any, ArrayLike_3D]]) -> Optional[Dict[str, ArrayLike_3D]]:
        """
        Function for checking, at Array creation, that the supplied data has the correct format :
            - all array-like objects in 'data' have the same shape
            - the shape of the array-like objects in 'data' match the parent vdata object's size
        :return: the data (dictionary of array-like objects), if correct
        """
        if data is None or not len(data):
            generalLogger.debug("  No data was given.")
            return None

        else:
            generalLogger.debug("  Data was found.")
            _data = {}
            _shape = self._parent.shape

            generalLogger.debug(f"  Reference shape is {_shape}.")

            for array_index, array in data.items():
                array_shape = (array.shape[0], [array[i].shape[0] for i in range(len(array))], array[0].shape[1])

                generalLogger.debug(f"  Checking array '{array_index}' with shape {array_shape}.")

                if not all([array[0].shape[1] == array[i].shape[1] for i in range(array.shape[0])]):
                    raise IncoherenceError(f"{self.name} '{array_index}' has arrays of different third dimension, "
                                           f"should all be the same.")

                if _shape != array_shape:

                    if _shape[0] != array_shape[0]:
                        raise IncoherenceError(f"{self.name} '{array_index}' has {array_shape[0]} "
                                               f"time point{'s' if array_shape[0] > 1 else ''}, "
                                               f"should have {_shape[0]}.")

                    elif self.name in ("layers", "obsm") and _shape[1] != array_shape[1]:
                        for i in range(len(array)):
                            if _shape[1][i] != array_shape[1][i]:
                                raise IncoherenceError(f"{self.name} '{array_index}' at time point {i} has"
                                                       f" {array_shape[1][i]} observations, "
                                                       f"should have {_shape[1][i]}.")

                    elif self.name in ("layers", "varm"):
                        raise IncoherenceError(f"{self.name} '{array_index}' has  {array_shape[2]} variables, "
                                               f"should have {_shape[2]}.")

                    else:
                        _data[str(array_index)] = np.array([arr.astype(self._parent.dtype) for arr in array],
                                                           dtype=object)

                else:
                    _data[str(array_index)] = np.array([arr.astype(self._parent.dtype) for arr in array], dtype=object)

            generalLogger.debug("  Data was OK.")
            return _data

    @abc.abstractmethod
    def get_idx_names(self) -> Collection:
        """
        Get index for the Array :
            - names of obs for layers and obsm
            - names of var for varm
        :return: index for the Array
        """
        pass

    @abc.abstractmethod
    def get_col_names(self, arr_name: Optional[str]) -> Collection:
        """
        Get columns for the Array :
            - names of var for layers
            - names of the columns for each array-like in obsm and varm
        :param arr_name: the name of the array in obsm or varm
        :return: columns for the Array
        """
        pass

    @property
    def shape(self) -> Tuple[int, int, int]:
        """
        The shape of the Array is computed from the shape of the array-like objects it contains.
        See __len__ for getting the number of array-like objects it contains.
        :return: shape of the contained array-like objects.
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


class VAxisArray(VBase3DArrayContainer):
    """
    Class for obsm and varm.
    These objects contain any number of 3D array-like objects, with shape (n_time_points, n_obs, any)
        and (n_var, any) respectively.
    The arrays-like objects can be accessed from the parent VData object by :
        VData.obsm['<array_name>'])
        VData.varm['<array_name>'])
    """

    def __init__(self, parent: "vdata.VData", axis: Literal['obs', 'var'],
                 data: Optional[Dict[str, ArrayLike_3D]] = None,
                 col_names: Optional[Dict[str, Collection]] = None):
        """
        :param parent: the parent VData object this Array is linked to
        :param data: a dictionary of array-like objects to store in this Array
        :col_names: a dictionary of collections of column names to describe array-like objects stored in the Array
        """
        generalLogger.debug(f"== Creating {axis}m VAxisArray. ==============================")
        self._axis = axis
        super().__init__(parent, data)
        self._col_names = self._check_col_names(col_names)

    def __repr__(self) -> str:
        if len(self):
            list_of_keys = "'" + "','".join(self.keys()) + "'"
            return f"VAxisArray of {self.name} with keys : {list_of_keys}."
        else:
            return f"Empty VAxisArray of {self.name}."

    def _check_col_names(self, col_names: Optional[Dict[str, Collection]]) -> Optional[Dict[str, Collection]]:
        """
        Function for checking that the supplied col names are of the right format :
            - same keys in 'col_names' as in 'data'
            - values in col_names are list of names for the columns of the array-like objects in 'data'.
        :param col_names: dictionary of column names per array-like object in 'data'
        :return: a properly formatted dictionary of column names per array-like object in 'data'
        """
        if self._data is None:
            if col_names is not None:
                raise VValueError("Can't set col names if no data is supplied.")

            else:
                return None

        else:
            _col_names: Dict[str, Collection] = {}

            if col_names is None:
                for k, v in self._data.items():
                    _col_names[k] = range(v.shape[2])
                return _col_names

            else:
                if not isinstance(col_names, dict):
                    raise VTypeError("'col_names' must be a dictionary with same keys as 'data' and values as lists "
                                     "of column names for 'data'.")

                elif col_names.keys() != self._data.keys():
                    raise VValueError("'col_names' must be the same as 'data' keys.")

                else:
                    for k, v in col_names.items():
                        _col_names[str(k)] = list(v)
                    return _col_names

    @property
    def name(self):
        """
        Name for the Array, either obsm or varm.
        :return: name of the array
        """
        return f"{self._axis}m"

    def get_idx_names(self) -> pd.Index:
        """
        Get index for the Array :
            - names of obs for layers and obsm
            - names of var for varm
        :return: index for the Array
        """
        return getattr(self._parent, self._axis).index

    def get_col_names(self, arr_name: Optional[str]) -> Collection:
        """
        Get columns for the Array :
            - names of var for layers
            - names of the columns for each array-like in obsm and varm
        :param arr_name: the name of the array in obsm or varm
        :return: columns for the Array
        """
        if arr_name is None:
            raise VValueError("No array-like name supplied.")
        return self._col_names[arr_name] if self._col_names is not None else []


class VLayersArrays(VBase3DArrayContainer):
    """
    Class for layers.
    This object contains any number of 3D array-like objects, with shapes (n_time_points, n_obs, n_var).
    The arrays-like objects can be accessed from the parent VData object by :
        VData.layers['<array_name>']
    """

    def __init__(self, parent: "vdata.VData", data: Optional[Dict[str, ArrayLike_3D]]):
        """
        :param parent: the parent VData object this Array is linked to
        :param data: a dictionary of array-like objects to store in this Array
        """
        generalLogger.debug(f"== Creating VLayersArrays. ================================")
        super().__init__(parent, data)

    def __repr__(self) -> str:
        if len(self):
            list_of_keys = "'" + "','".join(self.keys()) + "'"
            return f"VLayersArrays with keys : {list_of_keys}."
        else:
            return "Empty VLayersArrays of layers."

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


class VPairwiseArray(VBaseArrayContainer):
    """
    Class for obsp and varp.
    This object contains any number of 2D array-like objects, with shapes (n_time_points, n_obs, n_obs)
        and (n_time_points, n_var, n_var)
    respectively.
    The arrays-like objects can be accessed from the parent VData object by :
        VData.obsp['<array_name>']
        VData.obsp['<array_name>']
    """

    def __init__(self, parent: "vdata.VData", axis: Literal['obs', 'var'], data: Optional[Dict[Any, ArrayLike_2D]]):
        """
        :param parent: the parent VData object this Array is linked to
        :param axis: the axis this Array must conform to (obs or var)
        :param data: a dictionary of array-like objects to store in this Array
        """
        generalLogger.debug(f"== Creating {axis}p VPairwiseArray. ==========================")
        self._axis = axis
        super().__init__(parent, data)

    def __repr__(self) -> str:
        if len(self):
            list_of_keys = "'" + "','".join(self.keys()) + "'"
            return f"VPairwiseArray of {self.name} with keys : {list_of_keys}."
        else:
            return f"Empty VPairwiseArray of {self.name}."

    def __setitem__(self, key: str, value: ArrayLike_2D) -> None:
        """
        Set specific array-like in _data. The given array-like must have a square shape (n_obs, n_obs)
        for obsm and (n_var, n_var) for varm.
        :param key: key for storing array-like in _data
        :param value: an array-like to store
        """
        # first check that value has the correct shape
        shape_parent = getattr(self._parent, f"n_{self._axis}")

        if value.shape[0] == value.shape[1]:
            if value.shape[0] == shape_parent:
                if self._data is None:
                    self._data = {}
                self._data[key] = value

            else:
                raise IncoherenceError(f"The supplied array-like object has incorrect shape {value.shape}, "
                                       f"expected ({shape_parent}, {shape_parent})")

        else:
            raise ShapeError("The supplied array-like object is not square.")

    def _check_init_data(self, data: Optional[Dict[Any, ArrayLike_2D]]) -> Optional[Dict[str, ArrayLike_2D]]:
        """
        Function for checking, at Array creation, that the supplied data has the correct format:
            - all array-like objects in 'data' are square
            - their shape match the parent VData object's n_obs or n_var
        :param data: dictionary of array-like objects.
        :return: the data (dictionary of array-like objects), if correct
        """
        if data is None or not len(data):
            generalLogger.debug("  No data was given.")
            return None

        else:
            shape_parent = getattr(self._parent, f"n_{self._axis}")
            _data = {}

            for array_index, array in data.items():
                if array.shape[0] != array.shape[1]:
                    raise ShapeError(f"The array-like object '{array_index}' supplied to {self.name} is not square.")

                elif array.shape[0] != shape_parent:
                    raise IncoherenceError(f"The array-like object '{array_index}' supplied to {self.name} has shape "
                                           f"{array.shape}, it should have shape ({shape_parent}, {shape_parent})")

                else:
                    _data[str(array_index)] = array.astype(self._parent.dtype)

            generalLogger.debug("  Data was OK.")
            return _data

    @property
    def name(self) -> str:
        """
        Name for the Array, either obsp or varp.
        :return: name of the array
        """
        return f"{self._axis}p"

    @property
    def shape(self) -> Tuple[int, int]:
        """
        The shape of the Array is computed from the shape of the array-like objects it contains.
        See __len__ for getting the number of array-like objects it contains.
        :return: shape of the contained array-like objects.
        """
        if self._data is not None:
            return self._data[list(self._data.keys())[0]].shape

        else:
            return (self._parent.n_obs, self._parent.n_obs) \
                if self._axis == "obs" else (self._parent.n_var, self._parent.n_var)

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

        idx = getattr(self._parent, self._axis).index

        for arr_name, arr in self.items():
            pd.DataFrame(arr, index=idx, columns=idx).to_csv(f"{directory / self.name / arr_name}.csv",
                                                             sep, na_rep, index=index, header=header)
