# coding: utf-8
# Created on 15/01/2021 12:57
# Author : matteo

# ====================================================
# imports
import numpy as np
import abc
from pathlib import Path
from typing import Tuple, Dict, Union, KeysView, ValuesView, ItemsView, List, Iterator

from vdata.NameUtils import DataFrame
from ..arrays import VBaseArrayContainer, VLayerArrayContainer, D
from ..._TDF.views import dataframe
from ..._IO import generalLogger


# ====================================================
# code
# Base Containers -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
class ViewVBaseArrayContainer(abc.ABC):
    """
    A base abstract class for views of VBaseArrayContainers.
    This class is used to create views on VLayerArrayContainer, VAxisArrays and VPairwiseArrays.
    """

    def __init__(self, array_container: VBaseArrayContainer):
        """
        :param array_container: a VBaseArrayContainer object to build a view on.
        """
        self._array_container = array_container

    def __repr__(self) -> str:
        """
        Description for this view  to print.
        :return: a description of this view.
        """
        return f"View of {self._array_container}"

    @abc.abstractmethod
    def __getitem__(self, item: str) -> D:
        """
        Get a specific data item stored in this view.
        :param item: key in _data linked to a data item.
        :return: data item stored in _data under the given key.
        """
        pass

    @abc.abstractmethod
    def __setitem__(self, key: str, value: D) -> None:
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
    @abc.abstractmethod
    def name(self) -> str:
        """
        Name for this view.
        :return: the name of this view.
        """
        return f"{self._array_container.name}_view"

    @property
    @abc.abstractmethod
    def shape(self) -> Union[
        Tuple[int, int, int],
        Tuple[int, int, List[int]],
        Tuple[int, int, List[int], int],
        Tuple[int, int, List[int], List[int]]
    ]:
        """
        The shape of this view is computed from the shape of the Arrays it contains.
        See __len__ for getting the number of Arrays it contains.
        :return: the shape of this view.
        """
        pass

    @property
    @abc.abstractmethod
    def data(self) -> Dict[str, D]:
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

    def values(self) -> ValuesView[D]:
        """
        ValuesView of data items in this view.
        :return: ValuesView of this view.
        """
        return self.data.values()

    def items(self) -> ItemsView[str, D]:
        """
        ItemsView of pairs of keys and data items in this view.
        :return: ItemsView of this view.
        """
        return self.data.items()

    @abc.abstractmethod
    def dict_copy(self) -> Dict[str, D]:
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
class ViewVBase3DArrayContainer(ViewVBaseArrayContainer):
    """
    Base abstract class for views of ArrayContainers.
    It is based on VBaseArrayContainer and defines some functions shared by obsm and layers.
    """

    def __init__(self, array_container: VBaseArrayContainer):
        """
        :param array_container: a VBaseArrayContainer object to build a view on.
        """
        super().__init__(array_container)

    def __getitem__(self, item: str) -> D:
        pass

    def __setitem__(self, key: str, value: D) -> None:
        pass

    @property
    def empty(self) -> bool:
        pass

    @property
    def name(self) -> str:
        pass

    @property
    def shape(self) -> Union[
        Tuple[int, int, int],
        Tuple[int, int, List[int]],
        Tuple[int, int, List[int], int],
        Tuple[int, int, List[int], List[int]]
    ]:
        pass

    @property
    def data(self) -> Dict[str, D]:
        pass

    def dict_copy(self) -> Dict[str, D]:
        pass

    def to_csv(self, directory: Path, sep: str = ",", na_rep: str = "", index: bool = True, header: bool = True,
               spacer: str = '') -> None:
        pass


class ViewVLayerArrayContainer(ViewVBaseArrayContainer):
    """
    A view of a VLayerArrayContainer object.
    """

    def __init__(self, array_container: VLayerArrayContainer, time_points_slicer: np.ndarray,
                 obs_slicer: np.ndarray, var_slicer: np.ndarray):
        """
        :param array_container: a VLayerArrayContainer object to build a view on
        :param obs_slicer: the list of observations to view
        :param var_slicer: the list of variables to view
        :param time_points_slicer: the list of time points to view
        """
        generalLogger.debug("== Creating ViewVLayersArraysContainer. ================================")

        _view_array_container = {name: arr[time_points_slicer, obs_slicer, var_slicer]
                                 for name, arr in array_container.items()}

        super().__init__(_view_array_container)

        # self._time_points_slicer = time_points_slicer
        # generalLogger.debug(f"  1. Time points slicer is : {repr_array(self._time_points_slicer)}.")
        #
        # self._obs_slicer = obs_slicer
        # generalLogger.debug(f"  2. Obs slicer is : {repr_array(self._obs_slicer)}.")
        #
        # self._var_slicer = var_slicer
        # generalLogger.debug(f"  3. Var slicer is : {repr_array(self._var_slicer)}.")

    def __getitem__(self, array_name: str) -> dataframe.ViewTemporalDataFrame:
        """
        Get a specific Array in this view.
        :param array_name: the name of the Array to get
        """
        return self._view_array_container[array_name]

    # def __setitem__(self, array_name: str, values: ArrayLike_3D) -> None:
    #     """
    #     Set values for a specific Array in this view with an array-like object.
    #     Shapes of the view and of the array-like object must match.
    #     :param array_name: the name of the Array for which to set values
    #     :param values: an array-like object of values with shape matching the view's.
    #     """
    #     # TODO : update
    #     if not isinstance(values, np.ndarray):
    #         raise VTypeError("Values must be a 3D array-like object (numpy arrays)")
    #
    #     elif not values.shape == (len(self._time_points_slicer), len(self._obs_slicer), len(self._var_slicer)):
    #         raise ShapeError(f"Cannot set values, array-like object {values.shape} should have shape "
    #                          f"({len(self._time_points_slicer)}, {len(self._obs_slicer)}, {len(self._var_slicer)})")
    #
    #     else:
    #         self._array_container[array_name][np.ix_(self._time_points_slicer,
    #         self._obs_slicer, self._var_slicer)] = values

    def __len__(self) -> int:
        """
        Get the length of this ViewVLayerArrayContainer (the number of viewed layers).
        """
        return len(self._view_array_container)

    @property
    def shape(self) -> Tuple[int, int, List[int], int]:
        """
        The shape is computed from the shape of the Arrays viewed in this ViewVLayerArrayContainer.
        See __len__ for getting the number of Arrays it views.
        :return: shape of the contained Arrays.
        """
        _first_TDF: dataframe.ViewTemporalDataFrame = self[list(self.keys())[0]]
        _shape_TDF = _first_TDF.shape
        _shape = len(self), _shape_TDF[0], _shape_TDF[1], _shape_TDF[2]
        return _shape

    def dict_copy(self) -> Dict[str, DataFrame]:
        """
        Dictionary of keys and data items in this ArrayContainer.
        :return: Dictionary of this ArrayContainer.
        """
        return {k: v.copy() for k, v in self.items()}


# class ViewVAxisArrayContainer(ViewVBaseArrayContainer):
#     """
#     A view of a VAxisArrayContainer object.
#     """
#
#     def __init__(self, arrays: VAxisArrayContainer, time_points_slicer: np.ndarray, axis_slicer: np.ndarray):
#         """
#         :param arrays: a VAxisArrayContainer object to build a view on
#         :param axis_slicer: the list of observations/variables to view
#         :param time_points_slicer: the list of time points to view
#         """
#         super().__init__(arrays)
#         self._axis_slicer = axis_slicer
#         self._time_points_slicer = time_points_slicer
#
#     def __getitem__(self, array_name: str) -> ArrayLike_3D:
#         """
#         Get a specific Array in this view.
#         :param array_name: the name of the Array to get
#         """
#         return self._arrays[array_name][np.ix_(self._time_points_slicer, self._axis_slicer)]
#
#     def __setitem__(self, array_name: str, values: ArrayLike_3D) -> None:
#         """
#         Set values for a specific Array in this view with an array-like object.
#         Shapes of the view and of the array-like object must match.
#         :param array_name: the name of the Array for which to set values
#         :param values: an array-like object of values with shape matching the view's.
#         """
#         array_shape = (len(self._time_points_slicer), len(self._axis_slicer), self._arrays[array_name].shape[2])
#
#         if not isinstance(values, np.ndarray):
#             raise VTypeError("Values must be a 3D array-like object (numpy arrays)")
#
#         elif not values.shape == array_shape:
#             raise ShapeError(f"Cannot set values, array-like object {values.shape} should have shape {array_shape}")
#
#         else:
#             self._arrays[array_name][np.ix_(self._time_points_slicer, self._axis_slicer)] = values
#
#
# class ViewVPairwiseArrayContainer(ViewVBaseArrayContainer):
#     """
#     A view of a VPairwiseArrayContainer object.
#     """
#
#     def __init__(self, arrays: VPairwiseArrayContainer, axis_slicer: np.ndarray):
#         """
#         :param arrays: a VPairwiseArrayContainer object to build a view on
#         :param axis_slicer: the list of observations/variables to view
#         """
#         super().__init__(arrays)
#         self._axis_slicer = axis_slicer
#
#     def __getitem__(self, array_name: str) -> ArrayLike_2D:
#         """
#         Get a specific Array in this view.
#         :param array_name: the name of the Array to get
#         """
#         return self._arrays[array_name][np.ix_(self._axis_slicer, self._axis_slicer)]
#
#     def __setitem__(self, array_name: str, values: ArrayLike_2D) -> None:
#         """
#         Set values for a specific Array in this view with an array-like object.
#         Shapes of the view and of the array-like object must match.
#         :param array_name: the name of the Array for which to set values
#         :param values: an array-like object of values with shape matching the view's.
#         """
#         array_shape = (len(self._axis_slicer), len(self._axis_slicer))
#
#         if not isinstance(values, np.ndarray):
#             raise VTypeError("Values must be a 2D array-like object "
#                              "(pandas DataFrame or numpy array)")
#
#         elif not values.shape == array_shape:
#             raise ShapeError(f"Cannot set values, array-like object {values.shape} should have shape {array_shape}")
#
#         else:
#             self._arrays[array_name][np.ix_(self._axis_slicer, self._axis_slicer)] = values
