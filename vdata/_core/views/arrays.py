# coding: utf-8
# Created on 15/01/2021 12:57
# Author : matteo

# ====================================================
# imports
import numpy as np
import abc
from typing import Tuple, Dict, Union, KeysView, ValuesView, ItemsView, List

from vdata.NameUtils import DataFrame
from ..arrays import VLayerArrayContainer  # VAxisArrayContainer, VPairwiseArrayContainer, VPairwiseArray
from ..._TDF.views import dataframe
from ..._IO import generalLogger  # VTypeError, ShapeError


# ====================================================
# code
class ViewVBaseArrayContainer(abc.ABC):
    """
    A base view of a VBaseArrayContainer.
    This class is used to create views on VLayerArrayContainer, VAxisArrays and VPairwiseArrays.
    """

    def __init__(self, view_array_container: Dict[str, dataframe.ViewTemporalDataFrame]):
        """
        :param view_array_container: a VBaseArrayContainer object to build a view on.
        """
        self._view_array_container = view_array_container

    def __repr__(self) -> str:
        """
        Description for this view of a VBaseArrayContainer object to print.
        :return: a description of this view
        """
        return f"View of {self._view_array_container}"

    def keys(self) -> Union[Tuple[()], KeysView]:
        """
        Get keys of the VBaseArrayContainer.
        """
        return self._view_array_container.keys()

    def values(self) -> Union[Tuple[()], ValuesView]:
        """
        Get values of the VBaseArrayContainer.
        """
        return self._view_array_container.values()

    def items(self) -> Union[Tuple[()], ItemsView]:
        """
        Get items of the VBaseArrayContainer.
        """
        return self._view_array_container.items()

    def dict_copy(self) -> Dict[str, Union[DataFrame]]:  # Dict[str, Union[DataFrame, VPairwiseArray]]:
        """
        Build an actual copy of this Array view in dict format.
        :return: Dictionary of (keys, ArrayLike) in this Array view.
        """
        return dict([(arr, self._view_array_container[arr]) for arr in self._view_array_container.keys()])


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
