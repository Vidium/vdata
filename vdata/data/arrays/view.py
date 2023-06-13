from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, ItemsView, Iterator, KeysView, ValuesView

import numpy as np

from vdata._typing import NDArray_IFS
from vdata.data.arrays.base import ArrayContainerMixin, D, D_copy, VBaseArrayContainer
from vdata.IO import generalLogger
from vdata.tdf import TemporalDataFrame, TemporalDataFrameView
from vdata.timepoint import TimePointArray
from vdata.utils import first


class VBaseArrayContainerView(ABC, ArrayContainerMixin[D, D_copy]):
    """
    A base abstract class for views of VBaseArrayContainers.
    This class is used to create views on array containers.
    """

    # region magic methods
    def __init__(self, data: dict[str, D], array_container: VBaseArrayContainer[D, D_copy]):
        """
        Args:
            array_container: a VBaseArrayContainer object to build a view on.
        """
        generalLogger.debug(f"== Creating {type(self).__name__}. ================================")

        self._data: dict[str, D] = data
        self._array_container: VBaseArrayContainer[D, D_copy] = array_container

    def __repr__(self) -> str:
        """Description for this view  to print."""
        return f"View of {self._array_container}"

    def __getitem__(self, key: str) -> D:
        """Get a specific data item stored in this view."""
        return self.data[key]
    
    @abstractmethod
    def __setitem__(self, key: str, value: D) -> None:
        """Set a specific data item in this view. The given data item must have the correct shape."""
    
    def __delitem__(self, __key: str) -> None:
        raise TypeError('Cannot delete data from view.')

    def __len__(self) -> int:
        """Length of this view : the number of data items in the VBaseArrayContainer."""
        return len(self.keys())

    def __iter__(self) -> Iterator[str]:
        """Iterate on this view's keys."""
        return iter(self.keys())

    # endregion
    
    # region attributes
    @property
    def name(self) -> str:
        """Name for this view."""
        return f"{self._array_container.name}_view"

    @property
    @abstractmethod
    def shape(self) -> tuple[int, int, int] | \
        tuple[int, int, list[int]] | \
        tuple[int, int, list[int], int] | \
        tuple[int, int, list[int], list[int]]:
        """
        The shape of this view is computed from the shape of the Arrays it contains.
        See __len__ for getting the number of Arrays it contains.
        """
        pass

    @property
    def data(self) -> dict[str, D]:
        """Data of this view."""
        self._check_data_has_not_changed()
        return self._data

    # endregion
    
    # region methods
    @abstractmethod
    def _check_data_has_not_changed(self) -> None:
        pass
    
    def keys(self) -> KeysView[str]:
        """KeysView of keys for getting the data items in this view."""
        return self.data.keys()

    def values(self) -> ValuesView[D]:
        """ValuesView of data items in this view."""
        return self.data.values()

    def items(self) -> ItemsView[str, D]:
        """ItemsView of pairs of keys and data items in this view."""
        return self.data.items()

    # endregion
    
    
def get_tp_hash(array_container: VBaseArrayContainer[Any, Any]) -> int:
    return hash(tuple(array_container._vdata.timepoints.value.values))


def get_obs_hash(array_container: VBaseArrayContainer[Any, Any]) -> int:
    return hash(tuple(array_container._vdata.obs.index))

    
def get_var_hash(array_container: VBaseArrayContainer[Any, Any]) -> int:
    return hash(tuple(array_container._vdata.var.index))



class VTDFArrayContainerView(VBaseArrayContainerView[TemporalDataFrame | TemporalDataFrameView, TemporalDataFrame]):
    """
    Base abstract class for views of ArrayContainers that contain TemporalDataFrames (layers and obsm).
    It is based on VBaseArrayContainer.
    """

    # region magic methods
    def __init__(self, 
                 array_container: VBaseArrayContainer[TemporalDataFrame | TemporalDataFrameView, TemporalDataFrame],
                 timepoints_slicer: TimePointArray,
                 obs_slicer: NDArray_IFS,
                 var_slicer: NDArray_IFS | slice):
        """
        Args:
            array_container: a VBaseArrayContainer object to build a view on.
            timepoints_slicer: the list of time points to view.
            obs_slicer: the list of observations to view.
            var_slicer: the list of variables to view.
        """
        super().__init__(data={key: tdf[timepoints_slicer, obs_slicer, var_slicer]
                               for key, tdf in array_container.items()},
                         array_container=array_container)
        
        self._vdata_timepoints_hash = get_tp_hash(array_container)
        self._vdata_obs_hash = get_obs_hash(array_container)
        self._vdata_var_hash = get_var_hash(array_container)

    def __setitem__(self, key: str, value: TemporalDataFrame | TemporalDataFrameView) -> None:
        """Set a specific data item in this view. The given data item must have the correct shape."""
        self.data[key] = value

    # endregion
    
    # region attributes
    @property
    def has_repeating_index(self) -> bool:
        if self.empty:
            return False

        return first(self.data).has_repeating_index

    @property
    def shape(self) -> tuple[int, int, list[int], int]:
        """
        The shape of this view is computed from the shape of the Arrays it contains.
        See __len__ for getting the number of Arrays it contains.
        """
        if not len(self):
            return 0, 0, [], 0
        
        return len(self), *first(self.data).shape

    # endregion
    
    # region methods
    def _check_data_has_not_changed(self) -> None:
        if get_tp_hash(self._array_container) != self._vdata_timepoints_hash or \
                get_obs_hash(self._array_container) != self._vdata_obs_hash or \
                get_var_hash(self._array_container) != self._vdata_var_hash:
            raise ValueError("View no longer valid since parent's VData has changed.")
    
    def set_index(self, values: NDArray_IFS, repeating_index: bool) -> None:
        """Set a new index for rows."""
        for layer in self.values():
            layer.unlock_indices()
            layer.set_index(values, repeating_index)
            layer.lock_indices()
    
    def set_columns(self, values: NDArray_IFS) -> None:
        """Set a new index for columns."""
        for layer in self.values():
            layer.unlock_columns()
            layer.columns = np.array(values)
            layer.lock_columns()
    
    # endregion
