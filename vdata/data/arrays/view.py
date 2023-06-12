from __future__ import annotations

from abc import ABC, abstractmethod
from typing import (
    Collection,
    Generic,
    ItemsView,
    Iterator,
    KeysView,
    TypeVar,
    ValuesView,
)

import numpy as np

import vdata.data.arrays as arrays
from vdata._typing import IFS, NDArray_IFS
from vdata.data.arrays.base import ArrayContainerMixin, D, VBaseArrayContainer
from vdata.IO import generalLogger
from vdata.tdf import TemporalDataFrame, TemporalDataFrameView
from vdata.timepoint import TimePointArray
from vdata.utils import first
from vdata.vdataframe import VDataFrame

D_view = TypeVar('D_view', TemporalDataFrameView, VDataFrame)


class VBaseArrayContainerView(ABC, ArrayContainerMixin[D_view], Generic[D_view, D]):
    """
    A base abstract class for views of VBaseArrayContainers.
    This class is used to create views on array containers.
    """

    # region magic methods
    def __init__(self, data: dict[str, D_view], array_container: VBaseArrayContainer[D]):
        """
        Args:
            array_container: a VBaseArrayContainer object to build a view on.
        """
        generalLogger.debug(f"== Creating {type(self).__name__}. ================================")

        self._data: dict[str, D_view] = data
        self._array_container: VBaseArrayContainer[D] = array_container

    def __repr__(self) -> str:
        """Description for this view  to print."""
        return f"View of {self._array_container}"

    def __getitem__(self, key: str) -> D_view:
        """Get a specific data item stored in this view."""
        return self.data[key]
    
    @abstractmethod
    def __setitem__(self, key: str, value: D_view) -> None:
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
    def data(self) -> dict[str, D_view]:
        """Data of this view."""
        if isinstance(self, arrays.VLayersArrayContainerView):
            if hash(tuple(self._array_container._vdata.timepoints.value.values)) != self._vdata_timepoints_hash or \
                    hash(tuple(self._array_container._vdata.obs.index)) != self._vdata_obs_hash or \
                    hash(tuple(self._array_container._vdata.var.index)) != self._vdata_var_hash:
                raise ValueError("View no longer valid since parent's VData has changed.")

        elif isinstance(self, arrays.VObsmArrayContainerView):
            if hash(tuple(self._array_container._vdata.timepoints.value.values)) != self._vdata_timepoints_hash or \
                    hash(tuple(self._array_container._vdata.obs.index)) != self._vdata_obs_hash:
                raise ValueError("View no longer valid since parent's VData has changed.")
        
        return self._data

    # endregion
    
    # region methods
    def keys(self) -> KeysView[str]:
        """KeysView of keys for getting the data items in this view."""
        return self.data.keys()

    def values(self) -> ValuesView[D_view]:
        """ValuesView of data items in this view."""
        return self.data.values()

    def items(self) -> ItemsView[str, D_view]:
        """ItemsView of pairs of keys and data items in this view."""
        return self.data.items()

    # endregion

class VTDFArrayContainerView(VBaseArrayContainerView[TemporalDataFrameView, TemporalDataFrame]):
    """
    Base abstract class for views of ArrayContainers that contain TemporalDataFrames (layers and obsm).
    It is based on VBaseArrayContainer.
    """

    # region magic methods
    def __init__(self, 
                 array_container: VBaseArrayContainer[TemporalDataFrame],
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
        
        self._vdata_timepoints_hash: int = hash(tuple(self._array_container._vdata.timepoints.value.values))
        self._vdata_obs_hash: int = hash(tuple(self._array_container._vdata.obs.index))
        self._vdata_var_hash: int = hash(tuple(self._array_container._vdata.var.index))

    def __setitem__(self, key: str, value: TemporalDataFrameView) -> None:
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
    def set_index(self, values: Collection[IFS], repeating_index: bool) -> None:
        """Set a new index for rows."""
        for layer in self.values():
            layer.unlock_indices()
            layer.set_index(values, repeating_index)
            layer.lock_indices()
    
    def set_columns(self, values: Collection[IFS]) -> None:
        """Set a new index for columns."""
        for layer in self.values():
            layer.unlock_columns()
            layer.columns = np.array(values)
            layer.lock_columns()
    
    # endregion
