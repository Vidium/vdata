from __future__ import annotations

import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Callable, ItemsView, Iterator, KeysView, MutableMapping, TypeVar, ValuesView

import ch5mpy as ch
import numpy as np
import numpy.typing as npt
import pandas as pd

import vdata
import vdata.data.arrays as arrays
from vdata._typing import IFS_NP
from vdata.IO import VClosedFileError, VReadOnlyError, generalLogger
from vdata.tdf import TemporalDataFrame, TemporalDataFrameView
from vdata.timepoint import TimePointArray
from vdata.utils import first
from vdata.vdataframe import VDataFrame


class TimedDict(dict, MutableMapping[str, VDataFrame]):
        
    # region magic methods
    def __init__(self,
                 parent: 'vdata.VData',
                 **kwargs: Any):
        dict.__init__(kwargs)

        self._parent = parent

    def __getitem__(self,
                    key: str) -> VDataFrame:
        vdf = dict.__getitem__(self, key[1])

        if isinstance(key[0], slice):
            return vdf

        index = self._parent.obs.index_at(key[0])
        return vdf[index, index]

    def __setitem__(self,
                    key: str,
                    value: VDataFrame) -> None:
        dict.__setitem__(self, key, value)

    # endregion


D = TypeVar('D', TemporalDataFrame, VDataFrame, TimedDict)
D_view = TypeVar('D_view', TemporalDataFrameView, VDataFrame, TimedDict)
TD_ = TypeVar('TD_', bound=TimedDict)


# Base Containers -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
class VBaseArrayContainer(ABC, MutableMapping[str, D]):
    """
    Base abstract class for ArrayContainers linked to a VData object (obsm, obsp, varm, varp, layers).
    All Arrays have a '_parent' attribute for linking them to a VData and a '_data' dictionary
    attribute for storing 2D/3D arrays.
    """

    # region magic methods
    def __init__(self, 
                 parent: vdata.VData, 
                 data: dict[str, D] | ch.H5Dict[D] | None):
        """
        Args:
            parent: the parent VData object this ArrayContainer is linked to.
            data: a dictionary of data items (pandas DataFrames, TemporalDataFrames or dictionaries of pandas
            DataFrames) to store in this ArrayContainer.
        """
        generalLogger.debug(f"== Creating {self.__class__.__name__}. ==========================")

        self._parent = parent
        self._data: dict[str, D] | ch.H5Dict[D] = self._check_init_data(data)

    @abstractmethod
    def _check_init_data(self, data: dict[str, D] | ch.H5Dict[D] | None) -> dict[str, D] | ch.H5Dict[D]:
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
        if not len(self):
            return f"Empty {type(self).__name__}."

        return f"{type(self).__name__} with keys: {', '.join(map(repr, self.keys()))}."

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

        if item not in self.keys():
            raise AttributeError(f"This {type(self).__name__} has no attribute '{item}'")

        return self._data[item]

    @abstractmethod
    def __setitem__(self, key: str, value: D) -> None:
        """
        Set a specific data item in _data. The given data item must have the correct shape.

        Args:
            key: key for storing a data item in this ArrayContainer.
            value: a data item to store.
        """
        pass

    def __delitem__(self, key: str) -> None:
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
        return len(self._data.keys())

    def __iter__(self) -> Iterator[str]:
        """
        Iterate on this ArrayContainer's keys.
        :return: an iterator over this ArrayContainer's keys.
        """
        if self.is_closed:
            raise VClosedFileError

        return iter(self.keys())

    # endregion

    # region predicates
    @property
    def is_closed(self) -> bool:
        """
        Is the parent's file closed ?
        """
        return isinstance(self._data, ch.H5Dict) and self._data.is_closed

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

    # endregion

    # region attributes
    @property
    def name(self) -> str:
        return type(self).__name__[1:-14].lower()

    @property
    @abstractmethod
    def shape(self) -> tuple[int, int, int] | \
                       tuple[int, int, list[int]] | \
                       tuple[int, int, list[int], int] | \
                       tuple[int, int, list[int], list[int]]:
        """
        The shape of this ArrayContainer is computed from the shape of the Arrays it contains.
        See __len__ for getting the number of Arrays it contains.

        Returns:
            The shape of this ArrayContainer.
        """
        pass

    @property
    def data(self) -> dict[str, D]:
        """
        Data of this ArrayContainer.

        Returns:
            The data of this ArrayContainer.
        """
        if self.is_closed:
            raise VClosedFileError

        return self._data

    # endregion

    # region methods
    @abstractmethod
    def update_dtype(self, dtype: npt.DTypeLike) -> None:
        """
        Update the data type of Arrays stored in this ArrayContainer.

        Args:
            dtype: the new data type.
        """
        pass

    def keys(self) -> KeysView[str]:
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

    def items(self) -> ItemsView[str, D]:
        """
        ItemsView of pairs of keys and data items in this ArrayContainer.

        Returns:
            ItemsView of this ArrayContainer.
        """
        if self.is_closed:
            raise VClosedFileError

        return self._data.items()

    @abstractmethod
    def dict_copy(self) -> dict[str, D]:
        """
        Dictionary of keys and data items in this ArrayContainer.

        Returns:
            Dictionary of this ArrayContainer.
        """
        pass

    @abstractmethod
    def to_csv(self,
               directory: Path,
               sep: str = ",",
               na_rep: str = "",
               index: bool = True,
               header: bool = True,
               spacer: str = '') -> None:
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

    # endregion


def _check_parent_has_not_changed(func: Callable[..., Any]) -> Callable[..., Any]:
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        self = args[0]

        if isinstance(self, arrays.VLayersArrayContainerView):
            if hash(tuple(self._array_container._parent.timepoints.value.values)) != self._parent_timepoints_hash or \
                    hash(tuple(self._array_container._parent.obs.index)) != self._parent_obs_hash or \
                    hash(tuple(self._array_container._parent.var.index)) != self._parent_var_hash:
                raise ValueError("View no longer valid since parent's VData has changed.")

        elif isinstance(self, arrays.VObsmArrayContainerView):
            if hash(tuple(self._array_container._parent.timepoints.value.values)) != self._parent_timepoints_hash or \
                    hash(tuple(self._array_container._parent.obs.index)) != self._parent_obs_hash:
                raise ValueError("View no longer valid since parent's VData has changed.")

        return func(*args, **kwargs)
    return wrapper


class VBaseArrayContainerView(ABC, MutableMapping[str, D_view]):
    """
    A base abstract class for views of VBaseArrayContainers.
    This class is used to create views on array containers.
    """

    # region magic methods
    def __init__(self, array_container: VBaseArrayContainer[TemporalDataFrame] | 
                                        VBaseArrayContainer[VDataFrame] | 
                                        VBaseArrayContainer[TimedDict]):
        """
        Args:
            array_container: a VBaseArrayContainer object to build a view on.
        """
        generalLogger.debug(f"== Creating {type(self).__name__}. ================================")

        self._array_container = array_container

    @_check_parent_has_not_changed
    def __repr__(self) -> str:
        """
        Description for this view  to print.
        :return: a description of this view.
        """
        return f"View of {self._array_container}"

    @abstractmethod
    def __getitem__(self, item: str) -> D_view:
        """
        Get a specific data item stored in this view.
        :param item: key in _data linked to a data item.
        :return: data item stored in _data under the given key.
        """
        pass

    @abstractmethod
    def __setitem__(self, key: str, value: D_view) -> None:
        """
        Set a specific data item in this view. The given data item must have the correct shape.
        :param key: key for storing a data item in this view.
        :param value: a data item to store.
        """
        pass
    
    def __delitem__(self, __key: str) -> None:
        raise TypeError('Cannot delete column from view.')

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

    # endregion
    
    # region attributes
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
    def shape(self) -> tuple[int, int, int] | \
        tuple[int, int, list[int]] | \
        tuple[int, int, list[int], int] | \
        tuple[int, int, list[int], list[int]]:
        """
        The shape of this view is computed from the shape of the Arrays it contains.
        See __len__ for getting the number of Arrays it contains.
        :return: the shape of this view.
        """
        pass

    @property
    @abstractmethod
    def data(self) -> dict[str, D_view]:
        """
        Data of this view.
        :return: the data of this view.
        """
        pass

    # endregion
    
    # region methods
    @_check_parent_has_not_changed
    def keys(self) -> KeysView[str]:
        """
        KeysView of keys for getting the data items in this view.
        :return: KeysView of this view.
        """
        return self._array_container.keys()

    @_check_parent_has_not_changed
    def values(self) -> ValuesView[D_view]:
        """
        ValuesView of data items in this view.
        :return: ValuesView of this view.
        """
        return self.data.values()

    @_check_parent_has_not_changed
    def items(self) -> ItemsView[str, D_view]:
        """
        ItemsView of pairs of keys and data items in this view.
        :return: ItemsView of this view.
        """
        return self.data.items()

    @abstractmethod
    def dict_copy(self) -> dict[str, D_view]:
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

    # endregion

class VTDFArrayContainerView(VBaseArrayContainerView[TemporalDataFrameView]):
    """
    Base abstract class for views of ArrayContainers that contain TemporalDataFrames (layers and obsm).
    It is based on VBaseArrayContainer.
    """

    # region magic methods
    def __init__(self, 
                 array_container: VBaseArrayContainer[TemporalDataFrame],
                 timepoints_slicer: TimePointArray,
                 obs_slicer: npt.NDArray[IFS_NP],
                 var_slicer: npt.NDArray[IFS_NP] | slice):
        """
        Args:
            array_container: a VBaseArrayContainer object to build a view on.
        """
        super().__init__(array_container)
        
        self._data = {}

        for key, tdf in array_container.items():
            self._data[key] = tdf[timepoints_slicer, obs_slicer, var_slicer]

        self._parent_timepoints_hash: int = hash(tuple(self._array_container._parent.timepoints.value.values))
        self._parent_obs_hash: int = hash(tuple(self._array_container._parent.obs.index))

    @_check_parent_has_not_changed
    def __getitem__(self, key: str) -> TemporalDataFrameView:
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

    # endregion
    
    # region attributes
    @property
    @_check_parent_has_not_changed
    def empty(self) -> bool:
        """
        Whether this view is empty or not.

        Returns:
            Is this view empty ?
        """
        return all([VTDF.empty for VTDF in self.values()])

    @property
    @_check_parent_has_not_changed
    def has_repeating_index(self) -> bool:
        if self.empty:
            return False

        return list(self.values())[0].has_repeating_index

    @property
    @_check_parent_has_not_changed
    def shape(self) -> tuple[int, int, list[int], int]:
        """
        The shape of this view is computed from the shape of the Arrays it contains.
        See __len__ for getting the number of Arrays it contains.

        Returns:
            The shape of this view.
        """
        if len(self):
            return len(self), *first(self.data).shape
            
            # _first_VTDF = list(self.values())[0]
            # _shape_VTDF = _first_VTDF.shape
            # return len(self), _shape_VTDF[0], _shape_VTDF[1], _shape_VTDF[2]

        else:
            return 0, 0, [], 0

    @property
    @_check_parent_has_not_changed
    def data(self) -> dict[str, TemporalDataFrameView]:
        """
        Data of this view.

        Returns:
            The data of this view.
        """
        return self._data

    # endregion
    
    # region methods
    @_check_parent_has_not_changed
    def dict_copy(self) -> dict[str, TemporalDataFrame]:
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

    # endregion

# 2D Containers -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -
class VBase2DArrayContainer(VBaseArrayContainer[VDataFrame], ABC):
    """
    Base abstract class for ArrayContainers linked to a VData object that contain DataFrames (varm and varp)
    It is based on VBaseArrayContainer and defines some functions shared by varm and varp.
    """

    # region magic methods
    def __init__(self, 
                 parent: vdata.VData, 
                 data: dict[str, pd.DataFrame] | None):
        """
        Args:
            parent: the parent VData object this ArrayContainer is linked to.
            data: a dictionary of DataFrames in this ArrayContainer.
        """
        self._file = parent.file
        
        super().__init__(parent, data)

    # endregion
    
    # region attributes
    @property
    def empty(self) -> bool:
        """
        Whether this ArrayContainer is empty or not.
        :return: is this ArrayContainer empty ?
        """
        return all([DF.empty for DF in self.values()])

    # endregion
    
    # region methods
    def update_dtype(self, type_: npt.DTypeLike) -> None:
        """
        Update the data type of TemporalDataFrames stored in this ArrayContainer.

        Args:
            type_: the new data type.
        """
        for arr_name, arr in self.items():
            self[arr_name] = arr.astype(type_)

    def dict_copy(self) -> dict[str, VDataFrame]:
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

    # endregion

class VBase2DArrayContainerView(VBaseArrayContainerView[VDataFrame], ABC):
    """
    Base abstract class for views of ArrayContainers that contain DataFrames (varm and varp)
    It is based on VBaseArrayContainer.
    """

    # region magic methods
    def __init__(self, array_container: VBaseArrayContainer[VDataFrame], var_slicer: npt.NDArray[IFS_NP]):
        """
        :param array_container: a VBaseArrayContainer object to build a view on.
        :param var_slicer: the list of variables to view.
        """
        super().__init__(array_container)

        self._var_slicer = var_slicer

    def __getitem__(self, item: str) -> VDataFrame:
        """
        Get a specific data item stored in this view.
        :param item: key in _data linked to a data item.
        :return: data item stored in _data under the given key.
        """
        return self._array_container[item].loc[self._var_slicer]

    # endregion
    
    # region attributes
    @property
    def empty(self) -> bool:
        """
        Whether this view is empty or not.
        :return: is this view empty ?
        """
        return all([DF.empty for DF in self.values()])

    # endregion
    
    # region methods
    def values(self) -> ValuesView[pd.DataFrame]:
        """
        ValuesView of data items in this view.
        :return: ValuesView of this view.
        """
        return super().values()

    def dict_copy(self) -> dict[str, VDataFrame]:
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

    # endregion

# 3D Containers -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -
class VBase3DArrayContainer(VBaseArrayContainer[TemporalDataFrame], ABC):
    """
    Base abstract class for ArrayContainers linked to a VData object that contain TemporalDataFrames (obsm and layers).
    It is based on VBaseArrayContainer and defines some functions shared by obsm and layers.
    """

    # region methods
    def __init__(self, 
                 parent: vdata.VData,
                 data: dict[str, TemporalDataFrame] | 
                       ch.H5Dict[TemporalDataFrame] | 
                       None):
        """
        Args:
            parent: the parent VData object this ArrayContainer is linked to.
            data: a dictionary of TemporalDataFrames in this ArrayContainer.
        """
        super().__init__(parent, data)

    def __setitem__(self, 
                    key: str,
                    value: TemporalDataFrame) -> None:
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

        if not np.array_equal(self._parent.timepoints_values, value.timepoints):
            raise ValueError("Time-points do not match.")

        if not np.array_equal(self._parent.obs.index, value.index):
            raise ValueError("Index does not match.")

        self._data[key] = value

    # endregion

    # region predicates
    @property
    def empty(self) -> bool:
        """
        Whether this ArrayContainer is empty or not.
        """
        return len(self._data.keys()) == 0

    @property
    def has_repeating_index(self) -> bool:
        if self.empty:
            return False

        return list(self.values())[0].has_repeating_index

    # endregion

    # region attributes
    @property
    def shape(self) -> tuple[int, int, list[int], list[int]]:
        """
        The shape of this ArrayContainer is computed from the shape of the TemporalDataFrames it contains.
        See __len__ for getting the number of TemporalDataFrames it contains.
        
        Returns:
            (<nb TDFs>, <nb time-points>, <nb obs>, <nb vars>)
        """        
        if not len(self):
            return 0, 0, [], []

        _shape_TDF = first(self._data).shape
        return len(self._data), _shape_TDF[0], _shape_TDF[1], [d.shape[2] for d in self._data.values()]

    # endregion
    
    # region methods
    def update_dtype(self, dtype: npt.DTypeLike) -> None:
        """
        Update the data type of TemporalDataFrames stored in this ArrayContainer.

        Args:
            dtype: the new data type.
        """
        for arr in self.values():
            arr.astype(dtype)               # type: ignore[operator]

    def dict_copy(self) -> dict[str, TemporalDataFrame]:
        """
        Dictionary of keys and data items in this ArrayContainer.
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

    # endregion
