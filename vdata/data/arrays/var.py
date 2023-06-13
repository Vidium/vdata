from __future__ import annotations

from typing import Collection, cast

import ch5mpy as ch
import pandas as pd

from vdata._typing import IFS, AnyDictLike, DictLike, NDArray_IFS
from vdata.data.arrays.base import VBaseArrayContainer
from vdata.data.arrays.view import VBaseArrayContainerView, get_var_hash
from vdata.IO import (
    IncoherenceError,
    ShapeError,
    generalLogger,
)
from vdata.utils import first
from vdata.vdataframe import VDataFrame


class VVarmArrayContainer(VBaseArrayContainer[VDataFrame, pd.DataFrame]):
    """
    Class for varm.
    This object contains any number of DataFrames, with shape (n_var, any).
    The DataFrames can be accessed from the parent VData object by :
        VData.varm[<array_name>])
    """

    # region magic methods
    def _check_init_data(self, data: AnyDictLike[VDataFrame]) -> AnyDictLike[VDataFrame]:
        """
        Function for checking, at VVarmArrayContainer creation, that the supplied data has the correct format :
            - the index of the DataFrames in 'data' match the index of the parent VData's var DataFrame.
        """
        if not len(data):
            generalLogger.debug("  No data was given.")
            return dict()

        generalLogger.debug("  Data was found.")
        
        _index = self._vdata.var.index
        _data: DictLike[VDataFrame] = {} if not isinstance(data, ch.H5Dict) else data

        for DF_index, DF in data.items():
            if not _index.equals(DF.index):
                raise IncoherenceError(f"Index of DataFrame '{DF_index}' does not  match var's index. ({_index})")

            if isinstance(data, dict):
                _data[str(DF_index)] = VDataFrame(DF)

        generalLogger.debug("  Data was OK.")
        return _data

    def __setitem__(self, key: str, value: VDataFrame | pd.DataFrame) -> None:
        """
        Set a specific DataFrame in _data. The given DataFrame must have the correct shape.

        Args:
            key: key for storing a DataFrame in this VVarmArrayContainer.
            value: a DataFrame to store.
        """
        if not isinstance(value, VDataFrame):
            value = VDataFrame(value)

        if not self.shape[1] == value.shape[0]:
            raise ShapeError(f"Cannot set varm '{key}' because of shape mismatch.")

        if not self._vdata.var.index.equals(value.index):
            raise ValueError("Index does not match.")

        self.data[key] = value

    # endregion
    
    # region atributes
    @property
    def shape(self) -> tuple[int, int, list[int]]:
        """
        The shape of this VVarmArrayContainer is computed from the shape of the DataFrames it contains.
        See __len__ for getting the number of TemporalDataFrames it contains.
        """
        if not len(self):
            return 0, self._vdata.n_var, []
                        
        return len(self), first(self).shape[0], [DF.shape[1] for DF in self.values()]   
        
    # endregion
    
    # region methods
    def set_index(self, values: Collection[IFS]) -> None:
        """
        Set a new index for rows and columns.

        Args:
            values: collection of new index values.
        """
        for arr in self.values():
            arr.index = pd.Index(values)

    # endregion


class VVarmArrayContainerView(VBaseArrayContainerView[VDataFrame, pd.DataFrame]):

    # region magic methods
    def __init__(self, 
                 array_container: VVarmArrayContainer,
                 var_slicer: NDArray_IFS):
        super().__init__(data={key: cast(VDataFrame, vdf.loc[var_slicer])
                               for key, vdf in array_container.items()},
                         array_container=array_container)
        
        self._var_slicer = var_slicer
        self._vdata_var_hash = get_var_hash(array_container)
    
    def __setitem__(self, key: str, value: VDataFrame | pd.DataFrame) -> None:
        """
        Set a specific data item in this view. The given data item must have the correct shape.
        
        Args:
            key: key for storing a data item in this view.
            value: a data item to store.
        """
        if not isinstance(value, VDataFrame):
            value = VDataFrame(value)
            
        if not self.shape[1] == value.shape[0]:
            raise ShapeError(f"Cannot set varm '{key}' because of shape mismatch.")

        if not pd.Index(self._var_slicer).equals(value.index):
            raise ValueError("Index does not match.")

        self.data[key] = value

    # endregion
    
    # region attributes
    @property
    def shape(self) -> tuple[int, int, list[int]]:
        """
        The shape of this view is computed from the shape of the Arrays it contains.
        See __len__ for getting the number of Arrays it contains.
        """
        if not len(self):
            return 0, len(self._var_slicer), []
        
        return len(self), len(self._var_slicer), [DF.shape[1] for DF in self.values()]

    # endregion
    
    # region methods
    def _check_data_has_not_changed(self) -> None:
        if get_var_hash(self._array_container) != self._vdata_var_hash:
            raise ValueError("View no longer valid since parent's VData has changed.")
    
    def set_index(self, values: Collection[IFS]) -> None:
        """
        Set a new index for rows and columns.

        Args:
            values: collection of new index values.
        """
        for arr in self.values():
            arr.index = pd.Index(values)

    # endregion

class VVarpArrayContainer(VBaseArrayContainer[VDataFrame, pd.DataFrame]):
    """
    Class for varp.
    This object contains any number of DataFrames, with shape (n_var, n_var).
    The DataFrames can be accessed from the parent VData object by :
        VData.varp[<array_name>])
    """

    # region magic methods
    def _check_init_data(self, data: AnyDictLike[VDataFrame]) -> AnyDictLike[VDataFrame]:
        """
        Function for checking, at ArrayContainer creation, that the supplied data has the correct format :
            - the index and column names of the DataFrames in 'data' match the index of the parent VData's var
            DataFrame.
        """
        if not len(data):
            generalLogger.debug("  No data was given.")
            return dict()

        generalLogger.debug("  Data was found.")
        
        _index = self._vdata.var.index
        _data: DictLike[VDataFrame] = {} if not isinstance(data, ch.H5Dict) else data

        for DF_index, DF in data.items():
            if not _index.equals(DF.index):
                raise IncoherenceError(f"Index of DataFrame '{DF_index}' does not  match var's index. ({_index})")

            if not _index.equals(DF.columns):
                raise IncoherenceError(
                    f"Columns of DataFrame '{DF_index}' do not  match var's index. ({_index})")

            if isinstance(data, dict):
                _data[str(DF_index)] = VDataFrame(DF)

        generalLogger.debug("  Data was OK.")
        return _data

    def __setitem__(self, key: str, value: VDataFrame | pd.DataFrame) -> None:
        """
        Set a specific DataFrame in _data. The given DataFrame must have the correct shape.

        Args:
            key: key for storing a DataFrame in this VVarpArrayContainer.
            value: a DataFrame to store.
        """
        if not isinstance(value, VDataFrame):
            value = VDataFrame(value)

        if not self.shape[1:] == value.shape:
            raise ShapeError(f"Cannot set varp '{key}' because of shape mismatch.")

        if not self._vdata.var.index.equals(value.index):
            raise ValueError("Index does not match.")

        if not self._vdata.var.index.equals(value.columns):
            raise ValueError("column names do not match.")

        self.data[key] = value

    # endregion
    
    # region attributes
    @property
    def shape(self) -> tuple[int, int, int]:
        """
        The shape of this VVarpArrayContainer is computed from the shape of the DataFrames it contains.
        See __len__ for getting the number of TemporalDataFrames it contains.
        """
        if not len(self):
            return 0, self._vdata.n_var, self._vdata.n_var
            
        return len(self), first(self).shape[0], first(self).shape[1]

    # endregion
    
    # region methods
    def set_index(self, values: Collection[IFS]) -> None:
        """Set a new index for rows and columns."""
        for arr in self.values():
            arr.index = pd.Index(values)
            arr.columns = pd.Index(values)

    # endregion

class VVarpArrayContainerView(VBaseArrayContainerView[VDataFrame, pd.DataFrame]):

    # region magic methods
    def __init__(self,
                 array_container: VVarpArrayContainer,
                 var_slicer: NDArray_IFS):
        super().__init__(data={key: cast(VDataFrame, vdf.loc[var_slicer, var_slicer])
                               for key, vdf in array_container.items()}, 
                         array_container=array_container)
        
        self._var_slicer = var_slicer
        self._vdata_var_hash = get_var_hash(array_container)

    def __setitem__(self, key: str, value: VDataFrame | pd.DataFrame) -> None:
        """
        Set a specific data item in this view. The given data item must have the correct shape.
        
        Args:
            key: key for storing a data item in this view.
            value: a data item to store.
        """
        if not isinstance(value, VDataFrame):
            value = VDataFrame(value)

        if not self.shape[1:] == value.shape:
            raise ShapeError(f"Cannot set varp '{key}' because of shape mismatch.")

        _index = pd.Index(self._var_slicer)

        if not _index.equals(value.index):
            raise ValueError("Index does not match.")

        if not _index.equals(value.columns):
            raise ValueError("column names do not match.")

        self[key] = value

    # endregion
    
    # region attributes
    @property
    def shape(self) -> tuple[int, int, int]:
        """
        The shape of this view is computed from the shape of the Arrays it contains.
        See __len__ for getting the number of Arrays it contains.
        """
        return len(self), len(self._var_slicer), len(self._var_slicer)

    # endregion

    # region methods
    def _check_data_has_not_changed(self) -> None:
        if get_var_hash(self._array_container) != self._vdata_var_hash:
            raise ValueError("View no longer valid since parent's VData has changed.")
    
    
    def set_index(self, values: Collection[IFS]) -> None:
        """Set a new index for rows and columns."""
        for arr in self.values():
            arr.index = pd.Index(values)
            arr.columns = pd.Index(values)
            
    # endregion