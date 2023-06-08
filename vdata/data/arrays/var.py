from __future__ import annotations

from typing import Any, Collection

import ch5mpy as ch
import pandas as pd

import vdata
from vdata.data.arrays.base import VBase2DArrayContainer, VBase2DArrayContainerView
from vdata.IO import (
    IncoherenceError,
    ShapeError,
    VClosedFileError,
    VReadOnlyError,
    generalLogger,
)
from vdata.vdataframe import VDataFrame


class VVarmArrayContainer(VBase2DArrayContainer):
    """
    Class for varm.
    This object contains any number of DataFrames, with shape (n_var, any).
    The DataFrames can be accessed from the parent VData object by :
        VData.varm[<array_name>])
    """

    def _check_init_data(self, data: dict[str, pd.DataFrame] | None) -> dict[str, VDataFrame]:
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
            _index = self._parent.var.index
            _data = {}

            for DF_index, DF in data.items():
                # check that indexes match
                if not _index.equals(DF.index):
                    raise IncoherenceError(f"Index of DataFrame '{DF_index}' does not  match var's index. ({_index})")

                _data[DF_index] = VDataFrame(DF)

            generalLogger.debug("  Data was OK.")
            return _data

    def __getitem__(self, item: str) -> VDataFrame:
        """
        Get a specific DataFrame stored in this VVarmArrayContainer.
        :param item: key in _data linked to a DataFrame.
        :return: DataFrame stored in _data under the given key.
        """
        return super().__getitem__(item)

    def __setitem__(self, 
                    key: str,
                    value: VDataFrame | pd.DataFrame) -> None:
        """
        Set a specific DataFrame in _data. The given DataFrame must have the correct shape.

        Args:
            key: key for storing a DataFrame in this VVarmArrayContainer.
            value: a DataFrame to store.
        """
        if self.is_closed:
            raise VClosedFileError

        if self.is_read_only:
            raise VReadOnlyError

        if not isinstance(value, (pd.DataFrame, VDataFrame)):
            raise TypeError(f"Cannot set varm '{key}' from non pandas DataFrame object.")

        if isinstance(value, pd.DataFrame):
            value = VDataFrame(value)

        if not self.shape[1] == value.shape[0]:
            raise ShapeError(f"Cannot set varm '{key}' because of shape mismatch.")

        if not self._parent.var.index.equals(value.index):
            raise ValueError("Index does not match.")

        self._data[key] = value

    @property
    def shape(self) -> tuple[int, int, list[int]]:
        """
        The shape of this VVarmArrayContainer is computed from the shape of the DataFrames it contains.
        See __len__ for getting the number of TemporalDataFrames it contains.
        :return: the shape of this VVarmArrayContainer.
        """
        if len(self):
            _first_DF = list(self.values())[0]
            return len(self), _first_DF.shape[0], [DF.shape[1] for DF in self.values()]

        else:
            return 0, self._parent.n_var, []


class VVarmArrayContainerView(VBase2DArrayContainerView):

    def __setitem__(self, key: str, value: pd.DataFrame) -> None:
        """
        Set a specific data item in this view. The given data item must have the correct shape.
        :param key: key for storing a data item in this view.
        :param value: a data item to store.
        """
        if not isinstance(value, pd.DataFrame):
            raise TypeError(f"Cannot set varm view '{key}' from non pandas DataFrame object.")

        if not self.shape[1] == value.shape[0]:
            raise ShapeError(f"Cannot set varm '{key}' because of shape mismatch.")

        if not pd.Index(self._var_slicer).equals(value.index):
            raise ValueError("Index does not match.")

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
    def data(self) -> dict[str, VDataFrame]:
        """
        Data of this view.
        :return: the data of this view.
        """
        return {key: DF.loc[self._var_slicer] for key, DF in self._array_container.items()}


class VVarpArrayContainer(VBase2DArrayContainer):
    """
    Class for varp.
    This object contains any number of DataFrames, with shape (n_var, n_var).
    The DataFrames can be accessed from the parent VData object by :
        VData.varp[<array_name>])
    """

    def __init__(self, 
                 parent: vdata.VData,
                 data: dict[str, VDataFrame] | None = None):
        """
        Args:
            parent: the parent VData object this VVarmArrayContainer is linked to.
            data: a dictionary of DataFrames in this VVarmArrayContainer.
        """
        super().__init__(parent, data)

    def _check_init_data(self, data: dict[str, pd.DataFrame] | None) -> dict[str, VDataFrame]:
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
            _index = self._parent.var.index
            _data = {}

            for DF_index, DF in data.items():
                # check that indexes match
                if not _index.equals(DF.index):
                    raise IncoherenceError(f"Index of DataFrame '{DF_index}' does not  match var's index. ({_index})")

                # check that columns match
                if not _index.equals(DF.columns):
                    raise IncoherenceError(
                        f"Columns of DataFrame '{DF_index}' do not  match var's index. ({_index})")

                _data[DF_index] = VDataFrame(DF, 
                                             file=self._file[DF_index] if isinstance(self._file, ch.Group) else None)

            generalLogger.debug("  Data was OK.")
            return _data

    def __getitem__(self, item: str) -> VDataFrame:
        """
        Get a specific DataFrame stored in this VVarpArrayContainer.

        Args:
            item: key in _data linked to a DataFrame.

        Returns:
            DataFrame stored in _data under the given key.
        """
        return super().__getitem__(item)

    def __setitem__(self, key: str, value: VDataFrame | pd.DataFrame) -> None:
        """
        Set a specific DataFrame in _data. The given DataFrame must have the correct shape.

        Args:
            key: key for storing a DataFrame in this VVarpArrayContainer.
            value: a DataFrame to store.
        """
        if self.is_closed:
            raise VClosedFileError

        if self.is_read_only:
            raise VReadOnlyError

        if not isinstance(value, (pd.DataFrame, VDataFrame)):
            raise TypeError(f"Cannot set varp '{key}' from non pandas DataFrame object.")

        if isinstance(value, pd.DataFrame):
            value = VDataFrame(value)

        if not self.shape[1:] == value.shape:
            raise ShapeError(f"Cannot set varp '{key}' because of shape mismatch.")

        if not self._parent.var.index.equals(value.index):
            raise ValueError("Index does not match.")

        if not self._parent.var.index.equals(value.columns):
            raise ValueError("column names do not match.")

        self._data[key] = value

    @property
    def shape(self) -> tuple[int, int, int]:
        """
        The shape of this VVarpArrayContainer is computed from the shape of the DataFrames it contains.
        See __len__ for getting the number of TemporalDataFrames it contains.
        :return: the shape of this VVarpArrayContainer.
        """
        if len(self):
            _first_DF = self[list(self.keys())[0]]
            return len(self), _first_DF.shape[0], _first_DF.shape[1]

        else:
            return 0, self._parent.n_var, self._parent.n_var

    def set_index(self, values: Collection[Any]) -> None:
        """
        Set a new index for rows and columns.

        Args:
            values: collection of new index values.
        """
        for arr in self.values():

            arr.index = values
            arr.columns = values


class VVarpArrayContainerView(VBase2DArrayContainerView):

    def __getitem__(self, item: str) -> VDataFrame:
        """
        Get a specific data item stored in this view.
        :param item: key in _data linked to a data item.
        :return: data item stored in _data under the given key.
        """
        return self._array_container[item].loc[self._var_slicer, self._var_slicer]

    def __setitem__(self, key: str, value: pd.DataFrame) -> None:
        """
        Set a specific data item in this view. The given data item must have the correct shape.
        :param key: key for storing a data item in this view.
        :param value: a data item to store.
        """
        if not isinstance(value, pd.DataFrame):
            raise TypeError(f"Cannot set varp view '{key}' from non pandas DataFrame object.")

        if not self.shape[1:] == value.shape:
            raise ShapeError(f"Cannot set varp '{key}' because of shape mismatch.")

        _index = pd.Index(self._var_slicer)

        if not _index.equals(value.index):
            raise ValueError("Index does not match.")

        if not _index.equals(value.columns):
            raise ValueError("column names do not match.")

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
    def data(self) -> dict[str, VDataFrame]:
        """
        Data of this view.
        :return: the data of this view.
        """
        return {key: DF.loc[self._var_slicer, self._var_slicer] for key, DF in self._array_container.items()}
