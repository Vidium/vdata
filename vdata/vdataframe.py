# coding: utf-8
# Created on 04/03/2021 15:14
# Author : matteo

"""
VDataFrame wrapper around pandas DataFrames.
"""

# ====================================================
# imports
from typing import Optional, Collection, Sequence, Union

import numpy as np
import pandas as pd
from pandas._typing import Axes, Dtype

from typing import Literal

from .IO import VTypeError, VValueError
from .h5pickle import File, Group


# ====================================================
# code
class VDataFrame(pd.DataFrame):
    """
    Simple wrapper around pandas DataFrames for managing index and columns modification when the DataFrame is read
    from a h5 file.
    """

    _internal_names_set = {"_file"} | pd.DataFrame._internal_names_set

    def __init__(self,
                 data=None,
                 index: Optional[Axes] = None,
                 columns: Optional[Axes] = None,
                 dtype: Optional[Dtype] = None,
                 copy: bool = False,
                 file: Optional[Union[File, Group]] = None):
        """
        Args:
            file: an optional h5py group where this VDataFrame is read from.
        """
        super().__init__(data, index, columns, dtype, copy)

        self._file = file

    @property
    def is_backed(self) -> bool:
        """
        Is this VDataFrame backed on a h5 file ?

        Returns:
            Is this VDataFrame backed on a h5 file ?
        """
        return self._file is not None

    @property
    def file(self) -> Optional[Union[File, Group]]:
        """
        Get the h5 file this VDataFrame is backed on.
        :return: the h5 file this VDataFrame is backed on.
        """
        return self._file

    @file.setter
    def file(self, new_file: Union[File, Group]) -> None:
        """
        Set the h5 file to back this VDataFrame on.

        Args:
            new_file: a h5 file to back this VDataFrame on.
        """
        if not isinstance(new_file, (File, Group)):
            raise VTypeError(f"Cannot back this VDataFrame with an object of type '{type(new_file)}'.")

        self._file = new_file

    @property
    def index(self) -> pd.Index:
        """
        Get the index.
        """
        return super().index

    @index.setter
    def index(self, values: Collection) -> None:
        """
        Set the index (and write modifications to h5 file if backed).
        Args:
            values: new index to set.
        """
        self._set_axis(1, pd.Index(values))

        if self._file is not None and self._file.file.mode == 'r+':
            self._file["index"][()] = list(values)
            self._file.file.flush()

    @property
    def columns(self) -> pd.Index:
        """
        Get the columns.
        """
        return super().columns

    @columns.setter
    def columns(self, values: Sequence) -> None:
        """
        Set the columns (and write modifications to h5 file if backed).

        Args:
            values: new column names to set.
        """
        if self._file is not None and self._file.file.mode == 'r+':
            self._file.attrs["column_order"] = list(values)

            for col_index, col in enumerate(values):
                self._file.move(self.axes[1][col_index], str(col))

            self._file.file.flush()

        self._set_axis(0, pd.Index(values))


def _check_parent_has_not_changed(func):
    def wrapper(*args, **kwargs):
        self = args[0]

        if hash(tuple(object.__getattribute__(self, '_parent').index)) != \
                object.__getattribute__(self, '_parent_index_hash') or \
                hash(tuple(object.__getattribute__(self, '_parent').columns)) != \
                object.__getattribute__(self, '_parent_columns_hash'):
            raise VValueError("View no longer valid since parent's VDataFrame has changed.")

        return func(*args, **kwargs)

    return wrapper


class ViewVDataFrame:
    """
    View of a VDataFrame.
    """

    def __init__(self,
                 parent: VDataFrame,
                 index_slicer: Union[np.ndarray, slice] = slice(None),
                 column_slicer: Union[np.ndarray, slice] = slice(None)):
        """
        Args:
            parent: TODO
            index_slicer:
            column_slicer:
        """
        object.__setattr__(self, '_parent', parent)

        object.__setattr__(self, '_parent_index_hash', hash(tuple(parent.index)))
        object.__setattr__(self, '_parent_columns_hash', hash(tuple(parent.columns)))

        object.__setattr__(self, '_DataFrame', parent.loc[index_slicer, column_slicer])

    def __repr__(self) -> str:
        return repr(self._DataFrame)

    @property
    def is_backed(self) -> Literal[False]:
        """
        Is this view of a VDataFrame backed on a h5 file ?

        Returns:
            False
        """
        return False

    @_check_parent_has_not_changed
    def __getattr__(self, item):
        return getattr(self._DataFrame, item)

    @_check_parent_has_not_changed
    def __setattr__(self, key, value):
        setattr(self._DataFrame, key, value)

    @_check_parent_has_not_changed
    def __delattr__(self, item):
        delattr(self._DataFrame, item)

    @_check_parent_has_not_changed
    def __getitem__(self, item):
        return self._DataFrame.__getitem__(item)

    @_check_parent_has_not_changed
    def __setitem__(self, key, value):
        self._DataFrame.__setitem__(key, value)

    @_check_parent_has_not_changed
    def __delitem__(self, key):
        self._DataFrame.__delitem__(key)

    @_check_parent_has_not_changed
    def __len__(self):
        return len(self._DataFrame)

    @_check_parent_has_not_changed
    def to_pandas(self) -> pd.DataFrame:
        return self._DataFrame
