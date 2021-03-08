# coding: utf-8
# Created on 04/03/2021 15:14
# Author : matteo

# ====================================================
# imports
import h5py
import pandas as pd
from pandas._typing import Axes, Dtype
from typing import Optional, Collection


# ====================================================
# code

class VDataFrame(pd.DataFrame):
    """
    Simple wrapper around pandas DataFrames for managing index and columns modification when the DataFrame is read
    from a .h5 file.
    """

    _internal_names_set = {"_file"} | pd.DataFrame._internal_names_set

    def __init__(self,
                 data=None,
                 index: Optional[Axes] = None,
                 columns: Optional[Axes] = None,
                 dtype: Optional[Dtype] = None,
                 copy: bool = False,
                 file: Optional[h5py.Group] = None):
        """
        :param file: an optional h5py group where this VDataFrame is read from.
        """
        super().__init__(data, index, columns, dtype, copy)

        self._file = file

    @property
    def is_backed(self) -> bool:
        """
        Is this VDataFrame backed on a .h5 file ?
        :return: is this VDataFrame backed on a .h5 file ?
        """
        return self._file is not None

    @property
    def index(self) -> pd.Index:
        """
        Get the index.
        """
        return super().index

    @index.setter
    def index(self, values: Collection) -> None:
        """
        Set the index (and write modifications to .h5 file if backed).
        :param values: new index to set.
        """
        self._set_axis(1, pd.Index(values))

        if self.is_backed:
            self._file.attrs["index"] = list(self.index)

    @property
    def columns(self) -> pd.Index:
        """
        Get the columns.
        """
        return super().columns

    @columns.setter
    def columns(self, values: Collection) -> None:
        """
        Set the columns (and write modifications to .h5 file if backed).
        :param values: new column names to set.
        """
        self._set_axis(0, pd.Index(values))

        if self.is_backed:
            self._file.attrs["column_order"] = list(self.index)
