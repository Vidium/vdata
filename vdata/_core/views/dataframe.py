# coding: utf-8
# Created on 15/01/2021 12:57
# Author : matteo

# ====================================================
# imports
import numpy as np
import pandas as pd
from typing import Collection, Optional, Union, Tuple, Any, Dict, List, NoReturn, Hashable, IO, Iterable
from typing_extensions import Literal

import vdata
from vdata.NameUtils import PreSlicer, DType
from ..NameUtils import TemporalDataFrame_internal_attributes
from ..utils import repr_array, repr_index, reformat_index, match_time_points
from ..._IO import generalLogger
from ..._IO.errors import VValueError, VAttributeError


# ==========================================
# code
class ViewTemporalDataFrame:
    """
    A view of a TemporalDataFrame, created on sub-setting operations.
    """

    _internal_attributes = TemporalDataFrame_internal_attributes + ['parent', '_tp_slicer']

    def __init__(self, parent: 'vdata.TemporalDataFrame', tp_slicer: Collection[str], index_slicer: Collection,
                 column_slicer: Collection):
        """
        :param parent: a parent TemporalDataFrame to view.
        :param tp_slicer: a collection of time points to view.
        :param index_slicer: a pandas Index of rows to view.
        :param column_slicer: a pandas Index of columns to view.
        """
        generalLogger.debug(f"\u23BE ViewTemporalDataFrame '{parent.name}':{id(self)} creation : begin "
                            f"---------------------------------------- ")

        # set attributes on init using object's __setattr__ method to avoid self's __setattr__ which would provoke bugs
        object.__setattr__(self, '_parent', parent)
        object.__setattr__(self, 'index', index_slicer)
        object.__setattr__(self, 'columns', column_slicer)

        object.__setattr__(self, '_tp_slicer', np.array(tp_slicer)[
            match_time_points(tp_slicer, self.parent_data.loc[self.index]['__TPID'].values)])

        generalLogger.debug(f"  1. Refactored time point slicer to : {repr_array(self._tp_slicer)}")

        object.__setattr__(self, 'index', np.array(index_slicer)[np.isin(index_slicer, self.parent_data[
            self.parent_data['__TPID'].isin(self._tp_slicer)].index)])

        generalLogger.debug(f"  2. Refactored index slicer to : {repr_array(self.index)}")

        generalLogger.debug(f"\u23BF ViewTemporalDataFrame '{parent.name}':{id(self)} creation : end "
                            f"---------------------------------------- ")

    def __repr__(self):
        """
        Description for this view of a TemporalDataFrame object to print.
        :return: a description of this view of a TemporalDataFrame object
        """
        if self.n_time_points:
            repr_str = ""
            for TP in self.time_points:
                repr_str += f"\033[4mTime point : {TP}\033[0m\n"
                repr_str += f"{self.one_TP_repr(TP)}\n\n"

        else:
            repr_str = f"Empty View of a TemporalDataFrame\n" \
                       f"Columns: {[col for col in self.columns]}\n" \
                       f"Index: {[idx for idx in self.index]}"

        return repr_str

    def one_TP_repr(self, time_point: str, n: Optional[int] = None, func: Literal['head', 'tail'] = 'head'):
        """
        Representation of a single time point in this TemporalDataFrame to print.
        :param time_point: the time point to represent.
        :param n: the number of rows to print. Defaults to all.
        :param func: the name of the function to use to limit the output ('head' or 'tail')
        :return: a representation of a single time point in this TemporalDataFrame object
        """
        mask = match_time_points(self.parent_data['__TPID'], time_point)
        if len(mask):
            mask &= np.array(self.index_bool)
            return repr(self.parent_data.loc[mask, self.columns].__getattr__(func)(n=n))

        else:
            repr_str = f"Empty DataFrame\n" \
                       f"Columns: {[col for col in self.columns]}\n" \
                       f"Index: {[idx for idx in self.index]}"
            return repr_str

    def __getitem__(self, index: Union[PreSlicer,
                                       Tuple[PreSlicer],
                                       Tuple[PreSlicer, PreSlicer],
                                       Tuple[PreSlicer, PreSlicer, PreSlicer]]) \
            -> 'ViewTemporalDataFrame':
        """
        Get a sub-view from this view using an index with the usual sub-setting mechanics.
        :param index: A sub-setting index.
            See TemporalDataFrame's '__getitem__' method for more details.
        """
        generalLogger.debug(f"ViewTemporalDataFrame '{self._parent.name}':{id(self)} sub-setting "
                            f"- - - - - - - - - - - - - -")
        generalLogger.debug(f'  Got index : {repr_index(index)}')

        index = reformat_index(index, self.time_points, self.index, self.columns)

        generalLogger.debug(f'  Refactored index to : {repr_index(index)}')

        return ViewTemporalDataFrame(self._parent, index[0], index[1], index[2])

    def __setitem__(self, index: Union[PreSlicer, Tuple[PreSlicer], Tuple[PreSlicer, Collection[bool]]],
                    df: Union[pd.DataFrame, 'vdata.TemporalDataFrame', 'ViewTemporalDataFrame']) -> None:
        """
        Set values in the parent TemporalDataFrame from this view with a DataFrame.
        The columns and the rows must match.
        :param index: a sub-setting index. (see __getitem__ for more details)
        :param df: a DataFrame with values to set.
        """
        self[index].set(df)

    def __getattribute__(self, attr: str) -> Any:
        """
        Get attribute from this TemporalDataFrame in obj.attr fashion.
        This is called before __getattr__.
        :param attr: an attribute's name to get.
        :return: self.attr
        """
        if attr not in ViewTemporalDataFrame._internal_attributes:
            raise AttributeError

        return object.__getattribute__(self, attr)

    def __getattr__(self, attr: str) -> Any:
        """
        Get columns in the DataFrame (this is done for maintaining the pandas DataFrame behavior).
        :param attr: an attribute's name
        :return: a column with name <attr> from the DataFrame
        """
        if attr in self.columns:
            return self._parent.loc[self.index, attr]

        else:
            return object.__getattribute__(self, attr)

    def __setattr__(self, attr: str, value: Any) -> None:
        """
        Set value for a regular attribute of for a column in the DataFrame.
        :param attr: an attribute's name
        :param value: a value to be set into the attribute
        """
        if attr in ViewTemporalDataFrame._internal_attributes:
            self._parent.__setattr__(attr, value)

        elif attr in self.columns:
            self.parent_data.loc[self.index, attr] = value

        else:
            raise AttributeError(f"'{attr}' is not a valid attribute name.")

    def __len__(self):
        """
        Returns the length of info axis.
        :return: the length of info axis.
        """
        return len(self.index)

    def set(self, df: Union[pd.DataFrame, 'vdata.TemporalDataFrame', 'ViewTemporalDataFrame']) -> None:
        """
        Set values for this ViewTemporalDataFrame.
        Values can be given in a pandas.DataFrame, a TemporalDataFrame or an other ViewTemporalDataFrame.
        To set values, the columns and indexes must match ALL columns and indexes in this ViewTemporalDataFrame.
        :param df: a pandas.DataFrame, TemporalDataFrame or ViewTemporalDataFrame with new values to set.
        """
        assert isinstance(df, (pd.DataFrame, vdata.TemporalDataFrame, ViewTemporalDataFrame)), \
            "Cannot set values from non DataFrame object."
        # This is done to prevent introduction of NaNs
        assert self.n_columns == len(df.columns), "Columns must match."
        assert self.columns.equals(df.columns), "Columns must match."
        assert len(self) == len(df), "Number of rows must match."
        assert self.index.equals(df.index), "Indexes must match."

        if isinstance(df, pd.DataFrame):
            self.parent_data.loc[self.index, self.columns] = df

        else:
            self.parent_data.loc[self.index, self.columns] = df.df_data

    @property
    def parent_data(self) -> pd.DataFrame:
        """
        Get parent's _df.
        :return: parent's _df.
        """
        return getattr(self._parent, '_df')

    @property
    def df_data(self) -> pd.DataFrame:
        """
        Get a view on the parent TemporalDataFrame's raw pandas.DataFrame.
        :return: a view on the parent TemporalDataFrame's raw pandas.DataFrame.
        """
        return self.parent_data.loc[self.index, self.columns]

    def to_pandas(self) -> Any:
        """
        TODO
        """
        index = self.index[0] if len(self.index) == 1 else self.index
        columns = self.columns[0] if len(self.columns) == 1 else self.n_columns

        return self.parent_data.loc[index, columns]

    @property
    def parent_time_points_col(self) -> str:
        """
        Get parent's _time_points_col.
        :return: parent's _time_points_col
        """
        return getattr(self._parent, '_time_points_col')

    @property
    def index_bool(self) -> List[bool]:
        """
        Returns a list of booleans indicating whether the parental DataFrame's indexes are present in this view
        :return: a list of booleans indicating whether the parental DataFrame's indexes are present in this view
        """
        return [idx in self.index for idx in self._parent.index]

    @property
    def time_points(self) -> List[str]:
        """
        Get the list of time points in this view of a TemporalDataFrame
        :return: the list of time points in this view of a TemporalDataFrame
        """
        return self._tp_slicer

    @property
    def n_time_points(self) -> int:
        """
        :return: the number of time points
        """
        return len(self.time_points)

    def len_index(self, time_point: str) -> int:
        """
        :return: the length of the index at a given time point
        """
        return len(self[time_point].index)

    @property
    def n_columns(self) -> int:
        """
        :return: the number of columns
        """
        return len(self.columns)

    @property
    def dtypes(self) -> None:
        """
        Return the dtypes in the DataFrame.
        :return: the dtypes in the DataFrame.
        """
        return self.parent_data[self.columns].dtypes

    def astype(self, dtype: Union[DType, Dict[str, DType]]) -> NoReturn:
        """
        Reference to TemporalDataFrame's astype method. This cannot be done in a view.
        """
        raise VAttributeError('Cannot set data type from a view of a TemporalDataFrame.')

    def info(self, verbose: Optional[bool] = None, buf: Optional[IO[str]] = None, max_cols: Optional[int] = None,
             memory_usage: Optional[Union[bool, str]] = None, null_counts: Optional[bool] = None) -> None:
        """
        This method prints information about a DataFrame including the index dtype and columns, non-null values and
        memory usage.
        :return: a concise summary of a DataFrame.
        """
        return self._parent.info(verbose=verbose, buf=buf, max_cols=max_cols, memory_usage=memory_usage,
                                 null_counts=null_counts)

    @property
    def values(self) -> np.ndarray:
        """
        Return a Numpy representation of the DataFrame.
        :return: a Numpy representation of the DataFrame.
        """
        return self.parent_data[self.columns].values

    @property
    def axes(self) -> List[pd.Index]:
        """
        Return a list of the row axis labels.
        :return: a list of the row axis labels.
        """
        return self.parent_data[self.columns].axes

    @property
    def ndim(self) -> Literal[3]:
        """
        Return an int representing the number of axes / array dimensions.
        :return: 3
        """
        return 3

    @property
    def size(self) -> int:
        """
        Return the number of rows times number of columns.
        :return: an int representing the number of elements in this object.
        """
        return self.parent_data[self.columns].size

    @property
    def shape(self) -> Tuple[int, List[int], int]:
        """
        Return a tuple representing the dimensionality of the DataFrame
        (nb_time_points, [len_index for all time points], nb_col)
        :return: a tuple representing the dimensionality of the DataFrame
        """
        return self.n_time_points, [self.len_index(TP) for TP in self.time_points], self.n_columns

    def memory_usage(self, index: bool = True, deep: bool = False):
        """
        Return the memory usage of each column in bytes.
        The memory usage can optionally include the contribution of the index and elements of object dtype.
        :return: the memory usage of each column in bytes.
        """
        return self._parent.memory_usage(index=index, deep=deep)

    @property
    def empty(self) -> bool:
        """
        Indicator whether DataFrame is empty.
        :return: True if this TemporalDataFrame is empty.
        """
        if not self.n_time_points:
            return True

        for TP in self.time_points:
            mask = match_time_points(self.parent_data['__TPID'], TP) & np.array(self.index_bool)
            if not self.parent_data[mask].empty:
                return False

        return True

    def head(self, n: int = 5, time_points: PreSlicer = slice(None, None, None)) -> str:
        """
        This function returns the first n rows for the object based on position.

        For negative values of n, this function returns all rows except the last n rows.
        :return: the first n rows.
        """
        sub_TDF = self[time_points]

        if sub_TDF.n_time_points:
            repr_str = ""
            for TP in sub_TDF.time_points:
                repr_str += f"\033[4mTime point : {TP}\033[0m\n"
                repr_str += f"{sub_TDF[TP].one_TP_repr(TP, n)}\n"

        else:
            repr_str = f"Empty TemporalDataFrame\n" \
                       f"Columns: {[col for col in sub_TDF.columns]}\n" \
                       f"Index: {[idx for idx in sub_TDF.index]}"

        return repr_str

    def tail(self, n: int = 5, time_points: PreSlicer = slice(None, None, None)) -> str:
        """
        This function returns the last n rows for the object based on position.

        For negative values of n, this function returns all rows except the first n rows.
        :return: the last n rows.
        """
        sub_TDF = self[time_points]

        if sub_TDF.n_time_points:
            repr_str = ""
            for TP in sub_TDF.time_points:
                repr_str += f"\033[4mTime point : {TP}\033[0m\n"
                repr_str += f"{sub_TDF[TP].one_TP_repr(TP, n, func='tail')}\n"

        else:
            repr_str = f"Empty TemporalDataFrame\n" \
                       f"Columns: {[col for col in sub_TDF.columns]}\n" \
                       f"Index: {[idx for idx in sub_TDF.index]}"

        return repr_str

    # @property
    # def at(self) -> '_ViewVAtIndexer':
    #     """
    #     Access a single value for a row/column label pair.
    #     :return: a single value for a row/column label pair.
    #     """
    #     return _ViewVAtIndexer(self._parent)
    #
    # @property
    # def iat(self) -> '_ViewViAtIndexer':
    #     """
    #     Access a single value for a row/column pair by integer position.
    #     :return: a single value for a row/column pair by integer position.
    #     """
    #     return _ViewViAtIndexer(self._parent)

    # TODO : test loc and iloc for value setting
    # @property
    # def loc(self) -> '_VLocIndexer':
    #     """
    #     Access a group of rows and columns by label(s) or a boolean array.
    #
    #     Allowed inputs are:
    #         - A single label, e.g. 5 or 'a', (note that 5 is interpreted as a label of the index, and never as an
    #         integer position along the index).
    #         - A list or array of labels, e.g. ['a', 'b', 'c'].
    #         - A slice object with labels, e.g. 'a':'f'.
    #         - A boolean array of the same length as the axis being sliced, e.g. [True, False, True].
    #         - A callable function with one argument (the calling Series or DataFrame) and that returns valid output
    #         for indexing (one of the above)
    #
    #     :return: a group of rows and columns
    #     """
    #     return _VLocIndexer(self.parent_data[self.index_bool], self.parent_data[self.index_bool].loc,
    #                         self.parent_time_points_col)

    # @property
    # def iloc(self) -> '_ViLocIndexer':
    #     """
    #     Purely integer-location based indexing for selection by position (from 0 to length-1 of the axis).
    #
    #     Allowed inputs are:
    #         - An integer, e.g. 5.
    #         - A list or array of integers, e.g. [4, 3, 0].
    #         - A slice object with ints, e.g. 1:7.
    #         - A boolean array.
    #         - A callable function with one argument (the calling Series or DataFrame) and that returns valid output
    #         for indexing (one of the above). This is useful in method chains, when you don’t have a reference to the
    #         calling object, but would like to base your selection on some value.
    #
    #     :return: a group of rows and columns
    #     """
    #     return _ViLocIndexer(self.parent_data[self.index_bool].iloc)

    def insert(self, *args, **kwargs) -> NoReturn:
        """
        TODO
        """
        raise VValueError("Cannot insert a column from a view.")

    def items(self) -> List[Tuple[Optional[Hashable], pd.Series]]:
        """
        Iterate over (column name, Series) pairs.
        :return: a tuple with the column name and the content as a Series.
        """
        return list(self.parent_data[self.index_bool].items())[1:]

    def keys(self) -> List[Optional[Hashable]]:
        """
        Get the ‘info axis’.
        :return: the ‘info axis’.
        """
        keys = list(self.parent_data[self.index_bool].keys())

        if '__TPID' in keys:
            keys.remove('__TPID')

        return keys

    def isin(self, values: Union[Iterable, pd.Series, pd.DataFrame, Dict]) -> 'vdata.TemporalDataFrame':
        """
        Whether each element in the DataFrame is contained in values.
        :return: whether each element in the DataFrame is contained in values.
        """
        if self._time_points_col == '__TPID':
            time_points = self.parent_data[self.index_bool]['__TPID'].tolist()
            time_col = None

        else:
            time_points = self.parent_data[self.index_bool][self._time_points_col].tolist()
            time_col = self.parent_time_points_col

        return vdata.TemporalDataFrame(self.parent_data.isin(values)[self.columns], time_points=time_points,
                                       time_col=time_col)

    def eq(self, other: Any, axis: Literal[0, 1, 'index', 'column'] = 'columns',
           level: Any = None) -> 'vdata.TemporalDataFrame':
        """
        Get Equal to of dataframe and other, element-wise (binary operator eq).
        Equivalent to '=='.
        :param other: Any single or multiple element data structure, or list-like object.
        :param axis: {0 or ‘index’, 1 or ‘columns’}
        :param level: int or label
        """
        if self._time_points_col == '__TPID':
            time_points = self._df['__TPID'].tolist()
            time_col = None

        else:
            time_points = self._df[self._time_points_col].tolist()
            time_col = self._time_points_col

        return vdata.TemporalDataFrame(self._df.eq(other, axis, level)[self.columns],
                                       time_points=time_points, time_col=time_col)

    # TODO : copy method
