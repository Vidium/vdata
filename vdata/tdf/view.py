from __future__ import annotations

from typing import Any, Iterable, Literal, NoReturn

import ch5mpy as ch
import numpy as np

from vdata._typing import IFS, Indexer, NDArray_IFS
from vdata.array_view import ArrayGetter, NDArrayView
from vdata.tdf.base import TemporalDataFrameBase
from vdata.tdf.dataframe import TemporalDataFrame


def _as_view(
    container: TemporalDataFrameBase,
    name: str,
    index: Indexer | tuple[Indexer, ...],
    exposed_attributes: Iterable[str] = (),
) -> ch.H5Array[Any] | NDArrayView[Any]:
    if isinstance(getattr(container, name), ch.H5Array):
        return getattr(container, name)[index]

    return NDArrayView(ArrayGetter(container, name), index, exposed_attributes)


class TemporalDataFrameView(TemporalDataFrameBase):
    """
    Abstract base class for views on a TemporalDataFrame.
    """

    _attributes = TemporalDataFrameBase._attributes.union(
        {"_parent", "_numerical_selection", "_string_selection", "_inverted"}
    )

    # region magic methods
    def __init__(
        self,
        parent: TemporalDataFrameBase,
        numerical_selection: ch.indexing.Selection,
        string_selection: ch.indexing.Selection,
        *,
        inverted: bool = False,
    ):
        if isinstance(parent, TemporalDataFrameView):
            numerical_selection = numerical_selection.cast_on(parent._numerical_selection)
            string_selection = string_selection.cast_on(parent._string_selection)

            parent = parent.parent

        super().__init__(
            index=_as_view(parent, "index", ch.indexing.get_indexer(numerical_selection[0], enforce_1d=True)),
            timepoints_array=_as_view(
                parent, "timepoints_column", ch.indexing.get_indexer(numerical_selection[0], enforce_1d=True), {"unit"}
            ),
            numerical_array=_as_view(parent, "values_num", numerical_selection.get_indexers()),
            string_array=_as_view(parent, "values_str", string_selection.get_indexers()),
            columns_numerical=_as_view(
                parent, "columns_num", ch.indexing.get_indexer(numerical_selection[1], enforce_1d=True)
            ),
            columns_string=_as_view(
                parent, "columns_str", ch.indexing.get_indexer(string_selection[1], enforce_1d=True)
            ),
            attr_dict=parent._attr_dict,
            data=parent.data,
        )

        self._parent = parent
        self._numerical_selection = numerical_selection
        self._string_selection = string_selection
        self._inverted = inverted

    def __delattr__(self, _column_name: str) -> NoReturn:
        raise TypeError("Cannot delete columns from a view.")

    def __invert__(self) -> TemporalDataFrameView:
        """
        Invert the getitem selection behavior : all elements NOT present in the slicers will be selected.
        """
        return type(self)(
            parent=self._parent,
            numerical_selection=self._numerical_selection,
            string_selection=self._string_selection,
            inverted=not self._inverted,
        )

    # endregion

    # region attributes
    @property
    def full_name(self) -> str:
        """
        Get the full name.
        """
        parts = []
        if self.empty:
            parts.append("empty")

        if self.is_inverted:
            parts.append("inverted")

        parent_full_name = self._parent.full_name
        if not parent_full_name.startswith("TemporalDataFrame"):
            parent_full_name = parent_full_name[0].lower() + parent_full_name[1:]

        parts += ["view of", parent_full_name]

        parts[0] = parts[0].capitalize()

        return " ".join(parts)

    @property
    def parent(self) -> TemporalDataFrame:
        """Get the parent TemporalDataFrame of this view."""
        return self._parent

    # endregion

    # region predicates
    @property
    def is_view(self) -> Literal[True]:
        """
        Is this a view on a TemporalDataFrame ?
        """
        return True

    @property
    def is_inverted(self) -> bool:
        """
        Whether this view of a TemporalDataFrame is inverted or not.
        """
        return self._inverted

    @property
    def has_repeating_index(self) -> bool:
        """
        Is the index repeated at each time-point ?
        """
        if not self.parent.has_repeating_index:
            return False

        # must be computed since view on TDF with repeating index could still not have repeating index
        for tp in self.timepoints[1:]:
            if not np.array_equal(self.index_at(tp), self.index_at(self.tp0)):
                return False
        return True

    # endregion

    # region methods
    def _append_column(self, column_name: IFS, values: NDArray_IFS):
        raise NotImplementedError

    def lock_indices(self) -> None:
        """Lock the "index" axis to prevent modifications."""
        self._parent.lock_indices()

    def unlock_indices(self) -> None:
        """Unlock the "index" axis to allow modifications."""
        self._parent.unlock_indices()

    def lock_columns(self) -> None:
        """Lock the "columns" axis to prevent modifications."""
        self._parent.lock_columns()

    def unlock_columns(self) -> None:
        """Unlock the "columns" axis to allow modifications."""
        self._parent.unlock_columns()

    def set_index(self, values: NDArray_IFS, repeating_index: bool = False) -> None:
        """Set new index values."""
        raise NotImplementedError

    def reindex(self, order: NDArray_IFS, repeating_index: bool = False) -> None:
        """Re-order rows in this TemporalDataFrame so that their index matches the new given order."""
        raise NotImplementedError

    # endregion

    # region data methods
    def merge(self, other: TemporalDataFrameBase, name: str | None = None) -> TemporalDataFrame:
        """
        Merge two TemporalDataFrames together, by rows. The column names and time points must match.

        Args:
            other: a TemporalDataFrame to merge with this one.
            name: a name for the merged TemporalDataFrame.

        Returns:
            A new merged TemporalDataFrame.
        """
        raise NotImplementedError

    # endregion
