from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Collection, Literal

import pandas as pd

from vdata._typing import NDArray_IFS

if TYPE_CHECKING:
    from vdata.vdataframe.vdataframe import VDataFrame


def _check_parent_has_not_changed(func: Callable[..., Any]) -> Callable[..., Any]:
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        self = args[0]

        if hash(tuple(object.__getattribute__(self, '_parent').index)) != \
                object.__getattribute__(self, '_parent_index_hash') or \
                hash(tuple(object.__getattribute__(self, '_parent').columns)) != \
                object.__getattribute__(self, '_parent_columns_hash'):
            raise ValueError("View no longer valid since parent's VDataFrame has changed.")

        return func(*args, **kwargs)

    return wrapper


class ViewVDataFrame:
    """
    View of a VDataFrame.
    """
    _DataFrame: pd.DataFrame

    def __init__(self,
                 parent: VDataFrame,
                 index_slicer: NDArray_IFS | Collection[bool] | slice = slice(None),
                 column_slicer: NDArray_IFS | slice = slice(None)):
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
    def __getattr__(self, item: str) -> Any:
        return getattr(self._DataFrame, item)

    @_check_parent_has_not_changed
    def __setattr__(self, key: str, value: Any) -> None:
        setattr(self._DataFrame, key, value)

    @_check_parent_has_not_changed
    def __delattr__(self, item: str) -> None:
        delattr(self._DataFrame, item)

    @_check_parent_has_not_changed
    def __getitem__(self, item: str) -> Any:
        return self._DataFrame.__getitem__(item)

    @_check_parent_has_not_changed
    def __setitem__(self, key: str, value: Any) -> None:
        self._DataFrame.__setitem__(key, value)             # type: ignore[no-untyped-call]

    @_check_parent_has_not_changed
    def __delitem__(self, key: str) -> None:
        self._DataFrame.__delitem__(key)

    @_check_parent_has_not_changed
    def __len__(self) -> int:
        return len(self._DataFrame)

    @_check_parent_has_not_changed
    def to_pandas(self) -> pd.DataFrame:
        return self._DataFrame
