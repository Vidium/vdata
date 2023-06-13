from __future__ import annotations

from typing import Iterable, Union, cast

import numpy as np
import pandas as pd
import pandas.core.indexing

import vdata.vdataframe as vdf

IFS = Union[int, np.int_, float, np.float_, str, np.str_]


class _LocIndexer(pandas.core.indexing._LocIndexer):
    
    def __getitem__(self, key: slice | IFS | Iterable[IFS] |
                               tuple[slice | IFS | Iterable[IFS], 
                                     slice | IFS | Iterable[IFS]]) \
            -> vdf.VDataFrame | np.int_ | np.float_ | np.str_:
        res = super().__getitem__(key)      # type: ignore[no-untyped-call]
        
        if isinstance(res, pd.DataFrame):
            return vdf.VDataFrame(res)
        
        return cast(Union[np.int_, np.float_, np.str_],
                    res)
    
    
class _iLocIndexer(pandas.core.indexing._iLocIndexer):
    
    def __getitem__(self, key: slice | int | Iterable[int] | Iterable[bool] |
                               tuple[slice | int | Iterable[int] | Iterable[bool], 
                                     slice | int | Iterable[int] | Iterable[bool]]) \
            -> vdf.VDataFrame | np.int_ | np.float_ | np.str_:
        res = super().__getitem__(key)      # type: ignore[no-untyped-call]
        
        if isinstance(res, pd.DataFrame):
            return vdf.VDataFrame(res)
        
        return cast(Union[np.int_, np.float_, np.str_],
                    res)
