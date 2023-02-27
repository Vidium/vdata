# coding: utf-8
# Created on 22/10/2022 20:02
# Author : matteo

# ====================================================
# imports
from __future__ import annotations

from vdata.time_point import TimePoint
from vdata.core.attribute_proxy import AttributeProxy
from vdata.core.tdf.name_utils import H5Data


# ====================================================
# code
def parse_data_h5(data: H5Data,
                  lock: tuple[bool, bool] | None,
                  name: str) -> AttributeProxy:
    _attributes = AttributeProxy(data)

    if lock is not None:
        _attributes['locked_indices'], _attributes['locked_columns'] = bool(lock[0]), bool(lock[1])

    if name != 'No_Name':
        _attributes['name'] = str(name)

    return _attributes
