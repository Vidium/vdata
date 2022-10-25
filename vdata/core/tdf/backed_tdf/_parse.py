# coding: utf-8
# Created on 22/10/2022 20:02
# Author : matteo

# ====================================================
# imports
from __future__ import annotations

from vdata.core.attribute_proxy import AttributeProxy
from vdata.core.dataset_proxy import DatasetProxy, num_, str_, tp_
from vdata.core.tdf.name_utils import H5Data


# ====================================================
# code
def parse_data_h5(data: H5Data,
                  lock: tuple[bool, bool] | None,
                  name: str) -> tuple[
    DatasetProxy, DatasetProxy, DatasetProxy, DatasetProxy, DatasetProxy, DatasetProxy, AttributeProxy
]:
    _numerical_array = DatasetProxy(data['values_numerical'], dtype=num_)
    _string_array = DatasetProxy(data['values_string'], dtype=str_)
    _index = DatasetProxy(data['index'])
    _columns_numerical = DatasetProxy(data['columns_numerical'])
    _columns_string = DatasetProxy(data['columns_string'])
    _timepoints_array = DatasetProxy(data['timepoints'], dtype=tp_)
    _attributes = AttributeProxy(data)

    if lock is not None:
        _attributes['locked_indices'], _attributes['locked_columns'] = bool(lock[0]), bool(lock[1])

    if name != 'No_Name':
        _attributes['name'] = str(name)

    return _numerical_array, _string_array, _index, _columns_numerical, _columns_string, _timepoints_array, _attributes
