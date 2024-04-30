from __future__ import annotations

import pickle
from typing import Any

import ch5mpy as ch
import numpy as np

import vdata
from vdata.timepoint import TimePointArray, TimePointIndex
from vdata.timepoint.array import as_timepointarray
from vdata.update.array import update_array


def _update_tdf_v0_to_v1(data: ch.H5Dict[Any]) -> None:
    data.attributes.set(
        __h5_type__="object",
        __h5_class__=np.void(pickle.dumps(vdata.TemporalDataFrame, protocol=pickle.HIGHEST_PROTOCOL)),
    )

    del data.attributes["type"]

    if data.attributes["timepoints_column_name"] in ("__ATTRIBUTE_None__", "__TDF_None__"):
        data.attributes["timepoints_column_name"] = "__h5_NONE__"

    data.file.move("timepoints", "timepoints_array")
    data.file.move("values_numerical", "numerical_array")
    data.file.move("values_string", "string_array")

    for array_data in data.values():
        update_array[0](array_data)


def _update_tdf_v1_to_v2(data: ch.H5Dict[Any]) -> None:
    if data.attributes["timepoints_column_name"] is None:
        data.attributes["timepoints_column_name"] = "__h5_NONE__"

    timepoints_index = TimePointIndex.from_array(as_timepointarray(data.timepoints_array))

    data["timepoints_index"] = {
        "ranges": timepoints_index.ranges,
        "timepoints": {
            "array": timepoints_index.timepoints,
        },
    }

    data["timepoints_index"].attributes.set(
        __h5_type__="object",
        __h5_class__=np.void(pickle.dumps(TimePointIndex, protocol=pickle.HIGHEST_PROTOCOL)),
    )
    (data @ "timepoints_index" @ "timepoints").attributes.set(
        __h5_type__="object",
        __h5_class__=np.void(pickle.dumps(TimePointArray, protocol=pickle.HIGHEST_PROTOCOL)),
        unit="h",
    )

    del data["timepoints_array"]


update_tdf = {
    0: _update_tdf_v0_to_v1,
    1: _update_tdf_v1_to_v2,
}
