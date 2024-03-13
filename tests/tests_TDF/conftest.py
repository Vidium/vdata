import pickle
from tempfile import NamedTemporaryFile
from typing import Any, Generator

import numpy as np
import pytest
from ch5mpy import File, H5Mode
from h5py import string_dtype

from vdata.tdf import TemporalDataFrame, TemporalDataFrameBase, TemporalDataFrameView
from vdata.timepoint import TimePointArray, TimePointIndex

REFERENCE_BACKED_DATA = {
    "type": "tdf",
    "locked_indices": False,
    "locked_columns": False,
    "timepoints_column_name": None,
    "index": np.concatenate((np.arange(50, 100), np.arange(0, 50))),
    "repeating_index": False,
    "columns_numerical": np.array(["col1", "col2"], dtype=np.dtype("O")),
    "columns_string": np.array(["col3", "col4"], dtype=np.dtype("O")),
    "timepoints_index": {"timepoints": np.array([0.0, 1.0]), "ranges": np.array([50, 100])},
    "values_numerical": np.vstack(
        (
            np.concatenate((np.arange(50, 100), np.arange(0, 50))),
            np.concatenate((np.arange(150, 200), np.arange(100, 150))),
        )
    ).T.astype(float),
    "values_string": np.vstack(
        (
            np.concatenate((np.arange(250, 300), np.arange(200, 250))),
            np.concatenate((np.arange(350, 400), np.arange(300, 350))),
        )
    )
    .T.astype(str)
    .astype("O"),
}


def get_TDF(name: str = "1") -> TemporalDataFrame:
    return TemporalDataFrame(
        {
            "col1": np.array([i for i in range(100)]),
            "col2": np.array([i for i in range(100, 200)]),
            "col3": np.array([str(i) for i in range(200, 300)]),
            "col4": np.array([str(i) for i in range(300, 400)]),
        },
        name=name,
        timepoints=["1h" for _ in range(50)] + ["0h" for _ in range(50)],
    )


def get_backed_TDF(file: NamedTemporaryFile, name: str) -> TemporalDataFrame:  # pyright: ignore[reportGeneralTypeIssues]
    with File(file.name, H5Mode.WRITE_TRUNCATE) as h5_file:
        h5_file.attrs["name"] = name
        h5_file.attrs["locked_indices"] = REFERENCE_BACKED_DATA["locked_indices"]
        h5_file.attrs["locked_columns"] = REFERENCE_BACKED_DATA["locked_columns"]
        h5_file.attrs["timepoints_column_name"] = REFERENCE_BACKED_DATA["timepoints_column_name"]
        h5_file.attrs["repeating_index"] = REFERENCE_BACKED_DATA["repeating_index"]

        h5_file.create_dataset("index", data=REFERENCE_BACKED_DATA["index"])

        h5_file.create_dataset(
            "columns_numerical",
            data=REFERENCE_BACKED_DATA["columns_numerical"],
            chunks=True,
            maxshape=(None,),
            dtype=string_dtype(),
        )
        h5_file["columns_numerical"].attrs["dtype"] = "<U4"

        h5_file.create_dataset(
            "columns_string",
            data=REFERENCE_BACKED_DATA["columns_string"],
            chunks=True,
            maxshape=(None,),
            dtype=string_dtype(),
        )
        h5_file["columns_string"].attrs["dtype"] = "<U4"

        h5_file.create_dataset(
            "numerical_array", data=REFERENCE_BACKED_DATA["values_numerical"], chunks=True, maxshape=(None, None)
        )

        h5_file.create_dataset(
            "string_array",
            data=REFERENCE_BACKED_DATA["values_string"],
            dtype=string_dtype(),
            chunks=True,
            maxshape=(None, None),
        )
        h5_file["string_array"].attrs["dtype"] = "<U4"

        h5_file.create_group("timepoints_index")
        h5_file["timepoints_index"].attrs["__h5_type__"] = "object"
        h5_file["timepoints_index"].attrs["__h5_class__"] = np.void(
            pickle.dumps(TimePointIndex, protocol=pickle.HIGHEST_PROTOCOL)
        )
        h5_file["timepoints_index"].create_dataset("ranges", data=REFERENCE_BACKED_DATA["timepoints_index"]["ranges"])

        h5_file["timepoints_index"].create_group("timepoints")
        h5_file["timepoints_index"]["timepoints"].attrs["__h5_type__"] = "object"
        h5_file["timepoints_index"]["timepoints"].attrs["__h5_class__"] = np.void(
            pickle.dumps(TimePointArray, protocol=pickle.HIGHEST_PROTOCOL)
        )
        h5_file["timepoints_index"]["timepoints"].attrs["unit"] = "h"

        h5_file["timepoints_index"]["timepoints"].create_dataset(
            "array", data=REFERENCE_BACKED_DATA["timepoints_index"]["timepoints"]
        )

    # read tdf from file
    return TemporalDataFrame.read(file.name, mode=H5Mode.READ_WRITE)


@pytest.fixture(scope="function")
def TDF(request: Any) -> Generator[TemporalDataFrameBase, None, None]:
    if hasattr(request, "param"):
        which = request.param

    else:
        which = "plain"

    if "backed" in which:
        with NamedTemporaryFile() as tmp_file:
            if "view" in which:
                tdfv: TemporalDataFrameView = get_backed_TDF(tmp_file, "1")[:, np.arange(10, 90), ["col1", "col4"]]
                yield tdfv
                tdfv.parent.close()

            else:
                tdf: TemporalDataFrame = get_backed_TDF(tmp_file, "1")
                yield tdf
                tdf.close()

    else:
        if "view" in which:
            yield get_TDF()[:, np.arange(10, 90), ["col1", "col4"]]

        else:
            yield get_TDF()


@pytest.fixture(scope="class")
def class_TDF_backed(request: Any) -> Generator[None, None, None]:
    with NamedTemporaryFile() as tmp_file:
        tdf = get_backed_TDF(tmp_file, "1")
        request.cls.TDF = tdf
        yield


@pytest.fixture
def TDF1(request: Any) -> Generator[TemporalDataFrameBase, None, None]:
    if hasattr(request, "param"):
        which = request.param

    else:
        which = "plain"

    if "backed" in which:
        with NamedTemporaryFile() as tmp_file:
            if "view" in which:
                tdfv: TemporalDataFrameView = get_backed_TDF(tmp_file, "1")[:, np.r_[0:40, 50:90]]
                yield tdfv
                tdfv.parent.close()

            else:
                tdf = get_backed_TDF(tmp_file, "1")
                yield tdf
                tdf.close()

    else:
        if "view" in which:
            yield get_TDF()[:, np.r_[0:40, 50:90]]

        else:
            yield get_TDF()


@pytest.fixture(scope="class")
def class_TDF1(request: Any) -> Generator[None, None, None]:
    if hasattr(request, "param"):
        which = request.param

    else:
        which = "plain"

    if "backed" in which:
        with NamedTemporaryFile() as tmp_file:
            if "view" in which:
                tdfv: TemporalDataFrameView = get_backed_TDF(tmp_file, "1")[:, np.r_[0:40, 50:90]]
                request.cls.TDF = tdfv
                yield
                tdfv.parent.close()

            else:
                tdf = get_backed_TDF(tmp_file, "1")
                request.cls.TDF = tdf
                yield
                tdf.close()

    else:
        if "view" in which:
            request.cls.TDF = get_TDF()[:, np.r_[0:40, 50:90]]
            yield

        else:
            request.cls.TDF = get_TDF()
            yield


@pytest.fixture
def TDF2(request: Any) -> Generator[TemporalDataFrameBase, None, None]:
    if hasattr(request, "param"):
        which = request.param

    else:
        which = "plain"

    if "backed" in which:
        with NamedTemporaryFile() as tmp_file:
            if "view" in which:
                tdfv: TemporalDataFrameView = get_backed_TDF(tmp_file, "2")[:, np.r_[0:40, 50:90]]
                yield tdfv
                tdfv.parent.close()

            else:
                tdf = get_backed_TDF(tmp_file, "2")
                yield tdf
                tdf.close()

    else:
        if "view" in which:
            yield get_TDF("2")[:, np.r_[0:40, 50:90]]

        else:
            yield get_TDF("2")


@pytest.fixture(scope="class")
def h5_file(request: Any) -> Generator[None, None, None]:
    if hasattr(request, "param"):
        which = request.param

    else:
        which = "plain"

    with NamedTemporaryFile(suffix=".vd") as tmp_write:
        if "backed" in which:
            with NamedTemporaryFile() as tmp_backed:
                if "view" in which:
                    tdfv: TemporalDataFrameView = get_backed_TDF(tmp_backed, "1")[
                        :, np.arange(10, 90), ["col1", "col4"]
                    ]
                    tdfv.write(tmp_write.name)
                    tdfv.parent.close()

                else:
                    tdf = get_backed_TDF(tmp_backed, "1")
                    tdf.write(tmp_write.name)
                    tdf.close()

        else:
            if "view" in which:
                tdfv: TemporalDataFrameView = get_TDF()[:, np.arange(10, 90), ["col1", "col4"]]
                tdfv.write(tmp_write.name)

            else:
                tdf = get_TDF()
                tdf.write(tmp_write.name)

        request.cls.h5_file = File(tmp_write.name, H5Mode.READ)
        yield
