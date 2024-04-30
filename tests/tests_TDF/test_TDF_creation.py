from typing import Any

import numpy as np
import pandas as pd
import pytest

import vdata.timepoint as tp
from vdata import RepeatingIndex
from vdata.tdf import TemporalDataFrame, TemporalDataFrameBase


def test_empty_TDF_creation() -> None:
    TDF = TemporalDataFrame(data=None, timepoints=None, time_col_name=None, index=None)

    assert TDF.empty
    assert repr(TDF) == "Empty TemporalDataFrame No_Name\n" "Time points: []\n" "Columns: []\n" "Index: []"


def test_empty_TDF_with_index() -> None:
    TDF = TemporalDataFrame(data=None, timepoints=None, time_col_name=None, index=["a", "b", "c"])

    assert TDF.empty
    assert (
        repr(TDF) == "Empty TemporalDataFrame No_Name\n"
        "Time point : 0.0 hours\n"
        "‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾\n"
        "  Time-point   \n"
        "a       0.0h  ｜\n"
        "b       0.0h  ｜\n"
        "c       0.0h  ｜\n"
        "[3 rows x 0 columns]\n\n"
    )


def test_empty_TDF_with_timepoints() -> None:
    TDF = TemporalDataFrame(data=None, timepoints=["0h", "5h"], time_col_name=None, index=None)

    assert TDF.empty
    assert (
        repr(TDF) == "Empty TemporalDataFrame No_Name\n"
        "Time point : 0.0 hours\n"
        "‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾\n"
        "  Time-point   \n"
        "0       0.0h  ｜\n"
        "[1 rows x 0 columns]\n"
        "\n"
        "Time point : 5.0 hours\n"
        "‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾\n"
        "  Time-point   \n"
        "1       5.0h  ｜\n"
        "[1 rows x 0 columns]\n\n"
    )


def test_empty_TDF_with_tp_and_index() -> None:
    TDF = TemporalDataFrame(data=None, timepoints=["0h", "5h"], time_col_name=None, index=["a", "b"])

    assert TDF.empty
    assert (
        repr(TDF) == "Empty TemporalDataFrame No_Name\n"
        "Time point : 0.0 hours\n"
        "‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾\n"
        "  Time-point   \n"
        "a       0.0h  ｜\n"
        "[1 rows x 0 columns]\n"
        "\n"
        "Time point : 5.0 hours\n"
        "‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾\n"
        "  Time-point   \n"
        "b       5.0h  ｜\n"
        "[1 rows x 0 columns]\n\n"
    )


def test_TDF_is_not_backed(TDF: TemporalDataFrameBase) -> None:
    assert not TDF.is_backed


@pytest.fixture(scope="class")
def class_data(request: pytest.FixtureRequest) -> None:
    which = request.param

    if "dict" in which:
        if "numerical" in which:
            request.cls.data = {"col1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]}

        elif "string" in which:
            request.cls.data = {"col1": ["a", "b", "c", "d", "e", "f", "g", "h", "i"]}

        elif "both" in which:
            request.cls.data = {
                "col2": ["a", "b", "c", "d", "e", "f", "g", "h", "i"],
                "col1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
            }

        else:
            raise ValueError

    elif "dataframe" in which:
        if "numerical" in which:
            request.cls.data = pd.DataFrame({"col1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]})

        elif "string" in which:
            request.cls.data = pd.DataFrame({"col1": ["a", "b", "c", "d", "e", "f", "g", "h", "i"]})

        elif "both" in which:
            request.cls.data = pd.DataFrame(
                {
                    "col2": ["a", "b", "c", "d", "e", "f", "g", "h", "i"],
                    "col1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
                }
            )

        else:
            raise ValueError

    else:
        raise ValueError


@pytest.mark.usefixtures("class_data")
@pytest.mark.parametrize("class_data", ["dict numerical", "dataframe numerical"], indirect=True)
class TestCreationFromNumericalData:
    data: pd.DataFrame | dict[str, Any]

    def test_creation(self) -> None:
        TDF = TemporalDataFrame(data=self.data, timepoints=None, time_col_name=None, index=None)

        assert (
            repr(TDF) == "TemporalDataFrame No_Name\n"
            "Time point : 0.0 hours\n"
            "‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾\n"
            "  Time-point    col1\n"
            "0       0.0h  ｜  1.0\n"
            "1       0.0h  ｜  2.0\n"
            "2       0.0h  ｜  3.0\n"
            "3       0.0h  ｜  4.0\n"
            "4       0.0h  ｜  5.0\n"
            "[9 rows x 1 columns]\n\n"
        )

    def test_creation_with_index(self) -> None:
        TDF = TemporalDataFrame(
            data=self.data, timepoints=None, time_col_name=None, index=["a", "b", "c", "d", "e", "f", "g", "h", "i"]
        )

        assert (
            repr(TDF) == "TemporalDataFrame No_Name\n"
            "Time point : 0.0 hours\n"
            "‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾\n"
            "  Time-point    col1\n"
            "a       0.0h  ｜  1.0\n"
            "b       0.0h  ｜  2.0\n"
            "c       0.0h  ｜  3.0\n"
            "d       0.0h  ｜  4.0\n"
            "e       0.0h  ｜  5.0\n"
            "[9 rows x 1 columns]\n\n"
        )

    def test_creation_with_timepoints(self) -> None:
        TDF = TemporalDataFrame(
            data=self.data,
            timepoints=["0h", "0h", "0h", "5h", "5h", "5h", "10h", "10h", "10h"],
            time_col_name=None,
            index=None,
        )

        assert (
            repr(TDF) == "TemporalDataFrame No_Name\n"
            "Time point : 0.0 hours\n"
            "‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾\n"
            "  Time-point    col1\n"
            "0       0.0h  ｜  1.0\n"
            "1       0.0h  ｜  2.0\n"
            "2       0.0h  ｜  3.0\n"
            "[3 rows x 1 columns]\n"
            "\n"
            "Time point : 5.0 hours\n"
            "‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾\n"
            "  Time-point    col1\n"
            "3       5.0h  ｜  4.0\n"
            "4       5.0h  ｜  5.0\n"
            "5       5.0h  ｜  6.0\n"
            "[3 rows x 1 columns]\n"
            "\n"
            "Time point : 10.0 hours\n"
            "‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾\n"
            "  Time-point    col1\n"
            "6      10.0h  ｜  7.0\n"
            "7      10.0h  ｜  8.0\n"
            "8      10.0h  ｜  9.0\n"
            "[3 rows x 1 columns]\n\n"
        )

    def test_creation_with_index_and_timepoints(self) -> None:
        TDF = TemporalDataFrame(
            data=self.data,
            timepoints=["0h", "0h", "0h", "5h", "5h", "5h", "10h", "10h", "10h"],
            time_col_name=None,
            index=["a", "b", "c", "d", "e", "f", "g", "h", "i"],
        )
        assert (
            repr(TDF) == "TemporalDataFrame No_Name\n"
            "Time point : 0.0 hours\n"
            "‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾\n"
            "  Time-point    col1\n"
            "a       0.0h  ｜  1.0\n"
            "b       0.0h  ｜  2.0\n"
            "c       0.0h  ｜  3.0\n"
            "[3 rows x 1 columns]\n"
            "\n"
            "Time point : 5.0 hours\n"
            "‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾\n"
            "  Time-point    col1\n"
            "d       5.0h  ｜  4.0\n"
            "e       5.0h  ｜  5.0\n"
            "f       5.0h  ｜  6.0\n"
            "[3 rows x 1 columns]\n"
            "\n"
            "Time point : 10.0 hours\n"
            "‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾\n"
            "  Time-point    col1\n"
            "g      10.0h  ｜  7.0\n"
            "h      10.0h  ｜  8.0\n"
            "i      10.0h  ｜  9.0\n"
            "[3 rows x 1 columns]\n\n"
        )

    def test_creation_with_same_index_for_all_timepoints(self) -> None:
        TDF = TemporalDataFrame(
            data=self.data,
            timepoints=["0h", "0h", "0h", "5h", "5h", "5h", "10h", "10h", "10h"],
            time_col_name=None,
            index=RepeatingIndex(["a", "b", "c"], repeats=3),
        )
        assert (
            repr(TDF) == "TemporalDataFrame No_Name\n"
            "Time point : 0.0 hours\n"
            "‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾\n"
            "  Time-point    col1\n"
            "a       0.0h  ｜  1.0\n"
            "b       0.0h  ｜  2.0\n"
            "c       0.0h  ｜  3.0\n"
            "[3 rows x 1 columns]\n"
            "\n"
            "Time point : 5.0 hours\n"
            "‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾\n"
            "  Time-point    col1\n"
            "a       5.0h  ｜  4.0\n"
            "b       5.0h  ｜  5.0\n"
            "c       5.0h  ｜  6.0\n"
            "[3 rows x 1 columns]\n"
            "\n"
            "Time point : 10.0 hours\n"
            "‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾\n"
            "  Time-point    col1\n"
            "a      10.0h  ｜  7.0\n"
            "b      10.0h  ｜  8.0\n"
            "c      10.0h  ｜  9.0\n"
            "[3 rows x 1 columns]\n\n"
        )


@pytest.mark.usefixtures("class_data")
@pytest.mark.parametrize("class_data", ["dict string", "dataframe string"], indirect=True)
class TestCreationFromStringData:
    data: pd.DataFrame | dict

    def test_creation(self) -> None:
        TDF = TemporalDataFrame(data=self.data, timepoints=None, time_col_name=None, index=None)

        assert (
            repr(TDF) == "TemporalDataFrame No_Name\n"
            "Time point : 0.0 hours\n"
            "‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾\n"
            "  Time-point    col1\n"
            "0       0.0h  ｜    a\n"
            "1       0.0h  ｜    b\n"
            "2       0.0h  ｜    c\n"
            "3       0.0h  ｜    d\n"
            "4       0.0h  ｜    e\n"
            "[9 rows x 1 columns]\n\n"
        )

    def test_creation_with_index(self) -> None:
        TDF = TemporalDataFrame(
            data=self.data, timepoints=None, time_col_name=None, index=["a", "b", "c", "d", "e", "f", "g", "h", "i"]
        )
        assert (
            repr(TDF) == "TemporalDataFrame No_Name\n"
            "Time point : 0.0 hours\n"
            "‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾\n"
            "  Time-point    col1\n"
            "a       0.0h  ｜    a\n"
            "b       0.0h  ｜    b\n"
            "c       0.0h  ｜    c\n"
            "d       0.0h  ｜    d\n"
            "e       0.0h  ｜    e\n"
            "[9 rows x 1 columns]\n\n"
        )

    def test_creation_with_timepoints(self) -> None:
        TDF = TemporalDataFrame(
            data=self.data,
            timepoints=["0h", "0h", "0h", "5h", "5h", "5h", "10h", "10h", "10h"],
            time_col_name=None,
            index=None,
        )
        assert (
            repr(TDF) == "TemporalDataFrame No_Name\n"
            "Time point : 0.0 hours\n"
            "‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾\n"
            "  Time-point    col1\n"
            "0       0.0h  ｜    a\n"
            "1       0.0h  ｜    b\n"
            "2       0.0h  ｜    c\n"
            "[3 rows x 1 columns]\n"
            "\n"
            "Time point : 5.0 hours\n"
            "‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾\n"
            "  Time-point    col1\n"
            "3       5.0h  ｜    d\n"
            "4       5.0h  ｜    e\n"
            "5       5.0h  ｜    f\n"
            "[3 rows x 1 columns]\n"
            "\n"
            "Time point : 10.0 hours\n"
            "‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾\n"
            "  Time-point    col1\n"
            "6      10.0h  ｜    g\n"
            "7      10.0h  ｜    h\n"
            "8      10.0h  ｜    i\n"
            "[3 rows x 1 columns]\n\n"
        )

    def test_creation_with_index_and_timepoints(self) -> None:
        TDF = TemporalDataFrame(
            data=self.data,
            timepoints=["0h", "0h", "0h", "5h", "5h", "5h", "10h", "10h", "10h"],
            time_col_name=None,
            index=["a", "b", "c", "d", "e", "f", "g", "h", "i"],
        )
        assert (
            repr(TDF) == "TemporalDataFrame No_Name\n"
            "Time point : 0.0 hours\n"
            "‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾\n"
            "  Time-point    col1\n"
            "a       0.0h  ｜    a\n"
            "b       0.0h  ｜    b\n"
            "c       0.0h  ｜    c\n"
            "[3 rows x 1 columns]\n"
            "\n"
            "Time point : 5.0 hours\n"
            "‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾\n"
            "  Time-point    col1\n"
            "d       5.0h  ｜    d\n"
            "e       5.0h  ｜    e\n"
            "f       5.0h  ｜    f\n"
            "[3 rows x 1 columns]\n"
            "\n"
            "Time point : 10.0 hours\n"
            "‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾\n"
            "  Time-point    col1\n"
            "g      10.0h  ｜    g\n"
            "h      10.0h  ｜    h\n"
            "i      10.0h  ｜    i\n"
            "[3 rows x 1 columns]\n\n"
        )

    def test_creation_with_same_index_for_all_timepoints(self) -> None:
        TDF = TemporalDataFrame(
            data=self.data,
            timepoints=["0h", "0h", "0h", "5h", "5h", "5h", "10h", "10h", "10h"],
            time_col_name=None,
            index=RepeatingIndex(["a", "b", "c"], repeats=3),
        )
        assert (
            repr(TDF) == "TemporalDataFrame No_Name\n"
            "Time point : 0.0 hours\n"
            "‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾\n"
            "  Time-point    col1\n"
            "a       0.0h  ｜    a\n"
            "b       0.0h  ｜    b\n"
            "c       0.0h  ｜    c\n"
            "[3 rows x 1 columns]\n"
            "\n"
            "Time point : 5.0 hours\n"
            "‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾\n"
            "  Time-point    col1\n"
            "a       5.0h  ｜    d\n"
            "b       5.0h  ｜    e\n"
            "c       5.0h  ｜    f\n"
            "[3 rows x 1 columns]\n"
            "\n"
            "Time point : 10.0 hours\n"
            "‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾\n"
            "  Time-point    col1\n"
            "a      10.0h  ｜    g\n"
            "b      10.0h  ｜    h\n"
            "c      10.0h  ｜    i\n"
            "[3 rows x 1 columns]\n\n"
        )


@pytest.mark.usefixtures("class_data")
@pytest.mark.parametrize("class_data", ["dict both", "dataframe both"], indirect=True)
class TestCreationFromBothNumericalAndStringData:
    data: pd.DataFrame | dict

    def test_creation(self) -> None:
        TDF = TemporalDataFrame(data=self.data, timepoints=None, time_col_name=None, index=None)
        assert (
            repr(TDF) == "TemporalDataFrame No_Name\n"
            "Time point : 0.0 hours\n"
            "‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾\n"
            "  Time-point    col1    col2\n"
            "0       0.0h  ｜  1.0  ｜    a\n"
            "1       0.0h  ｜  2.0  ｜    b\n"
            "2       0.0h  ｜  3.0  ｜    c\n"
            "3       0.0h  ｜  4.0  ｜    d\n"
            "4       0.0h  ｜  5.0  ｜    e\n"
            "[9 rows x 2 columns]\n\n"
        )

    def test_creation_with_index(self) -> None:
        TDF = TemporalDataFrame(
            data=self.data, timepoints=None, time_col_name=None, index=["a", "b", "c", "d", "e", "f", "g", "h", "i"]
        )
        assert (
            repr(TDF) == "TemporalDataFrame No_Name\n"
            "Time point : 0.0 hours\n"
            "‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾\n"
            "  Time-point    col1    col2\n"
            "a       0.0h  ｜  1.0  ｜    a\n"
            "b       0.0h  ｜  2.0  ｜    b\n"
            "c       0.0h  ｜  3.0  ｜    c\n"
            "d       0.0h  ｜  4.0  ｜    d\n"
            "e       0.0h  ｜  5.0  ｜    e\n"
            "[9 rows x 2 columns]\n\n"
        )

    def test_creation_with_timepoints(self) -> None:
        TDF = TemporalDataFrame(
            data=self.data,
            timepoints=["0h", "0h", "0h", "5h", "5h", "5h", "10h", "10h", "10h"],
            time_col_name=None,
            index=None,
        )
        assert (
            repr(TDF) == "TemporalDataFrame No_Name\n"
            "Time point : 0.0 hours\n"
            "‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾\n"
            "  Time-point    col1    col2\n"
            "0       0.0h  ｜  1.0  ｜    a\n"
            "1       0.0h  ｜  2.0  ｜    b\n"
            "2       0.0h  ｜  3.0  ｜    c\n"
            "[3 rows x 2 columns]\n"
            "\n"
            "Time point : 5.0 hours\n"
            "‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾\n"
            "  Time-point    col1    col2\n"
            "3       5.0h  ｜  4.0  ｜    d\n"
            "4       5.0h  ｜  5.0  ｜    e\n"
            "5       5.0h  ｜  6.0  ｜    f\n"
            "[3 rows x 2 columns]\n"
            "\n"
            "Time point : 10.0 hours\n"
            "‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾\n"
            "  Time-point    col1    col2\n"
            "6      10.0h  ｜  7.0  ｜    g\n"
            "7      10.0h  ｜  8.0  ｜    h\n"
            "8      10.0h  ｜  9.0  ｜    i\n"
            "[3 rows x 2 columns]\n\n"
        )

    def test_creation_with_index_and_timepoints(self) -> None:
        TDF = TemporalDataFrame(
            data=self.data,
            timepoints=["0h", "0h", "0h", "5h", "5h", "5h", "10h", "10h", "10h"],
            time_col_name=None,
            index=["a", "b", "c", "d", "e", "f", "g", "h", "i"],
        )
        assert (
            repr(TDF) == "TemporalDataFrame No_Name\n"
            "Time point : 0.0 hours\n"
            "‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾\n"
            "  Time-point    col1    col2\n"
            "a       0.0h  ｜  1.0  ｜    a\n"
            "b       0.0h  ｜  2.0  ｜    b\n"
            "c       0.0h  ｜  3.0  ｜    c\n"
            "[3 rows x 2 columns]\n"
            "\n"
            "Time point : 5.0 hours\n"
            "‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾\n"
            "  Time-point    col1    col2\n"
            "d       5.0h  ｜  4.0  ｜    d\n"
            "e       5.0h  ｜  5.0  ｜    e\n"
            "f       5.0h  ｜  6.0  ｜    f\n"
            "[3 rows x 2 columns]\n"
            "\n"
            "Time point : 10.0 hours\n"
            "‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾\n"
            "  Time-point    col1    col2\n"
            "g      10.0h  ｜  7.0  ｜    g\n"
            "h      10.0h  ｜  8.0  ｜    h\n"
            "i      10.0h  ｜  9.0  ｜    i\n"
            "[3 rows x 2 columns]\n\n"
        )

    def test_creation_with_same_index_for_all_timepoints(self) -> None:
        TDF = TemporalDataFrame(
            data=self.data,
            timepoints=["0h", "0h", "0h", "5h", "5h", "5h", "10h", "10h", "10h"],
            time_col_name=None,
            index=RepeatingIndex(["a", "b", "c"], repeats=3),
        )
        assert (
            repr(TDF) == "TemporalDataFrame No_Name\n"
            "Time point : 0.0 hours\n"
            "‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾\n"
            "  Time-point    col1    col2\n"
            "a       0.0h  ｜  1.0  ｜    a\n"
            "b       0.0h  ｜  2.0  ｜    b\n"
            "c       0.0h  ｜  3.0  ｜    c\n"
            "[3 rows x 2 columns]\n"
            "\n"
            "Time point : 5.0 hours\n"
            "‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾\n"
            "  Time-point    col1    col2\n"
            "a       5.0h  ｜  4.0  ｜    d\n"
            "b       5.0h  ｜  5.0  ｜    e\n"
            "c       5.0h  ｜  6.0  ｜    f\n"
            "[3 rows x 2 columns]\n"
            "\n"
            "Time point : 10.0 hours\n"
            "‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾\n"
            "  Time-point    col1    col2\n"
            "a      10.0h  ｜  7.0  ｜    g\n"
            "b      10.0h  ｜  8.0  ｜    h\n"
            "c      10.0h  ｜  9.0  ｜    i\n"
            "[3 rows x 2 columns]\n\n"
        )


@pytest.fixture(scope="class")
def unsorted_TDF(request: pytest.FixtureRequest) -> None:
    request.cls.TDF = TemporalDataFrame(
        data={
            "col2": ["a", "b", "c", "d", "e", "f", "g", "h", "i"],
            "col1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
        },
        timepoints=["10h", "10h", "10h", "0h", "0h", "0h", "5h", "5h", "5h"],
        time_col_name=None,
        index=["a", "b", "c", "d", "e", "f", "g", "h", "i"],
    )


@pytest.mark.usefixtures("unsorted_TDF")
class TestCreationFromUnsortedData:
    TDF: TemporalDataFrame

    def test_creation(self) -> None:
        assert (
            repr(self.TDF) == "TemporalDataFrame No_Name\n"
            "Time point : 0.0 hours\n"
            "‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾\n"
            "  Time-point    col1    col2\n"
            "d       0.0h  ｜  4.0  ｜    d\n"
            "e       0.0h  ｜  5.0  ｜    e\n"
            "f       0.0h  ｜  6.0  ｜    f\n"
            "[3 rows x 2 columns]\n"
            "\n"
            "Time point : 5.0 hours\n"
            "‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾\n"
            "  Time-point    col1    col2\n"
            "g       5.0h  ｜  7.0  ｜    g\n"
            "h       5.0h  ｜  8.0  ｜    h\n"
            "i       5.0h  ｜  9.0  ｜    i\n"
            "[3 rows x 2 columns]\n"
            "\n"
            "Time point : 10.0 hours\n"
            "‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾\n"
            "  Time-point    col1    col2\n"
            "a      10.0h  ｜  1.0  ｜    a\n"
            "b      10.0h  ｜  2.0  ｜    b\n"
            "c      10.0h  ｜  3.0  ｜    c\n"
            "[3 rows x 2 columns]\n\n"
        )

    def test_index_in_correct_order(self) -> None:
        assert np.array_equal(self.TDF.index, np.array(["d", "e", "f", "g", "h", "i", "a", "b", "c"]))

    def test_columns_in_correct_order(self) -> None:
        assert np.all(self.TDF.columns == np.array(["col1", "col2"]))

    def test_timepoints(self) -> None:
        assert np.all(self.TDF.timepoints == tp.TimePointArray([0, 5, 10], unit="h"))

    def test_timepoints_column(self) -> None:
        assert np.all(self.TDF.timepoints_column == tp.TimePointArray([0, 0, 0, 5, 5, 5, 10, 10, 10], unit="h"))

    def test_values_numerical_in_correct_order(self) -> None:
        assert np.all(self.TDF.values_num == np.array([[4], [5], [6], [7], [8], [9], [1], [2], [3]]))

    def test_values_string_in_correct_order(self) -> None:
        assert np.all(self.TDF.values_str == np.array([["d"], ["e"], ["f"], ["g"], ["h"], ["i"], ["a"], ["b"], ["c"]]))


@pytest.mark.usefixtures("class_TDF_backed")
class TestCreationBackedTDF:
    TDF: TemporalDataFrame

    def test_creation(self) -> None:
        assert (
            repr(self.TDF) == "Backed TemporalDataFrame 1\n"
            "Time point : 0.0 hours\n"
            "‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾\n"
            "   Time-point     col1   col2    col3 col4\n"
            "50       0.0h  ｜  50.0  150.0  ｜  250  350\n"
            "51       0.0h  ｜  51.0  151.0  ｜  251  351\n"
            "52       0.0h  ｜  52.0  152.0  ｜  252  352\n"
            "53       0.0h  ｜  53.0  153.0  ｜  253  353\n"
            "54       0.0h  ｜  54.0  154.0  ｜  254  354\n"
            "[50 rows x 4 columns]\n"
            "\n"
            "Time point : 1.0 hour\n"
            "‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾\n"
            "  Time-point    col1   col2    col3 col4\n"
            "0       1.0h  ｜  0.0  100.0  ｜  200  300\n"
            "1       1.0h  ｜  1.0  101.0  ｜  201  301\n"
            "2       1.0h  ｜  2.0  102.0  ｜  202  302\n"
            "3       1.0h  ｜  3.0  103.0  ｜  203  303\n"
            "4       1.0h  ｜  4.0  104.0  ｜  204  304\n"
            "[50 rows x 4 columns]\n\n"
        )

    def test_is_backed(self) -> None:
        assert self.TDF.is_backed


@pytest.mark.parametrize("TDF", ["plain", "backed"], indirect=True)
def todo_test_inverted_TDF_creation(TDF: TemporalDataFrameBase) -> None:
    assert (
        repr(~TDF["0h", :, :]) == "Inverted view of TemporalDataFrame 1\n"
        "Time point : 0.0 hours\n"
        "‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾\n"
        "   Time-point     col1   col2    col3 col4\n"
        "50       0.0h  ｜  50.0  150.0  ｜  250  350\n"
        "51       0.0h  ｜  51.0  151.0  ｜  251  351\n"
        "52       0.0h  ｜  52.0  152.0  ｜  252  352\n"
        "53       0.0h  ｜  53.0  153.0  ｜  253  353\n"
        "54       0.0h  ｜  54.0  154.0  ｜  254  354\n"
        "[50 rows x 4 columns]\n"
        "\n"
        "Time point : 1.0 hour\n"
        "‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾\n"
        "  Time-point    col1   col2    col3 col4\n"
        "0       1.0h  ｜  0.0  100.0  ｜  200  300\n"
        "1       1.0h  ｜  1.0  101.0  ｜  201  301\n"
        "2       1.0h  ｜  2.0  102.0  ｜  202  302\n"
        "3       1.0h  ｜  3.0  103.0  ｜  203  303\n"
        "4       1.0h  ｜  4.0  104.0  ｜  204  304\n"
        "[50 rows x 4 columns]\n\n"
    )


@pytest.mark.parametrize("TDF", ["view", "backed view"], indirect=True)
def todo_test_inverted_TDF_view_creation(TDF: TemporalDataFrameBase) -> None:
    assert (
        repr(~TDF["0h", "0", :]) == "Inverted view of TemporalDataFrame 1\n"
        "Time point : 0.0 hours\n"
        "Time point : 1.0 hour\n"
        "  Time-point    col1    col4\n"
        "1       1.0h  ｜  1.0  ｜  301\n"
        "2       1.0h  ｜  2.0  ｜  302\n"
        "3       1.0h  ｜  3.0  ｜  303\n"
        "4       1.0h  ｜  4.0  ｜  304\n"
        "4       1.0h  ｜  4.0  ｜  304\n"
        "[39 rows x 2 columns]\n\n"
    )
