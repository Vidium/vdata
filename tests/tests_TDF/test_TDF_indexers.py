from typing import Any

import numpy as np
import numpy.typing as npt
import pytest

from vdata.tdf import TemporalDataFrameBase


@pytest.mark.parametrize(
    "TDF,expected", [("plain", 10), ("view", 10), ("backed", 10), ("backed view", 10)], indirect=["TDF"]
)
def test_at_indexer_gets_correct_value(TDF: TemporalDataFrameBase, expected: int) -> None:
    assert TDF.at[10, "col1"] == expected


@pytest.mark.parametrize("TDF", ["plain", "view", "backed", "backed view"], indirect=["TDF"])
def test_at_indexer_sets_corret_value(TDF: TemporalDataFrameBase) -> None:
    TDF.at[10, "col1"] = -1

    if TDF.is_view:
        index = 0
    else:
        index = 60

    assert TDF.values_num[index, 0] == -1


@pytest.mark.parametrize(
    "TDF,expected", [("plain", 10), ("view", 10), ("backed", 10), ("backed view", 10)], indirect=["TDF"]
)
def test_iat_indexer_gets_correct_value(TDF: TemporalDataFrameBase, expected: int) -> None:
    if TDF.is_view:
        index = 0
    else:
        index = 60

    assert TDF.iat[index, 0] == expected


@pytest.mark.parametrize("TDF", ["plain", "view", "backed", "backed view"], indirect=["TDF"])
def test_iat_indexer_sets_corret_value(TDF: TemporalDataFrameBase) -> None:
    TDF.iat[10, 0] = -1

    assert TDF.values_num[10, 0] == -1


@pytest.mark.parametrize(
    "TDF, expected",
    [
        ("plain", np.array([[10.0, 110.0, "210", "310"]], dtype=object)),
        ("view", np.array([[10.0, "310"]], dtype=object)),
        ("backed", np.array([[10.0, 110.0, "210", "310"]], dtype=object)),
        ("backed view", np.array([[10.0, "310"]], dtype=object)),
    ],
    indirect=["TDF"],
)
def test_loc_indexer_gets_correct_value_from_index(TDF: TemporalDataFrameBase, expected: npt.NDArray[Any]) -> None:
    assert np.all(TDF.loc[10].values == expected)


@pytest.mark.parametrize("TDF", ["plain", "backed"], indirect=True)
def test_loc_indexer_sets_correct_value_from_index(TDF: TemporalDataFrameBase) -> None:
    TDF.loc[10] = [-10, -110, "A", "B"]
    assert np.all(TDF.values_num[60] == [-10, -110]) and np.all(np.char.equal(TDF.values_str[60], ["A", "B"]))


@pytest.mark.parametrize("TDF", ["view", "backed view"], indirect=True)
def test_loc_indexer_sets_correct_value_from_index_for_view(TDF: TemporalDataFrameBase) -> None:
    TDF.loc[10] = [-10, "B"]

    TDF.values_num[0] == -10

    assert TDF.values_num[0] == -10 and TDF.values_str[0] == "B"


@pytest.mark.parametrize("TDF", ["plain", "backed"], indirect=True)
def test_loc_indexer_sets_correct_value_from_slice(TDF: TemporalDataFrameBase) -> None:
    TDF.loc[10:20] = np.tile([-10, -110, "A", "B"], (10, 1))
    assert np.all(TDF.values_num[60:70] == [-10, -110]) and np.all(np.char.equal(TDF.values_str[60:70], ["A", "B"]))


@pytest.mark.parametrize("TDF", ["view", "backed view"], indirect=True)
def test_loc_indexer_sets_correct_value_from_slice_for_view(TDF: TemporalDataFrameBase) -> None:
    TDF.loc[10:20] = np.tile([-10, "B"], (10, 1))
    assert np.all(TDF.values_num[0:10] == -10) and np.all(TDF.values_str[0:10] == "B")


@pytest.mark.parametrize("TDF", ["plain", "backed"], indirect=True)
def test_loc_indexer_sets_correct_value_from_list(TDF: TemporalDataFrameBase) -> None:
    TDF.loc[[10, 12, 14]] = np.tile([-10, -110, "A", "B"], (3, 1))
    assert np.all(TDF.values_num[[60, 62, 64]] == [-10, -110]) and np.all(
        np.char.equal(TDF.values_str[[60, 62, 64]], ["A", "B"])
    )


@pytest.mark.parametrize("TDF", ["view", "backed view"], indirect=True)
def test_loc_indexer_sets_correct_value_from_list_for_view(TDF: TemporalDataFrameBase) -> None:
    TDF.loc[[10, 12, 14]] = np.tile([-10, "B"], (3, 1))
    assert np.all(TDF.values_num[[0, 2, 4]] == -10) and np.all(TDF.values_str[[0, 2, 4]] == "B")


@pytest.mark.parametrize("TDF", ["plain", "backed"], indirect=True)
def test_loc_indexer_sets_correct_value_from_boolean_array(TDF: TemporalDataFrameBase) -> None:
    mask = np.zeros(100, dtype=bool)
    mask[[60, 62, 64]] = True

    TDF.loc[mask] = np.tile([-10, -110, "A", "B"], (3, 1))
    assert np.all(TDF.values_num[[60, 62, 64]] == [-10, -110]) and np.all(
        np.char.equal(TDF.values_str[[60, 62, 64]], ["A", "B"])
    )


@pytest.mark.parametrize("TDF", ["view", "backed view"], indirect=True)
def test_loc_indexer_sets_correct_value_from_boolean_array_for_view(TDF: TemporalDataFrameBase) -> None:
    mask = np.zeros(80, dtype=bool)
    mask[[40, 42, 44]] = True

    TDF.loc[mask] = np.tile([-10, "B"], (3, 1))
    assert np.all(TDF.values_num[[40, 42, 44]] == -10) and np.all(TDF.values_str[[40, 42, 44]] == "B")


@pytest.mark.parametrize("TDF", ["plain", "backed"], indirect=True)
def test_loc_indexer_sets_correct_value_with_single_column(TDF: TemporalDataFrameBase) -> None:
    TDF.loc[10:20, "col2"] = [-110] * 10
    assert np.all(TDF.values_num[60:70, 1] == -110)


@pytest.mark.parametrize("TDF", ["view", "backed view"], indirect=True)
def test_loc_indexer_sets_correct_value_with_single_column_for_view(TDF: TemporalDataFrameBase) -> None:
    TDF.loc[10:20, "col1"] = [-10] * 10
    assert np.all(TDF.values_num[0:10, 0] == -10)


@pytest.mark.parametrize("TDF", ["plain", "backed"], indirect=True)
def test_iloc_indexer_gets_correct_value_with_index(TDF: TemporalDataFrameBase) -> None:
    view = TDF.iloc[60]

    np.char.equal(view.values_str, ["210", "310"])

    assert np.all(view.values_num == [10, 110]) and np.all(np.char.equal(view.values_str, ["210", "310"]))


@pytest.mark.parametrize("TDF", ["view", "backed view"], indirect=True)
def test_view_iloc_indexer_gets_correct_value_with_index(TDF: TemporalDataFrameBase) -> None:
    view = TDF.iloc[40]
    assert view.values_num == 50 and view.values_str == "350"


@pytest.mark.parametrize("TDF", ["plain", "backed"], indirect=True)
def test_iloc_indexer_gets_correct_value_with_index_and_column(TDF: TemporalDataFrameBase) -> None:
    view = TDF.iloc[60:70, [0, 2]]
    assert np.all(view.values_num == np.arange(10, 20).reshape((10, 1))) and np.all(
        np.char.equal(view.values_str, np.arange(210, 220).astype(str).reshape((10, 1)))
    )


@pytest.mark.parametrize("TDF", ["view", "backed view"], indirect=True)
def test_iloc_indexer_gets_correct_value_with_index_and_column_for_view(TDF: TemporalDataFrameBase) -> None:
    view = TDF.iloc[40:50, 0]
    assert np.all(view.values_num == np.arange(50, 60).reshape((10, 1))) and view.values_str.size == 0


@pytest.mark.parametrize("TDF", ["plain", "backed"], indirect=True)
def test_iloc_indexer_sets_correct_value_with_index_and_column(TDF: TemporalDataFrameBase) -> None:
    TDF.iloc[60:70, [True, False, False, True]] = np.zeros((10, 2))
    assert (
        np.all(TDF.values_num[60:70, 0] == np.zeros(10))
        and np.all(TDF.values_num[60:70, 1] == np.arange(110, 120))
        and np.all(np.char.equal(TDF.values_str[60:70, 0], np.arange(210, 220).astype(str)))
        and np.all(np.char.equal(TDF.values_str[60:70, 1], np.zeros(10).astype(str)))
    )


@pytest.mark.parametrize("TDF", ["view", "backed view"], indirect=True)
def test_iloc_indexer_sets_correct_value_with_index_and_column_for_view(TDF: TemporalDataFrameBase) -> None:
    TDF.iloc[40:50, 1:] = np.arange(-1, -11, -1)
    assert np.all(TDF.values_num[40:50] == np.arange(50, 60).reshape((10, 1))) and np.all(
        np.char.equal(TDF.values_str[40:50], np.arange(-1, -11, -1).astype(str).reshape((10, 1)))
    )
