# coding: utf-8
# Created on 05/04/2022 09:11
# Author : matteo

# ====================================================
# imports
import numpy as np
import pytest

from vdata.tdf import TemporalDataFrame, TemporalDataFrameBase, TemporalDataFrameView


# ====================================================
# code
@pytest.mark.parametrize("TDF", ["plain", "backed"], indirect=True)
def test_delete_numerical_column(TDF: TemporalDataFrame) -> None:
    if TDF.is_backed:
        TDF.unlock_columns()

    del TDF.col1

    prefix = "Backed " if TDF.is_backed else ""
    assert (
        repr(TDF) == f"{prefix}TemporalDataFrame 1\n"
        "Time point : 0.0 hours\n"
        "‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾\n"
        "   Time-point      col2    col3 col4\n"
        "50       0.0h  ｜  150.0  ｜  250  350\n"
        "51       0.0h  ｜  151.0  ｜  251  351\n"
        "52       0.0h  ｜  152.0  ｜  252  352\n"
        "53       0.0h  ｜  153.0  ｜  253  353\n"
        "54       0.0h  ｜  154.0  ｜  254  354\n"
        "[50 rows x 3 columns]\n"
        "\n"
        "Time point : 1.0 hour\n"
        "‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾\n"
        "  Time-point      col2    col3 col4\n"
        "0       1.0h  ｜  100.0  ｜  200  300\n"
        "1       1.0h  ｜  101.0  ｜  201  301\n"
        "2       1.0h  ｜  102.0  ｜  202  302\n"
        "3       1.0h  ｜  103.0  ｜  203  303\n"
        "4       1.0h  ｜  104.0  ｜  204  304\n"
        "[50 rows x 3 columns]\n\n"
    )

    assert np.all(TDF.values_num == np.concatenate((np.arange(150, 200), np.arange(100, 150)))[:, None])
    assert np.all(
        TDF.values_str
        == np.vstack(
            (
                np.concatenate((np.arange(250, 300).astype(str), np.arange(200, 250).astype(str))),
                np.concatenate((np.arange(350, 400).astype(str), np.arange(300, 350).astype(str))),
            )
        ).T
    )


@pytest.mark.parametrize("TDF", ["plain", "backed"], indirect=True)
def test_delete_string_column(TDF: TemporalDataFrame) -> None:
    if TDF.is_backed:
        TDF.unlock_columns()

    del TDF.col4

    prefix = "Backed " if TDF.is_backed else ""
    assert (
        repr(TDF) == f"{prefix}TemporalDataFrame 1\n"
        "Time point : 0.0 hours\n"
        "‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾\n"
        "   Time-point     col1   col2    col3\n"
        "50       0.0h  ｜  50.0  150.0  ｜  250\n"
        "51       0.0h  ｜  51.0  151.0  ｜  251\n"
        "52       0.0h  ｜  52.0  152.0  ｜  252\n"
        "53       0.0h  ｜  53.0  153.0  ｜  253\n"
        "54       0.0h  ｜  54.0  154.0  ｜  254\n"
        "[50 rows x 3 columns]\n"
        "\n"
        "Time point : 1.0 hour\n"
        "‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾\n"
        "  Time-point    col1   col2    col3\n"
        "0       1.0h  ｜  0.0  100.0  ｜  200\n"
        "1       1.0h  ｜  1.0  101.0  ｜  201\n"
        "2       1.0h  ｜  2.0  102.0  ｜  202\n"
        "3       1.0h  ｜  3.0  103.0  ｜  203\n"
        "4       1.0h  ｜  4.0  104.0  ｜  204\n"
        "[50 rows x 3 columns]\n\n"
    )

    assert np.all(
        TDF.values_num
        == np.vstack(
            (
                np.concatenate((np.arange(50, 100), np.arange(0, 50))),
                np.concatenate((np.arange(150, 200), np.arange(100, 150))),
            )
        ).T.astype(float)
    )
    assert np.all(
        TDF.values_str == np.concatenate((np.arange(250, 300).astype(str), np.arange(200, 250).astype(str)))[:, None]
    )


@pytest.mark.parametrize("TDF", ["plain", "backed"], indirect=True)
def test_modify_numerical_values_of_existing_column(TDF: TemporalDataFrame) -> None:
    TDF.col1 = np.arange(500, 600)

    prefix = "Backed " if TDF.is_backed else ""
    assert (
        repr(TDF) == f"{prefix}TemporalDataFrame 1\n"
        "Time point : 0.0 hours\n"
        "‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾\n"
        "   Time-point      col1   col2    col3 col4\n"
        "50       0.0h  ｜  500.0  150.0  ｜  250  350\n"
        "51       0.0h  ｜  501.0  151.0  ｜  251  351\n"
        "52       0.0h  ｜  502.0  152.0  ｜  252  352\n"
        "53       0.0h  ｜  503.0  153.0  ｜  253  353\n"
        "54       0.0h  ｜  504.0  154.0  ｜  254  354\n"
        "[50 rows x 4 columns]\n"
        "\n"
        "Time point : 1.0 hour\n"
        "‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾\n"
        "  Time-point      col1   col2    col3 col4\n"
        "0       1.0h  ｜  550.0  100.0  ｜  200  300\n"
        "1       1.0h  ｜  551.0  101.0  ｜  201  301\n"
        "2       1.0h  ｜  552.0  102.0  ｜  202  302\n"
        "3       1.0h  ｜  553.0  103.0  ｜  203  303\n"
        "4       1.0h  ｜  554.0  104.0  ｜  204  304\n"
        "[50 rows x 4 columns]\n\n"
    )


@pytest.mark.parametrize("TDF", ["plain", "backed"], indirect=True)
def test_create_new_column_numerical(TDF: TemporalDataFrame) -> None:
    TDF.col7 = np.arange(600, 700)

    prefix = "Backed " if TDF.is_backed else ""
    assert (
        repr(TDF) == f"{prefix}TemporalDataFrame 1\n"
        "Time point : 0.0 hours\n"
        "‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾\n"
        "   Time-point     col1   col2   col7    col3 col4\n"
        "50       0.0h  ｜  50.0  150.0  600.0  ｜  250  350\n"
        "51       0.0h  ｜  51.0  151.0  601.0  ｜  251  351\n"
        "52       0.0h  ｜  52.0  152.0  602.0  ｜  252  352\n"
        "53       0.0h  ｜  53.0  153.0  603.0  ｜  253  353\n"
        "54       0.0h  ｜  54.0  154.0  604.0  ｜  254  354\n"
        "[50 rows x 5 columns]\n"
        "\n"
        "Time point : 1.0 hour\n"
        "‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾\n"
        "  Time-point    col1   col2   col7    col3 col4\n"
        "0       1.0h  ｜  0.0  100.0  650.0  ｜  200  300\n"
        "1       1.0h  ｜  1.0  101.0  651.0  ｜  201  301\n"
        "2       1.0h  ｜  2.0  102.0  652.0  ｜  202  302\n"
        "3       1.0h  ｜  3.0  103.0  653.0  ｜  203  303\n"
        "4       1.0h  ｜  4.0  104.0  654.0  ｜  204  304\n"
        "[50 rows x 5 columns]\n\n"
    )


@pytest.mark.parametrize("TDF", ["plain", "backed"], indirect=True)
def test_modify_string_values_of_existing_column(TDF: TemporalDataFrame) -> None:
    TDF.col4 = np.arange(700, 800).astype(str)

    prefix = "Backed " if TDF.is_backed else ""
    assert (
        repr(TDF) == f"{prefix}TemporalDataFrame 1\n"
        "Time point : 0.0 hours\n"
        "‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾\n"
        "   Time-point     col1   col2    col3 col4\n"
        "50       0.0h  ｜  50.0  150.0  ｜  250  700\n"
        "51       0.0h  ｜  51.0  151.0  ｜  251  701\n"
        "52       0.0h  ｜  52.0  152.0  ｜  252  702\n"
        "53       0.0h  ｜  53.0  153.0  ｜  253  703\n"
        "54       0.0h  ｜  54.0  154.0  ｜  254  704\n"
        "[50 rows x 4 columns]\n"
        "\n"
        "Time point : 1.0 hour\n"
        "‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾\n"
        "  Time-point    col1   col2    col3 col4\n"
        "0       1.0h  ｜  0.0  100.0  ｜  200  750\n"
        "1       1.0h  ｜  1.0  101.0  ｜  201  751\n"
        "2       1.0h  ｜  2.0  102.0  ｜  202  752\n"
        "3       1.0h  ｜  3.0  103.0  ｜  203  753\n"
        "4       1.0h  ｜  4.0  104.0  ｜  204  754\n"
        "[50 rows x 4 columns]\n\n"
    )


@pytest.mark.parametrize("TDF", ["plain", "backed"], indirect=True)
def test_create_new_column_string(TDF: TemporalDataFrame) -> None:
    TDF.col8 = np.arange(800, 900).astype(str)

    prefix = "Backed " if TDF.is_backed else ""
    assert (
        repr(TDF) == f"{prefix}TemporalDataFrame 1\n"
        "Time point : 0.0 hours\n"
        "‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾\n"
        "   Time-point     col1   col2    col3 col4 col8\n"
        "50       0.0h  ｜  50.0  150.0  ｜  250  350  800\n"
        "51       0.0h  ｜  51.0  151.0  ｜  251  351  801\n"
        "52       0.0h  ｜  52.0  152.0  ｜  252  352  802\n"
        "53       0.0h  ｜  53.0  153.0  ｜  253  353  803\n"
        "54       0.0h  ｜  54.0  154.0  ｜  254  354  804\n"
        "[50 rows x 5 columns]\n"
        "\n"
        "Time point : 1.0 hour\n"
        "‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾\n"
        "  Time-point    col1   col2    col3 col4 col8\n"
        "0       1.0h  ｜  0.0  100.0  ｜  200  300  850\n"
        "1       1.0h  ｜  1.0  101.0  ｜  201  301  851\n"
        "2       1.0h  ｜  2.0  102.0  ｜  202  302  852\n"
        "3       1.0h  ｜  3.0  103.0  ｜  203  303  853\n"
        "4       1.0h  ｜  4.0  104.0  ｜  204  304  854\n"
        "[50 rows x 5 columns]\n\n"
    )


@pytest.mark.parametrize("TDF", ["plain", "backed"], indirect=True)
def test_insert_numerical_column(TDF: TemporalDataFrame) -> None:
    if TDF.is_backed:
        TDF.unlock_columns()

    TDF.insert("col5", np.arange(500, 600), loc=1)

    prefix = "Backed " if TDF.is_backed else ""
    assert (
        repr(TDF) == f"{prefix}TemporalDataFrame 1\n"
        "Time point : 0.0 hours\n"
        "‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾\n"
        "   Time-point     col1   col5   col2    col3 col4\n"
        "50       0.0h  ｜  50.0  500.0  150.0  ｜  250  350\n"
        "51       0.0h  ｜  51.0  501.0  151.0  ｜  251  351\n"
        "52       0.0h  ｜  52.0  502.0  152.0  ｜  252  352\n"
        "53       0.0h  ｜  53.0  503.0  153.0  ｜  253  353\n"
        "54       0.0h  ｜  54.0  504.0  154.0  ｜  254  354\n"
        "[50 rows x 5 columns]\n"
        "\n"
        "Time point : 1.0 hour\n"
        "‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾\n"
        "  Time-point    col1   col5   col2    col3 col4\n"
        "0       1.0h  ｜  0.0  550.0  100.0  ｜  200  300\n"
        "1       1.0h  ｜  1.0  551.0  101.0  ｜  201  301\n"
        "2       1.0h  ｜  2.0  552.0  102.0  ｜  202  302\n"
        "3       1.0h  ｜  3.0  553.0  103.0  ｜  203  303\n"
        "4       1.0h  ｜  4.0  554.0  104.0  ｜  204  304\n"
        "[50 rows x 5 columns]\n\n"
    )


@pytest.mark.parametrize("TDF", ["plain", "backed"], indirect=True)
def test_insert_string_column(TDF: TemporalDataFrame) -> None:
    if TDF.is_backed:
        TDF.unlock_columns()

    TDF.insert("col6", np.arange(700, 800).astype(str), loc=2)

    prefix = "Backed " if TDF.is_backed else ""
    assert (
        repr(TDF) == f"{prefix}TemporalDataFrame 1\n"
        "Time point : 0.0 hours\n"
        "‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾\n"
        "   Time-point     col1   col2    col3 col4 col6\n"
        "50       0.0h  ｜  50.0  150.0  ｜  250  350  700\n"
        "51       0.0h  ｜  51.0  151.0  ｜  251  351  701\n"
        "52       0.0h  ｜  52.0  152.0  ｜  252  352  702\n"
        "53       0.0h  ｜  53.0  153.0  ｜  253  353  703\n"
        "54       0.0h  ｜  54.0  154.0  ｜  254  354  704\n"
        "[50 rows x 5 columns]\n"
        "\n"
        "Time point : 1.0 hour\n"
        "‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾\n"
        "  Time-point    col1   col2    col3 col4 col6\n"
        "0       1.0h  ｜  0.0  100.0  ｜  200  300  750\n"
        "1       1.0h  ｜  1.0  101.0  ｜  201  301  751\n"
        "2       1.0h  ｜  2.0  102.0  ｜  202  302  752\n"
        "3       1.0h  ｜  3.0  103.0  ｜  203  303  753\n"
        "4       1.0h  ｜  4.0  104.0  ｜  204  304  754\n"
        "[50 rows x 5 columns]\n\n"
    )


@pytest.mark.parametrize("TDF", ["plain", "backed"], indirect=True)
def test_insert_into_empty_array(TDF: TemporalDataFrame) -> None:
    if TDF.is_backed:
        TDF.unlock_columns()

    del TDF.col1
    del TDF.col2

    TDF.insert("col1", np.arange(800, 900), loc=0)

    prefix = "Backed " if TDF.is_backed else ""
    assert (
        repr(TDF) == f"{prefix}TemporalDataFrame 1\n"
        "Time point : 0.0 hours\n"
        "‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾\n"
        "   Time-point      col1    col3 col4\n"
        "50       0.0h  ｜  800.0  ｜  250  350\n"
        "51       0.0h  ｜  801.0  ｜  251  351\n"
        "52       0.0h  ｜  802.0  ｜  252  352\n"
        "53       0.0h  ｜  803.0  ｜  253  353\n"
        "54       0.0h  ｜  804.0  ｜  254  354\n"
        "[50 rows x 3 columns]\n"
        "\n"
        "Time point : 1.0 hour\n"
        "‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾\n"
        "  Time-point      col1    col3 col4\n"
        "0       1.0h  ｜  850.0  ｜  200  300\n"
        "1       1.0h  ｜  851.0  ｜  201  301\n"
        "2       1.0h  ｜  852.0  ｜  202  302\n"
        "3       1.0h  ｜  853.0  ｜  203  303\n"
        "4       1.0h  ｜  854.0  ｜  204  304\n"
        "[50 rows x 3 columns]\n\n"
    )


@pytest.mark.parametrize("TDF", ["plain", "backed"], indirect=True)
def test_insert_single_numerical_value(TDF: TemporalDataFrame) -> None:
    TDF.insert("col5", -1, loc=0)

    assert TDF.n_columns_num == 3 and np.all(TDF.values_num[:, 0] == -1)


@pytest.mark.parametrize("TDF", ["plain", "backed"], indirect=True)
def test_insert_single_string_value(TDF: TemporalDataFrame) -> None:
    TDF.insert("col5", "a", loc=0)

    assert TDF.n_columns_str == 3 and np.all(TDF.values_str[:, 0] == "a")


@pytest.mark.parametrize("TDF", ["plain", "backed"], indirect=True)
def test_should_insert_numerical_column_as_last(TDF: TemporalDataFrame) -> None:
    TDF.insert("col5", -1, loc=-1)

    assert np.all(TDF.values_num[:, -1] == -1)


@pytest.mark.parametrize("TDF", ["plain", "backed"], indirect=True)
def test_should_insert_string_column_as_last(TDF: TemporalDataFrame) -> None:
    TDF.insert("col5", "a", loc=-1)

    assert np.all(TDF.values_str[:, -1] == "a")


@pytest.mark.parametrize("TDF", ["plain", "backed"], indirect=True)
def test_add_numerical_value(TDF: TemporalDataFrame) -> None:
    prefix = "Backed " if TDF.is_backed else ""
    assert (
        repr(TDF + 1) == f"TemporalDataFrame {prefix}TemporalDataFrame 1 + 1\n"
        "Time point : 0.0 hours\n"
        "‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾\n"
        "   Time-point     col1   col2    col3 col4\n"
        "50       0.0h  ｜  51.0  151.0  ｜  250  350\n"
        "51       0.0h  ｜  52.0  152.0  ｜  251  351\n"
        "52       0.0h  ｜  53.0  153.0  ｜  252  352\n"
        "53       0.0h  ｜  54.0  154.0  ｜  253  353\n"
        "54       0.0h  ｜  55.0  155.0  ｜  254  354\n"
        "[50 rows x 4 columns]\n"
        "\n"
        "Time point : 1.0 hour\n"
        "‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾\n"
        "  Time-point    col1   col2    col3 col4\n"
        "0       1.0h  ｜  1.0  101.0  ｜  200  300\n"
        "1       1.0h  ｜  2.0  102.0  ｜  201  301\n"
        "2       1.0h  ｜  3.0  103.0  ｜  202  302\n"
        "3       1.0h  ｜  4.0  104.0  ｜  203  303\n"
        "4       1.0h  ｜  5.0  105.0  ｜  204  304\n"
        "[50 rows x 4 columns]\n\n"
    )


@pytest.mark.parametrize("TDF", ["view", "backed view"], indirect=True)
def test_add_numerical_value_to_view(TDF: TemporalDataFrame) -> None:
    prefix = "backed " if TDF.is_backed else ""

    assert (
        repr(TDF + 1) == f"TemporalDataFrame View of {prefix}TemporalDataFrame 1 + 1\n"
        "Time point : 1.0 hour\n"
        "‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾\n"
        "   Time-point     col1    col4\n"
        "10       1.0h  ｜  11.0  ｜  310\n"
        "11       1.0h  ｜  12.0  ｜  311\n"
        "12       1.0h  ｜  13.0  ｜  312\n"
        "13       1.0h  ｜  14.0  ｜  313\n"
        "14       1.0h  ｜  15.0  ｜  314\n"
        "[40 rows x 2 columns]\n"
        "\n"
        "Time point : 0.0 hours\n"
        "‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾\n"
        "   Time-point     col1    col4\n"
        "50       0.0h  ｜  51.0  ｜  350\n"
        "51       0.0h  ｜  52.0  ｜  351\n"
        "52       0.0h  ｜  53.0  ｜  352\n"
        "53       0.0h  ｜  54.0  ｜  353\n"
        "54       0.0h  ｜  55.0  ｜  354\n"
        "[40 rows x 2 columns]\n\n"
    )


@pytest.mark.parametrize("TDF", ["plain", "backed"], indirect=True)
def test_add_string_value(TDF: TemporalDataFrame) -> None:
    prefix = "Backed " if TDF.is_backed else ""
    assert (
        repr(TDF + "_1") == f"TemporalDataFrame {prefix}TemporalDataFrame 1 + _1\n"
        "Time point : 0.0 hours\n"
        "‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾\n"
        "   Time-point     col1   col2      col3   col4\n"
        "50       0.0h  ｜  50.0  150.0  ｜  250_1  350_1\n"
        "51       0.0h  ｜  51.0  151.0  ｜  251_1  351_1\n"
        "52       0.0h  ｜  52.0  152.0  ｜  252_1  352_1\n"
        "53       0.0h  ｜  53.0  153.0  ｜  253_1  353_1\n"
        "54       0.0h  ｜  54.0  154.0  ｜  254_1  354_1\n"
        "[50 rows x 4 columns]\n"
        "\n"
        "Time point : 1.0 hour\n"
        "‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾\n"
        "  Time-point    col1   col2      col3   col4\n"
        "0       1.0h  ｜  0.0  100.0  ｜  200_1  300_1\n"
        "1       1.0h  ｜  1.0  101.0  ｜  201_1  301_1\n"
        "2       1.0h  ｜  2.0  102.0  ｜  202_1  302_1\n"
        "3       1.0h  ｜  3.0  103.0  ｜  203_1  303_1\n"
        "4       1.0h  ｜  4.0  104.0  ｜  204_1  304_1\n"
        "[50 rows x 4 columns]\n\n"
    )


@pytest.mark.parametrize("TDF", ["view", "backed view"], indirect=True)
def test_add_string_value_to_view(TDF: TemporalDataFrameView) -> None:
    prefix = "backed " if TDF.is_backed else ""
    assert (
        repr(TDF + "_1") == f"TemporalDataFrame View of {prefix}TemporalDataFrame 1 + _1\n"
        "Time point : 1.0 hour\n"
        "‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾\n"
        "   Time-point     col1      col4\n"
        "10       1.0h  ｜  10.0  ｜  310_1\n"
        "11       1.0h  ｜  11.0  ｜  311_1\n"
        "12       1.0h  ｜  12.0  ｜  312_1\n"
        "13       1.0h  ｜  13.0  ｜  313_1\n"
        "14       1.0h  ｜  14.0  ｜  314_1\n"
        "[40 rows x 2 columns]\n"
        "\n"
        "Time point : 0.0 hours\n"
        "‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾\n"
        "   Time-point     col1      col4\n"
        "50       0.0h  ｜  50.0  ｜  350_1\n"
        "51       0.0h  ｜  51.0  ｜  351_1\n"
        "52       0.0h  ｜  52.0  ｜  352_1\n"
        "53       0.0h  ｜  53.0  ｜  353_1\n"
        "54       0.0h  ｜  54.0  ｜  354_1\n"
        "[40 rows x 2 columns]\n\n"
    )


@pytest.mark.parametrize("TDF", ["plain", "backed"], indirect=True)
def test_subtract_value(TDF: TemporalDataFrame) -> None:
    prefix = "Backed " if TDF.is_backed else ""
    assert (
        repr(TDF - 1) == f"TemporalDataFrame {prefix}TemporalDataFrame 1 - 1\n"
        "Time point : 0.0 hours\n"
        "‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾\n"
        "   Time-point     col1   col2    col3 col4\n"
        "50       0.0h  ｜  49.0  149.0  ｜  250  350\n"
        "51       0.0h  ｜  50.0  150.0  ｜  251  351\n"
        "52       0.0h  ｜  51.0  151.0  ｜  252  352\n"
        "53       0.0h  ｜  52.0  152.0  ｜  253  353\n"
        "54       0.0h  ｜  53.0  153.0  ｜  254  354\n"
        "[50 rows x 4 columns]\n"
        "\n"
        "Time point : 1.0 hour\n"
        "‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾\n"
        "  Time-point    col1   col2    col3 col4\n"
        "0       1.0h  ｜ -1.0   99.0  ｜  200  300\n"
        "1       1.0h  ｜  0.0  100.0  ｜  201  301\n"
        "2       1.0h  ｜  1.0  101.0  ｜  202  302\n"
        "3       1.0h  ｜  2.0  102.0  ｜  203  303\n"
        "4       1.0h  ｜  3.0  103.0  ｜  204  304\n"
        "[50 rows x 4 columns]\n\n"
    )


@pytest.mark.parametrize("TDF", ["view", "backed view"], indirect=True)
def test_subtract_value_to_view(TDF: TemporalDataFrameView) -> None:
    prefix = "backed " if TDF.is_backed else ""
    assert (
        repr(TDF - 1) == f"TemporalDataFrame View of {prefix}TemporalDataFrame 1 - 1\n"
        "Time point : 1.0 hour\n"
        "‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾\n"
        "   Time-point     col1    col4\n"
        "10       1.0h  ｜   9.0  ｜  310\n"
        "11       1.0h  ｜  10.0  ｜  311\n"
        "12       1.0h  ｜  11.0  ｜  312\n"
        "13       1.0h  ｜  12.0  ｜  313\n"
        "14       1.0h  ｜  13.0  ｜  314\n"
        "[40 rows x 2 columns]\n"
        "\n"
        "Time point : 0.0 hours\n"
        "‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾\n"
        "   Time-point     col1    col4\n"
        "50       0.0h  ｜  49.0  ｜  350\n"
        "51       0.0h  ｜  50.0  ｜  351\n"
        "52       0.0h  ｜  51.0  ｜  352\n"
        "53       0.0h  ｜  52.0  ｜  353\n"
        "54       0.0h  ｜  53.0  ｜  354\n"
        "[40 rows x 2 columns]\n\n"
    )


@pytest.mark.parametrize("TDF", ["plain", "backed"], indirect=True)
def test_mutliply_by_value(TDF: TemporalDataFrame) -> None:
    prefix = "Backed " if TDF.is_backed else ""
    assert (
        repr(TDF * 2) == f"TemporalDataFrame {prefix}TemporalDataFrame 1 * 2\n"
        "Time point : 0.0 hours\n"
        "‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾\n"
        "   Time-point      col1   col2    col3 col4\n"
        "50       0.0h  ｜  100.0  300.0  ｜  250  350\n"
        "51       0.0h  ｜  102.0  302.0  ｜  251  351\n"
        "52       0.0h  ｜  104.0  304.0  ｜  252  352\n"
        "53       0.0h  ｜  106.0  306.0  ｜  253  353\n"
        "54       0.0h  ｜  108.0  308.0  ｜  254  354\n"
        "[50 rows x 4 columns]\n"
        "\n"
        "Time point : 1.0 hour\n"
        "‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾\n"
        "  Time-point    col1   col2    col3 col4\n"
        "0       1.0h  ｜  0.0  200.0  ｜  200  300\n"
        "1       1.0h  ｜  2.0  202.0  ｜  201  301\n"
        "2       1.0h  ｜  4.0  204.0  ｜  202  302\n"
        "3       1.0h  ｜  6.0  206.0  ｜  203  303\n"
        "4       1.0h  ｜  8.0  208.0  ｜  204  304\n"
        "[50 rows x 4 columns]\n\n"
    )


@pytest.mark.parametrize("TDF", ["view", "backed view"], indirect=True)
def test_mutliply_by_value_to_view(TDF: TemporalDataFrameView) -> None:
    prefix = "backed " if TDF.is_backed else ""
    assert (
        repr(TDF * 2) == f"TemporalDataFrame View of {prefix}TemporalDataFrame 1 * 2\n"
        "Time point : 1.0 hour\n"
        "‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾\n"
        "   Time-point     col1    col4\n"
        "10       1.0h  ｜  20.0  ｜  310\n"
        "11       1.0h  ｜  22.0  ｜  311\n"
        "12       1.0h  ｜  24.0  ｜  312\n"
        "13       1.0h  ｜  26.0  ｜  313\n"
        "14       1.0h  ｜  28.0  ｜  314\n"
        "[40 rows x 2 columns]\n"
        "\n"
        "Time point : 0.0 hours\n"
        "‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾\n"
        "   Time-point      col1    col4\n"
        "50       0.0h  ｜  100.0  ｜  350\n"
        "51       0.0h  ｜  102.0  ｜  351\n"
        "52       0.0h  ｜  104.0  ｜  352\n"
        "53       0.0h  ｜  106.0  ｜  353\n"
        "54       0.0h  ｜  108.0  ｜  354\n"
        "[40 rows x 2 columns]\n\n"
    )


@pytest.mark.parametrize("TDF", ["plain", "backed"], indirect=True)
def test_divide_by_value(TDF: TemporalDataFrame) -> None:
    prefix = "Backed " if TDF.is_backed else ""
    assert (
        repr(TDF / 2) == f"TemporalDataFrame {prefix}TemporalDataFrame 1 / 2\n"
        "Time point : 0.0 hours\n"
        "‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾\n"
        "   Time-point     col1  col2    col3 col4\n"
        "50       0.0h  ｜  25.0  75.0  ｜  250  350\n"
        "51       0.0h  ｜  25.5  75.5  ｜  251  351\n"
        "52       0.0h  ｜  26.0  76.0  ｜  252  352\n"
        "53       0.0h  ｜  26.5  76.5  ｜  253  353\n"
        "54       0.0h  ｜  27.0  77.0  ｜  254  354\n"
        "[50 rows x 4 columns]\n"
        "\n"
        "Time point : 1.0 hour\n"
        "‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾\n"
        "  Time-point    col1  col2    col3 col4\n"
        "0       1.0h  ｜  0.0  50.0  ｜  200  300\n"
        "1       1.0h  ｜  0.5  50.5  ｜  201  301\n"
        "2       1.0h  ｜  1.0  51.0  ｜  202  302\n"
        "3       1.0h  ｜  1.5  51.5  ｜  203  303\n"
        "4       1.0h  ｜  2.0  52.0  ｜  204  304\n"
        "[50 rows x 4 columns]\n\n"
    )


@pytest.mark.parametrize("TDF", ["view", "backed view"], indirect=True)
def test_divide_by_value_to_view(TDF: TemporalDataFrameView) -> None:
    prefix = "backed " if TDF.is_backed else ""
    assert (
        repr(TDF / 2) == f"TemporalDataFrame View of {prefix}TemporalDataFrame 1 / 2\n"
        "Time point : 1.0 hour\n"
        "‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾\n"
        "   Time-point    col1    col4\n"
        "10       1.0h  ｜  5.0  ｜  310\n"
        "11       1.0h  ｜  5.5  ｜  311\n"
        "12       1.0h  ｜  6.0  ｜  312\n"
        "13       1.0h  ｜  6.5  ｜  313\n"
        "14       1.0h  ｜  7.0  ｜  314\n"
        "[40 rows x 2 columns]\n"
        "\n"
        "Time point : 0.0 hours\n"
        "‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾\n"
        "   Time-point     col1    col4\n"
        "50       0.0h  ｜  25.0  ｜  350\n"
        "51       0.0h  ｜  25.5  ｜  351\n"
        "52       0.0h  ｜  26.0  ｜  352\n"
        "53       0.0h  ｜  26.5  ｜  353\n"
        "54       0.0h  ｜  27.0  ｜  354\n"
        "[40 rows x 2 columns]\n\n"
    )


@pytest.mark.parametrize("TDF", ["plain", "backed"], indirect=True)
def test_add_value_to_empty_array(TDF: TemporalDataFrame) -> None:
    if TDF.is_backed:
        TDF.unlock_columns()

    del TDF.col1
    del TDF.col2

    with pytest.raises(ValueError) as exc_info:
        TDF + 1

    assert str(exc_info.value) == "No numerical data to add to."


@pytest.mark.parametrize("TDF", ["plain", "backed"], indirect=True)
def test_add_value_to_empty_array_in_view(TDF: TemporalDataFrame) -> None:
    if TDF.is_backed:
        TDF.unlock_columns()

    del TDF.col1
    del TDF.col2

    view = TDF[:]

    with pytest.raises(ValueError) as exc_info:
        view + 1

    assert str(exc_info.value) == "No numerical data to add to."


@pytest.mark.parametrize("TDF", ["plain", "backed"], indirect=True)
def test_add_str_value_to_empty_array(TDF: TemporalDataFrame) -> None:
    if TDF.is_backed:
        TDF.unlock_columns()

    del TDF.col3
    del TDF.col4

    with pytest.raises(ValueError) as exc_info:
        TDF + "_1"

    assert str(exc_info.value) == "No string data to add to."


@pytest.mark.parametrize("TDF", ["plain", "backed"], indirect=True)
def test_add_str_value_to_empty_array_in_view(TDF: TemporalDataFrame) -> None:
    if TDF.is_backed:
        TDF.unlock_columns()

    del TDF.col3
    del TDF.col4

    view = TDF[:]

    with pytest.raises(ValueError) as exc_info:
        view + "_1"

    assert str(exc_info.value) == "No string data to add to."


@pytest.mark.parametrize("TDF1", ["plain", "backed"], indirect=True)
@pytest.mark.parametrize(
    "TDF2",
    ["plain", "backed"],
    indirect=True,
)
def test_add_TDF_to_TDF(TDF1: TemporalDataFrameBase, TDF2: TemporalDataFrameBase) -> None:
    sum_TDF = TDF1 + TDF2

    assert isinstance(sum_TDF, TemporalDataFrame)
    assert np.array_equal(sum_TDF.values_num, TDF1.values_num * 2) and np.array_equal(
        sum_TDF.values_str, np.char.add(TDF1.values_str, TDF1.values_str)
    )


@pytest.mark.parametrize("TDF1", ["view", "backed view"], indirect=True)
@pytest.mark.parametrize(
    "TDF2",
    ["view", "backed view"],
    indirect=True,
)
def test_add_TDF_view_to_TDF_view(TDF1: TemporalDataFrameBase, TDF2: TemporalDataFrameBase) -> None:
    sum_TDF = TDF1 + TDF2

    assert isinstance(sum_TDF, TemporalDataFrame)
    assert np.array_equal(sum_TDF.values_num, TDF1.values_num * 2) and np.array_equal(
        sum_TDF.values_str, np.char.add(TDF1.values_str, TDF1.values_str)
    )


@pytest.mark.parametrize("TDF1", ["plain", "backed"], indirect=True)
@pytest.mark.parametrize(
    "TDF2",
    ["plain", "backed"],
    indirect=True,
)
def test_subtract_TDF_to_TDF(TDF1: TemporalDataFrameBase, TDF2: TemporalDataFrameBase) -> None:
    sub_TDF = TDF1 - TDF2

    assert isinstance(sub_TDF, TemporalDataFrame)
    assert np.array_equal(sub_TDF.values_num, np.zeros_like(sub_TDF.values_num)) and np.array_equal(
        sub_TDF.values_str, TDF1.values_str
    )


@pytest.mark.parametrize("TDF1", ["view", "backed view"], indirect=True)
@pytest.mark.parametrize(
    "TDF2",
    ["view", "backed view"],
    indirect=True,
)
def test_subtract_TDF_view_to_TDF_view(TDF1: TemporalDataFrameBase, TDF2: TemporalDataFrameBase) -> None:
    sub_TDF = TDF1 - TDF2

    assert isinstance(sub_TDF, TemporalDataFrame)
    assert np.array_equal(sub_TDF.values_num, np.zeros_like(sub_TDF.values_num)) and np.array_equal(
        sub_TDF.values_str, TDF1.values_str
    )


@pytest.mark.parametrize("TDF1", ["plain", "backed"], indirect=True)
@pytest.mark.parametrize(
    "TDF2",
    ["plain", "backed"],
    indirect=True,
)
def test_multiply_TDF_with_TDF(TDF1: TemporalDataFrameBase, TDF2: TemporalDataFrameBase) -> None:
    mul_TDF = TDF1 * TDF2

    assert isinstance(mul_TDF, TemporalDataFrame)
    assert np.array_equal(mul_TDF.values_num, TDF1.values_num**2) and np.array_equal(
        mul_TDF.values_str, TDF1.values_str
    )


@pytest.mark.parametrize("TDF1", ["view", "backed view"], indirect=True)
@pytest.mark.parametrize(
    "TDF2",
    ["view", "backed view"],
    indirect=True,
)
def test_multiply_TDF_view_with_TDF_view(TDF1: TemporalDataFrameBase, TDF2: TemporalDataFrameBase) -> None:
    mul_TDF = TDF1 * TDF2

    assert isinstance(mul_TDF, TemporalDataFrame)
    assert np.array_equal(mul_TDF.values_num, TDF1.values_num**2) and np.array_equal(
        mul_TDF.values_str, TDF1.values_str
    )


@pytest.mark.parametrize("TDF1", ["plain", "backed"], indirect=True)
@pytest.mark.parametrize(
    "TDF2",
    ["plain", "backed"],
    indirect=True,
)
def test_divide_TDF_with_TDF(TDF1: TemporalDataFrameBase, TDF2: TemporalDataFrameBase) -> None:
    div_TDF = TDF1 / TDF2

    assert isinstance(div_TDF, TemporalDataFrame)

    eq = div_TDF.values_num == np.ones_like(div_TDF.values_num)
    assert np.all(eq | np.isnan(div_TDF.values_num[~eq])) and np.array_equal(div_TDF.values_str, TDF1.values_str)


@pytest.mark.parametrize("TDF1", ["view", "backed view"], indirect=True)
@pytest.mark.parametrize(
    "TDF2",
    ["view", "backed view"],
    indirect=True,
)
def test_divide_TDF_view_with_TDF_view(TDF1: TemporalDataFrameBase, TDF2: TemporalDataFrameBase) -> None:
    div_TDF = TDF1 / TDF2

    assert isinstance(div_TDF, TemporalDataFrame)

    eq = div_TDF.values_num == np.ones_like(div_TDF.values_num)
    assert np.all(eq | np.isnan(div_TDF.values_num[~eq])) and np.array_equal(div_TDF.values_str, TDF1.values_str)


@pytest.mark.parametrize("TDF", ["plain", "view", "backed", "backed view"], indirect=True)
def test_inplace_add_to_TDF(TDF: TemporalDataFrameBase) -> None:
    original_values_num = TDF.values_num.copy()
    original_values_str = TDF.values_str.copy()

    TDF += 1

    assert np.array_equal(TDF.values_num, original_values_num + 1) and np.array_equal(
        TDF.values_str, original_values_str
    )


@pytest.mark.parametrize("TDF", ["plain", "view", "backed", "backed view"], indirect=True)
def test_inplace_add_string_to_TDF(TDF: TemporalDataFrameBase) -> None:
    original_values_num = TDF.values_num.copy()
    original_values_str = TDF.values_str.copy()

    TDF += "_1"

    assert np.array_equal(TDF.values_num, original_values_num) and np.array_equal(
        TDF.values_str, np.char.add(original_values_str, "_1")
    )


@pytest.mark.parametrize("TDF", ["plain", "view", "backed", "backed view"], indirect=True)
def test_inplace_subtract_to_TDF(TDF: TemporalDataFrameBase) -> None:
    original_values_num = TDF.values_num.copy()
    original_values_str = TDF.values_str.copy()

    TDF -= 2

    assert np.array_equal(TDF.values_num, original_values_num - 2) and np.array_equal(
        TDF.values_str, original_values_str
    )


@pytest.mark.parametrize("TDF", ["plain", "view", "backed", "backed view"], indirect=True)
def test_inplace_multiply_TDF(TDF: TemporalDataFrameBase) -> None:
    original_values_num = TDF.values_num.copy()
    original_values_str = TDF.values_str.copy()

    TDF *= 2

    assert np.array_equal(TDF.values_num, original_values_num * 2) and np.array_equal(
        TDF.values_str, original_values_str
    )


@pytest.mark.parametrize("TDF", ["plain", "view", "backed", "backed view"], indirect=True)
def test_inplace_divide_TDF(TDF: TemporalDataFrameBase) -> None:
    original_values_num = TDF.values_num.copy()
    original_values_str = TDF.values_str.copy()

    TDF /= 2

    assert np.array_equal(TDF.values_num, original_values_num / 2) and np.array_equal(
        TDF.values_str, original_values_str
    )


@pytest.mark.parametrize("TDF", ["plain", "view", "backed", "backed view"], indirect=True)
def test_add_to_TDF_with_empty_array_whould_fail(TDF: TemporalDataFrameBase) -> None:
    if TDF.is_view:
        del TDF.parent.col1
        del TDF.parent.col2
        TDF = TDF.parent[:]

    else:
        del TDF.col1
        del TDF.col2

    with pytest.raises(ValueError) as exc_info:
        TDF += 1

    assert str(exc_info.value) == "No numerical data to add to."
