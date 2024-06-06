# coding: utf-8
# Created on 31/03/2022 16:33
# Author : matteo

# ====================================================
# imports
import numpy as np
import numpy.typing as npt
import pandas as pd
import pytest

from vdata.tdf import RepeatingIndex, TemporalDataFrame, TemporalDataFrameBase


# ====================================================
# code
@pytest.mark.usefixtures("class_TDF1")
@pytest.mark.parametrize("class_TDF1", ["plain", "backed"], indirect=True)
class TestSubGetting:
    TDF: TemporalDataFrameBase

    def test_subset_get_single_tp(self) -> None:
        backed = "backed " if self.TDF.is_backed else ""

        assert (
            repr(self.TDF["0h"]) == f"View of {backed}TemporalDataFrame 1\n"
            "Time point : 0.0 hours\n"
            "‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾\n"
            "   Time-point     col1   col2    col3 col4\n"
            "50       0.0h  ｜  50.0  150.0  ｜  250  350\n"
            "51       0.0h  ｜  51.0  151.0  ｜  251  351\n"
            "52       0.0h  ｜  52.0  152.0  ｜  252  352\n"
            "53       0.0h  ｜  53.0  153.0  ｜  253  353\n"
            "54       0.0h  ｜  54.0  154.0  ｜  254  354\n"
            "[50 rows x 4 columns]\n\n"
        )

        assert np.all(
            self.TDF["0h"].values_num == np.hstack((np.arange(50, 100)[:, None], np.arange(150, 200)[:, None]))
        )
        assert np.all(
            self.TDF["0h"].values_str
            == np.hstack((np.arange(250, 300).astype(str)[:, None], np.arange(350, 400).astype(str)[:, None]))
        )

    def test_subset_get_tp_not_in_tdf_should_fail(self) -> None:
        with pytest.raises(KeyError) as exc_info:
            self.TDF["1s"]

        assert str(exc_info.value) == '"Could not find [1.0s] in TemporalDataFrame\'s timepoints."'

    def test_subset_multiple_timepoints_not_in_tdf_should_fail(self) -> None:
        # subset multiple TPs, some not in tdf
        with pytest.raises(KeyError) as exc_info:
            repr(self.TDF[["0h", "1h", "2h"]])

        assert str(exc_info.value) == '"Could not find [2.0h] in TemporalDataFrame\'s timepoints."'

    def test_subset_single_row(self) -> None:
        backed = "backed " if self.TDF.is_backed else ""

        assert (
            repr(self.TDF[:, 10]) == f"View of {backed}TemporalDataFrame 1\n"
            "Time point : 1.0 hour\n"
            "‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾\n"
            "   Time-point     col1   col2    col3 col4\n"
            "10       1.0h  ｜  10.0  110.0  ｜  210  310\n"
            "[1 rows x 4 columns]\n\n"
        )

        assert np.all(self.TDF[:, 10].values_num == np.array([[10, 110]]))
        assert np.all(self.TDF[:, 10].values_str == np.array([["210", "310"]]))

    def test_subset_single_row_not_in_tdf_should_fail(self) -> None:
        # subset single row, not in tdf
        with pytest.raises(KeyError) as exc_info:
            repr(self.TDF[:, 500])

        assert str(exc_info.value) == '"Could not find [500] in TemporalDataFrame\'s index."'

    def test_subset_multiple_rows(self) -> None:
        backed = "backed " if self.TDF.is_backed else ""

        assert (
            repr(self.TDF[:, range(25, 75)]) == f"View of {backed}TemporalDataFrame 1\n"
            "Time point : 1.0 hour\n"
            "‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾\n"
            "   Time-point     col1   col2    col3 col4\n"
            "25       1.0h  ｜  25.0  125.0  ｜  225  325\n"
            "26       1.0h  ｜  26.0  126.0  ｜  226  326\n"
            "27       1.0h  ｜  27.0  127.0  ｜  227  327\n"
            "28       1.0h  ｜  28.0  128.0  ｜  228  328\n"
            "29       1.0h  ｜  29.0  129.0  ｜  229  329\n"
            "[25 rows x 4 columns]\n"
            "\n"
            "Time point : 0.0 hours\n"
            "‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾\n"
            "   Time-point     col1   col2    col3 col4\n"
            "50       0.0h  ｜  50.0  150.0  ｜  250  350\n"
            "51       0.0h  ｜  51.0  151.0  ｜  251  351\n"
            "52       0.0h  ｜  52.0  152.0  ｜  252  352\n"
            "53       0.0h  ｜  53.0  153.0  ｜  253  353\n"
            "54       0.0h  ｜  54.0  154.0  ｜  254  354\n"
            "[25 rows x 4 columns]\n\n"
        )

        assert np.all(
            self.TDF[:, range(25, 75)].values_num
            == np.hstack(
                (
                    np.concatenate((np.arange(25, 50), np.arange(50, 75)))[:, None],
                    np.concatenate((np.arange(125, 150), np.arange(150, 175)))[:, None],
                )
            )
        )
        assert np.all(
            self.TDF[:, range(25, 75)].values_str
            == np.hstack(
                (
                    np.concatenate((np.arange(225, 250), np.arange(250, 275))).astype(str)[:, None],
                    np.concatenate((np.arange(325, 350), np.arange(350, 375))).astype(str)[:, None],
                )
            )
        )

    def test_subset_multiple_rows_with_some_not_in_tdf_should_fail(self) -> None:
        # subset multiple rows, some not in tdf
        with pytest.raises(KeyError) as exc_info:
            self.TDF[:, 20:500:2]

        assert str(exc_info.value) == "\"Could not find '500' in TemporalDataFrame's index.\""

    def test_subset_single_column(self) -> None:
        backed = "backed " if self.TDF.is_backed else ""

        assert (
            repr(self.TDF[:, :, "col3"]) == f"View of {backed}TemporalDataFrame 1\n"
            "Time point : 0.0 hours\n"
            "‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾\n"
            "   Time-point    col3\n"
            "50       0.0h  ｜  250\n"
            "51       0.0h  ｜  251\n"
            "52       0.0h  ｜  252\n"
            "53       0.0h  ｜  253\n"
            "54       0.0h  ｜  254\n"
            "[50 rows x 1 columns]\n"
            "\n"
            "Time point : 1.0 hour\n"
            "‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾\n"
            "  Time-point    col3\n"
            "0       1.0h  ｜  200\n"
            "1       1.0h  ｜  201\n"
            "2       1.0h  ｜  202\n"
            "3       1.0h  ｜  203\n"
            "4       1.0h  ｜  204\n"
            "[50 rows x 1 columns]\n\n"
        )

        assert self.TDF[:, :, "col3"].values_num.size == 0
        assert np.all(
            self.TDF[:, :, "col3"].values_str
            == np.concatenate((np.arange(250, 300), np.arange(200, 250))).astype(str)[:, None]
        )

    def test_subset_single_column_with_getattr(self) -> None:
        backed = "backed " if self.TDF.is_backed else ""

        assert (
            repr(self.TDF.col2) == f"View of {backed}TemporalDataFrame 1\n"
            "Time point : 0.0 hours\n"
            "‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾\n"
            "   Time-point      col2\n"
            "50       0.0h  ｜  150.0\n"
            "51       0.0h  ｜  151.0\n"
            "52       0.0h  ｜  152.0\n"
            "53       0.0h  ｜  153.0\n"
            "54       0.0h  ｜  154.0\n"
            "[50 rows x 1 columns]\n"
            "\n"
            "Time point : 1.0 hour\n"
            "‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾\n"
            "  Time-point      col2\n"
            "0       1.0h  ｜  100.0\n"
            "1       1.0h  ｜  101.0\n"
            "2       1.0h  ｜  102.0\n"
            "3       1.0h  ｜  103.0\n"
            "4       1.0h  ｜  104.0\n"
            "[50 rows x 1 columns]\n\n"
        )

        assert np.all(self.TDF.col2.values_num == np.concatenate((np.arange(150, 200), np.arange(100, 150)))[:, None])
        assert self.TDF.col2.values_str.size == 0

    def test_subset_column_not_in_tdf_should_fail(self) -> None:
        with pytest.raises(KeyError) as exc_info:
            self.TDF[:, :, "col5"]

        assert str(exc_info.value) == "\"Could not find ['col5'] in TemporalDataFrame's columns.\""

    def test_subset_column_with_getattr_not_in_tdf_should_fail(self) -> None:
        with pytest.raises(AttributeError) as exc_info:
            self.TDF.col5

        assert str(exc_info.value.__cause__) == "\"Could not find ['col5'] in TemporalDataFrame's columns.\""

    def test_subset_multiple_columns(self) -> None:
        backed = "backed " if self.TDF.is_backed else ""

        assert (
            repr(self.TDF[:, :, ["col1", "col3"]]) == f"View of {backed}TemporalDataFrame 1\n"
            "Time point : 0.0 hours\n"
            "‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾\n"
            "   Time-point     col1    col3\n"
            "50       0.0h  ｜  50.0  ｜  250\n"
            "51       0.0h  ｜  51.0  ｜  251\n"
            "52       0.0h  ｜  52.0  ｜  252\n"
            "53       0.0h  ｜  53.0  ｜  253\n"
            "54       0.0h  ｜  54.0  ｜  254\n"
            "[50 rows x 2 columns]\n"
            "\n"
            "Time point : 1.0 hour\n"
            "‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾\n"
            "  Time-point    col1    col3\n"
            "0       1.0h  ｜  0.0  ｜  200\n"
            "1       1.0h  ｜  1.0  ｜  201\n"
            "2       1.0h  ｜  2.0  ｜  202\n"
            "3       1.0h  ｜  3.0  ｜  203\n"
            "4       1.0h  ｜  4.0  ｜  204\n"
            "[50 rows x 2 columns]\n\n"
        )

        assert np.all(
            self.TDF[:, :, ["col1", "col3"]].values_num
            == np.concatenate((np.arange(50, 100), np.arange(0, 50)))[:, None]
        )
        assert np.all(
            self.TDF[:, :, ["col1", "col3"]].values_str
            == np.concatenate((np.arange(250, 300), np.arange(200, 250))).astype(str)[:, None]
        )

    def test_subset_multiple_columns_with_some_not_in_tdf_should_fail(self) -> None:
        with pytest.raises(KeyError) as exc_info:
            repr(self.TDF[:, :, ["col1", "col3", "col5"]])

        assert str(exc_info.value) == "\"Could not find ['col5'] in TemporalDataFrame's columns.\""

    def test_subset_multiple_columns_shuffled(self) -> None:
        backed = "backed " if self.TDF.is_backed else ""

        assert (
            repr(self.TDF[:, :, ["col4", "col2", "col1"]]) == f"View of {backed}TemporalDataFrame 1\n"
            "Time point : 0.0 hours\n"
            "‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾\n"
            "   Time-point      col2  col1    col4\n"
            "50       0.0h  ｜  150.0  50.0  ｜  350\n"
            "51       0.0h  ｜  151.0  51.0  ｜  351\n"
            "52       0.0h  ｜  152.0  52.0  ｜  352\n"
            "53       0.0h  ｜  153.0  53.0  ｜  353\n"
            "54       0.0h  ｜  154.0  54.0  ｜  354\n"
            "[50 rows x 3 columns]\n"
            "\n"
            "Time point : 1.0 hour\n"
            "‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾\n"
            "  Time-point      col2 col1    col4\n"
            "0       1.0h  ｜  100.0  0.0  ｜  300\n"
            "1       1.0h  ｜  101.0  1.0  ｜  301\n"
            "2       1.0h  ｜  102.0  2.0  ｜  302\n"
            "3       1.0h  ｜  103.0  3.0  ｜  303\n"
            "4       1.0h  ｜  104.0  4.0  ｜  304\n"
            "[50 rows x 3 columns]\n\n"
        )

        assert np.all(
            self.TDF[:, :, ["col4", "col2", "col1"]].values_num
            == np.hstack(
                (
                    np.concatenate((np.arange(150, 200), np.arange(100, 150)))[:, None],
                    np.concatenate((np.arange(50, 100), np.arange(0, 50)))[:, None],
                )
            )
        )

        assert np.all(
            self.TDF[:, :, ["col4", "col2", "col1"]].values_str
            == np.concatenate((np.arange(350, 400), np.arange(300, 350))).astype(str)[:, None]
        )

    def test_subset_with_timepoints_rows_and_columns(self) -> None:
        backed = "backed " if self.TDF.is_backed else ""

        assert (
            repr(self.TDF["1h", 10:40:5, ["col1", "col3"]]) == f"View of {backed}TemporalDataFrame 1\n"
            "Time point : 1.0 hour\n"
            "‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾\n"
            "   Time-point     col1    col3\n"
            "10       1.0h  ｜  10.0  ｜  210\n"
            "15       1.0h  ｜  15.0  ｜  215\n"
            "20       1.0h  ｜  20.0  ｜  220\n"
            "25       1.0h  ｜  25.0  ｜  225\n"
            "30       1.0h  ｜  30.0  ｜  230\n"
            "[6 rows x 2 columns]\n\n"
        )

        assert np.all(
            self.TDF["1h", 10:40:5, ["col1", "col3"]].values_num == np.array([10, 15, 20, 25, 30, 35])[:, None]
        )
        assert np.all(
            self.TDF["1h", 10:40:5, ["col1", "col3"]].values_str
            == np.array([210, 215, 220, 225, 230, 235]).astype(str)[:, None]
        )


@pytest.mark.usefixtures("class_TDF1")
@pytest.mark.parametrize("class_TDF1", ["view", "backed view"], indirect=True)
class TestSubGettingView:
    TDF: TemporalDataFrameBase

    def test_subset_get_single_tp(self) -> None:
        backed = "backed " if self.TDF.is_backed else ""

        assert (
            repr(self.TDF["0h"]) == f"View of {backed}TemporalDataFrame 1\n"
            "Time point : 0.0 hours\n"
            "‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾\n"
            "   Time-point     col1   col2    col3 col4\n"
            "50       0.0h  ｜  50.0  150.0  ｜  250  350\n"
            "51       0.0h  ｜  51.0  151.0  ｜  251  351\n"
            "52       0.0h  ｜  52.0  152.0  ｜  252  352\n"
            "53       0.0h  ｜  53.0  153.0  ｜  253  353\n"
            "54       0.0h  ｜  54.0  154.0  ｜  254  354\n"
            "[40 rows x 4 columns]\n\n"
        )

        assert np.array_equal(
            self.TDF["0h"].values_num, np.hstack((np.arange(50, 90)[:, None], np.arange(150, 190)[:, None]))
        )
        assert np.array_equal(
            self.TDF["0h"].values_str,
            np.hstack((np.arange(250, 290).astype(str)[:, None], np.arange(350, 390).astype(str)[:, None])),
        )

    def test_subset_get_tp_not_in_tdf_should_fail(self) -> None:
        with pytest.raises(KeyError) as exc_info:
            self.TDF["1s"]

        assert str(exc_info.value) == '"Could not find [1.0s] in TemporalDataFrame\'s timepoints."'

    def test_subset_multiple_timepoints(self) -> None:
        backed = "backed " if self.TDF.is_backed else ""

        assert (
            repr(self.TDF[["0h", "1h"]]) == f"View of {backed}TemporalDataFrame 1\n"
            "Time point : 1.0 hour\n"
            "‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾\n"
            "  Time-point    col1   col2    col3 col4\n"
            "0       1.0h  ｜  0.0  100.0  ｜  200  300\n"
            "1       1.0h  ｜  1.0  101.0  ｜  201  301\n"
            "2       1.0h  ｜  2.0  102.0  ｜  202  302\n"
            "3       1.0h  ｜  3.0  103.0  ｜  203  303\n"
            "4       1.0h  ｜  4.0  104.0  ｜  204  304\n"
            "[40 rows x 4 columns]\n"
            "\n"
            "Time point : 0.0 hours\n"
            "‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾\n"
            "   Time-point     col1   col2    col3 col4\n"
            "50       0.0h  ｜  50.0  150.0  ｜  250  350\n"
            "51       0.0h  ｜  51.0  151.0  ｜  251  351\n"
            "52       0.0h  ｜  52.0  152.0  ｜  252  352\n"
            "53       0.0h  ｜  53.0  153.0  ｜  253  353\n"
            "54       0.0h  ｜  54.0  154.0  ｜  254  354\n"
            "[40 rows x 4 columns]\n\n"
        )

        assert np.all(
            self.TDF[["0h", "1h"]].values_num
            == np.hstack(
                (
                    np.concatenate((np.arange(0, 40), np.arange(50, 90)))[:, None],
                    np.concatenate((np.arange(100, 140), np.arange(150, 190)))[:, None],
                )
            )
        )
        assert np.all(
            self.TDF[["0h", "1h"]].values_str
            == np.hstack(
                (
                    np.concatenate((np.arange(200, 240), np.arange(250, 290))).astype(str)[:, None],
                    np.concatenate((np.arange(300, 340), np.arange(350, 390))).astype(str)[:, None],
                )
            )
        )

    def test_subset_multiple_timepoints_not_in_tdf_should_fail(self) -> None:
        # subset multiple TPs, some not in tdf
        with pytest.raises(KeyError) as exc_info:
            repr(self.TDF[["0h", "1h", "2h"]])

        assert str(exc_info.value) == '"Could not find [2.0h] in TemporalDataFrame\'s timepoints."'

    def test_subset_single_row(self) -> None:
        backed = "backed " if self.TDF.is_backed else ""

        assert (
            repr(self.TDF[:, 10]) == f"View of {backed}TemporalDataFrame 1\n"
            "Time point : 1.0 hour\n"
            "‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾\n"
            "   Time-point     col1   col2    col3 col4\n"
            "10       1.0h  ｜  10.0  110.0  ｜  210  310\n"
            "[1 rows x 4 columns]\n\n"
        )

        assert np.all(self.TDF[:, 10].values_num == np.array([[10, 110]]))
        assert np.all(self.TDF[:, 10].values_str == np.array([["210", "310"]]))

    def test_subset_single_row_not_in_tdf_should_fail(self) -> None:
        # subset single row, not in tdf
        with pytest.raises(KeyError) as exc_info:
            repr(self.TDF[:, 500])

        assert str(exc_info.value) == '"Could not find [500] in TemporalDataFrame\'s index."'

    def test_subset_multiple_rows(self) -> None:
        backed = "backed " if self.TDF.is_backed else ""
        rows = np.r_[range(25, 40), range(50, 75)]

        assert (
            repr(self.TDF[:, rows]) == f"View of {backed}TemporalDataFrame 1\n"
            "Time point : 1.0 hour\n"
            "‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾\n"
            "   Time-point     col1   col2    col3 col4\n"
            "25       1.0h  ｜  25.0  125.0  ｜  225  325\n"
            "26       1.0h  ｜  26.0  126.0  ｜  226  326\n"
            "27       1.0h  ｜  27.0  127.0  ｜  227  327\n"
            "28       1.0h  ｜  28.0  128.0  ｜  228  328\n"
            "29       1.0h  ｜  29.0  129.0  ｜  229  329\n"
            "[15 rows x 4 columns]\n"
            "\n"
            "Time point : 0.0 hours\n"
            "‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾\n"
            "   Time-point     col1   col2    col3 col4\n"
            "50       0.0h  ｜  50.0  150.0  ｜  250  350\n"
            "51       0.0h  ｜  51.0  151.0  ｜  251  351\n"
            "52       0.0h  ｜  52.0  152.0  ｜  252  352\n"
            "53       0.0h  ｜  53.0  153.0  ｜  253  353\n"
            "54       0.0h  ｜  54.0  154.0  ｜  254  354\n"
            "[25 rows x 4 columns]\n\n"
        )

        assert np.all(self.TDF[:, rows].values_num == np.hstack((rows[:, None], (rows + 100)[:, None])))
        assert np.all(
            self.TDF[:, rows].values_str
            == np.hstack(((rows + 200).astype(str)[:, None], (rows + 300).astype(str)[:, None]))
        )

    def test_subset_multiple_rows_with_some_not_in_tdf_should_fail(self) -> None:
        # subset multiple rows, some not in tdf
        with pytest.raises(KeyError) as exc_info:
            self.TDF[:, 20:500:2]

        assert str(exc_info.value) == "\"Could not find '500' in TemporalDataFrame's index.\""

    def test_subset_single_column(self) -> None:
        backed = "backed " if self.TDF.is_backed else ""

        assert (
            repr(self.TDF[:, :, "col3"]) == f"View of {backed}TemporalDataFrame 1\n"
            "Time point : 1.0 hour\n"
            "‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾\n"
            "  Time-point    col3\n"
            "0       1.0h  ｜  200\n"
            "1       1.0h  ｜  201\n"
            "2       1.0h  ｜  202\n"
            "3       1.0h  ｜  203\n"
            "4       1.0h  ｜  204\n"
            "[40 rows x 1 columns]\n"
            "\n"
            "Time point : 0.0 hours\n"
            "‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾\n"
            "   Time-point    col3\n"
            "50       0.0h  ｜  250\n"
            "51       0.0h  ｜  251\n"
            "52       0.0h  ｜  252\n"
            "53       0.0h  ｜  253\n"
            "54       0.0h  ｜  254\n"
            "[40 rows x 1 columns]\n\n"
        )

        assert self.TDF[:, :, "col3"].values_num.size == 0
        assert np.all(
            self.TDF[:, :, "col3"].values_str
            == np.concatenate((np.arange(200, 240), np.arange(250, 290))).astype(str)[:, None]
        )

    def test_subset_single_column_with_getattr(self) -> None:
        backed = "backed " if self.TDF.is_backed else ""

        assert (
            repr(self.TDF.col2) == f"View of {backed}TemporalDataFrame 1\n"
            "Time point : 1.0 hour\n"
            "‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾\n"
            "  Time-point      col2\n"
            "0       1.0h  ｜  100.0\n"
            "1       1.0h  ｜  101.0\n"
            "2       1.0h  ｜  102.0\n"
            "3       1.0h  ｜  103.0\n"
            "4       1.0h  ｜  104.0\n"
            "[40 rows x 1 columns]\n"
            "\n"
            "Time point : 0.0 hours\n"
            "‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾\n"
            "   Time-point      col2\n"
            "50       0.0h  ｜  150.0\n"
            "51       0.0h  ｜  151.0\n"
            "52       0.0h  ｜  152.0\n"
            "53       0.0h  ｜  153.0\n"
            "54       0.0h  ｜  154.0\n"
            "[40 rows x 1 columns]\n\n"
        )

        assert np.all(self.TDF.col2.values_num == np.concatenate((np.arange(100, 140), np.arange(150, 190)))[:, None])
        assert self.TDF.col2.values_str.size == 0

    def test_subset_column_not_in_tdf_should_fail(self) -> None:
        with pytest.raises(KeyError) as exc_info:
            self.TDF[:, :, "col5"]

        assert str(exc_info.value) == "\"Could not find ['col5'] in TemporalDataFrame's columns.\""

    def test_subset_column_with_getattr_not_in_tdf_should_fail(self) -> None:
        with pytest.raises(AttributeError) as exc_info:
            self.TDF.col5

        assert str(exc_info.value.__cause__) == "\"Could not find ['col5'] in TemporalDataFrame's columns.\""

    def test_subset_multiple_columns(self) -> None:
        backed = "backed " if self.TDF.is_backed else ""

        assert (
            repr(self.TDF[:, :, ["col1", "col3"]]) == f"View of {backed}TemporalDataFrame 1\n"
            "Time point : 1.0 hour\n"
            "‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾\n"
            "  Time-point    col1    col3\n"
            "0       1.0h  ｜  0.0  ｜  200\n"
            "1       1.0h  ｜  1.0  ｜  201\n"
            "2       1.0h  ｜  2.0  ｜  202\n"
            "3       1.0h  ｜  3.0  ｜  203\n"
            "4       1.0h  ｜  4.0  ｜  204\n"
            "[40 rows x 2 columns]\n"
            "\n"
            "Time point : 0.0 hours\n"
            "‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾\n"
            "   Time-point     col1    col3\n"
            "50       0.0h  ｜  50.0  ｜  250\n"
            "51       0.0h  ｜  51.0  ｜  251\n"
            "52       0.0h  ｜  52.0  ｜  252\n"
            "53       0.0h  ｜  53.0  ｜  253\n"
            "54       0.0h  ｜  54.0  ｜  254\n"
            "[40 rows x 2 columns]\n\n"
        )

        assert np.array_equal(
            self.TDF[:, :, ["col1", "col3"]].values_num, np.concatenate((np.arange(0, 40), np.arange(50, 90)))[:, None]
        )
        assert np.array_equal(
            self.TDF[:, :, ["col1", "col3"]].values_str,
            np.concatenate((np.arange(200, 240), np.arange(250, 290))).astype(str)[:, None],
        )

    def test_subset_multiple_columns_with_some_not_in_tdf_should_fail(self) -> None:
        with pytest.raises(KeyError) as exc_info:
            repr(self.TDF[:, :, ["col1", "col3", "col5"]])

        assert str(exc_info.value) == "\"Could not find ['col5'] in TemporalDataFrame's columns.\""

    def test_subset_multiple_columns_shuffled(self) -> None:
        backed = "backed " if self.TDF.is_backed else ""

        assert (
            repr(self.TDF[:, :, ["col4", "col2", "col1"]]) == f"View of {backed}TemporalDataFrame 1\n"
            "Time point : 1.0 hour\n"
            "‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾\n"
            "  Time-point      col2 col1    col4\n"
            "0       1.0h  ｜  100.0  0.0  ｜  300\n"
            "1       1.0h  ｜  101.0  1.0  ｜  301\n"
            "2       1.0h  ｜  102.0  2.0  ｜  302\n"
            "3       1.0h  ｜  103.0  3.0  ｜  303\n"
            "4       1.0h  ｜  104.0  4.0  ｜  304\n"
            "[40 rows x 3 columns]\n"
            "\n"
            "Time point : 0.0 hours\n"
            "‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾\n"
            "   Time-point      col2  col1    col4\n"
            "50       0.0h  ｜  150.0  50.0  ｜  350\n"
            "51       0.0h  ｜  151.0  51.0  ｜  351\n"
            "52       0.0h  ｜  152.0  52.0  ｜  352\n"
            "53       0.0h  ｜  153.0  53.0  ｜  353\n"
            "54       0.0h  ｜  154.0  54.0  ｜  354\n"
            "[40 rows x 3 columns]\n\n"
        )

        assert np.array_equal(
            self.TDF[:, :, ["col4", "col2", "col1"]].values_num,
            np.hstack(
                (
                    np.concatenate((np.arange(100, 140), np.arange(150, 190)))[:, None],
                    np.concatenate((np.arange(0, 40), np.arange(50, 90)))[:, None],
                )
            ),
        )

        assert np.array_equal(
            self.TDF[:, :, ["col4", "col2", "col1"]].values_str,
            np.concatenate((np.arange(300, 340), np.arange(350, 390))).astype(str)[:, None],
        )

    def test_subset_with_timepoints_rows_and_columns(self) -> None:
        backed = "backed " if self.TDF.is_backed else ""

        v = self.TDF["1h", 10:39:5, ["col1", "col3"]]

        assert (
            repr(v) == f"View of {backed}TemporalDataFrame 1\n"
            "Time point : 1.0 hour\n"
            "‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾\n"
            "   Time-point     col1    col3\n"
            "10       1.0h  ｜  10.0  ｜  210\n"
            "15       1.0h  ｜  15.0  ｜  215\n"
            "20       1.0h  ｜  20.0  ｜  220\n"
            "25       1.0h  ｜  25.0  ｜  225\n"
            "30       1.0h  ｜  30.0  ｜  230\n"
            "[6 rows x 2 columns]\n\n"
        )

        assert np.all(v.values_num == np.array([10, 15, 20, 25, 30, 35])[:, None])
        assert np.all(v.values_str == np.array([210, 215, 220, 225, 230, 235]).astype(str)[:, None])


@pytest.mark.parametrize("TDF1", ["plain", "view"], indirect=True)
def test_subset_with_timepoints_rows_and_columns_shuffled(TDF1: TemporalDataFrameBase) -> None:
    view = TDF1[["0h"], [39, 10, 80, 60, 20, 70, 50], ["col2", "col1", "col3"]]
    assert (
        repr(view) == "View of TemporalDataFrame 1\n"
        "Time point : 0.0 hours\n"
        "‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾\n"
        "   Time-point      col2  col1    col3\n"
        "80       0.0h  ｜  180.0  80.0  ｜  280\n"
        "60       0.0h  ｜  160.0  60.0  ｜  260\n"
        "70       0.0h  ｜  170.0  70.0  ｜  270\n"
        "50       0.0h  ｜  150.0  50.0  ｜  250\n"
        "[4 rows x 3 columns]\n\n"
    )

    assert np.all(view.values_num == np.array([[180, 80], [160, 60], [170, 70], [150, 50]]))
    assert np.all(view.values_str == np.array([["280"], ["260"], ["270"], ["250"]]))


@pytest.mark.parametrize("TDF1", ["plain", "backed"], indirect=True)
def test_subset_same_index_at_multiple_timepoints(TDF1: TemporalDataFrameBase) -> None:
    TDF1.set_index(RepeatingIndex(np.arange(0, 50), repeats=2))

    backed = "backed " if TDF1.is_backed else ""

    assert (
        repr(TDF1[:, [0, 2, 4]]) == f"View of {backed}TemporalDataFrame 1\n"
        "Time point : 0.0 hours\n"
        "‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾\n"
        "  Time-point     col1   col2    col3 col4\n"
        "0       0.0h  ｜  50.0  150.0  ｜  250  350\n"
        "2       0.0h  ｜  52.0  152.0  ｜  252  352\n"
        "4       0.0h  ｜  54.0  154.0  ｜  254  354\n"
        "[3 rows x 4 columns]\n"
        "\n"
        "Time point : 1.0 hour\n"
        "‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾\n"
        "  Time-point    col1   col2    col3 col4\n"
        "0       1.0h  ｜  0.0  100.0  ｜  200  300\n"
        "2       1.0h  ｜  2.0  102.0  ｜  202  302\n"
        "4       1.0h  ｜  4.0  104.0  ｜  204  304\n"
        "[3 rows x 4 columns]\n\n"
    )

    assert np.all(
        TDF1[:, [0, 2, 4]].values_num == np.array([[50, 150], [52, 152], [54, 154], [0, 100], [2, 102], [4, 104]])
    )
    assert np.all(
        TDF1[:, [0, 2, 4]].values_str
        == np.array([["250", "350"], ["252", "352"], ["254", "354"], ["200", "300"], ["202", "302"], ["204", "304"]])
    )


@pytest.mark.parametrize("TDF1", ["view", "backed view"], indirect=True)
def test_subset_same_index_at_multiple_timepoints_on_view(TDF1: TemporalDataFrameBase) -> None:
    TDF1.parent.set_index(RepeatingIndex(np.arange(0, 50), repeats=2))

    backed = "backed " if TDF1.is_backed else ""

    assert (
        repr(TDF1[:, [0, 2, 4]]) == f"View of {backed}TemporalDataFrame 1\n"
        "Time point : 1.0 hour\n"
        "‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾\n"
        "  Time-point    col1   col2    col3 col4\n"
        "0       1.0h  ｜  0.0  100.0  ｜  200  300\n"
        "2       1.0h  ｜  2.0  102.0  ｜  202  302\n"
        "4       1.0h  ｜  4.0  104.0  ｜  204  304\n"
        "[3 rows x 4 columns]\n"
        "\n"
        "Time point : 0.0 hours\n"
        "‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾\n"
        "  Time-point     col1   col2    col3 col4\n"
        "0       0.0h  ｜  50.0  150.0  ｜  250  350\n"
        "2       0.0h  ｜  52.0  152.0  ｜  252  352\n"
        "4       0.0h  ｜  54.0  154.0  ｜  254  354\n"
        "[3 rows x 4 columns]\n\n"
    )

    assert np.all(
        TDF1[:, [0, 2, 4]].values_num == np.array([[0, 100], [2, 102], [4, 104], [50, 150], [52, 152], [54, 154]])
    )
    assert np.all(
        TDF1[:, [0, 2, 4]].values_str
        == np.array([["200", "300"], ["202", "302"], ["204", "304"], ["250", "350"], ["252", "352"], ["254", "354"]])
    )


@pytest.mark.parametrize("TDF1", ["plain"], indirect=True)
def test_subset_same_shuffled_index_at_multiple_timepoints(TDF1: TemporalDataFrameBase) -> None:
    TDF1.set_index(RepeatingIndex(np.arange(0, 50), repeats=2))

    view = TDF1[:, [4, 0, 2]]

    assert np.all(view.values_num == np.array([[54, 154], [50, 150], [52, 152], [4, 104], [0, 100], [2, 102]]))
    assert np.all(
        view.values_str
        == np.array([["254", "354"], ["250", "350"], ["252", "352"], ["204", "304"], ["200", "300"], ["202", "302"]])
    )


@pytest.mark.parametrize("TDF1", ["view"], indirect=True)
def test_subset_same_shuffled_index_at_multiple_timepoints_on_view(TDF1: TemporalDataFrameBase) -> None:
    TDF1.parent.set_index(RepeatingIndex(np.arange(0, 50), repeats=2))

    view = TDF1[:, [4, 0, 2]]

    assert np.all(view.values_num == np.array([[4, 104], [0, 100], [2, 102], [54, 154], [50, 150], [52, 152]]))
    assert np.all(
        view.values_str
        == np.array([["204", "304"], ["200", "300"], ["202", "302"], ["254", "354"], ["250", "350"], ["252", "352"]])
    )


@pytest.mark.parametrize("TDF1", ["plain", "backed"], indirect=True)
def test_set_values_with_wrong_shape_should_fail(TDF1: TemporalDataFrameBase) -> None:
    with pytest.raises(ValueError) as exc_info:
        TDF1["0h", 10:70:2, ["col4", "col1"]] = np.ones((50, 50))

    assert str(exc_info.value) == "Can't set (10, 2) values from (50, 50) array."


@pytest.mark.parametrize("TDF1", ["plain", "backed"], indirect=True)
def test_set_values(TDF1: TemporalDataFrameBase) -> None:
    TDF1["0h", 10:70:2, ["col4", "col1"]] = np.array(
        [["a", -1], ["b", -2], ["c", -3], ["d", -4], ["e", -5], ["f", -6], ["g", -7], ["h", -8], ["i", -9], ["j", -10]]
    )

    assert np.all(
        TDF1.values_num
        == np.array(
            [
                [-1.0, 150.0],
                [51.0, 151.0],
                [-2.0, 152.0],
                [53.0, 153.0],
                [-3.0, 154.0],
                [55.0, 155.0],
                [-4.0, 156.0],
                [57.0, 157.0],
                [-5.0, 158.0],
                [59.0, 159.0],
                [-6.0, 160.0],
                [61.0, 161.0],
                [-7.0, 162.0],
                [63.0, 163.0],
                [-8.0, 164.0],
                [65.0, 165.0],
                [-9.0, 166.0],
                [67.0, 167.0],
                [-10.0, 168.0],
                [69.0, 169.0],
                [70.0, 170.0],
                [71.0, 171.0],
                [72.0, 172.0],
                [73.0, 173.0],
                [74.0, 174.0],
                [75.0, 175.0],
                [76.0, 176.0],
                [77.0, 177.0],
                [78.0, 178.0],
                [79.0, 179.0],
                [80.0, 180.0],
                [81.0, 181.0],
                [82.0, 182.0],
                [83.0, 183.0],
                [84.0, 184.0],
                [85.0, 185.0],
                [86.0, 186.0],
                [87.0, 187.0],
                [88.0, 188.0],
                [89.0, 189.0],
                [90.0, 190.0],
                [91.0, 191.0],
                [92.0, 192.0],
                [93.0, 193.0],
                [94.0, 194.0],
                [95.0, 195.0],
                [96.0, 196.0],
                [97.0, 197.0],
                [98.0, 198.0],
                [99.0, 199.0],
                [0.0, 100.0],
                [1.0, 101.0],
                [2.0, 102.0],
                [3.0, 103.0],
                [4.0, 104.0],
                [5.0, 105.0],
                [6.0, 106.0],
                [7.0, 107.0],
                [8.0, 108.0],
                [9.0, 109.0],
                [10.0, 110.0],
                [11.0, 111.0],
                [12.0, 112.0],
                [13.0, 113.0],
                [14.0, 114.0],
                [15.0, 115.0],
                [16.0, 116.0],
                [17.0, 117.0],
                [18.0, 118.0],
                [19.0, 119.0],
                [20.0, 120.0],
                [21.0, 121.0],
                [22.0, 122.0],
                [23.0, 123.0],
                [24.0, 124.0],
                [25.0, 125.0],
                [26.0, 126.0],
                [27.0, 127.0],
                [28.0, 128.0],
                [29.0, 129.0],
                [30.0, 130.0],
                [31.0, 131.0],
                [32.0, 132.0],
                [33.0, 133.0],
                [34.0, 134.0],
                [35.0, 135.0],
                [36.0, 136.0],
                [37.0, 137.0],
                [38.0, 138.0],
                [39.0, 139.0],
                [40.0, 140.0],
                [41.0, 141.0],
                [42.0, 142.0],
                [43.0, 143.0],
                [44.0, 144.0],
                [45.0, 145.0],
                [46.0, 146.0],
                [47.0, 147.0],
                [48.0, 148.0],
                [49.0, 149.0],
            ]
        )
    )
    assert np.all(
        TDF1.values_str
        == np.array(
            [
                ["250", "a"],
                ["251", "351"],
                ["252", "b"],
                ["253", "353"],
                ["254", "c"],
                ["255", "355"],
                ["256", "d"],
                ["257", "357"],
                ["258", "e"],
                ["259", "359"],
                ["260", "f"],
                ["261", "361"],
                ["262", "g"],
                ["263", "363"],
                ["264", "h"],
                ["265", "365"],
                ["266", "i"],
                ["267", "367"],
                ["268", "j"],
                ["269", "369"],
                ["270", "370"],
                ["271", "371"],
                ["272", "372"],
                ["273", "373"],
                ["274", "374"],
                ["275", "375"],
                ["276", "376"],
                ["277", "377"],
                ["278", "378"],
                ["279", "379"],
                ["280", "380"],
                ["281", "381"],
                ["282", "382"],
                ["283", "383"],
                ["284", "384"],
                ["285", "385"],
                ["286", "386"],
                ["287", "387"],
                ["288", "388"],
                ["289", "389"],
                ["290", "390"],
                ["291", "391"],
                ["292", "392"],
                ["293", "393"],
                ["294", "394"],
                ["295", "395"],
                ["296", "396"],
                ["297", "397"],
                ["298", "398"],
                ["299", "399"],
                ["200", "300"],
                ["201", "301"],
                ["202", "302"],
                ["203", "303"],
                ["204", "304"],
                ["205", "305"],
                ["206", "306"],
                ["207", "307"],
                ["208", "308"],
                ["209", "309"],
                ["210", "310"],
                ["211", "311"],
                ["212", "312"],
                ["213", "313"],
                ["214", "314"],
                ["215", "315"],
                ["216", "316"],
                ["217", "317"],
                ["218", "318"],
                ["219", "319"],
                ["220", "320"],
                ["221", "321"],
                ["222", "322"],
                ["223", "323"],
                ["224", "324"],
                ["225", "325"],
                ["226", "326"],
                ["227", "327"],
                ["228", "328"],
                ["229", "329"],
                ["230", "330"],
                ["231", "331"],
                ["232", "332"],
                ["233", "333"],
                ["234", "334"],
                ["235", "335"],
                ["236", "336"],
                ["237", "337"],
                ["238", "338"],
                ["239", "339"],
                ["240", "340"],
                ["241", "341"],
                ["242", "342"],
                ["243", "343"],
                ["244", "344"],
                ["245", "345"],
                ["246", "346"],
                ["247", "347"],
                ["248", "348"],
                ["249", "349"],
            ],
            dtype="<U3",
        )
    )


@pytest.mark.parametrize("TDF1", ["plain", "backed"], indirect=True)
def test_set_values_for_timepoints_not_in_tdf_should_fail(TDF1: TemporalDataFrameBase) -> None:
    with pytest.raises(KeyError) as exc_info:
        TDF1[["0h", "2h"], 10:70:2, ["col4", "col1"]] = np.array([["a", -1]])

    assert str(exc_info.value) == '"Could not find [2.0h] in TemporalDataFrame\'s timepoints."'


@pytest.mark.parametrize("TDF1", ["plain", "backed"], indirect=True)
def test_set_values_for_rows_not_in_tdf_should_fail(TDF1: TemporalDataFrameBase) -> None:
    with pytest.raises(KeyError) as exc_info:
        TDF1["0h", 0:200:20, ["col4", "col1"]] = np.array(
            [
                ["a", -1],
                ["b", -2],
                ["c", -3],
                ["d", -4],
                ["e", -5],
                ["f", -6],
                ["g", -7],
                ["h", -8],
                ["i", -9],
                ["j", -10],
            ]
        )

    assert str(exc_info.value) == "\"Could not find '200' in TemporalDataFrame's index.\""


@pytest.mark.parametrize("TDF1", ["plain", "backed"], indirect=True)
def test_set_values_forcolumns_not_in_TDF_should_fail(TDF1: TemporalDataFrameBase) -> None:
    # set values for columns, some not in tdf
    with pytest.raises(KeyError) as exc_info:
        TDF1["0h", 10:70:20, ["col4", "col1", "col5"]] = np.array(
            [
                ["a", -1, 0],
                ["b", -2, 0],
                ["c", -3, 0],
                ["d", -4, 0],
                ["e", -5, 0],
                ["f", -6, 0],
                ["g", -7, 0],
                ["h", -8, 0],
                ["i", -9, 0],
                ["j", -10, 0],
            ]
        )

    assert str(exc_info.value) == "\"Could not find ['col5'] in TemporalDataFrame's columns.\""


@pytest.mark.parametrize("TDF1", ["plain", "backed"], indirect=True)
def test_set_values_with_same_index_at_multiple_timepoints(TDF1: TemporalDataFrameBase) -> None:
    TDF1.set_index(RepeatingIndex(np.arange(0, 50), repeats=2))

    TDF1[:, [4, 0, 2]] = np.array(
        [
            [-100, -101, "A", "AA"],
            [-200, -201, "B", "BB"],
            [-300, -301, "C", "CC"],
            [-400, -401, "D", "DD"],
            [-500, -501, "E", "EE"],
            [-600, -601, "F", "FF"],
        ]
    )

    assert np.all(
        TDF1.values_num
        == np.array(
            [
                [-200.0, -201.0],
                [51.0, 151.0],
                [-300.0, -301.0],
                [53.0, 153.0],
                [-100.0, -101.0],
                [55.0, 155.0],
                [56.0, 156.0],
                [57.0, 157.0],
                [58.0, 158.0],
                [59.0, 159.0],
                [60.0, 160.0],
                [61.0, 161.0],
                [62.0, 162.0],
                [63.0, 163.0],
                [64.0, 164.0],
                [65.0, 165.0],
                [66.0, 166.0],
                [67.0, 167.0],
                [68.0, 168.0],
                [69.0, 169.0],
                [70.0, 170.0],
                [71.0, 171.0],
                [72.0, 172.0],
                [73.0, 173.0],
                [74.0, 174.0],
                [75.0, 175.0],
                [76.0, 176.0],
                [77.0, 177.0],
                [78.0, 178.0],
                [79.0, 179.0],
                [80.0, 180.0],
                [81.0, 181.0],
                [82.0, 182.0],
                [83.0, 183.0],
                [84.0, 184.0],
                [85.0, 185.0],
                [86.0, 186.0],
                [87.0, 187.0],
                [88.0, 188.0],
                [89.0, 189.0],
                [90.0, 190.0],
                [91.0, 191.0],
                [92.0, 192.0],
                [93.0, 193.0],
                [94.0, 194.0],
                [95.0, 195.0],
                [96.0, 196.0],
                [97.0, 197.0],
                [98.0, 198.0],
                [99.0, 199.0],
                [-500.0, -501.0],
                [1.0, 101.0],
                [-600.0, -601.0],
                [3.0, 103.0],
                [-400.0, -401.0],
                [5.0, 105.0],
                [6.0, 106.0],
                [7.0, 107.0],
                [8.0, 108.0],
                [9.0, 109.0],
                [10.0, 110.0],
                [11.0, 111.0],
                [12.0, 112.0],
                [13.0, 113.0],
                [14.0, 114.0],
                [15.0, 115.0],
                [16.0, 116.0],
                [17.0, 117.0],
                [18.0, 118.0],
                [19.0, 119.0],
                [20.0, 120.0],
                [21.0, 121.0],
                [22.0, 122.0],
                [23.0, 123.0],
                [24.0, 124.0],
                [25.0, 125.0],
                [26.0, 126.0],
                [27.0, 127.0],
                [28.0, 128.0],
                [29.0, 129.0],
                [30.0, 130.0],
                [31.0, 131.0],
                [32.0, 132.0],
                [33.0, 133.0],
                [34.0, 134.0],
                [35.0, 135.0],
                [36.0, 136.0],
                [37.0, 137.0],
                [38.0, 138.0],
                [39.0, 139.0],
                [40.0, 140.0],
                [41.0, 141.0],
                [42.0, 142.0],
                [43.0, 143.0],
                [44.0, 144.0],
                [45.0, 145.0],
                [46.0, 146.0],
                [47.0, 147.0],
                [48.0, 148.0],
                [49.0, 149.0],
            ]
        )
    )
    assert np.all(
        TDF1.values_str
        == np.array(
            [
                ["B", "BB"],
                ["251", "351"],
                ["C", "CC"],
                ["253", "353"],
                ["A", "AA"],
                ["255", "355"],
                ["256", "356"],
                ["257", "357"],
                ["258", "358"],
                ["259", "359"],
                ["260", "360"],
                ["261", "361"],
                ["262", "362"],
                ["263", "363"],
                ["264", "364"],
                ["265", "365"],
                ["266", "366"],
                ["267", "367"],
                ["268", "368"],
                ["269", "369"],
                ["270", "370"],
                ["271", "371"],
                ["272", "372"],
                ["273", "373"],
                ["274", "374"],
                ["275", "375"],
                ["276", "376"],
                ["277", "377"],
                ["278", "378"],
                ["279", "379"],
                ["280", "380"],
                ["281", "381"],
                ["282", "382"],
                ["283", "383"],
                ["284", "384"],
                ["285", "385"],
                ["286", "386"],
                ["287", "387"],
                ["288", "388"],
                ["289", "389"],
                ["290", "390"],
                ["291", "391"],
                ["292", "392"],
                ["293", "393"],
                ["294", "394"],
                ["295", "395"],
                ["296", "396"],
                ["297", "397"],
                ["298", "398"],
                ["299", "399"],
                ["E", "EE"],
                ["201", "301"],
                ["F", "FF"],
                ["203", "303"],
                ["D", "DD"],
                ["205", "305"],
                ["206", "306"],
                ["207", "307"],
                ["208", "308"],
                ["209", "309"],
                ["210", "310"],
                ["211", "311"],
                ["212", "312"],
                ["213", "313"],
                ["214", "314"],
                ["215", "315"],
                ["216", "316"],
                ["217", "317"],
                ["218", "318"],
                ["219", "319"],
                ["220", "320"],
                ["221", "321"],
                ["222", "322"],
                ["223", "323"],
                ["224", "324"],
                ["225", "325"],
                ["226", "326"],
                ["227", "327"],
                ["228", "328"],
                ["229", "329"],
                ["230", "330"],
                ["231", "331"],
                ["232", "332"],
                ["233", "333"],
                ["234", "334"],
                ["235", "335"],
                ["236", "336"],
                ["237", "337"],
                ["238", "338"],
                ["239", "339"],
                ["240", "340"],
                ["241", "341"],
                ["242", "342"],
                ["243", "343"],
                ["244", "344"],
                ["245", "345"],
                ["246", "346"],
                ["247", "347"],
                ["248", "348"],
                ["249", "349"],
            ],
            dtype="<U3",
        )
    )


@pytest.mark.parametrize("TDF1", ["plain", "backed"], indirect=True)
def test_set_single_value_broadcast(TDF1: TemporalDataFrameBase) -> None:
    # set single value
    TDF1[:, [4, 0, 2], ["col1", "col2"]] = 1000

    assert np.array_equal(
        TDF1.values_num,
        np.array(
            [
                [50.0, 150.0],
                [51.0, 151.0],
                [52.0, 152.0],
                [53.0, 153.0],
                [54.0, 154.0],
                [55.0, 155.0],
                [56.0, 156.0],
                [57.0, 157.0],
                [58.0, 158.0],
                [59.0, 159.0],
                [60.0, 160.0],
                [61.0, 161.0],
                [62.0, 162.0],
                [63.0, 163.0],
                [64.0, 164.0],
                [65.0, 165.0],
                [66.0, 166.0],
                [67.0, 167.0],
                [68.0, 168.0],
                [69.0, 169.0],
                [70.0, 170.0],
                [71.0, 171.0],
                [72.0, 172.0],
                [73.0, 173.0],
                [74.0, 174.0],
                [75.0, 175.0],
                [76.0, 176.0],
                [77.0, 177.0],
                [78.0, 178.0],
                [79.0, 179.0],
                [80.0, 180.0],
                [81.0, 181.0],
                [82.0, 182.0],
                [83.0, 183.0],
                [84.0, 184.0],
                [85.0, 185.0],
                [86.0, 186.0],
                [87.0, 187.0],
                [88.0, 188.0],
                [89.0, 189.0],
                [90.0, 190.0],
                [91.0, 191.0],
                [92.0, 192.0],
                [93.0, 193.0],
                [94.0, 194.0],
                [95.0, 195.0],
                [96.0, 196.0],
                [97.0, 197.0],
                [98.0, 198.0],
                [99.0, 199.0],
                [1000.0, 1000.0],
                [1.0, 101.0],
                [1000.0, 1000.0],
                [3.0, 103.0],
                [1000.0, 1000.0],
                [5.0, 105.0],
                [6.0, 106.0],
                [7.0, 107.0],
                [8.0, 108.0],
                [9.0, 109.0],
                [10.0, 110.0],
                [11.0, 111.0],
                [12.0, 112.0],
                [13.0, 113.0],
                [14.0, 114.0],
                [15.0, 115.0],
                [16.0, 116.0],
                [17.0, 117.0],
                [18.0, 118.0],
                [19.0, 119.0],
                [20.0, 120.0],
                [21.0, 121.0],
                [22.0, 122.0],
                [23.0, 123.0],
                [24.0, 124.0],
                [25.0, 125.0],
                [26.0, 126.0],
                [27.0, 127.0],
                [28.0, 128.0],
                [29.0, 129.0],
                [30.0, 130.0],
                [31.0, 131.0],
                [32.0, 132.0],
                [33.0, 133.0],
                [34.0, 134.0],
                [35.0, 135.0],
                [36.0, 136.0],
                [37.0, 137.0],
                [38.0, 138.0],
                [39.0, 139.0],
                [40.0, 140.0],
                [41.0, 141.0],
                [42.0, 142.0],
                [43.0, 143.0],
                [44.0, 144.0],
                [45.0, 145.0],
                [46.0, 146.0],
                [47.0, 147.0],
                [48.0, 148.0],
                [49.0, 149.0],
            ]
        ),
    )
    assert np.all(
        TDF1.values_str
        == np.array(
            [
                ["250", "350"],
                ["251", "351"],
                ["252", "352"],
                ["253", "353"],
                ["254", "354"],
                ["255", "355"],
                ["256", "356"],
                ["257", "357"],
                ["258", "358"],
                ["259", "359"],
                ["260", "360"],
                ["261", "361"],
                ["262", "362"],
                ["263", "363"],
                ["264", "364"],
                ["265", "365"],
                ["266", "366"],
                ["267", "367"],
                ["268", "368"],
                ["269", "369"],
                ["270", "370"],
                ["271", "371"],
                ["272", "372"],
                ["273", "373"],
                ["274", "374"],
                ["275", "375"],
                ["276", "376"],
                ["277", "377"],
                ["278", "378"],
                ["279", "379"],
                ["280", "380"],
                ["281", "381"],
                ["282", "382"],
                ["283", "383"],
                ["284", "384"],
                ["285", "385"],
                ["286", "386"],
                ["287", "387"],
                ["288", "388"],
                ["289", "389"],
                ["290", "390"],
                ["291", "391"],
                ["292", "392"],
                ["293", "393"],
                ["294", "394"],
                ["295", "395"],
                ["296", "396"],
                ["297", "397"],
                ["298", "398"],
                ["299", "399"],
                ["200", "300"],
                ["201", "301"],
                ["202", "302"],
                ["203", "303"],
                ["204", "304"],
                ["205", "305"],
                ["206", "306"],
                ["207", "307"],
                ["208", "308"],
                ["209", "309"],
                ["210", "310"],
                ["211", "311"],
                ["212", "312"],
                ["213", "313"],
                ["214", "314"],
                ["215", "315"],
                ["216", "316"],
                ["217", "317"],
                ["218", "318"],
                ["219", "319"],
                ["220", "320"],
                ["221", "321"],
                ["222", "322"],
                ["223", "323"],
                ["224", "324"],
                ["225", "325"],
                ["226", "326"],
                ["227", "327"],
                ["228", "328"],
                ["229", "329"],
                ["230", "330"],
                ["231", "331"],
                ["232", "332"],
                ["233", "333"],
                ["234", "334"],
                ["235", "335"],
                ["236", "336"],
                ["237", "337"],
                ["238", "338"],
                ["239", "339"],
                ["240", "340"],
                ["241", "341"],
                ["242", "342"],
                ["243", "343"],
                ["244", "344"],
                ["245", "345"],
                ["246", "346"],
                ["247", "347"],
                ["248", "348"],
                ["249", "349"],
            ],
            dtype="<U3",
        )
    )


@pytest.mark.parametrize("TDF1", ["view", "backed view"], indirect=True)
def test_set_values_with_wrong_shape_should_fail_view(TDF1: TemporalDataFrameBase) -> None:
    with pytest.raises(ValueError) as exc_info:
        TDF1["0h", 10:70:2, ["col4", "col1"]] = np.ones((50, 50))

    assert str(exc_info.value) == "Can't set (10, 2) values from (50, 50) array."


@pytest.mark.parametrize("TDF1", ["view", "backed view"], indirect=True)
def test_set_values_view(TDF1: TemporalDataFrameBase) -> None:
    TDF1["0h", 10:70:2, ["col4", "col1"]] = np.array(
        [["a", -1], ["b", -2], ["c", -3], ["d", -4], ["e", -5], ["f", -6], ["g", -7], ["h", -8], ["i", -9], ["j", -10]]
    )

    assert np.all(
        TDF1.values_num
        == np.array(
            [
                [0.0, 100.0],
                [1.0, 101.0],
                [2.0, 102.0],
                [3.0, 103.0],
                [4.0, 104.0],
                [5.0, 105.0],
                [6.0, 106.0],
                [7.0, 107.0],
                [8.0, 108.0],
                [9.0, 109.0],
                [10.0, 110.0],
                [11.0, 111.0],
                [12.0, 112.0],
                [13.0, 113.0],
                [14.0, 114.0],
                [15.0, 115.0],
                [16.0, 116.0],
                [17.0, 117.0],
                [18.0, 118.0],
                [19.0, 119.0],
                [20.0, 120.0],
                [21.0, 121.0],
                [22.0, 122.0],
                [23.0, 123.0],
                [24.0, 124.0],
                [25.0, 125.0],
                [26.0, 126.0],
                [27.0, 127.0],
                [28.0, 128.0],
                [29.0, 129.0],
                [30.0, 130.0],
                [31.0, 131.0],
                [32.0, 132.0],
                [33.0, 133.0],
                [34.0, 134.0],
                [35.0, 135.0],
                [36.0, 136.0],
                [37.0, 137.0],
                [38.0, 138.0],
                [39.0, 139.0],
                [-1.0, 150.0],
                [51.0, 151.0],
                [-2.0, 152.0],
                [53.0, 153.0],
                [-3.0, 154.0],
                [55.0, 155.0],
                [-4.0, 156.0],
                [57.0, 157.0],
                [-5.0, 158.0],
                [59.0, 159.0],
                [-6.0, 160.0],
                [61.0, 161.0],
                [-7.0, 162.0],
                [63.0, 163.0],
                [-8.0, 164.0],
                [65.0, 165.0],
                [-9.0, 166.0],
                [67.0, 167.0],
                [-10.0, 168.0],
                [69.0, 169.0],
                [70.0, 170.0],
                [71.0, 171.0],
                [72.0, 172.0],
                [73.0, 173.0],
                [74.0, 174.0],
                [75.0, 175.0],
                [76.0, 176.0],
                [77.0, 177.0],
                [78.0, 178.0],
                [79.0, 179.0],
                [80.0, 180.0],
                [81.0, 181.0],
                [82.0, 182.0],
                [83.0, 183.0],
                [84.0, 184.0],
                [85.0, 185.0],
                [86.0, 186.0],
                [87.0, 187.0],
                [88.0, 188.0],
                [89.0, 189.0],
            ]
        )
    )
    assert np.all(
        TDF1.values_str
        == np.array(
            [
                ["200", "300"],
                ["201", "301"],
                ["202", "302"],
                ["203", "303"],
                ["204", "304"],
                ["205", "305"],
                ["206", "306"],
                ["207", "307"],
                ["208", "308"],
                ["209", "309"],
                ["210", "310"],
                ["211", "311"],
                ["212", "312"],
                ["213", "313"],
                ["214", "314"],
                ["215", "315"],
                ["216", "316"],
                ["217", "317"],
                ["218", "318"],
                ["219", "319"],
                ["220", "320"],
                ["221", "321"],
                ["222", "322"],
                ["223", "323"],
                ["224", "324"],
                ["225", "325"],
                ["226", "326"],
                ["227", "327"],
                ["228", "328"],
                ["229", "329"],
                ["230", "330"],
                ["231", "331"],
                ["232", "332"],
                ["233", "333"],
                ["234", "334"],
                ["235", "335"],
                ["236", "336"],
                ["237", "337"],
                ["238", "338"],
                ["239", "339"],
                ["250", "a"],
                ["251", "351"],
                ["252", "b"],
                ["253", "353"],
                ["254", "c"],
                ["255", "355"],
                ["256", "d"],
                ["257", "357"],
                ["258", "e"],
                ["259", "359"],
                ["260", "f"],
                ["261", "361"],
                ["262", "g"],
                ["263", "363"],
                ["264", "h"],
                ["265", "365"],
                ["266", "i"],
                ["267", "367"],
                ["268", "j"],
                ["269", "369"],
                ["270", "370"],
                ["271", "371"],
                ["272", "372"],
                ["273", "373"],
                ["274", "374"],
                ["275", "375"],
                ["276", "376"],
                ["277", "377"],
                ["278", "378"],
                ["279", "379"],
                ["280", "380"],
                ["281", "381"],
                ["282", "382"],
                ["283", "383"],
                ["284", "384"],
                ["285", "385"],
                ["286", "386"],
                ["287", "387"],
                ["288", "388"],
                ["289", "389"],
            ],
            dtype="<U3",
        )
    )


@pytest.mark.parametrize("TDF1", ["view", "backed view"], indirect=True)
def test_set_values_for_timepoints_not_in_tdf_should_fail_view(TDF1: TemporalDataFrameBase) -> None:
    with pytest.raises(KeyError) as exc_info:
        TDF1[["0h", "2h"], 10:70:2, ["col4", "col1"]] = np.array([["a", -1]])

    assert str(exc_info.value) == '"Could not find [2.0h] in TemporalDataFrame\'s timepoints."'


@pytest.mark.parametrize("TDF1", ["view", "backed view"], indirect=True)
def test_set_values_for_rows_not_in_tdf_should_fail_view(TDF1: TemporalDataFrameBase) -> None:
    with pytest.raises(KeyError) as exc_info:
        TDF1["0h", 0:200:20, ["col4", "col1"]] = np.array(
            [
                ["a", -1],
                ["b", -2],
                ["c", -3],
                ["d", -4],
                ["e", -5],
                ["f", -6],
                ["g", -7],
                ["h", -8],
                ["i", -9],
                ["j", -10],
            ]
        )

    assert str(exc_info.value) == "\"Could not find '200' in TemporalDataFrame's index.\""


@pytest.mark.parametrize("TDF1", ["view", "backed view"], indirect=True)
def test_set_values_forcolumns_not_in_TDF_should_fail_view(TDF1: TemporalDataFrameBase) -> None:
    # set values for columns, some not in tdf
    with pytest.raises(KeyError) as exc_info:
        TDF1["0h", 10:70:20, ["col4", "col1", "col5"]] = np.array(
            [
                ["a", -1, 0],
                ["b", -2, 0],
                ["c", -3, 0],
                ["d", -4, 0],
                ["e", -5, 0],
                ["f", -6, 0],
                ["g", -7, 0],
                ["h", -8, 0],
                ["i", -9, 0],
                ["j", -10, 0],
            ]
        )

    assert str(exc_info.value) == "\"Could not find ['col5'] in TemporalDataFrame's columns.\""


@pytest.mark.parametrize("TDF1", ["view", "backed view"], indirect=True)
def test_set_values_with_same_index_at_multiple_timepoints_view(TDF1: TemporalDataFrameBase) -> None:
    TDF1.parent.set_index(RepeatingIndex(np.arange(0, 50), repeats=2))

    TDF1[:, [4, 0, 2]] = np.array(
        [
            [-100, -101, "A", "AA"],
            [-200, -201, "B", "BB"],
            [-300, -301, "C", "CC"],
            [-400, -401, "D", "DD"],
            [-500, -501, "E", "EE"],
            [-600, -601, "F", "FF"],
        ]
    )

    assert np.all(
        TDF1.values_num
        == np.array(
            [
                [-200.0, -201.0],
                [1.0, 101.0],
                [-300.0, -301.0],
                [3.0, 103.0],
                [-100.0, -101.0],
                [5.0, 105.0],
                [6.0, 106.0],
                [7.0, 107.0],
                [8.0, 108.0],
                [9.0, 109.0],
                [10.0, 110.0],
                [11.0, 111.0],
                [12.0, 112.0],
                [13.0, 113.0],
                [14.0, 114.0],
                [15.0, 115.0],
                [16.0, 116.0],
                [17.0, 117.0],
                [18.0, 118.0],
                [19.0, 119.0],
                [20.0, 120.0],
                [21.0, 121.0],
                [22.0, 122.0],
                [23.0, 123.0],
                [24.0, 124.0],
                [25.0, 125.0],
                [26.0, 126.0],
                [27.0, 127.0],
                [28.0, 128.0],
                [29.0, 129.0],
                [30.0, 130.0],
                [31.0, 131.0],
                [32.0, 132.0],
                [33.0, 133.0],
                [34.0, 134.0],
                [35.0, 135.0],
                [36.0, 136.0],
                [37.0, 137.0],
                [38.0, 138.0],
                [39.0, 139.0],
                [-500.0, -501.0],
                [51.0, 151.0],
                [-600.0, -601.0],
                [53.0, 153.0],
                [-400.0, -401.0],
                [55.0, 155.0],
                [56.0, 156.0],
                [57.0, 157.0],
                [58.0, 158.0],
                [59.0, 159.0],
                [60.0, 160.0],
                [61.0, 161.0],
                [62.0, 162.0],
                [63.0, 163.0],
                [64.0, 164.0],
                [65.0, 165.0],
                [66.0, 166.0],
                [67.0, 167.0],
                [68.0, 168.0],
                [69.0, 169.0],
                [70.0, 170.0],
                [71.0, 171.0],
                [72.0, 172.0],
                [73.0, 173.0],
                [74.0, 174.0],
                [75.0, 175.0],
                [76.0, 176.0],
                [77.0, 177.0],
                [78.0, 178.0],
                [79.0, 179.0],
                [80.0, 180.0],
                [81.0, 181.0],
                [82.0, 182.0],
                [83.0, 183.0],
                [84.0, 184.0],
                [85.0, 185.0],
                [86.0, 186.0],
                [87.0, 187.0],
                [88.0, 188.0],
                [89.0, 189.0],
            ]
        )
    )
    assert np.all(
        TDF1.values_str
        == np.array(
            [
                ["B", "BB"],
                ["201", "301"],
                ["C", "CC"],
                ["203", "303"],
                ["A", "AA"],
                ["205", "305"],
                ["206", "306"],
                ["207", "307"],
                ["208", "308"],
                ["209", "309"],
                ["210", "310"],
                ["211", "311"],
                ["212", "312"],
                ["213", "313"],
                ["214", "314"],
                ["215", "315"],
                ["216", "316"],
                ["217", "317"],
                ["218", "318"],
                ["219", "319"],
                ["220", "320"],
                ["221", "321"],
                ["222", "322"],
                ["223", "323"],
                ["224", "324"],
                ["225", "325"],
                ["226", "326"],
                ["227", "327"],
                ["228", "328"],
                ["229", "329"],
                ["230", "330"],
                ["231", "331"],
                ["232", "332"],
                ["233", "333"],
                ["234", "334"],
                ["235", "335"],
                ["236", "336"],
                ["237", "337"],
                ["238", "338"],
                ["239", "339"],
                ["E", "EE"],
                ["251", "351"],
                ["F", "FF"],
                ["253", "353"],
                ["D", "DD"],
                ["255", "355"],
                ["256", "356"],
                ["257", "357"],
                ["258", "358"],
                ["259", "359"],
                ["260", "360"],
                ["261", "361"],
                ["262", "362"],
                ["263", "363"],
                ["264", "364"],
                ["265", "365"],
                ["266", "366"],
                ["267", "367"],
                ["268", "368"],
                ["269", "369"],
                ["270", "370"],
                ["271", "371"],
                ["272", "372"],
                ["273", "373"],
                ["274", "374"],
                ["275", "375"],
                ["276", "376"],
                ["277", "377"],
                ["278", "378"],
                ["279", "379"],
                ["280", "380"],
                ["281", "381"],
                ["282", "382"],
                ["283", "383"],
                ["284", "384"],
                ["285", "385"],
                ["286", "386"],
                ["287", "387"],
                ["288", "388"],
                ["289", "389"],
            ],
            dtype="<U3",
        )
    )


@pytest.mark.parametrize("TDF1", ["view", "backed view"], indirect=True)
def test_set_single_value_broadcast_view(TDF1: TemporalDataFrameBase) -> None:
    TDF1[:, [4, 0, 2], ["col1", "col2"]] = 1000

    assert np.array_equal(
        TDF1.values_num,
        np.array(
            [
                [1000.0, 1000.0],
                [1.0, 101.0],
                [1000.0, 1000.0],
                [3.0, 103.0],
                [1000.0, 1000.0],
                [5.0, 105.0],
                [6.0, 106.0],
                [7.0, 107.0],
                [8.0, 108.0],
                [9.0, 109.0],
                [10.0, 110.0],
                [11.0, 111.0],
                [12.0, 112.0],
                [13.0, 113.0],
                [14.0, 114.0],
                [15.0, 115.0],
                [16.0, 116.0],
                [17.0, 117.0],
                [18.0, 118.0],
                [19.0, 119.0],
                [20.0, 120.0],
                [21.0, 121.0],
                [22.0, 122.0],
                [23.0, 123.0],
                [24.0, 124.0],
                [25.0, 125.0],
                [26.0, 126.0],
                [27.0, 127.0],
                [28.0, 128.0],
                [29.0, 129.0],
                [30.0, 130.0],
                [31.0, 131.0],
                [32.0, 132.0],
                [33.0, 133.0],
                [34.0, 134.0],
                [35.0, 135.0],
                [36.0, 136.0],
                [37.0, 137.0],
                [38.0, 138.0],
                [39.0, 139.0],
                [50.0, 150.0],
                [51.0, 151.0],
                [52.0, 152.0],
                [53.0, 153.0],
                [54.0, 154.0],
                [55.0, 155.0],
                [56.0, 156.0],
                [57.0, 157.0],
                [58.0, 158.0],
                [59.0, 159.0],
                [60.0, 160.0],
                [61.0, 161.0],
                [62.0, 162.0],
                [63.0, 163.0],
                [64.0, 164.0],
                [65.0, 165.0],
                [66.0, 166.0],
                [67.0, 167.0],
                [68.0, 168.0],
                [69.0, 169.0],
                [70.0, 170.0],
                [71.0, 171.0],
                [72.0, 172.0],
                [73.0, 173.0],
                [74.0, 174.0],
                [75.0, 175.0],
                [76.0, 176.0],
                [77.0, 177.0],
                [78.0, 178.0],
                [79.0, 179.0],
                [80.0, 180.0],
                [81.0, 181.0],
                [82.0, 182.0],
                [83.0, 183.0],
                [84.0, 184.0],
                [85.0, 185.0],
                [86.0, 186.0],
                [87.0, 187.0],
                [88.0, 188.0],
                [89.0, 189.0],
            ]
        ),
    )
    assert np.all(
        TDF1.values_str
        == np.array(
            [
                ["200", "300"],
                ["201", "301"],
                ["202", "302"],
                ["203", "303"],
                ["204", "304"],
                ["205", "305"],
                ["206", "306"],
                ["207", "307"],
                ["208", "308"],
                ["209", "309"],
                ["210", "310"],
                ["211", "311"],
                ["212", "312"],
                ["213", "313"],
                ["214", "314"],
                ["215", "315"],
                ["216", "316"],
                ["217", "317"],
                ["218", "318"],
                ["219", "319"],
                ["220", "320"],
                ["221", "321"],
                ["222", "322"],
                ["223", "323"],
                ["224", "324"],
                ["225", "325"],
                ["226", "326"],
                ["227", "327"],
                ["228", "328"],
                ["229", "329"],
                ["230", "330"],
                ["231", "331"],
                ["232", "332"],
                ["233", "333"],
                ["234", "334"],
                ["235", "335"],
                ["236", "336"],
                ["237", "337"],
                ["238", "338"],
                ["239", "339"],
                ["250", "350"],
                ["251", "351"],
                ["252", "352"],
                ["253", "353"],
                ["254", "354"],
                ["255", "355"],
                ["256", "356"],
                ["257", "357"],
                ["258", "358"],
                ["259", "359"],
                ["260", "360"],
                ["261", "361"],
                ["262", "362"],
                ["263", "363"],
                ["264", "364"],
                ["265", "365"],
                ["266", "366"],
                ["267", "367"],
                ["268", "368"],
                ["269", "369"],
                ["270", "370"],
                ["271", "371"],
                ["272", "372"],
                ["273", "373"],
                ["274", "374"],
                ["275", "375"],
                ["276", "376"],
                ["277", "377"],
                ["278", "378"],
                ["279", "379"],
                ["280", "380"],
                ["281", "381"],
                ["282", "382"],
                ["283", "383"],
                ["284", "384"],
                ["285", "385"],
                ["286", "386"],
                ["287", "387"],
                ["288", "388"],
                ["289", "389"],
            ],
            dtype="<U3",
        )
    )


@pytest.mark.parametrize("TDF1", ["plain", "backed"], indirect=True)
def test_reindex(TDF1: TemporalDataFrame) -> None:
    # all in index
    TDF1.reindex(np.arange(99, -1, -1))

    assert np.all(TDF1.index == np.arange(99, -1, -1))
    assert np.all(TDF1.values_num == np.vstack((np.arange(99, -1, -1), np.arange(199, 99, -1))).T)
    assert np.all(TDF1.values_str == np.vstack((np.arange(299, 199, -1), np.arange(399, 299, -1))).T.astype(str))


@pytest.mark.parametrize("TDF1", ["plain", "backed"], indirect=True)
def test_reindex_with_indices_not_in_original_index_should_fail(TDF1: TemporalDataFrame) -> None:
    with pytest.raises(ValueError) as exc_info:
        TDF1.reindex(np.arange(149, 49, -1))

    assert str(exc_info.value) == "New index contains values which are not in the current index."


@pytest.mark.parametrize("TDF1", ["plain", "backed"], indirect=True)
def test_reindex_with_repeating_index_should_fail(TDF1: TemporalDataFrame) -> None:
    with pytest.raises(ValueError) as exc_info:
        TDF1.reindex(RepeatingIndex(np.arange(0, 50), repeats=2))

    assert str(exc_info.value == "Cannot set repeating index on tdf with non-repeating index.")


@pytest.mark.parametrize("TDF1", ["plain", "backed"], indirect=True)
def test_reindex_with_repeating_index_on_tdf_with_repeating_index(TDF1: TemporalDataFrame) -> None:
    TDF1.set_index(RepeatingIndex(np.arange(0, 50), repeats=2))

    TDF1.reindex(RepeatingIndex(np.arange(49, -1, -1), repeats=2))

    assert np.all(TDF1.index == np.concatenate((np.arange(49, -1, -1), np.arange(49, -1, -1))))
    assert np.all(TDF1.values_num == np.vstack((np.arange(99, -1, -1), np.arange(199, 99, -1))).T)
    assert np.all(TDF1.values_str == np.vstack((np.arange(299, 199, -1), np.arange(399, 299, -1))).T.astype(str))


# TODO
# @pytest.mark.parametrize('provide_TDFs', [(False, 'test_sub_getting_inverted_TDF', 1, 'r'),
#                                           (True, 'test_sub_getting_inverted_TDF', 3, 'r')],
#                          indirect=True)
# def test_reversed_sub_getting(provide_TDFs) -> None:
#     TDF, backed_TDF = provide_TDFs
#
#     # tdf is not backed -------------------------------------------------------
#     # subset only on time-points
#     inverted_view = (~TDF)['0h']
#
#     assert np.array_equal(inverted_view.timepoints, [TimePoint('1h')])
#     assert np.array_equal(inverted_view.index, np.arange(50))
#     assert np.array_equal(inverted_view.columns, TDF.columns)
#
#     # subset only on columns
#     inverted_view = (~TDF)[:, :, ['col3', 'col2']]
#
#     assert np.array_equal(inverted_view.timepoints, TDF.timepoints)
#     assert np.array_equal(inverted_view.index, TDF.index)
#     assert np.array_equal(inverted_view.columns, ['col1', 'col4'])
#
#
#
# @pytest.mark.parametrize('provide_TDFs', [(False, 'test_sub_getting_inverted_TDF', 1, 'r+'),
#                                           (True, 'test_sub_getting_inverted_TDF', 3, 'r+')],
#                          indirect=True)
# def test_reversed_sub_setting(provide_TDFs) -> None:
#     TDF, backed_TDF = provide_TDFs
#
#     # tdf is not backed -------------------------------------------------------
#     # subset only on time-points
#     (~TDF)['0h'] = np.concatenate((
#         -1 * (~TDF)['0h'].values_num,
#         -1 * (~TDF)['0h'].values_str.astype(int)
#     ), axis=1)
#
#     assert np.array_equal(TDF.values_num, np.vstack((
#         np.concatenate((np.arange(50, 100), -1 * np.arange(50))),
#         np.concatenate((np.arange(150, 200), -1 * np.arange(100, 150)))
#     )).T)
#     assert np.array_equal(TDF.values_str, np.vstack((
#         np.concatenate((np.arange(250, 300).astype(str), (-1 * np.arange(200., 250.)).astype(str))),
#         np.concatenate((np.arange(350, 400).astype(str), (-1 * np.arange(300., 350.)).astype(str)))
#     )).T)
#
#     # subset only on columns
#     (~TDF)[:, :, ['col3', 'col4', 'col2']] = 2 * TDF.col1
#
#     assert np.array_equal(TDF.values_num, np.vstack((
#         np.concatenate((np.arange(100, 200, 2), -1 * np.arange(0, 100, 2))),
#         np.concatenate((np.arange(150, 200), -1 * np.arange(100, 150)))
#     )).T)
#     assert np.array_equal(TDF.values_str, np.vstack((
#         np.concatenate((np.arange(250, 300).astype(str), (-1 * np.arange(200., 250.)).astype(str))),
#         np.concatenate((np.arange(350, 400).astype(str), (-1 * np.arange(300., 350.)).astype(str)))
#     )).T)
#
#     # tdf is backed -----------------------------------------------------------
#     # subset only on time-points
#     (~backed_TDF)['0h'] = -1 * (~backed_TDF)['0h']
#
#     assert np.array_equal(backed_TDF.values_num, np.vstack((
#         np.concatenate((np.arange(0, 50, 2), -1 * np.arange(50, 100, 2))),
#         np.concatenate((np.arange(1, 51, 2), -1 * np.arange(51, 101, 2)))
#     )).T)
#     assert np.array_equal(backed_TDF.values_str, np.arange(100, 150).astype(str)[:, None])
#
#     # subset only on columns
#     (~backed_TDF)[:, :, ['col3', 'col2']] = np.arange(100, 150)
#
#     assert np.array_equal(backed_TDF.values_num, np.vstack((
#         np.arange(100, 150),
#         np.concatenate((np.arange(1, 51, 2), -1 * np.arange(51, 101, 2)))
#     )).T)
#     assert np.array_equal(backed_TDF.values_str, np.arange(100, 150).astype(str)[:, None])


@pytest.mark.parametrize("TDF1", ["plain", "backed"], indirect=True)
@pytest.mark.parametrize("operation,expected", [("min", 0.0), ("max", 199.0), ("mean", 99.5)])
def test_global_min_max_mean(TDF1: TemporalDataFrameBase, operation: str, expected: float) -> None:
    assert getattr(TDF1, operation)() == expected


@pytest.mark.parametrize("TDF1", ["view", "backed view"], indirect=True)
@pytest.mark.parametrize("operation,expected", [("min", 0.0), ("max", 189.0), ("mean", 94.5)])
def test_global_min_max_mean_on_view(TDF1: TemporalDataFrameBase, operation: str, expected: float) -> None:
    assert getattr(TDF1, operation)() == expected


@pytest.mark.parametrize("TDF1", ["plain", "backed"], indirect=True)
@pytest.mark.parametrize(
    "operation,expected",
    [
        ("min", np.array([[50.0, 150.0], [0.0, 100.0]])),
        ("max", np.array([[99.0, 199.0], [49.0, 149.0]])),
        ("mean", np.array([[74.5, 174.5], [24.5, 124.5]])),
    ],
)
def test_min_max_mean_on_rows(TDF1: TemporalDataFrameBase, operation: str, expected: npt.NDArray[np.float_]) -> None:
    assert getattr(TDF1, operation)(axis=1) == TemporalDataFrame(
        data=pd.DataFrame(expected, index=[operation, operation], columns=["col1", "col2"]),
        timepoints=TDF1.timepoints,
        time_col_name=TDF1.timepoints_column_name,
    )


@pytest.mark.parametrize("TDF1", ["view", "backed view"], indirect=True)
@pytest.mark.parametrize(
    "operation,expected",
    [
        ("min", np.array([[50.0, 150.0], [0.0, 100.0]])),
        ("max", np.array([[89.0, 189.0], [39.0, 139.0]])),
        ("mean", np.array([[69.5, 169.5], [19.5, 119.5]])),
    ],
)
def test_min_max_mean_on_rows_on_view(
    TDF1: TemporalDataFrameBase, operation: str, expected: npt.NDArray[np.float_]
) -> None:
    assert getattr(TDF1, operation)(axis=1) == TemporalDataFrame(
        data=pd.DataFrame(expected, index=[operation, operation], columns=["col1", "col2"]),
        timepoints=["0h", "1h"],
        time_col_name=TDF1.timepoints_column_name,
    )


@pytest.mark.parametrize("TDF1", ["plain", "view", "backed", "backed view"], indirect=True)
@pytest.mark.parametrize("operation", ["min", "max", "mean"])
def test_min_max_mean_on_columns(TDF1: TemporalDataFrameBase, operation: str) -> None:
    assert getattr(TDF1, operation)(axis=2) == TemporalDataFrame(
        data=pd.DataFrame(
            getattr(np, operation)(TDF1.values_num, axis=1)[:, None], index=TDF1.index, columns=[operation]
        ),
        timepoints=TDF1.timepoints_column[:],
        time_col_name=TDF1.timepoints_column_name,
    )


@pytest.mark.parametrize("TDF1", ["plain", "view", "backed", "backed view"], indirect=True)
@pytest.mark.parametrize("operation", ["min", "max", "mean"])
def test_min_max_mean_on_timepoints(TDF1: TemporalDataFrameBase, operation: str) -> None:
    if TDF1.is_view:
        TDF1.parent.set_index(RepeatingIndex(np.arange(0, 50), repeats=2))

    else:
        TDF1.set_index(RepeatingIndex(np.arange(0, 50), repeats=2))

    assert np.array_equal(
        getattr(TDF1, operation)(axis=0),
        pd.DataFrame(
            getattr(np, operation)([TDF1["0h"].values_num, TDF1["1h"].values_num], axis=0),
            index=TDF1.index_at(TDF1.tp0),
            columns=TDF1.columns_num,
        ),
    )


def test_should_check_contains_in_columns() -> None:
    from tempfile import NamedTemporaryFile

    tdf = TemporalDataFrame(
        {
            "col1": np.array([i for i in range(100)]),
            "col2": np.array([i for i in range(100, 200)]),
        },
        timepoints=["1h" for _ in range(50)] + ["0h" for _ in range(50)],
    )
    with NamedTemporaryFile() as file:
        tdf.write(file.name)

        assert "test" not in tdf.columns_str
