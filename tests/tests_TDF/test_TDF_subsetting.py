# coding: utf-8
# Created on 31/03/2022 16:33
# Author : matteo

# ====================================================
# imports
import pytest
import numpy as np
import pandas as pd
from pathlib import Path

from vdata import TemporalDataFrame, TimePoint
from vdata.name_utils import H5Mode
from vdata.time_point import mean as tp_mean


# ====================================================
# code
@pytest.mark.usefixtures('class_TDF1')
@pytest.mark.parametrize(
    'class_TDF1',
    ['plain', 'view', 'backed', 'backed view'],
    indirect=True
)
class TestSubGetting:
    def test_subset_get_single_tp(self):
        backed = 'backed ' if self.TDF.is_backed else ''

        assert repr(self.TDF['0h']) == f"View of {backed}TemporalDataFrame 1\n" \
                                       "\x1b[4mTime point : 0.0 hours\x1b[0m\n" \
                                       "   Time-point     col1   col2    col3 col4\n" \
                                       "50       0.0h  |  50.0  150.0  |  250  350\n" \
                                       "51       0.0h  |  51.0  151.0  |  251  351\n" \
                                       "52       0.0h  |  52.0  152.0  |  252  352\n" \
                                       "53       0.0h  |  53.0  153.0  |  253  353\n" \
                                       "54       0.0h  |  54.0  154.0  |  254  354\n" \
                                       "[50 x 4]\n\n"

        assert np.all(self.TDF['0h'].values_num == np.hstack((np.arange(50, 100)[:, None],
                                                              np.arange(150, 200)[:, None])))
        assert np.all(self.TDF['0h'].values_str == np.hstack((np.arange(250, 300).astype(str)[:, None],
                                                              np.arange(350, 400).astype(str)[:, None])))

    def test_subset_get_tp_not_in_tdf_should_fail(self):
        with pytest.raises(ValueError) as exc_info:
            repr(self.TDF['1s'])

        view = 'view of a ' if self.TDF.is_view else ''
        backed = 'backed ' if self.TDF.is_backed else ''

        assert str(exc_info.value) == f"Some time-points were not found in this {view}{backed}TemporalDataFrame " \
                                      f"([1.0 seconds] (1 value long))"

    def test_subset_multiple_timepoints(self):
        backed = 'backed ' if self.TDF.is_backed else ''

        assert repr(self.TDF[['0h', '1h']]) == f"View of {backed}TemporalDataFrame 1\n" \
                                               "\x1b[4mTime point : 0.0 hours\x1b[0m\n" \
                                               "   Time-point     col1   col2    col3 col4\n" \
                                               "50       0.0h  |  50.0  150.0  |  250  350\n" \
                                               "51       0.0h  |  51.0  151.0  |  251  351\n" \
                                               "52       0.0h  |  52.0  152.0  |  252  352\n" \
                                               "53       0.0h  |  53.0  153.0  |  253  353\n" \
                                               "54       0.0h  |  54.0  154.0  |  254  354\n" \
                                               "[50 x 4]\n" \
                                               "\n" \
                                               "\x1b[4mTime point : 1.0 hours\x1b[0m\n" \
                                               "  Time-point    col1   col2    col3 col4\n" \
                                               "0       1.0h  |  0.0  100.0  |  200  300\n" \
                                               "1       1.0h  |  1.0  101.0  |  201  301\n" \
                                               "2       1.0h  |  2.0  102.0  |  202  302\n" \
                                               "3       1.0h  |  3.0  103.0  |  203  303\n" \
                                               "4       1.0h  |  4.0  104.0  |  204  304\n" \
                                               "[50 x 4]\n\n"

        assert np.all(self.TDF[['0h', '1h']].values_num == np.hstack((
            np.concatenate((np.arange(50, 100), np.arange(0, 50)))[:, None],
            np.concatenate((np.arange(150, 200), np.arange(100, 150)))[:, None])
        ))
        assert np.all(self.TDF[['0h', '1h']].values_str == np.hstack((
            np.concatenate((np.arange(250, 300), np.arange(200, 250))).astype(str)[:, None],
            np.concatenate((np.arange(350, 400), np.arange(300, 350))).astype(str)[:, None])
        ))

    def test_subset_multiple_timepoints_not_in_tdf_should_fail(self):
        # subset multiple TPs, some not in tdf
        with pytest.raises(ValueError) as exc_info:
            repr(self.TDF[['0h', '1h', '2h']])

        view = 'view of a ' if self.TDF.is_view else ''
        backed = 'backed ' if self.TDF.is_backed else ''

        assert str(exc_info.value) == f"Some time-points were not found in this {view}{backed}TemporalDataFrame " \
                                      "([2.0 hours] (1 value long))"

    def test_subset_single_row(self):
        backed = 'backed ' if self.TDF.is_backed else ''

        assert repr(self.TDF[:, 10]) == f"View of {backed}TemporalDataFrame 1\n" \
                                        "\x1b[4mTime point : 1.0 hours\x1b[0m\n" \
                                        "   Time-point     col1   col2    col3 col4\n" \
                                        "10       1.0h  |  10.0  110.0  |  210  310\n" \
                                        "[1 x 4]\n\n"

        assert np.all(self.TDF[:, 10].values_num == np.array([[10, 110]]))
        assert np.all(self.TDF[:, 10].values_str == np.array([['210', '310']]))

    def test_subset_single_row_not_in_tdf_should_fail(self):
        # subset single row, not in tdf
        with pytest.raises(ValueError) as exc_info:
            repr(self.TDF[:, 500])

        view = 'view of a ' if self.TDF.is_view else ''
        backed = 'backed ' if self.TDF.is_backed else ''

        assert str(exc_info.value) == f"Some indices were not found in this {view}{backed}TemporalDataFrame " \
                                      f"([500] (1 value long))"

    def test_subset_multiple_rows(self):
        backed = 'backed ' if self.TDF.is_backed else ''

        assert repr(self.TDF[:, range(25, 75)]) == f"View of {backed}TemporalDataFrame 1\n" \
                                                   "\x1b[4mTime point : 0.0 hours\x1b[0m\n" \
                                                   "   Time-point     col1   col2    col3 col4\n" \
                                                   "50       0.0h  |  50.0  150.0  |  250  350\n" \
                                                   "51       0.0h  |  51.0  151.0  |  251  351\n" \
                                                   "52       0.0h  |  52.0  152.0  |  252  352\n" \
                                                   "53       0.0h  |  53.0  153.0  |  253  353\n" \
                                                   "54       0.0h  |  54.0  154.0  |  254  354\n" \
                                                   "[25 x 4]\n" \
                                                   "\n" \
                                                   "\x1b[4mTime point : 1.0 hours\x1b[0m\n" \
                                                   "   Time-point     col1   col2    col3 col4\n" \
                                                   "25       1.0h  |  25.0  125.0  |  225  325\n" \
                                                   "26       1.0h  |  26.0  126.0  |  226  326\n" \
                                                   "27       1.0h  |  27.0  127.0  |  227  327\n" \
                                                   "28       1.0h  |  28.0  128.0  |  228  328\n" \
                                                   "29       1.0h  |  29.0  129.0  |  229  329\n" \
                                                   "[25 x 4]\n\n"

        assert np.all(self.TDF[:, range(25, 75)].values_num == np.hstack((
            np.concatenate((np.arange(50, 75), np.arange(25, 50)))[:, None],
            np.concatenate((np.arange(150, 175), np.arange(125, 150)))[:, None])
        ))
        assert np.all(self.TDF[:, range(25, 75)].values_str == np.hstack((
            np.concatenate((np.arange(250, 275), np.arange(225, 250))).astype(str)[:, None],
            np.concatenate((np.arange(350, 375), np.arange(325, 350))).astype(str)[:, None])
        ))

    def test_subset_multiple_rows_with_some_not_in_tdf_should_fail(self):
        # subset multiple rows, some not in tdf
        with pytest.raises(ValueError) as exc_info:
            repr(self.TDF[:, 20:500:2])

        view = 'view of a ' if self.TDF.is_view else ''
        backed = 'backed ' if self.TDF.is_backed else ''

        assert str(exc_info.value) == f"Some indices were not found in this {view}{backed}TemporalDataFrame " \
                                      "([100 102 ... 496 498] (200 values long))"

    def test_subset_single_column(self):
        backed = 'backed ' if self.TDF.is_backed else ''

        assert repr(self.TDF[:, :, 'col3']) == f"View of {backed}TemporalDataFrame 1\n" \
                                               "\x1b[4mTime point : 0.0 hours\x1b[0m\n" \
                                               "   Time-point    col3\n" \
                                               "50       0.0h  |  250\n" \
                                               "51       0.0h  |  251\n" \
                                               "52       0.0h  |  252\n" \
                                               "53       0.0h  |  253\n" \
                                               "54       0.0h  |  254\n" \
                                               "[50 x 1]\n" \
                                               "\n" \
                                               "\x1b[4mTime point : 1.0 hours\x1b[0m\n" \
                                               "  Time-point    col3\n" \
                                               "0       1.0h  |  200\n" \
                                               "1       1.0h  |  201\n" \
                                               "2       1.0h  |  202\n" \
                                               "3       1.0h  |  203\n" \
                                               "4       1.0h  |  204\n" \
                                               "[50 x 1]\n\n"

        assert self.TDF[:, :, 'col3'].values_num.size == 0
        assert np.all(self.TDF[:, :, 'col3'].values_str ==
                      np.concatenate((np.arange(250, 300), np.arange(200, 250))).astype(str)[:, None])

    def test_subset_single_column_with_getattr(self):
        backed = 'backed ' if self.TDF.is_backed else ''

        assert repr(self.TDF.col2) == f"View of {backed}TemporalDataFrame 1\n" \
                                      "\x1b[4mTime point : 0.0 hours\x1b[0m\n" \
                                      "   Time-point      col2\n" \
                                      "50       0.0h  |  150.0\n" \
                                      "51       0.0h  |  151.0\n" \
                                      "52       0.0h  |  152.0\n" \
                                      "53       0.0h  |  153.0\n" \
                                      "54       0.0h  |  154.0\n" \
                                      "[50 x 1]\n" \
                                      "\n" \
                                      "\x1b[4mTime point : 1.0 hours\x1b[0m\n" \
                                      "  Time-point      col2\n" \
                                      "0       1.0h  |  100.0\n" \
                                      "1       1.0h  |  101.0\n" \
                                      "2       1.0h  |  102.0\n" \
                                      "3       1.0h  |  103.0\n" \
                                      "4       1.0h  |  104.0\n" \
                                      "[50 x 1]\n\n"

        assert np.all(self.TDF.col2.values_num == np.concatenate((np.arange(150, 200), np.arange(100, 150)))[:, None])
        assert self.TDF.col2.values_str.size == 0

    def test_subset_column_not_in_tdf_should_fail(self):
        with pytest.raises(ValueError) as exc_info:
            repr(self.TDF[:, :, 'col5'])

        view = 'view of a ' if self.TDF.is_view else ''
        backed = 'backed ' if self.TDF.is_backed else ''

        assert str(exc_info.value) == f"Some columns were not found in this {view}{backed}TemporalDataFrame " \
                                      f"(['col5'] (1 value long))"

    def test_subset_column_with_getattr_not_in_tdf_should_fail(self):
        with pytest.raises(AttributeError) as exc_info:
            repr(self.TDF.col5)

        view = 'view of a ' if self.TDF.is_view else ''
        backed = 'backed ' if self.TDF.is_backed else ''

        assert str(exc_info.value) == f"'col5' not found in this {view}{backed}TemporalDataFrame."

    def test_subset_multiple_columns(self):
        backed = 'backed ' if self.TDF.is_backed else ''

        assert repr(self.TDF[:, :, ['col1', 'col3']]) == f"View of {backed}TemporalDataFrame 1\n" \
                                                         "\x1b[4mTime point : 0.0 hours\x1b[0m\n" \
                                                         "   Time-point     col1    col3\n" \
                                                         "50       0.0h  |  50.0  |  250\n" \
                                                         "51       0.0h  |  51.0  |  251\n" \
                                                         "52       0.0h  |  52.0  |  252\n" \
                                                         "53       0.0h  |  53.0  |  253\n" \
                                                         "54       0.0h  |  54.0  |  254\n" \
                                                         "[50 x 2]\n" \
                                                         "\n" \
                                                         "\x1b[4mTime point : 1.0 hours\x1b[0m\n" \
                                                         "  Time-point    col1    col3\n" \
                                                         "0       1.0h  |  0.0  |  200\n" \
                                                         "1       1.0h  |  1.0  |  201\n" \
                                                         "2       1.0h  |  2.0  |  202\n" \
                                                         "3       1.0h  |  3.0  |  203\n" \
                                                         "4       1.0h  |  4.0  |  204\n" \
                                                         "[50 x 2]\n\n"

        assert np.all(self.TDF[:, :, ['col1', 'col3']].values_num == np.concatenate((
            np.arange(50, 100), np.arange(0, 50)))[:, None])
        assert np.all(self.TDF[:, :, ['col1', 'col3']].values_str == np.concatenate((
            np.arange(250, 300), np.arange(200, 250))).astype(str)[:, None])

    def test_subset_multiple_columns_with_some_not_in_tdf_should_fail(self):
        with pytest.raises(ValueError) as exc_info:
            repr(self.TDF[:, :, ['col1', 'col3', 'col5']])

        view = 'view of a ' if self.TDF.is_view else ''
        backed = 'backed ' if self.TDF.is_backed else ''

        assert str(exc_info.value) == f"Some columns were not found in this {view}{backed}TemporalDataFrame " \
                                      f"(['col5'] (1 value long))"

    def test_subset_multiple_columns_shuffled(self):
        backed = 'backed ' if self.TDF.is_backed else ''

        assert repr(self.TDF[:, :, ['col4', 'col2', 'col1']]) == f"View of {backed}TemporalDataFrame 1\n" \
                                                                 "\x1b[4mTime point : 0.0 hours\x1b[0m\n" \
                                                                 "   Time-point      col2  col1    col4\n" \
                                                                 "50       0.0h  |  150.0  50.0  |  350\n" \
                                                                 "51       0.0h  |  151.0  51.0  |  351\n" \
                                                                 "52       0.0h  |  152.0  52.0  |  352\n" \
                                                                 "53       0.0h  |  153.0  53.0  |  353\n" \
                                                                 "54       0.0h  |  154.0  54.0  |  354\n" \
                                                                 "[50 x 3]\n" \
                                                                 "\n" \
                                                                 "\x1b[4mTime point : 1.0 hours\x1b[0m\n" \
                                                                 "  Time-point      col2 col1    col4\n" \
                                                                 "0       1.0h  |  100.0  0.0  |  300\n" \
                                                                 "1       1.0h  |  101.0  1.0  |  301\n" \
                                                                 "2       1.0h  |  102.0  2.0  |  302\n" \
                                                                 "3       1.0h  |  103.0  3.0  |  303\n" \
                                                                 "4       1.0h  |  104.0  4.0  |  304\n" \
                                                                 "[50 x 3]\n\n"

        assert np.all(self.TDF[:, :, ['col4', 'col2', 'col1']].values_num == np.hstack((
            np.concatenate((np.arange(150, 200), np.arange(100, 150)))[:, None],
            np.concatenate((np.arange(50, 100), np.arange(0, 50)))[:, None]
        )))

        assert np.all(self.TDF[:, :, ['col4', 'col2', 'col1']].values_str == np.concatenate((
            np.arange(350, 400), np.arange(300, 350))).astype(str)[:, None])

    def test_subset_with_timepoints_rows_and_columns(self):
        backed = 'backed ' if self.TDF.is_backed else ''

        assert repr(self.TDF['1h', 10:40:5, ['col1', 'col3']]) == f"View of {backed}TemporalDataFrame 1\n" \
                                                                  "\x1b[4mTime point : 1.0 hours\x1b[0m\n" \
                                                                  "   Time-point     col1    col3\n" \
                                                                  "10       1.0h  |  10.0  |  210\n" \
                                                                  "15       1.0h  |  15.0  |  215\n" \
                                                                  "20       1.0h  |  20.0  |  220\n" \
                                                                  "25       1.0h  |  25.0  |  225\n" \
                                                                  "30       1.0h  |  30.0  |  230\n" \
                                                                  "[6 x 2]\n\n"

        assert np.all(
            self.TDF['1h', 10:40:5, ['col1', 'col3']].values_num == np.array([10, 15, 20, 25, 30, 35])[:, None])
        assert np.all(self.TDF['1h', 10:40:5, ['col1', 'col3']].values_str == np.array([210, 215, 220, 225, 230, 235
                                                                                        ]).astype(str)[:, None])


@pytest.mark.parametrize(
    'TDF1',
    ['plain', 'view'],
    indirect=True
)
def test_subset_with_timepoints_rows_and_columns_shuffled(TDF1):
    view = TDF1[['0h'], [40, 10, 80, 60, 20, 70, 50], ['col2', 'col1', 'col3']]
    assert repr(view) == \
           "View of TemporalDataFrame 1\n" \
           "\x1b[4mTime point : 0.0 hours\x1b[0m\n" \
           "   Time-point      col2  col1    col3\n" \
           "80       0.0h  |  180.0  80.0  |  280\n" \
           "60       0.0h  |  160.0  60.0  |  260\n" \
           "70       0.0h  |  170.0  70.0  |  270\n" \
           "50       0.0h  |  150.0  50.0  |  250\n" \
           "[4 x 3]\n\n"

    assert np.all(view.values_num == np.array([
        [180, 80],
        [160, 60],
        [170, 70],
        [150, 50]
    ]))
    assert np.all(view.values_str == np.array([
        ['280'],
        ['260'],
        ['270'],
        ['250']
    ]))


@pytest.mark.parametrize(
    'TDF1',
    ['backed', 'backed view'],
    indirect=True
)
@pytest.mark.xfail(raises=TypeError)
def test_subset_with_timepoints_rows_and_columns_shuffled_should_fail(TDF1):
    repr(TDF1[['0h'], [40, 10, 80, 60, 20, 70, 50], ['col2', 'col1', 'col3']])


@pytest.mark.parametrize(
    'TDF1',
    ['plain', 'view', 'backed', 'backed view'],
    indirect=True
)
def test_subset_same_index_at_multiple_timepoints(TDF1):
    if TDF1.is_view:
        TDF1.parent.set_index(np.concatenate((np.arange(0, 50), np.arange(0, 50))), repeating_index=True)

    else:
        TDF1.set_index(np.concatenate((np.arange(0, 50), np.arange(0, 50))), repeating_index=True)

    backed = 'backed ' if TDF1.is_backed else ''

    assert repr(TDF1[:, [0, 2, 4]]) == f"View of {backed}TemporalDataFrame 1\n" \
                                       "\x1b[4mTime point : 0.0 hours\x1b[0m\n" \
                                       "  Time-point     col1   col2    col3 col4\n" \
                                       "0       0.0h  |  50.0  150.0  |  250  350\n" \
                                       "2       0.0h  |  52.0  152.0  |  252  352\n" \
                                       "4       0.0h  |  54.0  154.0  |  254  354\n" \
                                       "[3 x 4]\n" \
                                       "\n" \
                                       "\x1b[4mTime point : 1.0 hours\x1b[0m\n" \
                                       "  Time-point    col1   col2    col3 col4\n" \
                                       "0       1.0h  |  0.0  100.0  |  200  300\n" \
                                       "2       1.0h  |  2.0  102.0  |  202  302\n" \
                                       "4       1.0h  |  4.0  104.0  |  204  304\n" \
                                       "[3 x 4]\n\n"

    assert np.all(TDF1[:, [0, 2, 4]].values_num == np.array([[50, 150],
                                                             [52, 152],
                                                             [54, 154],
                                                             [0, 100],
                                                             [2, 102],
                                                             [4, 104]]))
    assert np.all(TDF1[:, [0, 2, 4]].values_str == np.array([['250', '350'],
                                                             ['252', '352'],
                                                             ['254', '354'],
                                                             ['200', '300'],
                                                             ['202', '302'],
                                                             ['204', '304']]))


@pytest.mark.parametrize(
    'TDF1',
    ['plain', 'view'],
    indirect=True
)
def test_subset_same_shuffled_index_at_multiple_timepoints(TDF1):
    if TDF1.is_view:
        TDF1.parent.set_index(np.concatenate((np.arange(0, 50), np.arange(0, 50))), repeating_index=True)

    else:
        TDF1.set_index(np.concatenate((np.arange(0, 50), np.arange(0, 50))), repeating_index=True)

    view = TDF1[:, [4, 0, 2]]

    assert np.all(view.values_num == np.array([[54, 154],
                                               [50, 150],
                                               [52, 152],
                                               [4, 104],
                                               [0, 100],
                                               [2, 102]]))
    assert np.all(view.values_str == np.array([['254', '354'],
                                               ['250', '350'],
                                               ['252', '352'],
                                               ['204', '304'],
                                               ['200', '300'],
                                               ['202', '302']]))


@pytest.mark.parametrize(
    'TDF1',
    ['backed', 'backed view'],
    indirect=True
)
@pytest.mark.xfail(raises=TypeError)
def test_subset_shuffled_index_should_fail(TDF1):
    repr(TDF1[:, [4, 0, 2]])


@pytest.mark.usefixtures('class_TDF1')
@pytest.mark.parametrize(
    'class_TDF1',
    ['plain', 'view', 'backed', 'backed view'],
    indirect=True
)
class TestSubSetting:

    def test_set_values_with_wrong_shape_should_fail(self):
        with pytest.raises(ValueError) as exc_info:
            self.TDF['0h', 10:70:2, ['col4', 'col1']] = np.ones((50, 50))

        assert str(exc_info.value) == "Can't set 10 x 2 values from 50 x 50 array."

    def test_set_values(self):
        self.TDF['0h', 10:70:2, ['col4', 'col1']] = np.array([['a', -1],
                                                              ['b', -2],
                                                              ['c', -3],
                                                              ['d', -4],
                                                              ['e', -5],
                                                              ['f', -6],
                                                              ['g', -7],
                                                              ['h', -8],
                                                              ['i', -9],
                                                              ['j', -10]])

        assert np.all(self.TDF.values_num == np.array([[-1., 150.],
                                                       [51., 151.],
                                                       [-2., 152.],
                                                       [53., 153.],
                                                       [-3., 154.],
                                                       [55., 155.],
                                                       [-4., 156.],
                                                       [57., 157.],
                                                       [-5., 158.],
                                                       [59., 159.],
                                                       [-6., 160.],
                                                       [61., 161.],
                                                       [-7., 162.],
                                                       [63., 163.],
                                                       [-8., 164.],
                                                       [65., 165.],
                                                       [-9., 166.],
                                                       [67., 167.],
                                                       [-10., 168.],
                                                       [69., 169.],
                                                       [70., 170.],
                                                       [71., 171.],
                                                       [72., 172.],
                                                       [73., 173.],
                                                       [74., 174.],
                                                       [75., 175.],
                                                       [76., 176.],
                                                       [77., 177.],
                                                       [78., 178.],
                                                       [79., 179.],
                                                       [80., 180.],
                                                       [81., 181.],
                                                       [82., 182.],
                                                       [83., 183.],
                                                       [84., 184.],
                                                       [85., 185.],
                                                       [86., 186.],
                                                       [87., 187.],
                                                       [88., 188.],
                                                       [89., 189.],
                                                       [90., 190.],
                                                       [91., 191.],
                                                       [92., 192.],
                                                       [93., 193.],
                                                       [94., 194.],
                                                       [95., 195.],
                                                       [96., 196.],
                                                       [97., 197.],
                                                       [98., 198.],
                                                       [99., 199.],
                                                       [0., 100.],
                                                       [1., 101.],
                                                       [2., 102.],
                                                       [3., 103.],
                                                       [4., 104.],
                                                       [5., 105.],
                                                       [6., 106.],
                                                       [7., 107.],
                                                       [8., 108.],
                                                       [9., 109.],
                                                       [10., 110.],
                                                       [11., 111.],
                                                       [12., 112.],
                                                       [13., 113.],
                                                       [14., 114.],
                                                       [15., 115.],
                                                       [16., 116.],
                                                       [17., 117.],
                                                       [18., 118.],
                                                       [19., 119.],
                                                       [20., 120.],
                                                       [21., 121.],
                                                       [22., 122.],
                                                       [23., 123.],
                                                       [24., 124.],
                                                       [25., 125.],
                                                       [26., 126.],
                                                       [27., 127.],
                                                       [28., 128.],
                                                       [29., 129.],
                                                       [30., 130.],
                                                       [31., 131.],
                                                       [32., 132.],
                                                       [33., 133.],
                                                       [34., 134.],
                                                       [35., 135.],
                                                       [36., 136.],
                                                       [37., 137.],
                                                       [38., 138.],
                                                       [39., 139.],
                                                       [40., 140.],
                                                       [41., 141.],
                                                       [42., 142.],
                                                       [43., 143.],
                                                       [44., 144.],
                                                       [45., 145.],
                                                       [46., 146.],
                                                       [47., 147.],
                                                       [48., 148.],
                                                       [49., 149.]]))
        assert np.all(self.TDF.values_str == np.array([['250', 'a'],
                                                       ['251', '351'],
                                                       ['252', 'b'],
                                                       ['253', '353'],
                                                       ['254', 'c'],
                                                       ['255', '355'],
                                                       ['256', 'd'],
                                                       ['257', '357'],
                                                       ['258', 'e'],
                                                       ['259', '359'],
                                                       ['260', 'f'],
                                                       ['261', '361'],
                                                       ['262', 'g'],
                                                       ['263', '363'],
                                                       ['264', 'h'],
                                                       ['265', '365'],
                                                       ['266', 'i'],
                                                       ['267', '367'],
                                                       ['268', 'j'],
                                                       ['269', '369'],
                                                       ['270', '370'],
                                                       ['271', '371'],
                                                       ['272', '372'],
                                                       ['273', '373'],
                                                       ['274', '374'],
                                                       ['275', '375'],
                                                       ['276', '376'],
                                                       ['277', '377'],
                                                       ['278', '378'],
                                                       ['279', '379'],
                                                       ['280', '380'],
                                                       ['281', '381'],
                                                       ['282', '382'],
                                                       ['283', '383'],
                                                       ['284', '384'],
                                                       ['285', '385'],
                                                       ['286', '386'],
                                                       ['287', '387'],
                                                       ['288', '388'],
                                                       ['289', '389'],
                                                       ['290', '390'],
                                                       ['291', '391'],
                                                       ['292', '392'],
                                                       ['293', '393'],
                                                       ['294', '394'],
                                                       ['295', '395'],
                                                       ['296', '396'],
                                                       ['297', '397'],
                                                       ['298', '398'],
                                                       ['299', '399'],
                                                       ['200', '300'],
                                                       ['201', '301'],
                                                       ['202', '302'],
                                                       ['203', '303'],
                                                       ['204', '304'],
                                                       ['205', '305'],
                                                       ['206', '306'],
                                                       ['207', '307'],
                                                       ['208', '308'],
                                                       ['209', '309'],
                                                       ['210', '310'],
                                                       ['211', '311'],
                                                       ['212', '312'],
                                                       ['213', '313'],
                                                       ['214', '314'],
                                                       ['215', '315'],
                                                       ['216', '316'],
                                                       ['217', '317'],
                                                       ['218', '318'],
                                                       ['219', '319'],
                                                       ['220', '320'],
                                                       ['221', '321'],
                                                       ['222', '322'],
                                                       ['223', '323'],
                                                       ['224', '324'],
                                                       ['225', '325'],
                                                       ['226', '326'],
                                                       ['227', '327'],
                                                       ['228', '328'],
                                                       ['229', '329'],
                                                       ['230', '330'],
                                                       ['231', '331'],
                                                       ['232', '332'],
                                                       ['233', '333'],
                                                       ['234', '334'],
                                                       ['235', '335'],
                                                       ['236', '336'],
                                                       ['237', '337'],
                                                       ['238', '338'],
                                                       ['239', '339'],
                                                       ['240', '340'],
                                                       ['241', '341'],
                                                       ['242', '342'],
                                                       ['243', '343'],
                                                       ['244', '344'],
                                                       ['245', '345'],
                                                       ['246', '346'],
                                                       ['247', '347'],
                                                       ['248', '348'],
                                                       ['249', '349']], dtype='<U3'))

    def test_set_values_for_timepoints_not_in_tdf_should_fail(self):
        with pytest.raises(ValueError) as exc_info:
            self.TDF[['0h', '2h'], 10:70:2, ['col4', 'col1']] = np.array([['a', -1]])

        view = 'view of a ' if self.TDF.is_view else ''
        backed = 'backed ' if self.TDF.is_backed else ''

        assert str(exc_info.value) == f'Some time-points were not found in this {view}{backed}TemporalDataFrame ' \
                                      f'([2.0 hours] (1 value long))'

    def test_set_values_for_rows_not_in_tdf_should_fail(self):
        with pytest.raises(ValueError) as exc_info:
            self.TDF['0h', 0:200:20, ['col4', 'col1']] = np.array([['a', -1],
                                                                   ['b', -2],
                                                                   ['c', -3],
                                                                   ['d', -4],
                                                                   ['e', -5],
                                                                   ['f', -6],
                                                                   ['g', -7],
                                                                   ['h', -8],
                                                                   ['i', -9],
                                                                   ['j', -10]])

        view = 'view of a ' if self.TDF.is_view else ''
        backed = 'backed ' if self.TDF.is_backed else ''

        assert str(exc_info.value) == f'Some indices were not found in this {view}{backed}TemporalDataFrame ' \
                                      '([100 120 ... 160 180] (5 values long))'

    def test_set_values_forcolumns_not_in_TDF_should_fail(self):
        # set values for columns, some not in tdf
        with pytest.raises(ValueError) as exc_info:
            self.TDF['0h', 10:70:20, ['col4', 'col1', 'col5']] = np.array([['a', -1, 0],
                                                                           ['b', -2, 0],
                                                                           ['c', -3, 0],
                                                                           ['d', -4, 0],
                                                                           ['e', -5, 0],
                                                                           ['f', -6, 0],
                                                                           ['g', -7, 0],
                                                                           ['h', -8, 0],
                                                                           ['i', -9, 0],
                                                                           ['j', -10, 0]])

        view = 'view of a ' if self.TDF.is_view else ''
        backed = 'backed ' if self.TDF.is_backed else ''

        assert str(exc_info.value) == f"Some columns were not found in this {view}{backed}TemporalDataFrame " \
                                      f"(['col5'] (1 value long))"

    def test_set_values_with_same_index_at_multiple_timepoints(self):
        if self.TDF.is_view:
            self.TDF.parent.set_index(np.concatenate((np.arange(0, 50), np.arange(0, 50))), repeating_index=True)

        else:
            self.TDF.set_index(np.concatenate((np.arange(0, 50), np.arange(0, 50))), repeating_index=True)

        self.TDF[:, [4, 0, 2]] = np.array([[-100, -101, 'A', 'AA'],
                                           [-200, -201, 'B', 'BB'],
                                           [-300, -301, 'C', 'CC'],
                                           [-400, -401, 'D', 'DD'],
                                           [-500, -501, 'E', 'EE'],
                                           [-600, -601, 'F', 'FF']])

        assert np.all(self.TDF.values_num == np.array([[-200., -201.],
                                                       [51., 151.],
                                                       [-300., -301.],
                                                       [53., 153.],
                                                       [-100., -101.],
                                                       [55., 155.],
                                                       [-4., 156.],
                                                       [57., 157.],
                                                       [-5., 158.],
                                                       [59., 159.],
                                                       [-6., 160.],
                                                       [61., 161.],
                                                       [-7., 162.],
                                                       [63., 163.],
                                                       [-8., 164.],
                                                       [65., 165.],
                                                       [-9., 166.],
                                                       [67., 167.],
                                                       [-10., 168.],
                                                       [69., 169.],
                                                       [70., 170.],
                                                       [71., 171.],
                                                       [72., 172.],
                                                       [73., 173.],
                                                       [74., 174.],
                                                       [75., 175.],
                                                       [76., 176.],
                                                       [77., 177.],
                                                       [78., 178.],
                                                       [79., 179.],
                                                       [80., 180.],
                                                       [81., 181.],
                                                       [82., 182.],
                                                       [83., 183.],
                                                       [84., 184.],
                                                       [85., 185.],
                                                       [86., 186.],
                                                       [87., 187.],
                                                       [88., 188.],
                                                       [89., 189.],
                                                       [90., 190.],
                                                       [91., 191.],
                                                       [92., 192.],
                                                       [93., 193.],
                                                       [94., 194.],
                                                       [95., 195.],
                                                       [96., 196.],
                                                       [97., 197.],
                                                       [98., 198.],
                                                       [99., 199.],
                                                       [-500., -501.],
                                                       [1., 101.],
                                                       [-600., -601.],
                                                       [3., 103.],
                                                       [-400., -401.],
                                                       [5., 105.],
                                                       [6., 106.],
                                                       [7., 107.],
                                                       [8., 108.],
                                                       [9., 109.],
                                                       [10., 110.],
                                                       [11., 111.],
                                                       [12., 112.],
                                                       [13., 113.],
                                                       [14., 114.],
                                                       [15., 115.],
                                                       [16., 116.],
                                                       [17., 117.],
                                                       [18., 118.],
                                                       [19., 119.],
                                                       [20., 120.],
                                                       [21., 121.],
                                                       [22., 122.],
                                                       [23., 123.],
                                                       [24., 124.],
                                                       [25., 125.],
                                                       [26., 126.],
                                                       [27., 127.],
                                                       [28., 128.],
                                                       [29., 129.],
                                                       [30., 130.],
                                                       [31., 131.],
                                                       [32., 132.],
                                                       [33., 133.],
                                                       [34., 134.],
                                                       [35., 135.],
                                                       [36., 136.],
                                                       [37., 137.],
                                                       [38., 138.],
                                                       [39., 139.],
                                                       [40., 140.],
                                                       [41., 141.],
                                                       [42., 142.],
                                                       [43., 143.],
                                                       [44., 144.],
                                                       [45., 145.],
                                                       [46., 146.],
                                                       [47., 147.],
                                                       [48., 148.],
                                                       [49., 149.]]))
        assert np.all(self.TDF.values_str == np.array([['B', 'BB'],
                                                       ['251', '351'],
                                                       ['C', 'CC'],
                                                       ['253', '353'],
                                                       ['A', 'AA'],
                                                       ['255', '355'],
                                                       ['256', 'd'],
                                                       ['257', '357'],
                                                       ['258', 'e'],
                                                       ['259', '359'],
                                                       ['260', 'f'],
                                                       ['261', '361'],
                                                       ['262', 'g'],
                                                       ['263', '363'],
                                                       ['264', 'h'],
                                                       ['265', '365'],
                                                       ['266', 'i'],
                                                       ['267', '367'],
                                                       ['268', 'j'],
                                                       ['269', '369'],
                                                       ['270', '370'],
                                                       ['271', '371'],
                                                       ['272', '372'],
                                                       ['273', '373'],
                                                       ['274', '374'],
                                                       ['275', '375'],
                                                       ['276', '376'],
                                                       ['277', '377'],
                                                       ['278', '378'],
                                                       ['279', '379'],
                                                       ['280', '380'],
                                                       ['281', '381'],
                                                       ['282', '382'],
                                                       ['283', '383'],
                                                       ['284', '384'],
                                                       ['285', '385'],
                                                       ['286', '386'],
                                                       ['287', '387'],
                                                       ['288', '388'],
                                                       ['289', '389'],
                                                       ['290', '390'],
                                                       ['291', '391'],
                                                       ['292', '392'],
                                                       ['293', '393'],
                                                       ['294', '394'],
                                                       ['295', '395'],
                                                       ['296', '396'],
                                                       ['297', '397'],
                                                       ['298', '398'],
                                                       ['299', '399'],
                                                       ['E', 'EE'],
                                                       ['201', '301'],
                                                       ['F', 'FF'],
                                                       ['203', '303'],
                                                       ['D', 'DD'],
                                                       ['205', '305'],
                                                       ['206', '306'],
                                                       ['207', '307'],
                                                       ['208', '308'],
                                                       ['209', '309'],
                                                       ['210', '310'],
                                                       ['211', '311'],
                                                       ['212', '312'],
                                                       ['213', '313'],
                                                       ['214', '314'],
                                                       ['215', '315'],
                                                       ['216', '316'],
                                                       ['217', '317'],
                                                       ['218', '318'],
                                                       ['219', '319'],
                                                       ['220', '320'],
                                                       ['221', '321'],
                                                       ['222', '322'],
                                                       ['223', '323'],
                                                       ['224', '324'],
                                                       ['225', '325'],
                                                       ['226', '326'],
                                                       ['227', '327'],
                                                       ['228', '328'],
                                                       ['229', '329'],
                                                       ['230', '330'],
                                                       ['231', '331'],
                                                       ['232', '332'],
                                                       ['233', '333'],
                                                       ['234', '334'],
                                                       ['235', '335'],
                                                       ['236', '336'],
                                                       ['237', '337'],
                                                       ['238', '338'],
                                                       ['239', '339'],
                                                       ['240', '340'],
                                                       ['241', '341'],
                                                       ['242', '342'],
                                                       ['243', '343'],
                                                       ['244', '344'],
                                                       ['245', '345'],
                                                       ['246', '346'],
                                                       ['247', '347'],
                                                       ['248', '348'],
                                                       ['249', '349']], dtype='<U3'))

    def test_set_single_value_broadcast(self):
        # set single value
        self.TDF[:, [4, 0, 2], ['col1', 'col2']] = 1000

        assert np.array_equal(self.TDF.values_num, np.array([[1000., 1000.],
                                                             [51., 151.],
                                                             [1000., 1000.],
                                                             [53., 153.],
                                                             [1000., 1000.],
                                                             [55., 155.],
                                                             [-4., 156.],
                                                             [57., 157.],
                                                             [-5., 158.],
                                                             [59., 159.],
                                                             [-6., 160.],
                                                             [61., 161.],
                                                             [-7., 162.],
                                                             [63., 163.],
                                                             [-8., 164.],
                                                             [65., 165.],
                                                             [-9., 166.],
                                                             [67., 167.],
                                                             [-10., 168.],
                                                             [69., 169.],
                                                             [70., 170.],
                                                             [71., 171.],
                                                             [72., 172.],
                                                             [73., 173.],
                                                             [74., 174.],
                                                             [75., 175.],
                                                             [76., 176.],
                                                             [77., 177.],
                                                             [78., 178.],
                                                             [79., 179.],
                                                             [80., 180.],
                                                             [81., 181.],
                                                             [82., 182.],
                                                             [83., 183.],
                                                             [84., 184.],
                                                             [85., 185.],
                                                             [86., 186.],
                                                             [87., 187.],
                                                             [88., 188.],
                                                             [89., 189.],
                                                             [90., 190.],
                                                             [91., 191.],
                                                             [92., 192.],
                                                             [93., 193.],
                                                             [94., 194.],
                                                             [95., 195.],
                                                             [96., 196.],
                                                             [97., 197.],
                                                             [98., 198.],
                                                             [99., 199.],
                                                             [1000., 1000.],
                                                             [1., 101.],
                                                             [1000., 1000.],
                                                             [3., 103.],
                                                             [1000., 1000.],
                                                             [5., 105.],
                                                             [6., 106.],
                                                             [7., 107.],
                                                             [8., 108.],
                                                             [9., 109.],
                                                             [10., 110.],
                                                             [11., 111.],
                                                             [12., 112.],
                                                             [13., 113.],
                                                             [14., 114.],
                                                             [15., 115.],
                                                             [16., 116.],
                                                             [17., 117.],
                                                             [18., 118.],
                                                             [19., 119.],
                                                             [20., 120.],
                                                             [21., 121.],
                                                             [22., 122.],
                                                             [23., 123.],
                                                             [24., 124.],
                                                             [25., 125.],
                                                             [26., 126.],
                                                             [27., 127.],
                                                             [28., 128.],
                                                             [29., 129.],
                                                             [30., 130.],
                                                             [31., 131.],
                                                             [32., 132.],
                                                             [33., 133.],
                                                             [34., 134.],
                                                             [35., 135.],
                                                             [36., 136.],
                                                             [37., 137.],
                                                             [38., 138.],
                                                             [39., 139.],
                                                             [40., 140.],
                                                             [41., 141.],
                                                             [42., 142.],
                                                             [43., 143.],
                                                             [44., 144.],
                                                             [45., 145.],
                                                             [46., 146.],
                                                             [47., 147.],
                                                             [48., 148.],
                                                             [49., 149.]]))
        assert np.all(self.TDF.values_str == np.array([['B', 'BB'],
                                                       ['251', '351'],
                                                       ['C', 'CC'],
                                                       ['253', '353'],
                                                       ['A', 'AA'],
                                                       ['255', '355'],
                                                       ['256', 'd'],
                                                       ['257', '357'],
                                                       ['258', 'e'],
                                                       ['259', '359'],
                                                       ['260', 'f'],
                                                       ['261', '361'],
                                                       ['262', 'g'],
                                                       ['263', '363'],
                                                       ['264', 'h'],
                                                       ['265', '365'],
                                                       ['266', 'i'],
                                                       ['267', '367'],
                                                       ['268', 'j'],
                                                       ['269', '369'],
                                                       ['270', '370'],
                                                       ['271', '371'],
                                                       ['272', '372'],
                                                       ['273', '373'],
                                                       ['274', '374'],
                                                       ['275', '375'],
                                                       ['276', '376'],
                                                       ['277', '377'],
                                                       ['278', '378'],
                                                       ['279', '379'],
                                                       ['280', '380'],
                                                       ['281', '381'],
                                                       ['282', '382'],
                                                       ['283', '383'],
                                                       ['284', '384'],
                                                       ['285', '385'],
                                                       ['286', '386'],
                                                       ['287', '387'],
                                                       ['288', '388'],
                                                       ['289', '389'],
                                                       ['290', '390'],
                                                       ['291', '391'],
                                                       ['292', '392'],
                                                       ['293', '393'],
                                                       ['294', '394'],
                                                       ['295', '395'],
                                                       ['296', '396'],
                                                       ['297', '397'],
                                                       ['298', '398'],
                                                       ['299', '399'],
                                                       ['E', 'EE'],
                                                       ['201', '301'],
                                                       ['F', 'FF'],
                                                       ['203', '303'],
                                                       ['D', 'DD'],
                                                       ['205', '305'],
                                                       ['206', '306'],
                                                       ['207', '307'],
                                                       ['208', '308'],
                                                       ['209', '309'],
                                                       ['210', '310'],
                                                       ['211', '311'],
                                                       ['212', '312'],
                                                       ['213', '313'],
                                                       ['214', '314'],
                                                       ['215', '315'],
                                                       ['216', '316'],
                                                       ['217', '317'],
                                                       ['218', '318'],
                                                       ['219', '319'],
                                                       ['220', '320'],
                                                       ['221', '321'],
                                                       ['222', '322'],
                                                       ['223', '323'],
                                                       ['224', '324'],
                                                       ['225', '325'],
                                                       ['226', '326'],
                                                       ['227', '327'],
                                                       ['228', '328'],
                                                       ['229', '329'],
                                                       ['230', '330'],
                                                       ['231', '331'],
                                                       ['232', '332'],
                                                       ['233', '333'],
                                                       ['234', '334'],
                                                       ['235', '335'],
                                                       ['236', '336'],
                                                       ['237', '337'],
                                                       ['238', '338'],
                                                       ['239', '339'],
                                                       ['240', '340'],
                                                       ['241', '341'],
                                                       ['242', '342'],
                                                       ['243', '343'],
                                                       ['244', '344'],
                                                       ['245', '345'],
                                                       ['246', '346'],
                                                       ['247', '347'],
                                                       ['248', '348'],
                                                       ['249', '349']], dtype='<U3'))


@pytest.mark.parametrize(
    'TDF1',
    ['plain', 'backed'],
    indirect=True
)
def test_reindex(TDF1):
    # all in index
    TDF1.reindex(np.arange(99, -1, -1))

    assert np.all(TDF1.index == np.arange(99, -1, -1))
    assert np.all(TDF1.values_num == np.vstack((np.arange(99, -1, -1),
                                                np.arange(199, 99, -1))).T)
    assert np.all(TDF1.values_str == np.vstack((np.arange(299, 199, -1),
                                                np.arange(399, 299, -1))).T.astype(str))


@pytest.mark.parametrize(
    'TDF1',
    ['plain', 'backed'],
    indirect=True
)
def test_reindex_with_indices_not_in_original_index_should_fail(TDF1):
    with pytest.raises(ValueError) as exc_info:
        TDF1.reindex(np.arange(149, 49, -1))

    assert str(exc_info.value) == "New index contains values which are not in the current index."


@pytest.mark.parametrize(
    'TDF1',
    ['plain', 'backed'],
    indirect=True
)
def test_reindex_with_repeating_index_should_fail(TDF1):
    with pytest.raises(ValueError) as exc_info:
        TDF1.reindex(np.concatenate((np.arange(0, 50), np.arange(0, 50))), repeating_index=True)

    assert str(exc_info.value == "Cannot set repeating index on tdf with non-repeating index.")


@pytest.mark.parametrize(
    'TDF1',
    ['plain', 'backed'],
    indirect=True
)
def test_reindex_with_repeating_index_on_tdf_with_repeating_index(TDF1):
    TDF1.set_index(np.concatenate((np.arange(0, 50), np.arange(0, 50))), repeating_index=True)

    TDF1.reindex(np.concatenate((np.arange(49, -1, -1), np.arange(49, -1, -1))), repeating_index=True)

    assert np.all(TDF1.index == np.concatenate((np.arange(49, -1, -1), np.arange(49, -1, -1))))
    assert np.all(TDF1.values_num == np.vstack((np.arange(99, -1, -1),
                                                np.arange(199, 99, -1))).T)
    assert np.all(TDF1.values_str == np.vstack((np.arange(299, 199, -1),
                                                np.arange(399, 299, -1))).T.astype(str))


# TODO
# @pytest.mark.parametrize('provide_TDFs', [(False, 'test_sub_getting_inverted_TDF', 1, 'r'),
#                                           (True, 'test_sub_getting_inverted_TDF', 3, 'r')],
#                          indirect=True)
# def test_reversed_sub_getting(provide_TDFs):
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
# def test_reversed_sub_setting(provide_TDFs):
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


@pytest.mark.parametrize(
    'TDF1',
    ['plain', 'view', 'backed', 'backed view'],
    indirect=True
)
@pytest.mark.parametrize(
    'operation,expected',
    [
        ('min', 0),
        ('max', 199),
        ('mean', 99.5)
    ]
)
def test_global_min_max_mean(TDF1, operation, expected):
    assert getattr(TDF1, operation)() == expected


@pytest.mark.parametrize(
    'TDF1',
    ['plain', 'view', 'backed', 'backed view'],
    indirect=True
)
@pytest.mark.parametrize(
    'operation,expected',
    [
        ('min', np.array([[50, 150], [0, 100]])),
        ('max', np.array([[99, 199], [49, 149]])),
        ('mean', np.array([[74.5, 174.5], [24.5, 124.5]]))
    ]
)
def test_min_max_mean_on_rows(TDF1, operation, expected):
    assert getattr(TDF1, operation)(axis=1) == TemporalDataFrame(
        data=pd.DataFrame(expected,
                          index=[operation, operation],
                          columns=['col1', 'col2']),
        repeating_index=True,
        time_list=TDF1.timepoints,
        time_col_name=TDF1.timepoints_column_name
    )


@pytest.mark.parametrize(
    'TDF1',
    ['plain', 'view', 'backed', 'backed view'],
    indirect=True
)
@pytest.mark.parametrize('operation', ['min', 'max', 'mean'])
def test_min_max_mean_on_columns(TDF1, operation):
    assert getattr(TDF1, operation)(axis=2) == TemporalDataFrame(data=pd.DataFrame(
        getattr(np, operation)(TDF1.values_num, axis=1)[:, None],
        index=TDF1.index,
        columns=[operation]
    ),
        time_list=TDF1.timepoints_column[:],
        time_col_name=TDF1.timepoints_column_name)


@pytest.mark.parametrize(
    'TDF1',
    ['plain', 'view', 'backed', 'backed view'],
    indirect=True
)
@pytest.mark.parametrize('operation', ['min', 'max', 'mean'])
def test_min_max_mean_on_timepoints(TDF1, operation):
    if TDF1.is_view:
        TDF1.parent.set_index(np.concatenate((np.arange(0, 50), np.arange(0, 50))), repeating_index=True)

    else:
        TDF1.set_index(np.concatenate((np.arange(0, 50), np.arange(0, 50))), repeating_index=True)

    mmm_tp = {'min': min,
              'max': max,
              'mean': tp_mean}[operation](TDF1.timepoints)

    assert getattr(TDF1, operation)(axis=0) == TemporalDataFrame(data=pd.DataFrame(
        getattr(np, operation)([TDF1['0h'].values_num,
                                TDF1['1h'].values_num], axis=0),
        index=TDF1.index_at(TDF1.tp0),
        columns=TDF1.columns_num[:]
    ),
        time_list=[mmm_tp for _ in enumerate(TDF1.index_at(TDF1.tp0))],
        time_col_name=TDF1.timepoints_column_name)
