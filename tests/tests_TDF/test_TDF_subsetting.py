# coding: utf-8
# Created on 31/03/2022 16:33
# Author : matteo

# ====================================================
# imports
import pytest
import numpy as np
from pathlib import Path

from .utils import get_TDF, get_backed_TDF, cleanup
from vdata.name_utils import H5Mode
from vdata.time_point import TimePoint


# ====================================================
# code
@pytest.fixture
def provide_TDFs(request):
    view, filename, name, mode = request.param

    # setup
    input_file = Path(__file__).parent / filename
    cleanup([input_file])

    TDF, backed_TDF = get_TDF(str(name)), get_backed_TDF(input_file, str(name + 1), mode)

    if view:
        yield TDF[:], backed_TDF[:]

    else:
        yield TDF, backed_TDF

    backed_TDF.file.close()

    # cleanup
    cleanup([input_file])


def test_sub_getting():
    TDF = get_TDF('1')

    # subset single TP
    assert repr(TDF['0h']) == "View of TemporalDataFrame 1\n" \
                              "\x1b[4mTime point : 0.0 hours\x1b[0m\n" \
                              "   Time-point     col1   col2    col3 col4\n" \
                              "50       0.0h  |  50.0  150.0  |  250  350\n" \
                              "51       0.0h  |  51.0  151.0  |  251  351\n" \
                              "52       0.0h  |  52.0  152.0  |  252  352\n" \
                              "53       0.0h  |  53.0  153.0  |  253  353\n" \
                              "54       0.0h  |  54.0  154.0  |  254  354\n" \
                              "[50 x 4]\n\n"

    assert np.all(TDF['0h'].values_num == np.hstack((np.arange(50, 100)[:, None], np.arange(150, 200)[:, None])))
    assert np.all(TDF['0h'].values_str == np.hstack((np.arange(250, 300).astype(str)[:, None],
                                                     np.arange(350, 400).astype(str)[:, None])))

    # subset single TP, not in TDF
    with pytest.raises(ValueError) as exc_info:
        repr(TDF['1s'])

    assert str(exc_info.value) == "Some time-points were not found in this TemporalDataFrame ([1.0 seconds] (1 value " \
                                  "long))"

    # subset multiple TPs
    assert repr(TDF[['0h', '1h']]) == "View of TemporalDataFrame 1\n" \
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

    assert np.all(TDF[['0h', '1h']].values_num == np.hstack((
        np.concatenate((np.arange(50, 100), np.arange(0, 50)))[:, None],
        np.concatenate((np.arange(150, 200), np.arange(100, 150)))[:, None])
    ))
    assert np.all(TDF[['0h', '1h']].values_str == np.hstack((
        np.concatenate((np.arange(250, 300), np.arange(200, 250))).astype(str)[:, None],
        np.concatenate((np.arange(350, 400), np.arange(300, 350))).astype(str)[:, None])
    ))

    # subset multiple TPs, some not in TDF
    with pytest.raises(ValueError) as exc_info:
        repr(TDF[['0h', '1h', '2h']])

    assert str(exc_info.value) == "Some time-points were not found in this TemporalDataFrame ([2.0 hours] (1 value " \
                                  "long))"

    # subset single row
    assert repr(TDF[:, 10]) == "View of TemporalDataFrame 1\n" \
                               "\x1b[4mTime point : 1.0 hours\x1b[0m\n" \
                               "   Time-point     col1   col2    col3 col4\n" \
                               "10       1.0h  |  10.0  110.0  |  210  310\n" \
                               "[1 x 4]\n\n"

    assert np.all(TDF[:, 10].values_num == np.array([[10, 110]]))
    assert np.all(TDF[:, 10].values_str == np.array([['210', '310']]))

    # subset single row, not in TDF
    with pytest.raises(ValueError) as exc_info:
        repr(TDF[:, 500])

    assert str(exc_info.value) == "Some indices were not found in this TemporalDataFrame ([500] (1 value long))"

    # subset multiple rows
    assert repr(TDF[:, range(25, 75)]) == "View of TemporalDataFrame 1\n" \
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

    assert np.all(TDF[:, range(25, 75)].values_num == np.hstack((
        np.concatenate((np.arange(50, 75), np.arange(25, 50)))[:, None],
        np.concatenate((np.arange(150, 175), np.arange(125, 150)))[:, None])
    ))
    assert np.all(TDF[:, range(25, 75)].values_str == np.hstack((
        np.concatenate((np.arange(250, 275), np.arange(225, 250))).astype(str)[:, None],
        np.concatenate((np.arange(350, 375), np.arange(325, 350))).astype(str)[:, None])
    ))

    # subset multiple rows, some not in TDF
    with pytest.raises(ValueError) as exc_info:
        repr(TDF[:, 20:500:2])

    assert str(exc_info.value) == "Some indices were not found in this TemporalDataFrame ([100 102 ... 496 498] (200 " \
                                  "values long))"

    # subset multiple rows, not in order
    assert repr(TDF[:, [30, 10, 20, 80, 60, 70]]) == "View of TemporalDataFrame 1\n" \
                                                     "\x1b[4mTime point : 0.0 hours\x1b[0m\n" \
                                                     "   Time-point     col1   col2    col3 col4\n" \
                                                     "80       0.0h  |  80.0  180.0  |  280  380\n" \
                                                     "60       0.0h  |  60.0  160.0  |  260  360\n" \
                                                     "70       0.0h  |  70.0  170.0  |  270  370\n" \
                                                     "[3 x 4]\n" \
                                                     "\n" \
                                                     "\x1b[4mTime point : 1.0 hours\x1b[0m\n" \
                                                     "   Time-point     col1   col2    col3 col4\n" \
                                                     "30       1.0h  |  30.0  130.0  |  230  330\n" \
                                                     "10       1.0h  |  10.0  110.0  |  210  310\n" \
                                                     "20       1.0h  |  20.0  120.0  |  220  320\n" \
                                                     "[3 x 4]\n\n"

    assert np.all(TDF[:, [30, 10, 20, 80, 60, 70]].values_num == np.array([[80, 180],
                                                                           [60, 160],
                                                                           [70, 170],
                                                                           [30, 130],
                                                                           [10, 110],
                                                                           [20, 120]]))
    assert np.all(TDF[:, [30, 10, 20, 80, 60, 70]].values_str == np.array([['280', '380'],
                                                                           ['260', '360'],
                                                                           ['270', '370'],
                                                                           ['230', '330'],
                                                                           ['210', '310'],
                                                                           ['220', '320']]))

    # subset single column
    #   getitem
    assert repr(TDF[:, :, 'col3']) == "View of TemporalDataFrame 1\n" \
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

    assert TDF[:, :, 'col3'].values_num.size == 0
    assert np.all(TDF[:, :, 'col3'].values_str ==
                  np.concatenate((np.arange(250, 300), np.arange(200, 250))).astype(str)[:, None])

    #   getattr
    assert repr(TDF.col2) == "View of TemporalDataFrame 1\n" \
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

    assert np.all(TDF.col2.values_num == np.concatenate((np.arange(150, 200), np.arange(100, 150)))[:, None])
    assert TDF.col2.values_str.size == 0

    # subset single column, not in TDF
    #   getitem
    with pytest.raises(ValueError) as exc_info:
        repr(TDF[:, :, 'col5'])

    assert str(exc_info.value) == "Some columns were not found in this TemporalDataFrame (['col5'] (1 value long))"

    #   getattr
    with pytest.raises(AttributeError) as exc_info:
        repr(TDF.col5)

    assert str(exc_info.value) == "'col5' not found in this TemporalDataFrame."

    # subset multiple columns
    assert repr(TDF[:, :, ['col1', 'col3']]) == "View of TemporalDataFrame 1\n" \
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

    assert np.all(TDF[:, :, ['col1', 'col3']].values_num == np.concatenate((
        np.arange(50, 100), np.arange(0, 50)))[:, None])
    assert np.all(TDF[:, :, ['col1', 'col3']].values_str == np.concatenate((
        np.arange(250, 300), np.arange(200, 250))).astype(str)[:, None])

    # subset multiple columns, some not in TDF
    with pytest.raises(ValueError) as exc_info:
        repr(TDF[:, :, ['col1', 'col3', 'col5']])

    assert str(exc_info.value) == "Some columns were not found in this TemporalDataFrame (['col5'] (1 value long))"

    # subset multiple columns, not in order
    assert repr(TDF[:, :, ['col4', 'col2', 'col1']]) == "View of TemporalDataFrame 1\n" \
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

    assert np.all(TDF[:, :, ['col4', 'col2', 'col1']].values_num == np.hstack((
        np.concatenate((np.arange(150, 200), np.arange(100, 150)))[:, None],
        np.concatenate((np.arange(50, 100), np.arange(0, 50)))[:, None]
    )))

    assert np.all(TDF[:, :, ['col4', 'col2', 'col1']].values_str == np.concatenate((
        np.arange(350, 400), np.arange(300, 350))).astype(str)[:, None])

    # subset TP, rows, columns
    assert repr(TDF['1h', 10:40:5, ['col1', 'col3']]) == "View of TemporalDataFrame 1\n" \
                                                         "\x1b[4mTime point : 1.0 hours\x1b[0m\n" \
                                                         "   Time-point     col1    col3\n" \
                                                         "10       1.0h  |  10.0  |  210\n" \
                                                         "15       1.0h  |  15.0  |  215\n" \
                                                         "20       1.0h  |  20.0  |  220\n" \
                                                         "25       1.0h  |  25.0  |  225\n" \
                                                         "30       1.0h  |  30.0  |  230\n" \
                                                         "[6 x 2]\n\n"

    assert np.all(TDF['1h', 10:40:5, ['col1', 'col3']].values_num == np.array([10, 15, 20, 25, 30, 35])[:, None])
    assert np.all(TDF['1h', 10:40:5, ['col1', 'col3']].values_str == np.array([210, 215, 220, 225, 230, 235
                                                                               ]).astype(str)[:, None])

    # subset TPs, rows, columns
    assert repr(TDF[['0h'], 10:70:5, ['col1', 'col3']]) == "View of TemporalDataFrame 1\n" \
                                                           "\x1b[4mTime point : 0.0 hours\x1b[0m\n" \
                                                           "   Time-point     col1    col3\n" \
                                                           "50       0.0h  |  50.0  |  250\n" \
                                                           "55       0.0h  |  55.0  |  255\n" \
                                                           "60       0.0h  |  60.0  |  260\n" \
                                                           "65       0.0h  |  65.0  |  265\n" \
                                                           "[4 x 2]\n\n"

    assert np.all(TDF[['0h'], 10:70:5, ['col1', 'col3']].values_num == np.array([50, 55, 60, 65])[:, None])
    assert np.all(TDF[['0h'], 10:70:5, ['col1', 'col3']].values_str == np.array([250, 255, 260, 265]).astype(
        str)[:, None])

    # subset TPs, rows, columns, some not in TDF
    with pytest.raises(ValueError) as exc_info:
        repr(TDF[['0h', '1h', '2h'], 10:70:5, ['col1', 'col3']])

    assert str(exc_info.value) == "Some time-points were not found in this TemporalDataFrame ([2.0 hours] (1 value " \
                                  "long))"

    with pytest.raises(ValueError) as exc_info:
        repr(TDF[['0h', '1h'], 10:200:50, ['col1', 'col3']])

    assert str(exc_info.value) == "Some indices were not found in this TemporalDataFrame ([110 160] (2 values long))"

    with pytest.raises(ValueError) as exc_info:
        repr(TDF[['0h', '1h'], 10:70:5, ['col1', 'col3', 'col5']])

    assert str(exc_info.value) == "Some columns were not found in this TemporalDataFrame (['col5'] (1 value long))"

    # subset TPs, rows, columns, not in order
    assert repr(TDF[['0h'], [40, 10, 80, 60, 20, 70, 50], ['col2', 'col1', 'col3']]) == \
           "View of TemporalDataFrame 1\n" \
           "\x1b[4mTime point : 0.0 hours\x1b[0m\n" \
           "   Time-point      col2  col1    col3\n" \
           "80       0.0h  |  180.0  80.0  |  280\n" \
           "60       0.0h  |  160.0  60.0  |  260\n" \
           "70       0.0h  |  170.0  70.0  |  270\n" \
           "50       0.0h  |  150.0  50.0  |  250\n" \
           "[4 x 3]\n\n"

    assert np.all(TDF[['0h'], [40, 10, 80, 60, 20, 70, 50], ['col2', 'col1', 'col3']].values_num == np.array([
        [180, 80],
        [160, 60],
        [170, 70],
        [150, 50]
    ]))
    assert np.all(TDF[['0h'], [40, 10, 80, 60, 20, 70, 50], ['col2', 'col1', 'col3']].values_str == np.array([
        ['280'],
        ['260'],
        ['270'],
        ['250']
    ]))

    # subset rows, same index at multiple time points
    TDF.set_index(np.concatenate((np.arange(0, 50), np.arange(0, 50))),
                  repeating_index=True)

    assert repr(TDF[:, [4, 0, 2]]) == "View of TemporalDataFrame 1\n" \
                                      "\x1b[4mTime point : 0.0 hours\x1b[0m\n" \
                                      "  Time-point     col1   col2    col3 col4\n" \
                                      "4       0.0h  |  54.0  154.0  |  254  354\n" \
                                      "0       0.0h  |  50.0  150.0  |  250  350\n" \
                                      "2       0.0h  |  52.0  152.0  |  252  352\n" \
                                      "[3 x 4]\n" \
                                      "\n" \
                                      "\x1b[4mTime point : 1.0 hours\x1b[0m\n" \
                                      "  Time-point    col1   col2    col3 col4\n" \
                                      "4       1.0h  |  4.0  104.0  |  204  304\n" \
                                      "0       1.0h  |  0.0  100.0  |  200  300\n" \
                                      "2       1.0h  |  2.0  102.0  |  202  302\n" \
                                      "[3 x 4]\n\n"

    assert np.all(TDF[:, [4, 0, 2]].values_num == np.array([[54, 154],
                                                            [50, 150],
                                                            [52, 152],
                                                            [4, 104],
                                                            [0, 100],
                                                            [2, 102]]))
    assert np.all(TDF[:, [4, 0, 2]].values_str == np.array([['254', '354'],
                                                            ['250', '350'],
                                                            ['252', '352'],
                                                            ['204', '304'],
                                                            ['200', '300'],
                                                            ['202', '302']]))


def test_sub_setting():
    # TDF is not backed
    TDF = get_TDF('1')

    # set values, wrong shape
    with pytest.raises(ValueError) as exc_info:
        TDF['0h', 10:70:2, ['col4', 'col1']] = np.ones((50, 50))

    assert str(exc_info.value) == "Can't set 10 x 2 values from 50 x 50 array."

    # set values for TPs, rows, columns
    TDF['0h', 10:70:2, ['col4', 'col1']] = np.array([['a', -1],
                                                     ['b', -2],
                                                     ['c', -3],
                                                     ['d', -4],
                                                     ['e', -5],
                                                     ['f', -6],
                                                     ['g', -7],
                                                     ['h', -8],
                                                     ['i', -9],
                                                     ['j', -10]])

    assert np.all(TDF.values_num == np.array([[-1., 150.],
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
    assert np.all(TDF.values_str == np.array([['250', 'a'],
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

    # set values for TPs, some not in TDF
    with pytest.raises(ValueError) as exc_info:
        TDF[['0h', '2h'], 10:70:2, ['col4', 'col1']] = np.array([['a', -1]])

    assert str(exc_info.value) == 'Some time-points were not found in this TemporalDataFrame ([2.0 hours] ' \
                                  '(1 value long))'

    # set values for rows, some not in TDF
    with pytest.raises(ValueError) as exc_info:
        TDF['0h', 0:200:20, ['col4', 'col1']] = np.array([['a', -1],
                                                          ['b', -2],
                                                          ['c', -3],
                                                          ['d', -4],
                                                          ['e', -5],
                                                          ['f', -6],
                                                          ['g', -7],
                                                          ['h', -8],
                                                          ['i', -9],
                                                          ['j', -10]])

    assert str(exc_info.value) == 'Some indices were not found in this TemporalDataFrame ([100 120 ... 160 180] ' \
                                  '(5 values long))'

    # set values for columns, some not in TDF
    with pytest.raises(ValueError) as exc_info:
        TDF['0h', 10:70:20, ['col4', 'col1', 'col5']] = np.array([['a', -1, 0],
                                                                  ['b', -2, 0],
                                                                  ['c', -3, 0],
                                                                  ['d', -4, 0],
                                                                  ['e', -5, 0],
                                                                  ['f', -6, 0],
                                                                  ['g', -7, 0],
                                                                  ['h', -8, 0],
                                                                  ['i', -9, 0],
                                                                  ['j', -10, 0]])

    assert str(exc_info.value) == "Some columns were not found in this TemporalDataFrame (['col5'] (1 value long))"

    # set values in TDF with same index at multiple time-points
    TDF.set_index(np.concatenate((np.arange(0, 50), np.arange(0, 50))),
                  repeating_index=True)

    TDF[:, [4, 0, 2]] = np.array([[-100, -101, 'A', 'AA'],
                                  [-200, -201, 'B', 'BB'],
                                  [-300, -301, 'C', 'CC'],
                                  [-400, -401, 'D', 'DD'],
                                  [-500, -501, 'E', 'EE'],
                                  [-600, -601, 'F', 'FF']])

    assert np.all(TDF.values_num == np.array([[-200., -201.],
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
    assert np.all(TDF.values_str == np.array([['B', 'BB'],
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

    # set single value
    TDF[:, [4, 0, 2], ['col1', 'col2']] = 1000

    assert np.array_equal(TDF.values_num, np.array([[1000., 1000.],
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
    assert np.all(TDF.values_str == np.array([['B', 'BB'],
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

    # TDF is backed -----------------------------------------------------------
    input_file = Path(__file__).parent / 'test_subset_TDF'
    cleanup([input_file])

    TDF = get_backed_TDF(input_file, '2', mode=H5Mode.READ_WRITE)

    # set values for TPs, rows, columns
    TDF['0h', 10:50:2, ['col3', 'col1']] = np.array([['a', -1],
                                                     ['b', -2],
                                                     ['c', -3],
                                                     ['d', -4],
                                                     ['e', -5],
                                                     ['f', -6],
                                                     ['g', -7],
                                                     ['h', -8]])

    assert np.all(TDF.values_num == np.array([[0., 1.],
                                              [2., 3.],
                                              [4., 5.],
                                              [6., 7.],
                                              [8., 9.],
                                              [10., 11.],
                                              [12., 13.],
                                              [14., 15.],
                                              [16., 17.],
                                              [18., 19.],
                                              [-1., 21.],
                                              [22., 23.],
                                              [-2., 25.],
                                              [26., 27.],
                                              [-3., 29.],
                                              [30., 31.],
                                              [-4., 33.],
                                              [34., 35.],
                                              [-5., 37.],
                                              [38., 39.],
                                              [-6., 41.],
                                              [42., 43.],
                                              [-7., 45.],
                                              [46., 47.],
                                              [-8., 49.],
                                              [50., 51.],
                                              [52., 53.],
                                              [54., 55.],
                                              [56., 57.],
                                              [58., 59.],
                                              [60., 61.],
                                              [62., 63.],
                                              [64., 65.],
                                              [66., 67.],
                                              [68., 69.],
                                              [70., 71.],
                                              [72., 73.],
                                              [74., 75.],
                                              [76., 77.],
                                              [78., 79.],
                                              [80., 81.],
                                              [82., 83.],
                                              [84., 85.],
                                              [86., 87.],
                                              [88., 89.],
                                              [90., 91.],
                                              [92., 93.],
                                              [94., 95.],
                                              [96., 97.],
                                              [98., 99.]]))
    assert np.all(TDF.values_str == np.array([['100'],
                                              ['101'],
                                              ['102'],
                                              ['103'],
                                              ['104'],
                                              ['105'],
                                              ['106'],
                                              ['107'],
                                              ['108'],
                                              ['109'],
                                              ['a'],
                                              ['111'],
                                              ['b'],
                                              ['113'],
                                              ['c'],
                                              ['115'],
                                              ['d'],
                                              ['117'],
                                              ['e'],
                                              ['119'],
                                              ['f'],
                                              ['121'],
                                              ['g'],
                                              ['123'],
                                              ['h'],
                                              ['125'],
                                              ['126'],
                                              ['127'],
                                              ['128'],
                                              ['129'],
                                              ['130'],
                                              ['131'],
                                              ['132'],
                                              ['133'],
                                              ['134'],
                                              ['135'],
                                              ['136'],
                                              ['137'],
                                              ['138'],
                                              ['139'],
                                              ['140'],
                                              ['141'],
                                              ['142'],
                                              ['143'],
                                              ['144'],
                                              ['145'],
                                              ['146'],
                                              ['147'],
                                              ['148'],
                                              ['149']], dtype='<U3'))

    # set values for TPs, some not in TDF
    with pytest.raises(ValueError) as exc_info:
        TDF[['0h', '2h'], 10:50:2, ['col3', 'col1']] = np.array([['a', -1],
                                                                 ['b', -2],
                                                                 ['c', -3],
                                                                 ['d', -4],
                                                                 ['e', -5],
                                                                 ['f', -6],
                                                                 ['g', -7],
                                                                 ['h', -8]])

    assert str(exc_info.value) == "Some time-points were not found in this TemporalDataFrame ([2.0 hours] " \
                                  "(1 value long))"

    # set values for rows, some not in TDF
    with pytest.raises(ValueError) as exc_info:
        TDF['0h', 10:150:2, ['col3', 'col1']] = np.array([['a', -1],
                                                          ['b', -2],
                                                          ['c', -3],
                                                          ['d', -4],
                                                          ['e', -5],
                                                          ['f', -6],
                                                          ['g', -7],
                                                          ['h', -8]])

    assert str(exc_info.value) == "Some indices were not found in this TemporalDataFrame ([50 52 ... 146 148] " \
                                  "(50 values long))"

    # set values for columns, some not in TDF
    with pytest.raises(ValueError) as exc_info:
        TDF['0h', 10:50:2, ['col3', 'col1', 'col5']] = np.array([['a', -1],
                                                                 ['b', -2],
                                                                 ['c', -3],
                                                                 ['d', -4],
                                                                 ['e', -5],
                                                                 ['f', -6],
                                                                 ['g', -7],
                                                                 ['h', -8]])

    assert str(exc_info.value) == "Some columns were not found in this TemporalDataFrame (['col5'] " \
                                  "(1 value long))"

    # set values in TDF with same index at multiple time-points
    TDF.set_index(np.concatenate((np.arange(0, 25), np.arange(0, 25))),
                  repeating_index=True)

    TDF[:, [4, 0, 2]] = np.array([[-100, -101, 'A'],
                                  [-200, -201, 'B'],
                                  [-300, -301, 'C'],
                                  [-400, -401, 'D'],
                                  [-500, -501, 'E'],
                                  [-600, -601, 'F']])

    assert np.all(TDF.values_num == np.array([[-200., -201.],
                                              [2., 3.],
                                              [-300., -301.],
                                              [6., 7.],
                                              [-100., -101.],
                                              [10., 11.],
                                              [12., 13.],
                                              [14., 15.],
                                              [16., 17.],
                                              [18., 19.],
                                              [-1., 21.],
                                              [22., 23.],
                                              [-2., 25.],
                                              [26., 27.],
                                              [-3., 29.],
                                              [30., 31.],
                                              [-4., 33.],
                                              [34., 35.],
                                              [-5., 37.],
                                              [38., 39.],
                                              [-6., 41.],
                                              [42., 43.],
                                              [-7., 45.],
                                              [46., 47.],
                                              [-8., 49.],
                                              [-500., -501.],
                                              [52., 53.],
                                              [-600., -601.],
                                              [56., 57.],
                                              [-400., -401.],
                                              [60., 61.],
                                              [62., 63.],
                                              [64., 65.],
                                              [66., 67.],
                                              [68., 69.],
                                              [70., 71.],
                                              [72., 73.],
                                              [74., 75.],
                                              [76., 77.],
                                              [78., 79.],
                                              [80., 81.],
                                              [82., 83.],
                                              [84., 85.],
                                              [86., 87.],
                                              [88., 89.],
                                              [90., 91.],
                                              [92., 93.],
                                              [94., 95.],
                                              [96., 97.],
                                              [98., 99.]]))
    assert np.all(TDF.values_str == np.array([['B'],
                                              ['101'],
                                              ['C'],
                                              ['103'],
                                              ['A'],
                                              ['105'],
                                              ['106'],
                                              ['107'],
                                              ['108'],
                                              ['109'],
                                              ['a'],
                                              ['111'],
                                              ['b'],
                                              ['113'],
                                              ['c'],
                                              ['115'],
                                              ['d'],
                                              ['117'],
                                              ['e'],
                                              ['119'],
                                              ['f'],
                                              ['121'],
                                              ['g'],
                                              ['123'],
                                              ['h'],
                                              ['E'],
                                              ['126'],
                                              ['F'],
                                              ['128'],
                                              ['D'],
                                              ['130'],
                                              ['131'],
                                              ['132'],
                                              ['133'],
                                              ['134'],
                                              ['135'],
                                              ['136'],
                                              ['137'],
                                              ['138'],
                                              ['139'],
                                              ['140'],
                                              ['141'],
                                              ['142'],
                                              ['143'],
                                              ['144'],
                                              ['145'],
                                              ['146'],
                                              ['147'],
                                              ['148'],
                                              ['149']], dtype='<U3'))

    cleanup([input_file])

    # set for all indices
    TDF[:, :, ['col1']] = TDF[:, :, ['col2']].values_num

    assert np.all(TDF.values_num == np.array([[-201., -201.],
                                              [3., 3.],
                                              [-301., -301.],
                                              [7., 7.],
                                              [-101., -101.],
                                              [11., 11.],
                                              [13., 13.],
                                              [15., 15.],
                                              [17., 17.],
                                              [19., 19.],
                                              [21., 21.],
                                              [23., 23.],
                                              [25., 25.],
                                              [27., 27.],
                                              [29., 29.],
                                              [31., 31.],
                                              [33., 33.],
                                              [35., 35.],
                                              [37., 37.],
                                              [39., 39.],
                                              [41., 41.],
                                              [43., 43.],
                                              [45., 45.],
                                              [47., 47.],
                                              [49., 49.],
                                              [-501., -501.],
                                              [53., 53.],
                                              [-601., -601.],
                                              [57., 57.],
                                              [-401., -401.],
                                              [61., 61.],
                                              [63., 63.],
                                              [65., 65.],
                                              [67., 67.],
                                              [69., 69.],
                                              [71., 71.],
                                              [73., 73.],
                                              [75., 75.],
                                              [77., 77.],
                                              [79., 79.],
                                              [81., 81.],
                                              [83., 83.],
                                              [85., 85.],
                                              [87., 87.],
                                              [89., 89.],
                                              [91., 91.],
                                              [93., 93.],
                                              [95., 95.],
                                              [97., 97.],
                                              [99., 99.]]))
    assert np.all(TDF.values_str == np.array([['B'],
                                              ['101'],
                                              ['C'],
                                              ['103'],
                                              ['A'],
                                              ['105'],
                                              ['106'],
                                              ['107'],
                                              ['108'],
                                              ['109'],
                                              ['a'],
                                              ['111'],
                                              ['b'],
                                              ['113'],
                                              ['c'],
                                              ['115'],
                                              ['d'],
                                              ['117'],
                                              ['e'],
                                              ['119'],
                                              ['f'],
                                              ['121'],
                                              ['g'],
                                              ['123'],
                                              ['h'],
                                              ['E'],
                                              ['126'],
                                              ['F'],
                                              ['128'],
                                              ['D'],
                                              ['130'],
                                              ['131'],
                                              ['132'],
                                              ['133'],
                                              ['134'],
                                              ['135'],
                                              ['136'],
                                              ['137'],
                                              ['138'],
                                              ['139'],
                                              ['140'],
                                              ['141'],
                                              ['142'],
                                              ['143'],
                                              ['144'],
                                              ['145'],
                                              ['146'],
                                              ['147'],
                                              ['148'],
                                              ['149']], dtype='<U3'))


def test_view_sub_getting():
    TDF = get_TDF('2')
    view = TDF[:, range(10, 90), ['col1', 'col4']]

    # subset single TP
    assert repr(view['0h']) == "View of TemporalDataFrame 2\n" \
                               "\x1b[4mTime point : 0.0 hours\x1b[0m\n" \
                               "   Time-point     col1    col4\n" \
                               "50       0.0h  |  50.0  |  350\n" \
                               "51       0.0h  |  51.0  |  351\n" \
                               "52       0.0h  |  52.0  |  352\n" \
                               "53       0.0h  |  53.0  |  353\n" \
                               "54       0.0h  |  54.0  |  354\n" \
                               "[40 x 2]\n\n"

    assert np.all(view['0h'].values_num == np.arange(50, 90)[:, None])
    assert np.all(view['0h'].values_str == np.arange(350, 390).astype(str)[:, None])

    # subset single TP, not in view
    with pytest.raises(ValueError) as exc_info:
        repr(view['1s'])

    assert str(exc_info.value) == "Some time-points were not found in this ViewTemporalDataFrame ([1.0 seconds] (1 " \
                                  "value long))"

    # subset multiple TPs
    assert repr(view[['0h', '1h']]) == "View of TemporalDataFrame 2\n" \
                                       "\x1b[4mTime point : 0.0 hours\x1b[0m\n" \
                                       "   Time-point     col1    col4\n" \
                                       "50       0.0h  |  50.0  |  350\n" \
                                       "51       0.0h  |  51.0  |  351\n" \
                                       "52       0.0h  |  52.0  |  352\n" \
                                       "53       0.0h  |  53.0  |  353\n" \
                                       "54       0.0h  |  54.0  |  354\n" \
                                       "[40 x 2]\n" \
                                       "\n" \
                                       "\x1b[4mTime point : 1.0 hours\x1b[0m\n" \
                                       "   Time-point     col1    col4\n" \
                                       "10       1.0h  |  10.0  |  310\n" \
                                       "11       1.0h  |  11.0  |  311\n" \
                                       "12       1.0h  |  12.0  |  312\n" \
                                       "13       1.0h  |  13.0  |  313\n" \
                                       "14       1.0h  |  14.0  |  314\n" \
                                       "[40 x 2]\n\n"

    assert np.all(view[['0h', '1h']].values_num == np.concatenate((np.arange(50, 90), np.arange(10, 50)))[:, None])
    assert np.all(view[['0h', '1h']].values_str == np.concatenate((np.arange(350, 390),
                                                                   np.arange(310, 350))).astype(str)[:, None])

    # subset multiple TPs, some not in view
    with pytest.raises(ValueError) as exc_info:
        repr(view[['0h', '1h', '2h']])

    assert str(exc_info.value) == "Some time-points were not found in this ViewTemporalDataFrame ([2.0 hours] (1 " \
                                  "value long))"

    # subset single row
    assert repr(view[:, 10]) == "View of TemporalDataFrame 2\n" \
                                "\x1b[4mTime point : 1.0 hours\x1b[0m\n" \
                                "   Time-point     col1    col4\n" \
                                "10       1.0h  |  10.0  |  310\n" \
                                "[1 x 2]\n\n"

    assert np.all(view[:, 10].values_num == np.array([[10]]))
    assert np.all(view[:, 10].values_str == np.array([['310']]))

    # subset single row, not in view
    with pytest.raises(ValueError) as exc_info:
        repr(view[:, 500])

    assert str(exc_info.value) == "Some indices were not found in this ViewTemporalDataFrame ([500] (1 value long))"

    # subset multiple rows
    assert repr(view[:, range(25, 75)]) == "View of TemporalDataFrame 2\n" \
                                           "\x1b[4mTime point : 0.0 hours\x1b[0m\n" \
                                           "   Time-point     col1    col4\n" \
                                           "50       0.0h  |  50.0  |  350\n" \
                                           "51       0.0h  |  51.0  |  351\n" \
                                           "52       0.0h  |  52.0  |  352\n" \
                                           "53       0.0h  |  53.0  |  353\n" \
                                           "54       0.0h  |  54.0  |  354\n" \
                                           "[25 x 2]\n" \
                                           "\n" \
                                           "\x1b[4mTime point : 1.0 hours\x1b[0m\n" \
                                           "   Time-point     col1    col4\n" \
                                           "25       1.0h  |  25.0  |  325\n" \
                                           "26       1.0h  |  26.0  |  326\n" \
                                           "27       1.0h  |  27.0  |  327\n" \
                                           "28       1.0h  |  28.0  |  328\n" \
                                           "29       1.0h  |  29.0  |  329\n" \
                                           "[25 x 2]\n\n"

    assert np.all(view[:, range(25, 75)].values_num == np.concatenate((np.arange(50, 75), np.arange(25, 50)))[:, None])
    assert np.all(view[:, range(25, 75)].values_str == np.concatenate((np.arange(350, 375),
                                                                       np.arange(325, 350))).astype(str)[:, None])

    # subset multiple rows, some not in view
    with pytest.raises(ValueError) as exc_info:
        repr(view[:, 20:500:2])

    assert str(exc_info.value) == "Some indices were not found in this ViewTemporalDataFrame ([90 92 ... 496 498] (" \
                                  "205 values long))"

    # subset multiple rows, not in order
    assert repr(view[:, [30, 10, 20, 80, 60, 70]]) == "View of TemporalDataFrame 2\n" \
                                                      "\x1b[4mTime point : 0.0 hours\x1b[0m\n" \
                                                      "   Time-point     col1    col4\n" \
                                                      "80       0.0h  |  80.0  |  380\n" \
                                                      "60       0.0h  |  60.0  |  360\n" \
                                                      "70       0.0h  |  70.0  |  370\n" \
                                                      "[3 x 2]\n" \
                                                      "\n" \
                                                      "\x1b[4mTime point : 1.0 hours\x1b[0m\n" \
                                                      "   Time-point     col1    col4\n" \
                                                      "30       1.0h  |  30.0  |  330\n" \
                                                      "10       1.0h  |  10.0  |  310\n" \
                                                      "20       1.0h  |  20.0  |  320\n" \
                                                      "[3 x 2]\n\n"

    assert np.all(view[:, [30, 10, 20, 80, 60, 70]].values_num == np.array([[80],
                                                                            [60],
                                                                            [70],
                                                                            [30],
                                                                            [10],
                                                                            [20]]))
    assert np.all(view[:, [30, 10, 20, 80, 60, 70]].values_str == np.array([['380'],
                                                                            ['360'],
                                                                            ['370'],
                                                                            ['330'],
                                                                            ['310'],
                                                                            ['320']]))

    # subset single column
    #   getitem
    assert repr(view[:, :, 'col4']) == "View of TemporalDataFrame 2\n" \
                                       "\x1b[4mTime point : 0.0 hours\x1b[0m\n" \
                                       "   Time-point    col4\n" \
                                       "50       0.0h  |  350\n" \
                                       "51       0.0h  |  351\n" \
                                       "52       0.0h  |  352\n" \
                                       "53       0.0h  |  353\n" \
                                       "54       0.0h  |  354\n" \
                                       "[40 x 1]\n" \
                                       "\n" \
                                       "\x1b[4mTime point : 1.0 hours\x1b[0m\n" \
                                       "   Time-point    col4\n" \
                                       "10       1.0h  |  310\n" \
                                       "11       1.0h  |  311\n" \
                                       "12       1.0h  |  312\n" \
                                       "13       1.0h  |  313\n" \
                                       "14       1.0h  |  314\n" \
                                       "[40 x 1]\n\n"

    assert view[:, :, 'col4'].values_num.size == 0
    assert np.all(view[:, :, 'col4'].values_str ==
                  np.concatenate((np.arange(350, 390), np.arange(310, 350))).astype(str)[:, None])

    #   getattr
    assert repr(view.col1) == "View of TemporalDataFrame 2\n" \
                              "\x1b[4mTime point : 0.0 hours\x1b[0m\n" \
                              "   Time-point     col1\n" \
                              "50       0.0h  |  50.0\n" \
                              "51       0.0h  |  51.0\n" \
                              "52       0.0h  |  52.0\n" \
                              "53       0.0h  |  53.0\n" \
                              "54       0.0h  |  54.0\n" \
                              "[40 x 1]\n" \
                              "\n" \
                              "\x1b[4mTime point : 1.0 hours\x1b[0m\n" \
                              "   Time-point     col1\n" \
                              "10       1.0h  |  10.0\n" \
                              "11       1.0h  |  11.0\n" \
                              "12       1.0h  |  12.0\n" \
                              "13       1.0h  |  13.0\n" \
                              "14       1.0h  |  14.0\n" \
                              "[40 x 1]\n\n"

    assert np.all(view.col1.values_num == np.concatenate((np.arange(50, 90), np.arange(10, 50)))[:, None])
    assert view.col1.values_str.size == 0

    # subset single column, not in view
    #   getitem
    with pytest.raises(ValueError) as exc_info:
        repr(view[:, :, 'col5'])

    assert str(exc_info.value) == "Some columns were not found in this ViewTemporalDataFrame (['col5'] (1 value long))"

    #   getattr
    with pytest.raises(AttributeError) as exc_info:
        repr(view.col5)

    assert str(exc_info.value) == "'col5' not found in this view of a TemporalDataFrame."

    # subset multiple columns
    assert repr(view[:, :, ['col1', 'col4']]) == "View of TemporalDataFrame 2\n" \
                                                 "\x1b[4mTime point : 0.0 hours\x1b[0m\n" \
                                                 "   Time-point     col1    col4\n" \
                                                 "50       0.0h  |  50.0  |  350\n" \
                                                 "51       0.0h  |  51.0  |  351\n" \
                                                 "52       0.0h  |  52.0  |  352\n" \
                                                 "53       0.0h  |  53.0  |  353\n" \
                                                 "54       0.0h  |  54.0  |  354\n" \
                                                 "[40 x 2]\n" \
                                                 "\n" \
                                                 "\x1b[4mTime point : 1.0 hours\x1b[0m\n" \
                                                 "   Time-point     col1    col4\n" \
                                                 "10       1.0h  |  10.0  |  310\n" \
                                                 "11       1.0h  |  11.0  |  311\n" \
                                                 "12       1.0h  |  12.0  |  312\n" \
                                                 "13       1.0h  |  13.0  |  313\n" \
                                                 "14       1.0h  |  14.0  |  314\n" \
                                                 "[40 x 2]\n\n"

    assert np.all(view[:, :, ['col1', 'col4']].values_num == np.concatenate((np.arange(50, 90),
                                                                             np.arange(10, 50)))[:, None])
    assert np.all(view[:, :, ['col1', 'col4']].values_str == np.concatenate((np.arange(350, 390),
                                                                             np.arange(310, 350))).astype(str)[:, None])

    # subset multiple columns, some not in view
    with pytest.raises(ValueError) as exc_info:
        repr(view[:, :, ['col1', 'col3', 'col5']])

    assert str(exc_info.value) == "Some columns were not found in this ViewTemporalDataFrame (['col3' 'col5'] " \
                                  "(2 values long))"

    # subset multiple columns, not in order
    assert repr(TDF[:, range(10, 90), ['col1', 'col2', 'col4']][:, :, ['col4', 'col2', 'col1']]) == \
           "View of TemporalDataFrame 2\n" \
           "\x1b[4mTime point : 0.0 hours\x1b[0m\n" \
           "   Time-point      col2  col1    col4\n" \
           "50       0.0h  |  150.0  50.0  |  350\n" \
           "51       0.0h  |  151.0  51.0  |  351\n" \
           "52       0.0h  |  152.0  52.0  |  352\n" \
           "53       0.0h  |  153.0  53.0  |  353\n" \
           "54       0.0h  |  154.0  54.0  |  354\n" \
           "[40 x 3]\n" \
           "\n" \
           "\x1b[4mTime point : 1.0 hours\x1b[0m\n" \
           "   Time-point      col2  col1    col4\n" \
           "10       1.0h  |  110.0  10.0  |  310\n" \
           "11       1.0h  |  111.0  11.0  |  311\n" \
           "12       1.0h  |  112.0  12.0  |  312\n" \
           "13       1.0h  |  113.0  13.0  |  313\n" \
           "14       1.0h  |  114.0  14.0  |  314\n" \
           "[40 x 3]\n\n"

    assert np.all(TDF[:, range(10, 90), ['col1', 'col2', 'col4']][:, :, ['col4', 'col2', 'col1']].values_num ==
                  np.hstack((
                      np.concatenate((np.arange(150, 190), np.arange(110, 150)))[:, None],
                      np.concatenate((np.arange(50, 90), np.arange(10, 50)))[:, None]
                  )))

    assert np.all(TDF[:, range(10, 90), ['col1', 'col2', 'col4']][:, :, ['col4', 'col2', 'col1']].values_str ==
                  np.concatenate((np.arange(350, 390), np.arange(310, 350))).astype(str)[:, None])

    # subset TP, rows, columns
    assert repr(view['1h', 20:40:5, ['col1']]) == "View of TemporalDataFrame 2\n" \
                                                  "\x1b[4mTime point : 1.0 hours\x1b[0m\n" \
                                                  "   Time-point     col1\n" \
                                                  "20       1.0h  |  20.0\n" \
                                                  "25       1.0h  |  25.0\n" \
                                                  "30       1.0h  |  30.0\n" \
                                                  "35       1.0h  |  35.0\n" \
                                                  "[4 x 1]\n\n"

    assert np.all(view['1h', 20:40:5, ['col1']].values_num == np.array([20, 25, 30, 35])[:, None])
    assert view['1h', 20:40:5, ['col1']].values_str.size == 0

    # subset TPs, rows, columns
    assert repr(view[['0h'], 20:70:5, ['col1', 'col4']]) == "View of TemporalDataFrame 2\n" \
                                                            "\x1b[4mTime point : 0.0 hours\x1b[0m\n" \
                                                            "   Time-point     col1    col4\n" \
                                                            "50       0.0h  |  50.0  |  350\n" \
                                                            "55       0.0h  |  55.0  |  355\n" \
                                                            "60       0.0h  |  60.0  |  360\n" \
                                                            "65       0.0h  |  65.0  |  365\n" \
                                                            "[4 x 2]\n\n"

    assert np.all(view[['0h'], 10:70:5, ['col1', 'col4']].values_num == np.array([50, 55, 60, 65])[:, None])
    assert np.all(view[['0h'], 10:70:5, ['col1', 'col4']].values_str == np.array([350, 355, 360, 365]).astype(
        str)[:, None])

    # subset TPs, rows, columns, some not in view
    with pytest.raises(ValueError) as exc_info:
        repr(view[['0h', '1h', '2h'], 20:70:5, ['col1', 'col4']])

    assert str(exc_info.value) == "Some time-points were not found in this ViewTemporalDataFrame ([2.0 hours] (1 " \
                                  "value long))"

    with pytest.raises(ValueError) as exc_info:
        repr(view[['0h', '1h'], 20:200:50, ['col1', 'col4']])

    assert str(exc_info.value) == "Some indices were not found in this ViewTemporalDataFrame ([120 170] (2 values " \
                                  "long))"

    with pytest.raises(ValueError) as exc_info:
        repr(view[['0h', '1h'], 20:70:5, ['col1', 'col4', 'col5']])

    assert str(exc_info.value) == "Some columns were not found in this ViewTemporalDataFrame (['col5'] (1 value long))"

    # subset TPs, rows, columns, not in order
    assert repr(view[['0h'], [40, 10, 80, 60, 20, 70, 50], ['col1', 'col4']]) == \
           "View of TemporalDataFrame 2\n" \
           "\x1b[4mTime point : 0.0 hours\x1b[0m\n" \
           "   Time-point     col1    col4\n" \
           "80       0.0h  |  80.0  |  380\n" \
           "60       0.0h  |  60.0  |  360\n" \
           "70       0.0h  |  70.0  |  370\n" \
           "50       0.0h  |  50.0  |  350\n" \
           "[4 x 2]\n\n"

    assert np.all(view[['0h'], [40, 10, 80, 60, 20, 70, 50], ['col1', 'col4']].values_num == np.array([
        [80],
        [60],
        [70],
        [50]
    ]))
    assert np.all(view[['0h'], [40, 10, 80, 60, 20, 70, 50], ['col1', 'col4']].values_str == np.array([
        ['380'],
        ['360'],
        ['370'],
        ['350']
    ]))

    # subset rows, same index at multiple time points
    TDF.set_index(np.concatenate((np.arange(0, 50), np.arange(0, 50))),
                  repeating_index=True)

    assert repr(TDF[:, :, ['col2', 'col1']][:, [4, 0, 2]]) == "View of TemporalDataFrame 2\n" \
                                                              "\x1b[4mTime point : 0.0 hours\x1b[0m\n" \
                                                              "  Time-point      col2  col1\n" \
                                                              "4       0.0h  |  154.0  54.0\n" \
                                                              "0       0.0h  |  150.0  50.0\n" \
                                                              "2       0.0h  |  152.0  52.0\n" \
                                                              "[3 x 2]\n" \
                                                              "\n" \
                                                              "\x1b[4mTime point : 1.0 hours\x1b[0m\n" \
                                                              "  Time-point      col2 col1\n" \
                                                              "4       1.0h  |  104.0  4.0\n" \
                                                              "0       1.0h  |  100.0  0.0\n" \
                                                              "2       1.0h  |  102.0  2.0\n" \
                                                              "[3 x 2]\n\n"

    assert np.all(TDF[:, :, ['col2', 'col1']][:, [4, 0, 2]].values_num == np.array([[154, 54],
                                                                                    [150, 50],
                                                                                    [152, 52],
                                                                                    [104, 4],
                                                                                    [100, 0],
                                                                                    [102, 2]]))
    assert TDF[:, :, ['col2', 'col1']][:, [4, 0, 2]].values_str.size == 0


def test_view_sub_setting():
    # TDF is not backed
    TDF = get_TDF('1')

    view = TDF[:, range(10, 90), ['col1', 'col4']]

    # set values, wrong shape
    with pytest.raises(ValueError) as exc_info:
        view['0h', 10:70:2, ['col4', 'col1']] = np.ones((50, 50))

    assert str(exc_info.value) == "Can't set 10 x 2 values from 50 x 50 array."

    # set values for TPs, rows, columns
    view['0h', 60:70, ['col4', 'col1']] = np.array([['a', -1],
                                                    ['b', -2],
                                                    ['c', -3],
                                                    ['d', -4],
                                                    ['e', -5],
                                                    ['f', -6],
                                                    ['g', -7],
                                                    ['h', -8],
                                                    ['i', -9],
                                                    ['j', -10]])

    assert np.all(TDF.values_num == np.array([[50., 150.],
                                              [51., 151.],
                                              [52., 152.],
                                              [53., 153.],
                                              [54., 154.],
                                              [55., 155.],
                                              [56., 156.],
                                              [57., 157.],
                                              [58., 158.],
                                              [59., 159.],
                                              [-1., 160.],
                                              [-2., 161.],
                                              [-3., 162.],
                                              [-4., 163.],
                                              [-5., 164.],
                                              [-6., 165.],
                                              [-7., 166.],
                                              [-8., 167.],
                                              [-9., 168.],
                                              [-10., 169.],
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
    assert np.all(TDF.values_str == np.array([['250', '350'],
                                              ['251', '351'],
                                              ['252', '352'],
                                              ['253', '353'],
                                              ['254', '354'],
                                              ['255', '355'],
                                              ['256', '356'],
                                              ['257', '357'],
                                              ['258', '358'],
                                              ['259', '359'],
                                              ['260', 'a'],
                                              ['261', 'b'],
                                              ['262', 'c'],
                                              ['263', 'd'],
                                              ['264', 'e'],
                                              ['265', 'f'],
                                              ['266', 'g'],
                                              ['267', 'h'],
                                              ['268', 'i'],
                                              ['269', 'j'],
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

    # set values for TPs, some not in TDF
    with pytest.raises(ValueError) as exc_info:
        view[['0h', '2h'], 10:70:2, ['col4', 'col1']] = np.array([['a', -1]])

    assert str(exc_info.value) == 'Some time-points were not found in this ViewTemporalDataFrame ([2.0 hours] ' \
                                  '(1 value long))'

    # set values for rows, some not in TDF
    with pytest.raises(ValueError) as exc_info:
        view['0h', 0:70:10, ['col4', 'col1']] = np.array([['a', -1],
                                                          ['b', -2],
                                                          ['c', -3],
                                                          ['d', -4],
                                                          ['e', -5],
                                                          ['f', -6],
                                                          ['g', -7]])

    assert str(exc_info.value) == 'Some indices were not found in this ViewTemporalDataFrame ([0] ' \
                                  '(1 value long))'

    # set values for columns, some not in TDF
    with pytest.raises(ValueError) as exc_info:
        view['0h', 10:70:20, ['col4', 'col1', 'col5']] = np.array([['a', -1, 0],
                                                                   ['b', -2, 0],
                                                                   ['c', -3, 0],
                                                                   ['d', -4, 0],
                                                                   ['e', -5, 0],
                                                                   ['f', -6, 0],
                                                                   ['g', -7, 0],
                                                                   ['h', -8, 0],
                                                                   ['i', -9, 0],
                                                                   ['j', -10, 0]])

    assert str(exc_info.value) == "Some columns were not found in this ViewTemporalDataFrame (['col5'] (1 value long))"

    # set values in TDF with same index at multiple time-points
    TDF.set_index(np.concatenate((np.arange(0, 50), np.arange(0, 50))),
                  repeating_index=True)

    view[:, [14, 10, 12]] = np.array([[-100, 'A'],
                                      [-200, 'B'],
                                      [-300, 'C'],
                                      [-400, 'D'],
                                      [-500, 'E'],
                                      [-600, 'F']])

    assert np.all(TDF.values_num == np.array([[50., 150.],
                                              [51., 151.],
                                              [52., 152.],
                                              [53., 153.],
                                              [54., 154.],
                                              [55., 155.],
                                              [56., 156.],
                                              [57., 157.],
                                              [58., 158.],
                                              [59., 159.],
                                              [-200., 160.],
                                              [-2., 161.],
                                              [-300., 162.],
                                              [-4., 163.],
                                              [-100., 164.],
                                              [-6., 165.],
                                              [-7., 166.],
                                              [-8., 167.],
                                              [-9., 168.],
                                              [-10., 169.],
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
                                              [-500., 110.],
                                              [11., 111.],
                                              [-600., 112.],
                                              [13., 113.],
                                              [-400., 114.],
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
    assert np.all(TDF.values_str == np.array([['250', '350'],
                                              ['251', '351'],
                                              ['252', '352'],
                                              ['253', '353'],
                                              ['254', '354'],
                                              ['255', '355'],
                                              ['256', '356'],
                                              ['257', '357'],
                                              ['258', '358'],
                                              ['259', '359'],
                                              ['260', 'B'],
                                              ['261', 'b'],
                                              ['262', 'C'],
                                              ['263', 'd'],
                                              ['264', 'A'],
                                              ['265', 'f'],
                                              ['266', 'g'],
                                              ['267', 'h'],
                                              ['268', 'i'],
                                              ['269', 'j'],
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
                                              ['210', 'E'],
                                              ['211', '311'],
                                              ['212', 'F'],
                                              ['213', '313'],
                                              ['214', 'D'],
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

    # TDF is backed -----------------------------------------------------------
    input_file = Path(__file__).parent / 'test_subset_TDF'
    cleanup([input_file])

    TDF = get_backed_TDF(input_file, '2', mode=H5Mode.READ_WRITE)

    view = TDF['0h', 10:50:2, ['col3', 'col1']]

    # set values for TPs, rows, columns
    view[:, 10:24:4, ['col3', 'col1']] = np.array([['a', -1],
                                                   ['b', -2],
                                                   ['c', -3],
                                                   ['d', -4]])

    assert np.all(TDF.values_num == np.array([[0., 1.],
                                              [2., 3.],
                                              [4., 5.],
                                              [6., 7.],
                                              [8., 9.],
                                              [10., 11.],
                                              [12., 13.],
                                              [14., 15.],
                                              [16., 17.],
                                              [18., 19.],
                                              [-1., 21.],
                                              [22., 23.],
                                              [24., 25.],
                                              [26., 27.],
                                              [-2., 29.],
                                              [30., 31.],
                                              [32., 33.],
                                              [34., 35.],
                                              [-3., 37.],
                                              [38., 39.],
                                              [40., 41.],
                                              [42., 43.],
                                              [-4., 45.],
                                              [46., 47.],
                                              [48., 49.],
                                              [50., 51.],
                                              [52., 53.],
                                              [54., 55.],
                                              [56., 57.],
                                              [58., 59.],
                                              [60., 61.],
                                              [62., 63.],
                                              [64., 65.],
                                              [66., 67.],
                                              [68., 69.],
                                              [70., 71.],
                                              [72., 73.],
                                              [74., 75.],
                                              [76., 77.],
                                              [78., 79.],
                                              [80., 81.],
                                              [82., 83.],
                                              [84., 85.],
                                              [86., 87.],
                                              [88., 89.],
                                              [90., 91.],
                                              [92., 93.],
                                              [94., 95.],
                                              [96., 97.],
                                              [98., 99.]]))
    assert np.all(TDF.values_str == np.array([['100'],
                                              ['101'],
                                              ['102'],
                                              ['103'],
                                              ['104'],
                                              ['105'],
                                              ['106'],
                                              ['107'],
                                              ['108'],
                                              ['109'],
                                              ['a'],
                                              ['111'],
                                              ['112'],
                                              ['113'],
                                              ['b'],
                                              ['115'],
                                              ['116'],
                                              ['117'],
                                              ['c'],
                                              ['119'],
                                              ['120'],
                                              ['121'],
                                              ['d'],
                                              ['123'],
                                              ['124'],
                                              ['125'],
                                              ['126'],
                                              ['127'],
                                              ['128'],
                                              ['129'],
                                              ['130'],
                                              ['131'],
                                              ['132'],
                                              ['133'],
                                              ['134'],
                                              ['135'],
                                              ['136'],
                                              ['137'],
                                              ['138'],
                                              ['139'],
                                              ['140'],
                                              ['141'],
                                              ['142'],
                                              ['143'],
                                              ['144'],
                                              ['145'],
                                              ['146'],
                                              ['147'],
                                              ['148'],
                                              ['149']], dtype='<U3'))

    # set values for TPs, some not in TDF
    with pytest.raises(ValueError) as exc_info:
        view[['0h', '2h'], 10:24:4, ['col3', 'col1']] = np.array([['a', -1],
                                                                  ['b', -2],
                                                                  ['c', -3],
                                                                  ['d', -4]])

    assert str(exc_info.value) == "Some time-points were not found in this ViewTemporalDataFrame ([2.0 hours] " \
                                  "(1 value long))"

    # set values for rows, some not in TDF
    with pytest.raises(ValueError) as exc_info:
        view['0h', 0:50:10, ['col3', 'col1']] = np.array([['a', -1],
                                                          ['b', -2],
                                                          ['c', -3],
                                                          ['d', -4],
                                                          ['e', -5]])

    assert str(exc_info.value) == "Some indices were not found in this ViewTemporalDataFrame ([ 0 30 40] " \
                                  "(3 values long))"

    # set values for columns, some not in TDF
    with pytest.raises(ValueError) as exc_info:
        view['0h', 10:24:4, ['col3', 'col1', 'col5']] = np.array([['a', -1],
                                                                  ['b', -2],
                                                                  ['c', -3],
                                                                  ['d', -4]])

    assert str(exc_info.value) == "Some columns were not found in this ViewTemporalDataFrame (['col5'] " \
                                  "(1 value long))"

    # set values in TDF with same index at multiple time-points
    TDF.set_index(np.concatenate((np.arange(0, 25), np.arange(0, 25))),
                  repeating_index=True)

    view = TDF[:, 10:25:2, ['col3', 'col1']]

    view[:, [14, 10, 12]] = np.array([[-100, 'A'],
                                      [-200, 'B'],
                                      [-300, 'C'],
                                      [-400, 'D'],
                                      [-500, 'E'],
                                      [-600, 'F']])

    assert np.all(TDF.values_num == np.array([[0., 1.],
                                              [2., 3.],
                                              [4., 5.],
                                              [6., 7.],
                                              [8., 9.],
                                              [10., 11.],
                                              [12., 13.],
                                              [14., 15.],
                                              [16., 17.],
                                              [18., 19.],
                                              [-200., 21.],
                                              [22., 23.],
                                              [-300, 25.],
                                              [26., 27.],
                                              [-100., 29.],
                                              [30., 31.],
                                              [32., 33.],
                                              [34., 35.],
                                              [-3., 37.],
                                              [38., 39.],
                                              [40., 41.],
                                              [42., 43.],
                                              [-4., 45.],
                                              [46., 47.],
                                              [48., 49.],
                                              [50., 51.],
                                              [52., 53.],
                                              [54., 55.],
                                              [56., 57.],
                                              [58., 59.],
                                              [60., 61.],
                                              [62., 63.],
                                              [64., 65.],
                                              [66., 67.],
                                              [68., 69.],
                                              [-500, 71.],
                                              [72., 73.],
                                              [-600, 75.],
                                              [76., 77.],
                                              [-400, 79.],
                                              [80., 81.],
                                              [82., 83.],
                                              [84., 85.],
                                              [86., 87.],
                                              [88., 89.],
                                              [90., 91.],
                                              [92., 93.],
                                              [94., 95.],
                                              [96., 97.],
                                              [98., 99.]]))
    assert np.all(TDF.values_str == np.array([['100'],
                                              ['101'],
                                              ['102'],
                                              ['103'],
                                              ['104'],
                                              ['105'],
                                              ['106'],
                                              ['107'],
                                              ['108'],
                                              ['109'],
                                              ['B'],
                                              ['111'],
                                              ['C'],
                                              ['113'],
                                              ['A'],
                                              ['115'],
                                              ['116'],
                                              ['117'],
                                              ['c'],
                                              ['119'],
                                              ['120'],
                                              ['121'],
                                              ['d'],
                                              ['123'],
                                              ['124'],
                                              ['125'],
                                              ['126'],
                                              ['127'],
                                              ['128'],
                                              ['129'],
                                              ['130'],
                                              ['131'],
                                              ['132'],
                                              ['133'],
                                              ['134'],
                                              ['E'],
                                              ['136'],
                                              ['F'],
                                              ['138'],
                                              ['D'],
                                              ['140'],
                                              ['141'],
                                              ['142'],
                                              ['143'],
                                              ['144'],
                                              ['145'],
                                              ['146'],
                                              ['147'],
                                              ['148'],
                                              ['149']], dtype='<U3'))

    cleanup([input_file])


def test_reindex():
    # non repeating index
    TDF = get_TDF('1')

    # all in index
    TDF.reindex(np.arange(99, -1, -1))

    assert np.all(TDF.index == np.arange(99, -1, -1))
    assert np.all(TDF.values_num == np.vstack((np.arange(99, -1, -1),
                                               np.arange(199, 99, -1))).T)
    assert np.all(TDF.values_str == np.vstack((np.arange(299, 199, -1),
                                               np.arange(399, 299, -1))).T.astype(str))

    # some not in index
    TDF = get_TDF('2')

    with pytest.raises(ValueError) as exc_info:
        TDF.reindex(np.arange(149, 49, -1))

    assert str(exc_info.value) == "New index contains values which are not in the current index."

    # repeating index
    TDF = get_TDF('3')

    with pytest.raises(ValueError) as exc_info:
        TDF.reindex(np.concatenate((np.arange(0, 50), np.arange(0, 50))),
                    repeating_index=True)

    assert str(exc_info.value == "Cannot set repeating index on TDF with non-repeating index.")

    TDF.set_index(np.concatenate((np.arange(0, 50), np.arange(0, 50))),
                  repeating_index=True)

    TDF.reindex(np.concatenate((np.arange(49, -1, -1), np.arange(49, -1, -1))),
                repeating_index=True)

    assert np.all(TDF.index == np.concatenate((np.arange(49, -1, -1), np.arange(49, -1, -1))))
    assert np.all(TDF.values_num == np.vstack((np.arange(99, -1, -1),
                                               np.arange(199, 99, -1))).T)
    assert np.all(TDF.values_str == np.vstack((np.arange(299, 199, -1),
                                               np.arange(399, 299, -1))).T.astype(str))


@pytest.mark.parametrize('provide_TDFs', [(False, 'test_sub_getting_inverted_TDF', 1, 'r'),
                                          (True, 'test_sub_getting_inverted_TDF', 3, 'r')],
                         indirect=True)
def test_reversed_sub_getting(provide_TDFs):
    TDF, backed_TDF = provide_TDFs

    # TDF is not backed -------------------------------------------------------
    # subset only on time-points
    inverted_view = (~TDF)['0h']

    assert np.array_equal(inverted_view.timepoints, [TimePoint('1h')])
    assert np.array_equal(inverted_view.index, np.arange(50))
    assert np.array_equal(inverted_view.columns, TDF.columns)

    # subset only on columns
    inverted_view = (~TDF)[:, :, ['col3', 'col2']]

    assert np.array_equal(inverted_view.timepoints, TDF.timepoints)
    assert np.array_equal(inverted_view.index, TDF.index)
    assert np.array_equal(inverted_view.columns, ['col1', 'col4'])

    # TDF is backed -----------------------------------------------------------
    # subset only on time-points
    inverted_view = (~backed_TDF)['0h']

    assert np.array_equal(inverted_view.timepoints, [TimePoint('1h')])
    assert np.array_equal(inverted_view.index, np.arange(25, 50))
    assert np.array_equal(inverted_view.columns, backed_TDF.columns)

    # subset only on columns
    inverted_view = (~backed_TDF)[:, :, ['col3', 'col2']]

    assert np.array_equal(inverted_view.timepoints, backed_TDF.timepoints)
    assert np.array_equal(inverted_view.index, backed_TDF.index)
    assert np.array_equal(inverted_view.columns, ['col1'])


@pytest.mark.parametrize('provide_TDFs', [(False, 'test_sub_getting_inverted_TDF', 1, 'r+'),
                                          (True, 'test_sub_getting_inverted_TDF', 3, 'r+')],
                         indirect=True)
def test_reversed_sub_setting(provide_TDFs):
    TDF, backed_TDF = provide_TDFs

    # TDF is not backed -------------------------------------------------------
    # subset only on time-points
    (~TDF)['0h'] = np.concatenate((
        -1 * (~TDF)['0h'].values_num,
        -1 * (~TDF)['0h'].values_str.astype(int)
    ), axis=1)

    assert np.array_equal(TDF.values_num, np.vstack((
        np.concatenate((np.arange(50, 100), -1 * np.arange(50))),
        np.concatenate((np.arange(150, 200), -1 * np.arange(100, 150)))
    )).T)
    assert np.array_equal(TDF.values_str, np.vstack((
        np.concatenate((np.arange(250, 300).astype(str), (-1 * np.arange(200., 250.)).astype(str))),
        np.concatenate((np.arange(350, 400).astype(str), (-1 * np.arange(300., 350.)).astype(str)))
    )).T)

    # subset only on columns
    (~TDF)[:, :, ['col3', 'col4', 'col2']] = 2 * TDF.col1

    assert np.array_equal(TDF.values_num, np.vstack((
        np.concatenate((np.arange(100, 200, 2), -1 * np.arange(0, 100, 2))),
        np.concatenate((np.arange(150, 200), -1 * np.arange(100, 150)))
    )).T)
    assert np.array_equal(TDF.values_str, np.vstack((
        np.concatenate((np.arange(250, 300).astype(str), (-1 * np.arange(200., 250.)).astype(str))),
        np.concatenate((np.arange(350, 400).astype(str), (-1 * np.arange(300., 350.)).astype(str)))
    )).T)

    # TDF is backed -----------------------------------------------------------
    # subset only on time-points
    (~backed_TDF)['0h'] = -1 * (~backed_TDF)['0h']

    assert np.array_equal(backed_TDF.values_num, np.vstack((
        np.concatenate((np.arange(0, 50, 2), -1 * np.arange(50, 100, 2))),
        np.concatenate((np.arange(1, 51, 2), -1 * np.arange(51, 101, 2)))
    )).T)
    assert np.array_equal(backed_TDF.values_str, np.arange(100, 150).astype(str)[:, None])

    # subset only on columns
    (~backed_TDF)[:, :, ['col3', 'col2']] = np.arange(100, 150)

    assert np.array_equal(backed_TDF.values_num, np.vstack((
        np.arange(100, 150),
        np.concatenate((np.arange(1, 51, 2), -1 * np.arange(51, 101, 2)))
    )).T)
    assert np.array_equal(backed_TDF.values_str, np.arange(100, 150).astype(str)[:, None])


if __name__ == '__main__':
    test_sub_getting()
    test_sub_setting()
    test_view_sub_getting()
    test_view_sub_setting()
    test_reindex()
    test_reversed_sub_getting()
    test_reversed_sub_setting()
