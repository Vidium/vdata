# coding: utf-8
# Created on 31/03/2022 16:33
# Author : matteo

# ====================================================
# imports
import pytest
import numpy as np

from .utils import get_TDF


# ====================================================
# code
def test_sub_setting():
    TDF = get_TDF('1')

    # subset single TP
    assert repr(TDF['0h']) == "View of TemporalDataFrame '1'\n" \
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
    assert repr(TDF[['0h', '1h']]) == "View of TemporalDataFrame '1'\n" \
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
    assert repr(TDF[:, 10]) == "View of TemporalDataFrame '1'\n" \
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
    assert repr(TDF[:, range(25, 75)]) == "View of TemporalDataFrame '1'\n" \
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
    assert repr(TDF[:, [30, 10, 20, 80, 60, 70]]) == "View of TemporalDataFrame '1'\n"\
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
    assert repr(TDF[:, :, 'col3']) == "View of TemporalDataFrame '1'\n" \
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
    assert repr(TDF.col2) == "View of TemporalDataFrame '1'\n" \
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
    assert repr(TDF[:, :, ['col1', 'col3']]) == "View of TemporalDataFrame '1'\n" \
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
    assert repr(TDF[:, :, ['col4', 'col2', 'col1']]) == "View of TemporalDataFrame '1'\n" \
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
    assert repr(TDF['1h', 10:40:5, ['col1', 'col3']]) == "View of TemporalDataFrame '1'\n" \
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
    assert repr(TDF[['0h'], 10:70:5, ['col1', 'col3']]) == "View of TemporalDataFrame '1'\n" \
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
           "View of TemporalDataFrame '1'\n" \
           "\x1b[4mTime point : 0.0 hours\x1b[0m\n" \
           "   Time-point      col2  col1    col3\n" \
           "80       0.0h  |  180.0  80.0  |  280\n" \
           "60       0.0h  |  160.0  60.0  |  260\n" \
           "70       0.0h  |  170.0  70.0  |  270\n" \
           "50       0.0h  |  150.0  50.0  |  250\n" \
           "[4 x 3]\n\n"

    assert np.all(TDF[['0h'], [40, 10, 80, 60, 20, 70, 50], ['col2', 'col1', 'col3']].values_num == np.array([
        [180, 80],
        [160,  60],
        [170,  70],
        [150,  50]
    ]))
    assert np.all(TDF[['0h'], [40, 10, 80, 60, 20, 70, 50], ['col2', 'col1', 'col3']].values_str == np.array([
        ['280'],
        ['260'],
        ['270'],
        ['250']
    ]))

    # subset rows, same index at multiple time points
    TDF.index = np.concatenate((np.arange(0, 50), np.arange(0, 50)))

    assert repr(TDF[:, [4, 0, 2]]) == "View of TemporalDataFrame '1'\n" \
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


def test_view_sub_setting():
    TDF = get_TDF('2')
    view = TDF[:, range(10, 90), ['col1', 'col4']]

    # subset single TP
    assert repr(view['0h']) == "View of TemporalDataFrame '2'\n" \
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
    assert repr(view[['0h', '1h']]) == "View of TemporalDataFrame '2'\n" \
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
    assert repr(view[:, 10]) == "View of TemporalDataFrame '2'\n" \
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
    assert repr(view[:, range(25, 75)]) == "View of TemporalDataFrame '2'\n" \
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
    assert repr(view[:, [30, 10, 20, 80, 60, 70]]) == "View of TemporalDataFrame '2'\n" \
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
    assert repr(view[:, :, 'col4']) == "View of TemporalDataFrame '2'\n" \
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
    assert repr(view.col1) == "View of TemporalDataFrame '2'\n" \
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
    assert repr(view[:, :, ['col1', 'col4']]) == "View of TemporalDataFrame '2'\n" \
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
           "View of TemporalDataFrame '2'\n" \
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
    assert repr(view['1h', 20:40:5, ['col1']]) == "View of TemporalDataFrame '2'\n" \
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
    assert repr(view[['0h'], 20:70:5, ['col1', 'col4']]) == "View of TemporalDataFrame '2'\n" \
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
           "View of TemporalDataFrame '2'\n" \
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
    TDF.index = np.concatenate((np.arange(0, 50), np.arange(0, 50)))

    assert repr(TDF[:, :, ['col2', 'col1']][:, [4, 0, 2]]) == "View of TemporalDataFrame '2'\n" \
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


if __name__ == '__main__':
    test_sub_setting()
    test_view_sub_setting()
