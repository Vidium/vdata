# coding: utf-8
# Created on 04/04/2022 17:24
# Author : matteo

# ====================================================
# imports
import numpy as np
from pathlib import Path

from vdata.name_utils import H5Mode
from .utils import get_TDF, get_backed_TDF, cleanup


# ====================================================
# code
def test_convert():
    # TDF is not backed
    TDF = get_TDF('1')

    #   no time-points
    df = TDF.to_pandas()

    assert np.all(df.values[:, :2] == np.vstack((
        np.concatenate((np.arange(50, 100), np.arange(0, 50))),
        np.concatenate((np.arange(150, 200), np.arange(100, 150)))
    )).T)
    assert np.all(df.values[:, 2:] == np.vstack((
        np.concatenate((np.arange(250, 300).astype(str), np.arange(200, 250).astype(str))),
        np.concatenate((np.arange(350, 400).astype(str), np.arange(300, 350).astype(str)))
    )).T)

    #   with time-points
    df = TDF.to_pandas(with_timepoints='timepoints')

    assert np.all(df.timepoints == ['0.0h' for _ in range(50)] + ['1.0h' for _ in range(50)])
    assert np.all(df.values[:, 1:3] == np.vstack((
        np.concatenate((np.arange(50, 100), np.arange(0, 50))),
        np.concatenate((np.arange(150, 200), np.arange(100, 150)))
    )).T)
    assert np.all(df.values[:, 3:] == np.vstack((
        np.concatenate((np.arange(250, 300).astype(str), np.arange(200, 250).astype(str))),
        np.concatenate((np.arange(350, 400).astype(str), np.arange(300, 350).astype(str)))
    )).T)

    # TDF is backed
    input_file = Path(__file__).parent / 'test_convert_TDF'
    cleanup([input_file])

    TDF = get_backed_TDF(input_file, '2')

    #   no time-points
    df = TDF.to_pandas()

    assert np.all(df.values[:, :2] == np.array(range(100)).reshape((50, 2)))
    assert np.all(df.values[:, 2:] == np.array(list(map(str, range(100, 150))), dtype=np.dtype('O')).reshape((50, 1)))

    #   with time-points
    df = TDF.to_pandas(with_timepoints='timepoints')

    assert np.all(df.timepoints == ['0.0h' for _ in range(25)] + ['1.0h' for _ in range(25)])
    assert np.all(df.values[:, 1:3] == np.array(range(100)).reshape((50, 2)))
    assert np.all(df.values[:, 3:] == np.array(list(map(str, range(100, 150))), dtype=np.dtype('O')).reshape((50, 1)))

    cleanup([input_file])

    # -------------------------------------------------------------------------
    # TDF is a view
    # TDF is not backed
    TDF = get_TDF('3')

    view = TDF[:, range(10, 90), ['col1', 'col4']]

    #   no time-points
    df = view.to_pandas()

    assert np.all(df.values[:, 0] == np.concatenate((np.arange(50, 90), np.arange(10, 50))))
    assert np.all(df.values[:, 1] == np.concatenate((np.arange(350, 390).astype(str), np.arange(310, 350).astype(str))))

    #   with time-points
    df = view.to_pandas(with_timepoints='timepoints')

    assert np.all(df.timepoints == ['0.0h' for _ in range(40)] + ['1.0h' for _ in range(40)])
    assert np.all(df.values[:, 1] == np.concatenate((np.arange(50, 90), np.arange(10, 50))))
    assert np.all(df.values[:, 2] == np.concatenate((np.arange(350, 390).astype(str), np.arange(310, 350).astype(str))))

    # TDF is backed
    input_file = Path(__file__).parent / 'test_convert_TDF'
    cleanup([input_file])

    TDF = get_backed_TDF(input_file, '4')

    view = TDF[:, range(10, 40), ['col1', 'col3']]

    #   no time-points
    df = view.to_pandas()

    assert np.all(df.values[:, 0] == np.arange(20, 80, 2))
    assert np.all(df.values[:, 1] == np.arange(110, 140).astype(str))

    #   with time-points
    df = view.to_pandas(with_timepoints='timepoints')

    assert np.all(df.timepoints == ['0.0h' for _ in range(15)] + ['1.0h' for _ in range(15)])
    assert np.all(df.values[:, 1] == np.arange(20, 80, 2))
    assert np.all(df.values[:, 2] == np.arange(110, 140).astype(str))

    cleanup([input_file])


def test_copy():
    # TDF is not backed
    TDF = get_TDF('1')

    #   time-points column name is None
    TDF_copy = TDF.copy()

    assert repr(TDF_copy) == "TemporalDataFrame 'copy of 1'\n" \
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

    assert np.all(TDF_copy.values_num == TDF.values_num)
    assert np.all(TDF_copy.values_str == TDF.values_str)

    #   time-points column name is not None
    TDF.timepoints_column_name = 'Time'
    TDF_copy = TDF.copy()

    assert repr(TDF_copy) == "TemporalDataFrame 'copy of 1'\n" \
                             "\x1b[4mTime point : 0.0 hours\x1b[0m\n" \
                             "    Time     col1   col2    col3 col4\n" \
                             "50  0.0h  |  50.0  150.0  |  250  350\n" \
                             "51  0.0h  |  51.0  151.0  |  251  351\n" \
                             "52  0.0h  |  52.0  152.0  |  252  352\n" \
                             "53  0.0h  |  53.0  153.0  |  253  353\n" \
                             "54  0.0h  |  54.0  154.0  |  254  354\n" \
                             "[50 x 4]\n" \
                             "\n" \
                             "\x1b[4mTime point : 1.0 hours\x1b[0m\n" \
                             "   Time    col1   col2    col3 col4\n" \
                             "0  1.0h  |  0.0  100.0  |  200  300\n" \
                             "1  1.0h  |  1.0  101.0  |  201  301\n" \
                             "2  1.0h  |  2.0  102.0  |  202  302\n" \
                             "3  1.0h  |  3.0  103.0  |  203  303\n" \
                             "4  1.0h  |  4.0  104.0  |  204  304\n" \
                             "[50 x 4]\n\n"

    assert np.all(TDF_copy.values_num == TDF.values_num)
    assert np.all(TDF_copy.values_str == TDF.values_str)

    # TDF is backed
    input_file = Path(__file__).parent / 'test_copy_TDF'
    cleanup([input_file])

    TDF = get_backed_TDF(input_file, '2', mode=H5Mode.READ_WRITE)

    #   time-points column name is None
    TDF_copy = TDF.copy()

    assert repr(TDF_copy) == "TemporalDataFrame 'copy of 2'\n" \
                             "\x1b[4mTime point : 0.0 hours\x1b[0m\n" \
                             "  Time-point    col1 col2    col3\n" \
                             "0       0.0h  |  0.0  1.0  |  100\n" \
                             "1       0.0h  |  2.0  3.0  |  101\n" \
                             "2       0.0h  |  4.0  5.0  |  102\n" \
                             "3       0.0h  |  6.0  7.0  |  103\n" \
                             "4       0.0h  |  8.0  9.0  |  104\n" \
                             "[25 x 3]\n" \
                             "\n" \
                             "\x1b[4mTime point : 1.0 hours\x1b[0m\n" \
                             "   Time-point     col1  col2    col3\n" \
                             "25       1.0h  |  50.0  51.0  |  125\n" \
                             "26       1.0h  |  52.0  53.0  |  126\n" \
                             "27       1.0h  |  54.0  55.0  |  127\n" \
                             "28       1.0h  |  56.0  57.0  |  128\n" \
                             "29       1.0h  |  58.0  59.0  |  129\n" \
                             "[25 x 3]\n\n"

    assert np.all(TDF_copy.values_num == TDF.values_num)
    assert np.all(TDF_copy.values_str == TDF.values_str)

    #   time-points column name is not None
    TDF.timepoints_column_name = 'Time'
    TDF_copy = TDF.copy()

    assert repr(TDF_copy) == "TemporalDataFrame 'copy of 2'\n" \
                             "\x1b[4mTime point : 0.0 hours\x1b[0m\n" \
                             "   Time    col1 col2    col3\n" \
                             "0  0.0h  |  0.0  1.0  |  100\n" \
                             "1  0.0h  |  2.0  3.0  |  101\n" \
                             "2  0.0h  |  4.0  5.0  |  102\n" \
                             "3  0.0h  |  6.0  7.0  |  103\n" \
                             "4  0.0h  |  8.0  9.0  |  104\n" \
                             "[25 x 3]\n" \
                             "\n" \
                             "\x1b[4mTime point : 1.0 hours\x1b[0m\n" \
                             "    Time     col1  col2    col3\n" \
                             "25  1.0h  |  50.0  51.0  |  125\n" \
                             "26  1.0h  |  52.0  53.0  |  126\n" \
                             "27  1.0h  |  54.0  55.0  |  127\n" \
                             "28  1.0h  |  56.0  57.0  |  128\n" \
                             "29  1.0h  |  58.0  59.0  |  129\n" \
                             "[25 x 3]\n\n"

    assert np.all(TDF_copy.values_num == TDF.values_num)
    assert np.all(TDF_copy.values_str == TDF.values_str)

    cleanup([input_file])

    # -------------------------------------------------------------------------
    # TDF is a view
    # TDF is not backed
    TDF = get_TDF('3')

    view = TDF[:, range(10, 90), ['col1', 'col4']]

    #   time-points column name is None
    view_copy = view.copy()

    assert repr(view_copy) == "TemporalDataFrame 'copy of view of 3'\n" \
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

    assert np.all(view_copy.values_num == view.values_num)
    assert np.all(view_copy.values_str == view.values_str)

    #   time-points column name is not None
    TDF.timepoints_column_name = 'Time'

    view = TDF[:, range(10, 90), ['col1', 'col4']]

    view_copy = view.copy()

    assert repr(view_copy) == "TemporalDataFrame 'copy of view of 3'\n" \
                              "\x1b[4mTime point : 0.0 hours\x1b[0m\n" \
                              "    Time     col1    col4\n" \
                              "50  0.0h  |  50.0  |  350\n" \
                              "51  0.0h  |  51.0  |  351\n" \
                              "52  0.0h  |  52.0  |  352\n" \
                              "53  0.0h  |  53.0  |  353\n" \
                              "54  0.0h  |  54.0  |  354\n" \
                              "[40 x 2]\n" \
                              "\n" \
                              "\x1b[4mTime point : 1.0 hours\x1b[0m\n" \
                              "    Time     col1    col4\n" \
                              "10  1.0h  |  10.0  |  310\n" \
                              "11  1.0h  |  11.0  |  311\n" \
                              "12  1.0h  |  12.0  |  312\n" \
                              "13  1.0h  |  13.0  |  313\n" \
                              "14  1.0h  |  14.0  |  314\n" \
                              "[40 x 2]\n\n"

    assert np.all(view_copy.values_num == view.values_num)
    assert np.all(view_copy.values_str == view.values_str)

    # TDF is backed
    input_file = Path(__file__).parent / 'test_copy_TDF'
    cleanup([input_file])

    TDF = get_backed_TDF(input_file, '4', mode=H5Mode.READ_WRITE)

    view = TDF[:, range(10, 40), ['col1', 'col3']]

    #   time-points column name is None
    view_copy = view.copy()

    assert repr(view_copy) == "TemporalDataFrame 'copy of view of 4'\n" \
                              "\x1b[4mTime point : 0.0 hours\x1b[0m\n" \
                              "   Time-point     col1    col3\n" \
                              "10       0.0h  |  20.0  |  110\n" \
                              "11       0.0h  |  22.0  |  111\n" \
                              "12       0.0h  |  24.0  |  112\n" \
                              "13       0.0h  |  26.0  |  113\n" \
                              "14       0.0h  |  28.0  |  114\n" \
                              "[15 x 2]\n" \
                              "\n" \
                              "\x1b[4mTime point : 1.0 hours\x1b[0m\n" \
                              "   Time-point     col1    col3\n" \
                              "25       1.0h  |  50.0  |  125\n" \
                              "26       1.0h  |  52.0  |  126\n" \
                              "27       1.0h  |  54.0  |  127\n" \
                              "28       1.0h  |  56.0  |  128\n" \
                              "29       1.0h  |  58.0  |  129\n" \
                              "[15 x 2]\n\n"

    assert np.all(view_copy.values_num == view.values_num)
    assert np.all(view_copy.values_str == view.values_str)

    #   time-points column name is not None
    TDF.timepoints_column_name = 'Time'

    view = TDF[:, range(10, 40), ['col1', 'col3']]

    view_copy = view.copy()

    assert repr(view_copy) == "TemporalDataFrame 'copy of view of 4'\n" \
                              "\x1b[4mTime point : 0.0 hours\x1b[0m\n" \
                              "    Time     col1    col3\n" \
                              "10  0.0h  |  20.0  |  110\n" \
                              "11  0.0h  |  22.0  |  111\n" \
                              "12  0.0h  |  24.0  |  112\n" \
                              "13  0.0h  |  26.0  |  113\n" \
                              "14  0.0h  |  28.0  |  114\n" \
                              "[15 x 2]\n" \
                              "\n" \
                              "\x1b[4mTime point : 1.0 hours\x1b[0m\n" \
                              "    Time     col1    col3\n" \
                              "25  1.0h  |  50.0  |  125\n" \
                              "26  1.0h  |  52.0  |  126\n" \
                              "27  1.0h  |  54.0  |  127\n" \
                              "28  1.0h  |  56.0  |  128\n" \
                              "29  1.0h  |  58.0  |  129\n" \
                              "[15 x 2]\n\n"

    assert np.all(view_copy.values_num == view.values_num)
    assert np.all(view_copy.values_str == view.values_str)

    cleanup([input_file])


if __name__ == '__main__':
    test_convert()
    test_copy()
