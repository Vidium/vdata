# coding: utf-8
# Created on 04/04/2022 17:24
# Author : matteo

# ====================================================
# imports
import numpy as np
from pathlib import Path

from ..name_utils import H5Mode
from .utils import get_TDF, get_backed_TDF


# ====================================================
# code
def cleanup(path: Path) -> None:
    if path.exists():
        path.unlink(missing_ok=True)


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
    df = TDF.to_pandas(with_timepoints='time_points')

    assert np.all(df.time_points == ['0.0h' for _ in range(50)] + ['1.0h' for _ in range(50)])
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
    cleanup(input_file)

    TDF = get_backed_TDF(input_file, '2')

    #   no time-points
    df = TDF.to_pandas()

    assert np.all(df.values[:, :2] == np.array(range(100)).reshape((50, 2)))
    assert np.all(df.values[:, 2:] == np.array(list(map(str, range(100, 150))), dtype=np.dtype('O')).reshape((50, 1)))

    #   with time-points
    df = TDF.to_pandas(with_timepoints='time_points')

    assert np.all(df.time_points == ['0.0h' for _ in range(25)] + ['1.0h' for _ in range(25)])
    assert np.all(df.values[:, 1:3] == np.array(range(100)).reshape((50, 2)))
    assert np.all(df.values[:, 3:] == np.array(list(map(str, range(100, 150))), dtype=np.dtype('O')).reshape((50, 1)))

    cleanup(input_file)

    # -------------------------------------------------------------------------
    # TDF is a view

    # TODO


def test_copy():
    # TDF is not backed
    TDF = get_TDF('1')

    #   time-points column name is None
    TDF_copy = TDF.copy()

    assert repr(TDF_copy) == "TemporalDataFrame 'copy of 1'\n" \
                             "\x1b[4mTime point : 0.0 hours\x1b[0m\n" \
                             "   Time-point    col1 col2    col3 col4\n" \
                             "50       0.0h  |   50  150  |  250  350\n" \
                             "51       0.0h  |   51  151  |  251  351\n" \
                             "52       0.0h  |   52  152  |  252  352\n" \
                             "53       0.0h  |   53  153  |  253  353\n" \
                             "54       0.0h  |   54  154  |  254  354\n" \
                             "[50 x 4]\n" \
                             "\n" \
                             "\x1b[4mTime point : 1.0 hours\x1b[0m\n" \
                             "  Time-point    col1 col2    col3 col4\n" \
                             "0       1.0h  |    0  100  |  200  300\n" \
                             "1       1.0h  |    1  101  |  201  301\n" \
                             "2       1.0h  |    2  102  |  202  302\n" \
                             "3       1.0h  |    3  103  |  203  303\n" \
                             "4       1.0h  |    4  104  |  204  304\n" \
                             "[50 x 4]\n\n"

    assert np.all(TDF_copy.values_num == TDF.values_num)
    assert np.all(TDF_copy.values_str == TDF.values_str)

    #   time-points column name is not None
    TDF.timepoints_column_name = 'Time'
    TDF_copy = TDF.copy()

    assert repr(TDF_copy) == "TemporalDataFrame 'copy of 1'\n" \
                             "\x1b[4mTime point : 0.0 hours\x1b[0m\n" \
                             "    Time    col1 col2    col3 col4\n" \
                             "50  0.0h  |   50  150  |  250  350\n" \
                             "51  0.0h  |   51  151  |  251  351\n" \
                             "52  0.0h  |   52  152  |  252  352\n" \
                             "53  0.0h  |   53  153  |  253  353\n" \
                             "54  0.0h  |   54  154  |  254  354\n" \
                             "[50 x 4]\n" \
                             "\n" \
                             "\x1b[4mTime point : 1.0 hours\x1b[0m\n" \
                             "   Time    col1 col2    col3 col4\n" \
                             "0  1.0h  |    0  100  |  200  300\n" \
                             "1  1.0h  |    1  101  |  201  301\n" \
                             "2  1.0h  |    2  102  |  202  302\n" \
                             "3  1.0h  |    3  103  |  203  303\n" \
                             "4  1.0h  |    4  104  |  204  304\n" \
                             "[50 x 4]\n\n"

    assert np.all(TDF_copy.values_num == TDF.values_num)
    assert np.all(TDF_copy.values_str == TDF.values_str)

    # TDF is backed

    input_file = Path(__file__).parent / 'test_copy_TDF'
    cleanup(input_file)

    TDF = get_backed_TDF(input_file, '2', mode=H5Mode.READ_WRITE)

    #   time-points column name is None
    TDF_copy = TDF.copy()

    assert repr(TDF_copy) == "TemporalDataFrame 'copy of 2'\n" \
                             "\x1b[4mTime point : 0.0 hours\x1b[0m\n" \
                             "  Time-point    col1 col2    col3\n" \
                             "0       0.0h  |    0    1  |  100\n" \
                             "1       0.0h  |    2    3  |  101\n" \
                             "2       0.0h  |    4    5  |  102\n" \
                             "3       0.0h  |    6    7  |  103\n" \
                             "4       0.0h  |    8    9  |  104\n" \
                             "[25 x 3]\n" \
                             "\n" \
                             "\x1b[4mTime point : 1.0 hours\x1b[0m\n" \
                             "   Time-point    col1 col2    col3\n" \
                             "25       1.0h  |   50   51  |  125\n" \
                             "26       1.0h  |   52   53  |  126\n" \
                             "27       1.0h  |   54   55  |  127\n" \
                             "28       1.0h  |   56   57  |  128\n" \
                             "29       1.0h  |   58   59  |  129\n" \
                             "[25 x 3]\n\n"

    assert np.all(TDF_copy.values_num == TDF.values_num)
    assert np.all(TDF_copy.values_str == TDF.values_str)

    #   time-points column name is not None
    TDF.timepoints_column_name = 'Time'
    TDF_copy = TDF.copy()

    assert repr(TDF_copy) == "TemporalDataFrame 'copy of 2'\n" \
                             "\x1b[4mTime point : 0.0 hours\x1b[0m\n" \
                             "   Time    col1 col2    col3\n" \
                             "0  0.0h  |    0    1  |  100\n" \
                             "1  0.0h  |    2    3  |  101\n" \
                             "2  0.0h  |    4    5  |  102\n" \
                             "3  0.0h  |    6    7  |  103\n" \
                             "4  0.0h  |    8    9  |  104\n" \
                             "[25 x 3]\n" \
                             "\n" \
                             "\x1b[4mTime point : 1.0 hours\x1b[0m\n" \
                             "    Time    col1 col2    col3\n" \
                             "25  1.0h  |   50   51  |  125\n" \
                             "26  1.0h  |   52   53  |  126\n" \
                             "27  1.0h  |   54   55  |  127\n" \
                             "28  1.0h  |   56   57  |  128\n" \
                             "29  1.0h  |   58   59  |  129\n" \
                             "[25 x 3]\n\n"

    assert np.all(TDF_copy.values_num == TDF.values_num)
    assert np.all(TDF_copy.values_str == TDF.values_str)

    cleanup(input_file)

    # -------------------------------------------------------------------------
    # TDF is a view

    # TODO


if __name__ == '__main__':
    test_convert()
    test_copy()
