# coding: utf-8
# Created on 29/03/2022 17:08
# Author : matteo

# ====================================================
# imports
import numpy as np
import pandas as pd
from h5py import File, string_dtype
from pathlib import Path

from .utils import get_TDF, get_backed_TDF, reference_backed_data, cleanup
from vdata.name_utils import H5Mode
from vdata._core.TDF.dataframe import TemporalDataFrame


# ====================================================
# code
def test_TDF_creation():
    # data is None
    #   time_list is None
    #       index is None
    TDF = TemporalDataFrame(data=None, time_list=None, time_col_name=None, index=None, name=1)

    assert repr(TDF) == "Empty TemporalDataFrame '1'\n" \
                        "Time points: []\n" \
                        "Columns: []\n" \
                        "Index: []", repr(TDF)

    #       index is a Collection of values
    TDF = TemporalDataFrame(data=None, time_list=None, time_col_name=None, index=['a', 'b', 'c'], name=2)
    assert repr(TDF) == "Empty TemporalDataFrame '2'\n" \
                        "Time points: []\n" \
                        "Columns: []\n" \
                        "Index: ['a', 'b', 'c']", repr(TDF)

    #   time_list is a time point
    #       index is None
    TDF = TemporalDataFrame(data=None, time_list=['0h'], time_col_name=None, index=None, name=3)
    assert repr(TDF) == "Empty TemporalDataFrame '3'\n" \
                        "\033[4mTime point : '0h'\033[0m\n" \
                        "  Time-point   \n" \
                        "0         0h  |\n" \
                        "[1 x 0]\n\n", repr(TDF)

    #       index is a Collection of values
    TDF = TemporalDataFrame(data=None, time_list=['0h'], time_col_name=None, index=['a'], name=4)
    assert repr(TDF) == "Empty TemporalDataFrame '4'\n" \
                        "\033[4mTime point : '0h'\033[0m\n" \
                        "  Time-point   \n" \
                        "a         0h  |\n" \
                        "[1 x 0]\n\n", repr(TDF)

    #   time_list is a Collection of time points
    #       index is None
    TDF = TemporalDataFrame(data=None, time_list=['0h', '5h'], time_col_name=None, index=None, name=5)
    assert repr(TDF) == "Empty TemporalDataFrame '5'\n" \
                        "\033[4mTime point : '0h'\033[0m\n" \
                        "  Time-point   \n" \
                        "0         0h  |\n" \
                        "[1 x 0]\n" \
                        "\n" \
                        "\033[4mTime point : '5h'\033[0m\n" \
                        "  Time-point   \n" \
                        "1         5h  |\n" \
                        "[1 x 0]\n\n", repr(TDF)

    #       index is a Collection of values
    TDF = TemporalDataFrame(data=None, time_list=['0h', '5h'], time_col_name=None, index=['a', 'b'], name=6)
    assert repr(TDF) == "Empty TemporalDataFrame '6'\n" \
                        "\033[4mTime point : '0h'\033[0m\n" \
                        "  Time-point   \n" \
                        "a         0h  |\n" \
                        "[1 x 0]\n" \
                        "\n" \
                        "\033[4mTime point : '5h'\033[0m\n" \
                        "  Time-point   \n" \
                        "b         5h  |\n" \
                        "[1 x 0]\n\n", repr(TDF)

    def sub_test(_data):
        #   time_list is None
        #       index is None
        _TDF = TemporalDataFrame(data=data, time_list=None, time_col_name=None, index=None, name=7)
        assert repr(_TDF) == "TemporalDataFrame '7'\n" \
                             "\033[4mTime point : 0.0 hours\033[0m\n" \
                             "  Time-point    col1\n" \
                             "0       0.0h  |  1.0\n" \
                             "1       0.0h  |  2.0\n" \
                             "2       0.0h  |  3.0\n" \
                             "3       0.0h  |  4.0\n" \
                             "4       0.0h  |  5.0\n" \
                             "[9 x 1]\n\n", repr(_TDF)

        #       index is a Collection of values, same length as data
        _TDF = TemporalDataFrame(data=data, time_list=None, time_col_name=None,
                                 index=['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i'], name=8)
        assert repr(_TDF) == "TemporalDataFrame '8'\n" \
                             "\033[4mTime point : 0.0 hours\033[0m\n" \
                             "  Time-point    col1\n" \
                             "a       0.0h  |  1.0\n" \
                             "b       0.0h  |  2.0\n" \
                             "c       0.0h  |  3.0\n" \
                             "d       0.0h  |  4.0\n" \
                             "e       0.0h  |  5.0\n" \
                             "[9 x 1]\n\n", repr(_TDF)

        #   time_list is a Collection of time points
        #       index is None
        _TDF = TemporalDataFrame(data=data, time_list=['0h', '0h', '0h', '5h', '5h', '5h', '10h', '10h', '10h'],
                                 time_col_name=None, index=None, name=9)
        assert repr(_TDF) == "TemporalDataFrame '9'\n" \
                             "\033[4mTime point : 0.0 hours\033[0m\n" \
                             "  Time-point    col1\n" \
                             "0       0.0h  |  1.0\n" \
                             "1       0.0h  |  2.0\n" \
                             "2       0.0h  |  3.0\n" \
                             "[3 x 1]\n" \
                             "\n" \
                             "\033[4mTime point : 5.0 hours\033[0m\n" \
                             "  Time-point    col1\n" \
                             "3       5.0h  |  4.0\n" \
                             "4       5.0h  |  5.0\n" \
                             "5       5.0h  |  6.0\n" \
                             "[3 x 1]\n" \
                             "\n" \
                             "\033[4mTime point : 10.0 hours\033[0m\n" \
                             "  Time-point    col1\n" \
                             "6      10.0h  |  7.0\n" \
                             "7      10.0h  |  8.0\n" \
                             "8      10.0h  |  9.0\n" \
                             "[3 x 1]\n\n", repr(_TDF)

        #       index is a Collection of values, same length as data
        _TDF = TemporalDataFrame(data=data, time_list=['0h', '0h', '0h', '5h', '5h', '5h', '10h', '10h', '10h'],
                                 time_col_name=None, index=['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i'], name=10)
        assert repr(_TDF) == "TemporalDataFrame '10'\n" \
                             "\033[4mTime point : 0.0 hours\033[0m\n" \
                             "  Time-point    col1\n" \
                             "a       0.0h  |  1.0\n" \
                             "b       0.0h  |  2.0\n" \
                             "c       0.0h  |  3.0\n" \
                             "[3 x 1]\n" \
                             "\n" \
                             "\033[4mTime point : 5.0 hours\033[0m\n" \
                             "  Time-point    col1\n" \
                             "d       5.0h  |  4.0\n" \
                             "e       5.0h  |  5.0\n" \
                             "f       5.0h  |  6.0\n" \
                             "[3 x 1]\n" \
                             "\n" \
                             "\033[4mTime point : 10.0 hours\033[0m\n" \
                             "  Time-point    col1\n" \
                             "g      10.0h  |  7.0\n" \
                             "h      10.0h  |  8.0\n" \
                             "i      10.0h  |  9.0\n" \
                             "[3 x 1]\n\n", repr(_TDF)

        #       index is a Collection of values, divides data
        _TDF = TemporalDataFrame(data=data, time_list=['0h', '0h', '0h', '5h', '5h', '5h', '10h', '10h', '10h'],
                                 time_col_name=None, index=['a', 'b', 'c', 'a', 'b', 'c', 'a', 'b', 'c'],
                                 repeating_index=True, name=11)
        assert repr(_TDF) == "TemporalDataFrame '11'\n" \
                             "\033[4mTime point : 0.0 hours\033[0m\n" \
                             "  Time-point    col1\n" \
                             "a       0.0h  |  1.0\n" \
                             "b       0.0h  |  2.0\n" \
                             "c       0.0h  |  3.0\n" \
                             "[3 x 1]\n" \
                             "\n" \
                             "\033[4mTime point : 5.0 hours\033[0m\n" \
                             "  Time-point    col1\n" \
                             "a       5.0h  |  4.0\n" \
                             "b       5.0h  |  5.0\n" \
                             "c       5.0h  |  6.0\n" \
                             "[3 x 1]\n" \
                             "\n" \
                             "\033[4mTime point : 10.0 hours\033[0m\n" \
                             "  Time-point    col1\n" \
                             "a      10.0h  |  7.0\n" \
                             "b      10.0h  |  8.0\n" \
                             "c      10.0h  |  9.0\n" \
                             "[3 x 1]\n\n", repr(_TDF)

    def sub_test_str(_data):
        #   time_list is None
        #       index is None
        _TDF = TemporalDataFrame(data=data, time_list=None, time_col_name=None, index=None, name=12)
        assert repr(_TDF) == "TemporalDataFrame '12'\n" \
                             "\033[4mTime point : 0.0 hours\033[0m\n" \
                             "  Time-point    col1\n" \
                             "0       0.0h  |    a\n" \
                             "1       0.0h  |    b\n" \
                             "2       0.0h  |    c\n" \
                             "3       0.0h  |    d\n" \
                             "4       0.0h  |    e\n" \
                             "[9 x 1]\n\n", repr(_TDF)

        #       index is a Collection of values, same length as data
        _TDF = TemporalDataFrame(data=data, time_list=None, time_col_name=None,
                                 index=['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i'], name=13)
        assert repr(_TDF) == "TemporalDataFrame '13'\n" \
                             "\033[4mTime point : 0.0 hours\033[0m\n" \
                             "  Time-point    col1\n" \
                             "a       0.0h  |    a\n" \
                             "b       0.0h  |    b\n" \
                             "c       0.0h  |    c\n" \
                             "d       0.0h  |    d\n" \
                             "e       0.0h  |    e\n" \
                             "[9 x 1]\n\n", repr(_TDF)

        #   time_list is a Collection of time points
        #       index is None
        _TDF = TemporalDataFrame(data=data, time_list=['0h', '0h', '0h', '5h', '5h', '5h', '10h', '10h', '10h'],
                                 time_col_name=None, index=None, name=14)
        assert repr(_TDF) == "TemporalDataFrame '14'\n" \
                             "\033[4mTime point : 0.0 hours\033[0m\n" \
                             "  Time-point    col1\n" \
                             "0       0.0h  |    a\n" \
                             "1       0.0h  |    b\n" \
                             "2       0.0h  |    c\n" \
                             "[3 x 1]\n" \
                             "\n" \
                             "\033[4mTime point : 5.0 hours\033[0m\n" \
                             "  Time-point    col1\n" \
                             "3       5.0h  |    d\n" \
                             "4       5.0h  |    e\n" \
                             "5       5.0h  |    f\n" \
                             "[3 x 1]\n" \
                             "\n" \
                             "\033[4mTime point : 10.0 hours\033[0m\n" \
                             "  Time-point    col1\n" \
                             "6      10.0h  |    g\n" \
                             "7      10.0h  |    h\n" \
                             "8      10.0h  |    i\n" \
                             "[3 x 1]\n\n", repr(_TDF)

        #       index is a Collection of values, same length as data
        _TDF = TemporalDataFrame(data=data, time_list=['0h', '0h', '0h', '5h', '5h', '5h', '10h', '10h', '10h'],
                                 time_col_name=None, index=['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i'], name=15)
        assert repr(_TDF) == "TemporalDataFrame '15'\n" \
                             "\033[4mTime point : 0.0 hours\033[0m\n" \
                             "  Time-point    col1\n" \
                             "a       0.0h  |    a\n" \
                             "b       0.0h  |    b\n" \
                             "c       0.0h  |    c\n" \
                             "[3 x 1]\n" \
                             "\n" \
                             "\033[4mTime point : 5.0 hours\033[0m\n" \
                             "  Time-point    col1\n" \
                             "d       5.0h  |    d\n" \
                             "e       5.0h  |    e\n" \
                             "f       5.0h  |    f\n" \
                             "[3 x 1]\n" \
                             "\n" \
                             "\033[4mTime point : 10.0 hours\033[0m\n" \
                             "  Time-point    col1\n" \
                             "g      10.0h  |    g\n" \
                             "h      10.0h  |    h\n" \
                             "i      10.0h  |    i\n" \
                             "[3 x 1]\n\n", repr(_TDF)

        #       index is a Collection of values, divides data
        _TDF = TemporalDataFrame(data=data, time_list=['0h', '0h', '0h', '5h', '5h', '5h', '10h', '10h', '10h'],
                                 time_col_name=None, index=['a', 'b', 'c', 'a', 'b', 'c', 'a', 'b', 'c'],
                                 repeating_index=True, name=16)
        assert repr(_TDF) == "TemporalDataFrame '16'\n" \
                             "\033[4mTime point : 0.0 hours\033[0m\n" \
                             "  Time-point    col1\n" \
                             "a       0.0h  |    a\n" \
                             "b       0.0h  |    b\n" \
                             "c       0.0h  |    c\n" \
                             "[3 x 1]\n" \
                             "\n" \
                             "\033[4mTime point : 5.0 hours\033[0m\n" \
                             "  Time-point    col1\n" \
                             "a       5.0h  |    d\n" \
                             "b       5.0h  |    e\n" \
                             "c       5.0h  |    f\n" \
                             "[3 x 1]\n" \
                             "\n" \
                             "\033[4mTime point : 10.0 hours\033[0m\n" \
                             "  Time-point    col1\n" \
                             "a      10.0h  |    g\n" \
                             "b      10.0h  |    h\n" \
                             "c      10.0h  |    i\n" \
                             "[3 x 1]\n\n", repr(_TDF)

    def sub_test_both(_data):
        #   time_list is None
        #       index is None
        _TDF = TemporalDataFrame(data=data, time_list=None, time_col_name=None, index=None, name=17)
        assert repr(_TDF) == "TemporalDataFrame '17'\n" \
                             "\033[4mTime point : 0.0 hours\033[0m\n" \
                             "  Time-point    col1    col2\n" \
                             "0       0.0h  |  1.0  |    a\n" \
                             "1       0.0h  |  2.0  |    b\n" \
                             "2       0.0h  |  3.0  |    c\n" \
                             "3       0.0h  |  4.0  |    d\n" \
                             "4       0.0h  |  5.0  |    e\n" \
                             "[9 x 2]\n\n", repr(_TDF)

        #       index is a Collection of values, same length as data
        _TDF = TemporalDataFrame(data=data, time_list=None, time_col_name=None,
                                 index=['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i'], name=18)
        assert repr(_TDF) == "TemporalDataFrame '18'\n" \
                             "\033[4mTime point : 0.0 hours\033[0m\n" \
                             "  Time-point    col1    col2\n" \
                             "a       0.0h  |  1.0  |    a\n" \
                             "b       0.0h  |  2.0  |    b\n" \
                             "c       0.0h  |  3.0  |    c\n" \
                             "d       0.0h  |  4.0  |    d\n" \
                             "e       0.0h  |  5.0  |    e\n" \
                             "[9 x 2]\n\n", repr(_TDF)

        #   time_list is a Collection of time points
        #       index is None
        _TDF = TemporalDataFrame(data=data, time_list=['0h', '0h', '0h', '5h', '5h', '5h', '10h', '10h', '10h'],
                                 time_col_name=None, index=None, name=19)
        assert repr(_TDF) == "TemporalDataFrame '19'\n" \
                             "\033[4mTime point : 0.0 hours\033[0m\n" \
                             "  Time-point    col1    col2\n" \
                             "0       0.0h  |  1.0  |    a\n" \
                             "1       0.0h  |  2.0  |    b\n" \
                             "2       0.0h  |  3.0  |    c\n" \
                             "[3 x 2]\n" \
                             "\n" \
                             "\033[4mTime point : 5.0 hours\033[0m\n" \
                             "  Time-point    col1    col2\n" \
                             "3       5.0h  |  4.0  |    d\n" \
                             "4       5.0h  |  5.0  |    e\n" \
                             "5       5.0h  |  6.0  |    f\n" \
                             "[3 x 2]\n" \
                             "\n" \
                             "\033[4mTime point : 10.0 hours\033[0m\n" \
                             "  Time-point    col1    col2\n" \
                             "6      10.0h  |  7.0  |    g\n" \
                             "7      10.0h  |  8.0  |    h\n" \
                             "8      10.0h  |  9.0  |    i\n" \
                             "[3 x 2]\n\n", repr(_TDF)

        #       index is a Collection of values, same length as data
        _TDF = TemporalDataFrame(data=data, time_list=['0h', '0h', '0h', '5h', '5h', '5h', '10h', '10h', '10h'],
                                 time_col_name=None, index=['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i'], name=20)
        assert repr(_TDF) == "TemporalDataFrame '20'\n" \
                             "\033[4mTime point : 0.0 hours\033[0m\n" \
                             "  Time-point    col1    col2\n" \
                             "a       0.0h  |  1.0  |    a\n" \
                             "b       0.0h  |  2.0  |    b\n" \
                             "c       0.0h  |  3.0  |    c\n" \
                             "[3 x 2]\n" \
                             "\n" \
                             "\033[4mTime point : 5.0 hours\033[0m\n" \
                             "  Time-point    col1    col2\n" \
                             "d       5.0h  |  4.0  |    d\n" \
                             "e       5.0h  |  5.0  |    e\n" \
                             "f       5.0h  |  6.0  |    f\n" \
                             "[3 x 2]\n" \
                             "\n" \
                             "\033[4mTime point : 10.0 hours\033[0m\n" \
                             "  Time-point    col1    col2\n" \
                             "g      10.0h  |  7.0  |    g\n" \
                             "h      10.0h  |  8.0  |    h\n" \
                             "i      10.0h  |  9.0  |    i\n" \
                             "[3 x 2]\n\n", repr(_TDF)

        #       index is a Collection of values, divides data
        _TDF = TemporalDataFrame(data=data, time_list=['0h', '0h', '0h', '5h', '5h', '5h', '10h', '10h', '10h'],
                                 time_col_name=None, index=['a', 'b', 'c', 'a', 'b', 'c', 'a', 'b', 'c'],
                                 repeating_index=True, name=21)
        assert repr(_TDF) == "TemporalDataFrame '21'\n" \
                             "\033[4mTime point : 0.0 hours\033[0m\n" \
                             "  Time-point    col1    col2\n" \
                             "a       0.0h  |  1.0  |    a\n" \
                             "b       0.0h  |  2.0  |    b\n" \
                             "c       0.0h  |  3.0  |    c\n" \
                             "[3 x 2]\n" \
                             "\n" \
                             "\033[4mTime point : 5.0 hours\033[0m\n" \
                             "  Time-point    col1    col2\n" \
                             "a       5.0h  |  4.0  |    d\n" \
                             "b       5.0h  |  5.0  |    e\n" \
                             "c       5.0h  |  6.0  |    f\n" \
                             "[3 x 2]\n" \
                             "\n" \
                             "\033[4mTime point : 10.0 hours\033[0m\n" \
                             "  Time-point    col1    col2\n" \
                             "a      10.0h  |  7.0  |    g\n" \
                             "b      10.0h  |  8.0  |    h\n" \
                             "c      10.0h  |  9.0  |    i\n" \
                             "[3 x 2]\n\n", repr(_TDF)

    # data is a dictionary, numeric data only
    data = {'col1': [1., 2., 3., 4., 5., 6., 7., 8., 9.]}

    sub_test(data)

    # data is a pandas DataFrame
    data = pd.DataFrame({'col1': [1., 2., 3., 4., 5., 6., 7., 8., 9.]})

    sub_test(data)

    # data is a dictionary, string data only
    data = {'col1': ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i']}

    sub_test_str(data)

    # data is a dictionary, both numeric and string data
    data = {'col1': [1., 2., 3., 4., 5., 6., 7., 8., 9.],
            'col2': ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i']}

    sub_test_both(data)

    # data is not sorted
    data = {'col2': ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i'],
            'col1': [1., 2., 3., 4., 5., 6., 7., 8., 9.]}

    TDF = TemporalDataFrame(data=data, time_list=['10h', '10h', '10h', '0h', '0h', '0h', '5h', '5h', '5h'],
                            time_col_name=None, index=['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i'], name=22)
    assert repr(TDF) == "TemporalDataFrame '22'\n" \
                        "\033[4mTime point : 0.0 hours\033[0m\n" \
                        "  Time-point    col1    col2\n" \
                        "d       0.0h  |  4.0  |    d\n" \
                        "e       0.0h  |  5.0  |    e\n" \
                        "f       0.0h  |  6.0  |    f\n" \
                        "[3 x 2]\n" \
                        "\n" \
                        "\033[4mTime point : 5.0 hours\033[0m\n" \
                        "  Time-point    col1    col2\n" \
                        "g       5.0h  |  7.0  |    g\n" \
                        "h       5.0h  |  8.0  |    h\n" \
                        "i       5.0h  |  9.0  |    i\n" \
                        "[3 x 2]\n" \
                        "\n" \
                        "\033[4mTime point : 10.0 hours\033[0m\n" \
                        "  Time-point    col1    col2\n" \
                        "a      10.0h  |  1.0  |    a\n" \
                        "b      10.0h  |  2.0  |    b\n" \
                        "c      10.0h  |  3.0  |    c\n" \
                        "[3 x 2]\n\n", repr(TDF)

    assert np.all(TDF.index == np.array(['d', 'e', 'f', 'g', 'h', 'i', 'a', 'b', 'c']))
    assert np.all(TDF.columns == np.array(['col1', 'col2']))
    assert np.all(TDF.timepoints == np.array(['0h', '5h', '10h']))
    assert np.all(TDF.timepoints_column == np.array(['0h', '0h', '0h', '5h', '5h', '5h', '10h', '10h', '10h']))

    assert np.all(TDF.values_num == np.array([[4], [5], [6], [7], [8], [9], [1], [2], [3]]))
    assert np.all(TDF.values_str == np.array([['d'], ['e'], ['f'], ['g'], ['h'], ['i'], ['a'], ['b'], ['c']]))

    TDF = get_TDF('23')

    assert repr(TDF) == "TemporalDataFrame '23'\n" \
                        "\033[4mTime point : 0.0 hours\033[0m\n" \
                        "   Time-point     col1   col2    col3 col4\n" \
                        "50       0.0h  |  50.0  150.0  |  250  350\n" \
                        "51       0.0h  |  51.0  151.0  |  251  351\n" \
                        "52       0.0h  |  52.0  152.0  |  252  352\n" \
                        "53       0.0h  |  53.0  153.0  |  253  353\n" \
                        "54       0.0h  |  54.0  154.0  |  254  354\n" \
                        "[50 x 4]\n" \
                        "\n" \
                        "\033[4mTime point : 1.0 hours\033[0m\n" \
                        "  Time-point    col1   col2    col3 col4\n" \
                        "0       1.0h  |  0.0  100.0  |  200  300\n" \
                        "1       1.0h  |  1.0  101.0  |  201  301\n" \
                        "2       1.0h  |  2.0  102.0  |  202  302\n" \
                        "3       1.0h  |  3.0  103.0  |  203  303\n" \
                        "4       1.0h  |  4.0  104.0  |  204  304\n" \
                        "[50 x 4]\n\n"

    # data is in a H5 file
    input_file = Path(__file__).parent / 'test_convert_TDF'
    cleanup([input_file])

    h5_data = File(input_file, H5Mode.WRITE_TRUNCATE)
    h5_data.attrs['type'] = reference_backed_data['type']
    h5_data.attrs['name'] = '24'
    h5_data.attrs['locked_indices'] = reference_backed_data['locked_indices']
    h5_data.attrs['locked_columns'] = reference_backed_data['locked_columns']
    h5_data.attrs['timepoints_column_name'] = reference_backed_data['timepoints_column_name']
    h5_data.attrs['repeating_index'] = reference_backed_data['repeating_index']

    h5_data.create_dataset('index', data=reference_backed_data['index'])
    h5_data.create_dataset('columns_numerical', data=reference_backed_data['columns_numerical'],
                           chunks=True, maxshape=(None,), dtype=string_dtype())
    h5_data.create_dataset('columns_string', data=reference_backed_data['columns_string'],
                           chunks=True, maxshape=(None,), dtype=string_dtype())
    h5_data.create_dataset('timepoints', data=reference_backed_data['timepoints'], dtype=string_dtype())

    h5_data.create_dataset('values_numerical', data=reference_backed_data['values_numerical'],
                           chunks=True, maxshape=(None, None))

    h5_data.create_dataset('values_string', data=reference_backed_data['values_string'], dtype=string_dtype(),
                           chunks=True, maxshape=(None, None))

    TDF = TemporalDataFrame(data=h5_data)

    assert repr(TDF) == "Backed TemporalDataFrame '24'\n" \
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

    TDF.file.close()
    cleanup([input_file])


def test_inverted_TDF_creation():
    # TDF is not backed
    TDF = get_TDF('1')

    assert repr(~TDF) == "Inverted view of TemporalDataFrame '1'\n" \
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

    # TDF is backed
    input_file = Path(__file__).parent / 'test_creation_inverted_TDF'
    cleanup([input_file])

    TDF = get_backed_TDF(input_file, '2')

    assert repr(~TDF) == "Inverted view of backed TemporalDataFrame '2'\n" \
                         "\x1b[4mTime point : '0.0h'\x1b[0m\n" \
                         "  Time-point    col1 col2    col3\n" \
                         "0       0.0h  |  0.0  1.0  |  100\n" \
                         "1       0.0h  |  2.0  3.0  |  101\n" \
                         "2       0.0h  |  4.0  5.0  |  102\n" \
                         "3       0.0h  |  6.0  7.0  |  103\n" \
                         "4       0.0h  |  8.0  9.0  |  104\n" \
                         "[25 x 3]\n" \
                         "\n" \
                         "\x1b[4mTime point : '1.0h'\x1b[0m\n" \
                         "   Time-point     col1  col2    col3\n" \
                         "25       1.0h  |  50.0  51.0  |  125\n" \
                         "26       1.0h  |  52.0  53.0  |  126\n" \
                         "27       1.0h  |  54.0  55.0  |  127\n" \
                         "28       1.0h  |  56.0  57.0  |  128\n" \
                         "29       1.0h  |  58.0  59.0  |  129\n" \
                         "[25 x 3]\n" \
                         "\n"

    TDF.file.close()

    cleanup([input_file])


if __name__ == '__main__':
    test_TDF_creation()
