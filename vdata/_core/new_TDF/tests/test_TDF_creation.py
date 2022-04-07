# coding: utf-8
# Created on 29/03/2022 17:08
# Author : matteo

# ====================================================
# imports
import numpy as np
import pandas as pd

from .utils import get_TDF
from vdata._core.new_TDF.dataframe import TemporalDataFrame


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
    TDF = TemporalDataFrame(data=None, time_list='0h', time_col_name=None, index=None, name=3)
    assert repr(TDF) == "Empty TemporalDataFrame '3'\n" \
                        "Time points: []\n" \
                        "Columns: []\n" \
                        "Index: []", repr(TDF)

    #       index is a Collection of values
    TDF = TemporalDataFrame(data=None, time_list='0h', time_col_name=None, index=['a'], name=4)
    assert repr(TDF) == "Empty TemporalDataFrame '4'\n" \
                        "Time points: []\n" \
                        "Columns: []\n" \
                        "Index: ['a']", repr(TDF)

    #   time_list is a Collection of time points
    #       index is None
    TDF = TemporalDataFrame(data=None, time_list=['0h', '5h'], time_col_name=None, index=None, name=5)
    assert repr(TDF) == "Empty TemporalDataFrame '5'\n" \
                        "Time points: []\n" \
                        "Columns: []\n" \
                        "Index: []", repr(TDF)

    #       index is a Collection of values
    TDF = TemporalDataFrame(data=None, time_list=['0h', '5h'], time_col_name=None, index=['a', 'b'], name=6)
    assert repr(TDF) == "Empty TemporalDataFrame '6'\n" \
                        "Time points: []\n" \
                        "Columns: []\n" \
                        "Index: ['a', 'b']", repr(TDF)

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
                                 time_col_name=None, index=['a', 'b', 'c', 'a', 'b', 'c', 'a', 'b', 'c'], name=11)
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
                                 time_col_name=None, index=['a', 'b', 'c', 'a', 'b', 'c', 'a', 'b', 'c'], name=16)
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
                                 time_col_name=None, index=['a', 'b', 'c', 'a', 'b', 'c', 'a', 'b', 'c'], name=21)
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
                        "   Time-point    col1 col2    col3 col4\n" \
                        "50       0.0h  |   50  150  |  250  350\n" \
                        "51       0.0h  |   51  151  |  251  351\n" \
                        "52       0.0h  |   52  152  |  252  352\n" \
                        "53       0.0h  |   53  153  |  253  353\n" \
                        "54       0.0h  |   54  154  |  254  354\n" \
                        "[50 x 4]\n" \
                        "\n" \
                        "\033[4mTime point : 1.0 hours\033[0m\n" \
                        "  Time-point    col1 col2    col3 col4\n" \
                        "0       1.0h  |    0  100  |  200  300\n" \
                        "1       1.0h  |    1  101  |  201  301\n" \
                        "2       1.0h  |    2  102  |  202  302\n" \
                        "3       1.0h  |    3  103  |  203  303\n" \
                        "4       1.0h  |    4  104  |  204  304\n" \
                        "[50 x 4]\n\n"

    # data is in a H5 file
    # TODO


if __name__ == '__main__':
    test_TDF_creation()