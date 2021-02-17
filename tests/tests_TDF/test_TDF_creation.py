# coding: utf-8
# Created on 12/9/20 10:21 AM
# Author : matteo

# ====================================================
# imports
import pandas as pd

import vdata


# ====================================================
# code
def test_TDF_creation():
    # data is None
    #   time_list is None
    #       time_points is None
    #           index is None
    TDF = vdata.TemporalDataFrame(data=None, time_list=None, time_col_name=None, time_points=None, index=None, name=1)
    assert repr(TDF) == "Empty TemporalDataFrame '1'\n" \
                        "Time points: []\n" \
                        "Columns: []\n" \
                        "Index: []", repr(TDF)

    #           index is a Collection of values
    TDF = vdata.TemporalDataFrame(data=None, time_list=None, time_col_name=None, time_points=None,
                                  index=['a', 'b', 'c'], name=2)
    assert repr(TDF) == "Empty TemporalDataFrame '2'\n" \
                        "\033[4mTime point : 0.0 (no unit)\033[0m\n" \
                        "Empty DataFrame\n" \
                        "Columns: []\n" \
                        "Index: [a, b, c]\n" \
                        "\n" \
                        "[3 x 0]\n\n", repr(TDF)

    #       time_points is a Collection of time points
    #           index is None
    TDF = vdata.TemporalDataFrame(data=None, time_list=None, time_col_name=None, time_points=['0h', '5h'], index=None,
                                  name=3)
    assert repr(TDF) == "Empty TemporalDataFrame '3'\n" \
                        "\033[4mTime point : 0.0 hours\033[0m\n" \
                        "Empty DataFrame\n" \
                        "Columns: []\n" \
                        "Index: []\n" \
                        "\n" \
                        "\033[4mTime point : 5.0 hours\033[0m\n" \
                        "Empty DataFrame\n" \
                        "Columns: []\n" \
                        "Index: []\n\n", repr(TDF)

    #           index is a Collection of values
    TDF = vdata.TemporalDataFrame(data=None, time_list=None, time_col_name=None, time_points=['0h', '5h'],
                                  index=['a', 'b', 'c'], name=4)
    assert repr(TDF) == "Empty TemporalDataFrame '4'\n" \
                        "\033[4mTime point : 0.0 hours\033[0m\n" \
                        "Empty DataFrame\n" \
                        "Columns: []\n" \
                        "Index: [a, b, c]\n" \
                        "\n" \
                        "[3 x 0]\n" \
                        "\n" \
                        "\033[4mTime point : 5.0 hours\033[0m\n" \
                        "Empty DataFrame\n" \
                        "Columns: []\n" \
                        "Index: [a, b, c]\n" \
                        "\n" \
                        "[3 x 0]\n\n", repr(TDF)

    #   time_list is a time point
    #       time_points is None
    #           index is None
    TDF = vdata.TemporalDataFrame(data=None, time_list='0h', time_col_name=None, time_points=None, index=None, name=5)
    assert repr(TDF) == "Empty TemporalDataFrame '5'\n" \
                        "\033[4mTime point : 0.0 hours\033[0m\n" \
                        "Empty DataFrame\n" \
                        "Columns: []\n" \
                        "Index: [0]\n" \
                        "\n" \
                        "[1 x 0]\n\n", repr(TDF)

    #           index is a Collection of values
    TDF = vdata.TemporalDataFrame(data=None, time_list='0h', time_col_name=None, time_points=None, index=['a'], name=6)
    assert repr(TDF) == "Empty TemporalDataFrame '6'\n" \
                        "\033[4mTime point : 0.0 hours\033[0m\n" \
                        "Empty DataFrame\n" \
                        "Columns: []\n" \
                        "Index: [a]\n" \
                        "\n" \
                        "[1 x 0]\n\n", repr(TDF)

    #       time_points is a Collection of time points
    #           index is None
    TDF = vdata.TemporalDataFrame(data=None, time_list='0h', time_col_name=None, time_points=['0h', '5h'], index=None,
                                  name=7)
    assert repr(TDF) == "Empty TemporalDataFrame '7'\n" \
                        "\033[4mTime point : 0.0 hours\033[0m\n" \
                        "Empty DataFrame\n" \
                        "Columns: []\n" \
                        "Index: [0]\n" \
                        "\n" \
                        "[1 x 0]\n" \
                        "\n" \
                        "\033[4mTime point : 5.0 hours\033[0m\n" \
                        "Empty DataFrame\n" \
                        "Columns: []\n" \
                        "Index: []\n\n", repr(TDF)

    #           index is a Collection of values
    TDF = vdata.TemporalDataFrame(data=None, time_list='0h', time_col_name=None, time_points=['0h', '5h'], index=['a'],
                                  name=8)
    assert repr(TDF) == "Empty TemporalDataFrame '8'\n" \
                        "\033[4mTime point : 0.0 hours\033[0m\n" \
                        "Empty DataFrame\n" \
                        "Columns: []\n" \
                        "Index: [a]\n" \
                        "\n" \
                        "[1 x 0]\n" \
                        "\n" \
                        "\033[4mTime point : 5.0 hours\033[0m\n" \
                        "Empty DataFrame\n" \
                        "Columns: []\n" \
                        "Index: []\n\n", repr(TDF)

    #   time_list is a Collection of time points
    #       time_points is None
    #           index is None
    TDF = vdata.TemporalDataFrame(data=None, time_list=['0h', '5h'], time_col_name=None, time_points=None, index=None,
                                  name=9)
    assert repr(TDF) == "Empty TemporalDataFrame '9'\n" \
                        "\033[4mTime point : 0.0 hours\033[0m\n" \
                        "Empty DataFrame\n" \
                        "Columns: []\n" \
                        "Index: [0]\n" \
                        "\n" \
                        "[1 x 0]\n" \
                        "\n" \
                        "\033[4mTime point : 5.0 hours\033[0m\n" \
                        "Empty DataFrame\n" \
                        "Columns: []\n" \
                        "Index: [1]\n" \
                        "\n" \
                        "[1 x 0]\n\n", repr(TDF)

    #           index is a Collection of values
    TDF = vdata.TemporalDataFrame(data=None, time_list=['0h', '5h'], time_col_name=None, time_points=None,
                                  index=['a', 'b'], name=10)
    assert repr(TDF) == "Empty TemporalDataFrame '10'\n" \
                        "\033[4mTime point : 0.0 hours\033[0m\n" \
                        "Empty DataFrame\n" \
                        "Columns: []\n" \
                        "Index: [a]\n" \
                        "\n" \
                        "[1 x 0]\n" \
                        "\n" \
                        "\033[4mTime point : 5.0 hours\033[0m\n" \
                        "Empty DataFrame\n" \
                        "Columns: []\n" \
                        "Index: [b]\n" \
                        "\n" \
                        "[1 x 0]\n\n", repr(TDF)

    #       time_points is a Collection of time points
    #           index is None
    TDF = vdata.TemporalDataFrame(data=None, time_list=['0h', '5h'], time_col_name=None,
                                  time_points=['0h', '5h', '10h'], index=None, name=11)
    assert repr(TDF) == "Empty TemporalDataFrame '11'\n" \
                        "\033[4mTime point : 0.0 hours\033[0m\n" \
                        "Empty DataFrame\n" \
                        "Columns: []\n" \
                        "Index: [0]\n" \
                        "\n" \
                        "[1 x 0]\n" \
                        "\n" \
                        "\033[4mTime point : 5.0 hours\033[0m\n" \
                        "Empty DataFrame\n" \
                        "Columns: []\n" \
                        "Index: [1]\n" \
                        "\n" \
                        "[1 x 0]\n" \
                        "\n" \
                        "\033[4mTime point : 10.0 hours\033[0m\n" \
                        "Empty DataFrame\n" \
                        "Columns: []\n" \
                        "Index: []\n\n", repr(TDF)

    #           index is a Collection of values
    TDF = vdata.TemporalDataFrame(data=None, time_list=['0h', '5h'], time_col_name=None,
                                  time_points=['0h', '5h', '10h'], index=['a', 'b'], name=12)
    assert repr(TDF) == "Empty TemporalDataFrame '12'\n" \
                        "\033[4mTime point : 0.0 hours\033[0m\n" \
                        "Empty DataFrame\n" \
                        "Columns: []\n" \
                        "Index: [a]\n" \
                        "\n" \
                        "[1 x 0]\n" \
                        "\n" \
                        "\033[4mTime point : 5.0 hours\033[0m\n" \
                        "Empty DataFrame\n" \
                        "Columns: []\n" \
                        "Index: [b]\n" \
                        "\n" \
                        "[1 x 0]\n" \
                        "\n" \
                        "\033[4mTime point : 10.0 hours\033[0m\n" \
                        "Empty DataFrame\n" \
                        "Columns: []\n" \
                        "Index: []\n\n", repr(TDF)

    #   time_list is '*'
    #       time_points is None
    #           index is None
    TDF = vdata.TemporalDataFrame(data=None, time_list='*', time_col_name=None, time_points=None, index=None, name=13)
    assert repr(TDF) == "Empty TemporalDataFrame '13'\n" \
                        "\033[4mTime point : 0.0 (no unit)\033[0m\n" \
                        "Empty DataFrame\n" \
                        "Columns: []\n" \
                        "Index: [0]\n" \
                        "\n" \
                        "[1 x 0]\n\n", repr(TDF)

    #           index is a Collection of values
    TDF = vdata.TemporalDataFrame(data=None, time_list='*', time_col_name=None, time_points=None, index=['a'], name=14)
    assert repr(TDF) == "Empty TemporalDataFrame '14'\n" \
                        "\033[4mTime point : 0.0 (no unit)\033[0m\n" \
                        "Empty DataFrame\n" \
                        "Columns: []\n" \
                        "Index: [a]\n" \
                        "\n" \
                        "[1 x 0]\n\n", repr(TDF)

    #       time_points is a Collection of time points
    #           index is None
    TDF = vdata.TemporalDataFrame(data=None, time_list='*', time_col_name=None, time_points=['0h', '5h'], index=None,
                                  name=15)
    assert repr(TDF) == "Empty TemporalDataFrame '15'\n" \
                        "\033[4mTime point : 0.0 hours\033[0m\n" \
                        "Empty DataFrame\n" \
                        "Columns: []\n" \
                        "Index: [0]\n" \
                        "\n" \
                        "[1 x 0]\n" \
                        "\n" \
                        "\033[4mTime point : 5.0 hours\033[0m\n" \
                        "Empty DataFrame\n" \
                        "Columns: []\n" \
                        "Index: [0]\n" \
                        "\n" \
                        "[1 x 0]\n\n", repr(TDF)

    #           index is a Collection of values
    TDF = vdata.TemporalDataFrame(data=None, time_list='*', time_col_name=None, time_points=['0h', '5h'], index=['a'],
                                  name=16)
    assert repr(TDF) == "Empty TemporalDataFrame '16'\n" \
                        "\033[4mTime point : 0.0 hours\033[0m\n" \
                        "Empty DataFrame\n" \
                        "Columns: []\n" \
                        "Index: [a]\n" \
                        "\n" \
                        "[1 x 0]\n" \
                        "\n" \
                        "\033[4mTime point : 5.0 hours\033[0m\n" \
                        "Empty DataFrame\n" \
                        "Columns: []\n" \
                        "Index: [a]\n" \
                        "\n" \
                        "[1 x 0]\n" \
                        "\n", repr(TDF)

    #   time_list is a Collection of '*'
    #       time_points is None
    #           index is None
    TDF = vdata.TemporalDataFrame(data=None, time_list=['*', '*', '*'], time_col_name=None, time_points=None,
                                  index=None, name=17)
    assert repr(TDF) == "Empty TemporalDataFrame '17'\n" \
                        "\033[4mTime point : 0.0 (no unit)\033[0m\n" \
                        "Empty DataFrame\n" \
                        "Columns: []\n" \
                        "Index: [0, 1, 2]\n" \
                        "\n" \
                        "[3 x 0]\n\n", repr(TDF)

    #           index is a Collection of values
    TDF = vdata.TemporalDataFrame(data=None, time_list=['*', '*', '*'], time_col_name=None, time_points=None,
                                  index=['a', 'b', 'c'], name=18)
    assert repr(TDF) == "Empty TemporalDataFrame '18'\n" \
                        "\033[4mTime point : 0.0 (no unit)\033[0m\n" \
                        "Empty DataFrame\n" \
                        "Columns: []\n" \
                        "Index: [a, b, c]\n" \
                        "\n" \
                        "[3 x 0]\n\n", repr(TDF)

    #       time_points is a Collection of time points
    #           index is None
    TDF = vdata.TemporalDataFrame(data=None, time_list=['*', '*', '*'], time_col_name=None, time_points=['0h', '5h'],
                                  index=None, name=19)
    assert repr(TDF) == "Empty TemporalDataFrame '19'\n" \
                        "\033[4mTime point : 0.0 hours\033[0m\n" \
                        "Empty DataFrame\n" \
                        "Columns: []\n" \
                        "Index: [0, 1, 2]\n" \
                        "\n" \
                        "[3 x 0]\n" \
                        "\n" \
                        "\033[4mTime point : 5.0 hours\033[0m\n" \
                        "Empty DataFrame\n" \
                        "Columns: []\n" \
                        "Index: [0, 1, 2]\n" \
                        "\n" \
                        "[3 x 0]\n\n", repr(TDF)

    #           index is a Collection of values
    TDF = vdata.TemporalDataFrame(data=None, time_list=['*', '*', '*'], time_col_name=None, time_points=['0h', '5h'],
                                  index=['a', 'b', 'c'], name=20)
    assert repr(TDF) == "Empty TemporalDataFrame '20'\n" \
                        "\033[4mTime point : 0.0 hours\033[0m\n" \
                        "Empty DataFrame\n" \
                        "Columns: []\n" \
                        "Index: [a, b, c]\n" \
                        "\n" \
                        "[3 x 0]\n" \
                        "\n" \
                        "\033[4mTime point : 5.0 hours\033[0m\n" \
                        "Empty DataFrame\n" \
                        "Columns: []\n" \
                        "Index: [a, b, c]\n" \
                        "\n" \
                        "[3 x 0]\n\n", repr(TDF)

    def sub_test(_data):
        #   time_list is None
        #       time_points is None
        #           index is None
        _TDF = vdata.TemporalDataFrame(data=data, time_list=None, time_col_name=None, time_points=None, index=None,
                                      name=21)
        assert repr(_TDF) == "TemporalDataFrame '21'\n" \
                            "\033[4mTime point : 0.0 (no unit)\033[0m\n" \
                            "   col1\n" \
                            "0   1.0\n" \
                            "1   2.0\n" \
                            "2   3.0\n" \
                            "3   4.0\n" \
                            "4   5.0\n" \
                            "\n" \
                            "[9 x 1]\n\n", repr(_TDF)

        #           index is a Collection of values, same length as data
        _TDF = vdata.TemporalDataFrame(data=data, time_list=None, time_col_name=None, time_points=None,
                                      index=['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i'], name=22)
        assert repr(_TDF) == "TemporalDataFrame '22'\n" \
                            "\033[4mTime point : 0.0 (no unit)\033[0m\n" \
                            "   col1\n" \
                            "a   1.0\n" \
                            "b   2.0\n" \
                            "c   3.0\n" \
                            "d   4.0\n" \
                            "e   5.0\n" \
                            "\n" \
                            "[9 x 1]\n\n", repr(_TDF)

        #       time_points is a Collection of time points
        #           index is None
        _TDF = vdata.TemporalDataFrame(data=data, time_list=None, time_col_name=None, time_points=['0h', '5h', '10h'],
                                      index=None, name=23)
        assert repr(_TDF) == "TemporalDataFrame '23'\n" \
                            "\033[4mTime point : 0.0 hours\033[0m\n" \
                            "   col1\n" \
                            "0   1.0\n" \
                            "1   2.0\n" \
                            "2   3.0\n" \
                            "3   4.0\n" \
                            "4   5.0\n" \
                            "\n" \
                            "[9 x 1]\n" \
                            "\n" \
                            "\033[4mTime point : 5.0 hours\033[0m\n" \
                            "   col1\n" \
                            "0   1.0\n" \
                            "1   2.0\n" \
                            "2   3.0\n" \
                            "3   4.0\n" \
                            "4   5.0\n" \
                            "\n" \
                            "[9 x 1]\n" \
                            "\n" \
                            "\033[4mTime point : 10.0 hours\033[0m\n" \
                            "   col1\n" \
                            "0   1.0\n" \
                            "1   2.0\n" \
                            "2   3.0\n" \
                            "3   4.0\n" \
                            "4   5.0\n" \
                            "\n" \
                            "[9 x 1]\n\n", repr(_TDF)

        #           index is a Collection of values, same length as data
        _TDF = vdata.TemporalDataFrame(data=data, time_list=None, time_col_name=None, time_points=['0h', '5h', '10h'],
                                      index=['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i'], name=24)
        assert repr(_TDF) == "TemporalDataFrame '24'\n" \
                            "\033[4mTime point : 0.0 hours\033[0m\n" \
                            "   col1\n" \
                            "a   1.0\n" \
                            "b   2.0\n" \
                            "c   3.0\n" \
                            "d   4.0\n" \
                            "e   5.0\n" \
                            "\n" \
                            "[9 x 1]\n" \
                            "\n" \
                            "\033[4mTime point : 5.0 hours\033[0m\n" \
                            "   col1\n" \
                            "a   1.0\n" \
                            "b   2.0\n" \
                            "c   3.0\n" \
                            "d   4.0\n" \
                            "e   5.0\n" \
                            "\n" \
                            "[9 x 1]\n" \
                            "\n" \
                            "\033[4mTime point : 10.0 hours\033[0m\n" \
                            "   col1\n" \
                            "a   1.0\n" \
                            "b   2.0\n" \
                            "c   3.0\n" \
                            "d   4.0\n" \
                            "e   5.0\n" \
                            "\n" \
                            "[9 x 1]\n\n", repr(_TDF)

        #   time_list is a Collection of time points
        #       time_points is None
        #           index is None
        _TDF = vdata.TemporalDataFrame(data=data, time_list=['0h', '0h', '0h', '5h', '5h', '5h', '10h', '10h', '10h'],
                                      time_col_name=None, time_points=None, index=None, name=25)
        assert repr(_TDF) == "TemporalDataFrame '25'\n" \
                            "\033[4mTime point : 0.0 hours\033[0m\n" \
                            "   col1\n" \
                            "0   1.0\n" \
                            "1   2.0\n" \
                            "2   3.0\n" \
                            "\n" \
                            "[3 x 1]\n" \
                            "\n" \
                            "\033[4mTime point : 5.0 hours\033[0m\n" \
                            "   col1\n" \
                            "3   4.0\n" \
                            "4   5.0\n" \
                            "5   6.0\n" \
                            "\n" \
                            "[3 x 1]\n" \
                            "\n" \
                            "\033[4mTime point : 10.0 hours\033[0m\n" \
                            "   col1\n" \
                            "6   7.0\n" \
                            "7   8.0\n" \
                            "8   9.0\n" \
                            "\n" \
                            "[3 x 1]\n\n", repr(_TDF)

        #           index is a Collection of values, same length as data
        _TDF = vdata.TemporalDataFrame(data=data, time_list=['0h', '0h', '0h', '5h', '5h', '5h', '10h', '10h', '10h'],
                                      time_col_name=None, time_points=None,
                                      index=['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i'], name=26)
        assert repr(_TDF) == "TemporalDataFrame '26'\n" \
                            "\033[4mTime point : 0.0 hours\033[0m\n" \
                            "   col1\n" \
                            "a   1.0\n" \
                            "b   2.0\n" \
                            "c   3.0\n" \
                            "\n" \
                            "[3 x 1]\n" \
                            "\n" \
                            "\033[4mTime point : 5.0 hours\033[0m\n" \
                            "   col1\n" \
                            "d   4.0\n" \
                            "e   5.0\n" \
                            "f   6.0\n" \
                            "\n" \
                            "[3 x 1]\n" \
                            "\n" \
                            "\033[4mTime point : 10.0 hours\033[0m\n" \
                            "   col1\n" \
                            "g   7.0\n" \
                            "h   8.0\n" \
                            "i   9.0\n" \
                            "\n" \
                            "[3 x 1]\n\n", repr(_TDF)

        #           index is a Collection of values, divides data
        _TDF = vdata.TemporalDataFrame(data=data, time_list=['0h', '0h', '0h', '5h', '5h', '5h', '10h', '10h', '10h'],
                                      time_col_name=None, time_points=None,
                                      index=['a', 'b', 'c'], name=27)
        assert repr(_TDF) == "TemporalDataFrame '27'\n" \
                            "\033[4mTime point : 0.0 hours\033[0m\n" \
                            "   col1\n" \
                            "a   1.0\n" \
                            "b   2.0\n" \
                            "c   3.0\n" \
                            "\n" \
                            "[3 x 1]\n" \
                            "\n" \
                            "\033[4mTime point : 5.0 hours\033[0m\n" \
                            "   col1\n" \
                            "a   4.0\n" \
                            "b   5.0\n" \
                            "c   6.0\n" \
                            "\n" \
                            "[3 x 1]\n" \
                            "\n" \
                            "\033[4mTime point : 10.0 hours\033[0m\n" \
                            "   col1\n" \
                            "a   7.0\n" \
                            "b   8.0\n" \
                            "c   9.0\n" \
                            "\n" \
                            "[3 x 1]\n\n", repr(_TDF)

        #       time_points is a Collection of time points
        #           index is None
        _TDF = vdata.TemporalDataFrame(data=data, time_list=['0h', '0h', '0h', '5h', '5h', '5h', '10h', '10h', '10h'],
                                      time_col_name=None, time_points=['0h', '5h', '10h', '15h'],
                                      index=None, name=28)
        assert repr(_TDF) == "TemporalDataFrame '28'\n" \
                            "\033[4mTime point : 0.0 hours\033[0m\n" \
                            "   col1\n" \
                            "0   1.0\n" \
                            "1   2.0\n" \
                            "2   3.0\n" \
                            "\n" \
                            "[3 x 1]\n" \
                            "\n" \
                            "\033[4mTime point : 5.0 hours\033[0m\n" \
                            "   col1\n" \
                            "3   4.0\n" \
                            "4   5.0\n" \
                            "5   6.0\n" \
                            "\n" \
                            "[3 x 1]\n" \
                            "\n" \
                            "\033[4mTime point : 10.0 hours\033[0m\n" \
                            "   col1\n" \
                            "6   7.0\n" \
                            "7   8.0\n" \
                            "8   9.0\n" \
                            "\n" \
                            "[3 x 1]\n" \
                            "\n" \
                            "\033[4mTime point : 15.0 hours\033[0m\n" \
                            "Empty DataFrame\n" \
                            "Columns: ['col1']\n" \
                            "Index: []\n\n", repr(_TDF)

        #           index is a Collection of values, same length as data
        _TDF = vdata.TemporalDataFrame(data=data, time_list=['0h', '0h', '0h', '5h', '5h', '5h', '10h', '10h', '10h'],
                                      time_col_name=None, time_points=['0h', '5h', '10h', '15h'],
                                      index=['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i'], name=29)
        assert repr(_TDF) == "TemporalDataFrame '29'\n" \
                            "\033[4mTime point : 0.0 hours\033[0m\n" \
                            "   col1\n" \
                            "a   1.0\n" \
                            "b   2.0\n" \
                            "c   3.0\n" \
                            "\n" \
                            "[3 x 1]\n" \
                            "\n" \
                            "\033[4mTime point : 5.0 hours\033[0m\n" \
                            "   col1\n" \
                            "d   4.0\n" \
                            "e   5.0\n" \
                            "f   6.0\n" \
                            "\n" \
                            "[3 x 1]\n" \
                            "\n" \
                            "\033[4mTime point : 10.0 hours\033[0m\n" \
                            "   col1\n" \
                            "g   7.0\n" \
                            "h   8.0\n" \
                            "i   9.0\n" \
                            "\n" \
                            "[3 x 1]\n" \
                            "\n" \
                            "\033[4mTime point : 15.0 hours\033[0m\n" \
                            "Empty DataFrame\n" \
                            "Columns: ['col1']\n" \
                            "Index: []\n\n", repr(_TDF)

        #           index is a Collection of values, divides data
        _TDF = vdata.TemporalDataFrame(data=data, time_list=['0h', '0h', '0h', '5h', '5h', '5h', '10h', '10h', '10h'],
                                      time_col_name=None, time_points=['0h', '5h', '10h'],
                                      index=['a', 'b', 'c'], name=30)
        assert repr(_TDF) == "TemporalDataFrame '30'\n" \
                            "\033[4mTime point : 0.0 hours\033[0m\n" \
                            "   col1\n" \
                            "a   1.0\n" \
                            "b   2.0\n" \
                            "c   3.0\n" \
                            "\n" \
                            "[3 x 1]\n" \
                            "\n" \
                            "\033[4mTime point : 5.0 hours\033[0m\n" \
                            "   col1\n" \
                            "a   4.0\n" \
                            "b   5.0\n" \
                            "c   6.0\n" \
                            "\n" \
                            "[3 x 1]\n" \
                            "\n" \
                            "\033[4mTime point : 10.0 hours\033[0m\n" \
                            "   col1\n" \
                            "a   7.0\n" \
                            "b   8.0\n" \
                            "c   9.0\n" \
                            "\n" \
                            "[3 x 1]\n\n", repr(_TDF)

    # data is a dictionary
    data = {'col1': [1., 2., 3., 4., 5., 6., 7., 8., 9.]}

    sub_test(data)

    # data is a pandas DataFrame
    data = pd.DataFrame({'col1': [1., 2., 3., 4., 5., 6., 7., 8., 9.]})

    sub_test(data)


if __name__ == '__main__':
    vdata.setLoggingLevel('DEBUG')

    test_TDF_creation()
