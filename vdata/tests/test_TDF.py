# coding: utf-8
# Created on 12/9/20 10:21 AM
# Author : matteo

# ====================================================
# imports
import pandas as pd

from vdata import TemporalDataFrame, setLoggingLevel

setLoggingLevel('INFO')


# ====================================================
# code
def test_TDF_creation():
    # data is None
    #   time_list is None
    #       time_points is None
    #           index is None
    TDF = TemporalDataFrame(data=None, time_list=None, time_col=None, time_points=None, index=None)
    assert repr(TDF) == "Empty TemporalDataFrame 'No_Name'\n" \
                        "Columns: []\n" \
                        "Index: []", repr(TDF)

    #           index is a Collection of values
    TDF = TemporalDataFrame(data=None, time_list=None, time_col=None, time_points=None, index=['a', 'b', 'c'])
    assert repr(TDF) == "TemporalDataFrame 'No_Name'\n" \
                        "\033[4mTime point : 0 (no unit)\033[0m\n" \
                        "Empty DataFrame\n" \
                        "Columns: []\n" \
                        "Index: [a, b, c]\n\n", repr(TDF)

    #       time_points is a Collection of time points
    #           index is None
    TDF = TemporalDataFrame(data=None, time_list=None, time_col=None, time_points=['0h', '5h'], index=None)
    assert repr(TDF) == "TemporalDataFrame 'No_Name'\n" \
                        "\033[4mTime point : 0 hours\033[0m\n" \
                        "Empty DataFrame\n" \
                        "Columns: []\n" \
                        "Index: []\n" \
                        "\n" \
                        "\033[4mTime point : 5 hours\033[0m\n" \
                        "Empty DataFrame\n" \
                        "Columns: []\n" \
                        "Index: []\n\n", repr(TDF)

    #           index is a Collection of values
    TDF = TemporalDataFrame(data=None, time_list=None, time_col=None, time_points=['0h', '5h'], index=['a', 'b', 'c'])
    assert repr(TDF) == "TemporalDataFrame 'No_Name'\n" \
                        "\033[4mTime point : 0 hours\033[0m\n" \
                        "Empty DataFrame\n" \
                        "Columns: []\n" \
                        "Index: [a, b, c]\n" \
                        "\n" \
                        "\033[4mTime point : 5 hours\033[0m\n" \
                        "Empty DataFrame\n" \
                        "Columns: []\n" \
                        "Index: [a, b, c]\n\n", repr(TDF)

    #   time_list is a time point
    #       time_points is None
    #           index is None
    TDF = TemporalDataFrame(data=None, time_list='0h', time_col=None, time_points=None, index=None)
    assert repr(TDF) == "TemporalDataFrame 'No_Name'\n" \
                        "\033[4mTime point : 0 hours\033[0m\n" \
                        "Empty DataFrame\n" \
                        "Columns: []\n" \
                        "Index: [0]\n\n", repr(TDF)

    #           index is a Collection of values
    TDF = TemporalDataFrame(data=None, time_list='0h', time_col=None, time_points=None, index=['a'])
    assert repr(TDF) == "TemporalDataFrame 'No_Name'\n" \
                        "\033[4mTime point : 0 hours\033[0m\n" \
                        "Empty DataFrame\n" \
                        "Columns: []\n" \
                        "Index: [a]\n\n", repr(TDF)

    #       time_points is a Collection of time points
    #           index is None
    TDF = TemporalDataFrame(data=None, time_list='0h', time_col=None, time_points=['0h', '5h'], index=None)
    assert repr(TDF) == "TemporalDataFrame 'No_Name'\n" \
                        "\033[4mTime point : 0 hours\033[0m\n" \
                        "Empty DataFrame\n" \
                        "Columns: []\n" \
                        "Index: [0]\n" \
                        "\n" \
                        "\033[4mTime point : 5 hours\033[0m\n" \
                        "Empty DataFrame\n" \
                        "Columns: []\n" \
                        "Index: []\n\n", repr(TDF)

    #           index is a Collection of values
    TDF = TemporalDataFrame(data=None, time_list='0h', time_col=None, time_points=['0h', '5h'], index=['a'])
    assert repr(TDF) == "TemporalDataFrame 'No_Name'\n" \
                        "\033[4mTime point : 0 hours\033[0m\n" \
                        "Empty DataFrame\n" \
                        "Columns: []\n" \
                        "Index: [a]\n" \
                        "\n" \
                        "\033[4mTime point : 5 hours\033[0m\n" \
                        "Empty DataFrame\n" \
                        "Columns: []\n" \
                        "Index: []\n\n", repr(TDF)

    #   time_list is a Collection of time points
    #       time_points is None
    #           index is None
    TDF = TemporalDataFrame(data=None, time_list=['0h', '5h'], time_col=None, time_points=None, index=None)
    assert repr(TDF) == "TemporalDataFrame 'No_Name'\n" \
                        "\033[4mTime point : 0 hours\033[0m\n" \
                        "Empty DataFrame\n" \
                        "Columns: []\n" \
                        "Index: [0]\n" \
                        "\n" \
                        "\033[4mTime point : 5 hours\033[0m\n" \
                        "Empty DataFrame\n" \
                        "Columns: []\n" \
                        "Index: [1]\n\n", repr(TDF)

    #           index is a Collection of values
    TDF = TemporalDataFrame(data=None, time_list=['0h', '5h'], time_col=None, time_points=None, index=['a', 'b'])
    assert repr(TDF) == "TemporalDataFrame 'No_Name'\n" \
                        "\033[4mTime point : 0 hours\033[0m\n" \
                        "Empty DataFrame\n" \
                        "Columns: []\n" \
                        "Index: [a]\n" \
                        "\n" \
                        "\033[4mTime point : 5 hours\033[0m\n" \
                        "Empty DataFrame\n" \
                        "Columns: []\n" \
                        "Index: [b]\n\n", repr(TDF)

    #       time_points is a Collection of time points
    #           index is None
    TDF = TemporalDataFrame(data=None, time_list=['0h', '5h'], time_col=None, time_points=['0h', '5h', '10h'],
                            index=None)
    assert repr(TDF) == "TemporalDataFrame 'No_Name'\n" \
                        "\033[4mTime point : 0 hours\033[0m\n" \
                        "Empty DataFrame\n" \
                        "Columns: []\n" \
                        "Index: [0]\n" \
                        "\n" \
                        "\033[4mTime point : 5 hours\033[0m\n" \
                        "Empty DataFrame\n" \
                        "Columns: []\n" \
                        "Index: [1]\n\n" \
                        "\033[4mTime point : 10 hours\033[0m\n" \
                        "Empty DataFrame\n" \
                        "Columns: []\n" \
                        "Index: []\n\n", repr(TDF)

    #           index is a Collection of values
    TDF = TemporalDataFrame(data=None, time_list=['0h', '5h'], time_col=None, time_points=['0h', '5h', '10h'],
                            index=['a', 'b'])
    assert repr(TDF) == "TemporalDataFrame 'No_Name'\n" \
                        "\033[4mTime point : 0 hours\033[0m\n" \
                        "Empty DataFrame\n" \
                        "Columns: []\n" \
                        "Index: [a]\n" \
                        "\n" \
                        "\033[4mTime point : 5 hours\033[0m\n" \
                        "Empty DataFrame\n" \
                        "Columns: []\n" \
                        "Index: [b]\n\n" \
                        "\033[4mTime point : 10 hours\033[0m\n" \
                        "Empty DataFrame\n" \
                        "Columns: []\n" \
                        "Index: []\n\n", repr(TDF)

    #   time_list is '*'
    #       time_points is None
    #           index is None
    TDF = TemporalDataFrame(data=None, time_list='*', time_col=None, time_points=None, index=None)
    assert repr(TDF) == "TemporalDataFrame 'No_Name'\n" \
                        "\033[4mTime point : 0 (no unit)\033[0m\n" \
                        "Empty DataFrame\n" \
                        "Columns: []\n" \
                        "Index: [0]\n\n", repr(TDF)

    #           index is a Collection of values
    TDF = TemporalDataFrame(data=None, time_list='*', time_col=None, time_points=None, index=['a'])
    assert repr(TDF) == "TemporalDataFrame 'No_Name'\n" \
                        "\033[4mTime point : 0 (no unit)\033[0m\n" \
                        "Empty DataFrame\n" \
                        "Columns: []\n" \
                        "Index: [a]\n\n", repr(TDF)

    #       time_points is a Collection of time points
    #           index is None
    TDF = TemporalDataFrame(data=None, time_list='*', time_col=None, time_points=['0h', '5h'], index=None)
    assert repr(TDF) == "TemporalDataFrame 'No_Name'\n" \
                        "\033[4mTime point : 0 hours\033[0m\n" \
                        "Empty DataFrame\n" \
                        "Columns: []\n" \
                        "Index: [0]\n" \
                        "\n" \
                        "\033[4mTime point : 5 hours\033[0m\n" \
                        "Empty DataFrame\n" \
                        "Columns: []\n" \
                        "Index: [0]\n\n", repr(TDF)

    #           index is a Collection of values
    TDF = TemporalDataFrame(data=None, time_list='*', time_col=None, time_points=['0h', '5h'], index=['a'])
    assert repr(TDF) == "TemporalDataFrame 'No_Name'\n" \
                        "\033[4mTime point : 0 hours\033[0m\n" \
                        "Empty DataFrame\n" \
                        "Columns: []\n" \
                        "Index: [a]\n" \
                        "\n" \
                        "\033[4mTime point : 5 hours\033[0m\n" \
                        "Empty DataFrame\n" \
                        "Columns: []\n" \
                        "Index: [a]\n\n", repr(TDF)

    #   time_list is a Collection of '*'
    #       time_points is None
    #           index is None
    TDF = TemporalDataFrame(data=None, time_list=['*', '*', '*'], time_col=None, time_points=None, index=None)
    assert repr(TDF) == "TemporalDataFrame 'No_Name'\n" \
                        "\033[4mTime point : 0 (no unit)\033[0m\n" \
                        "Empty DataFrame\n" \
                        "Columns: []\n" \
                        "Index: [0, 1, 2]\n\n", repr(TDF)

    #           index is a Collection of values
    TDF = TemporalDataFrame(data=None, time_list=['*', '*', '*'], time_col=None, time_points=None,
                            index=['a', 'b', 'c'])
    assert repr(TDF) == "TemporalDataFrame 'No_Name'\n" \
                        "\033[4mTime point : 0 (no unit)\033[0m\n" \
                        "Empty DataFrame\n" \
                        "Columns: []\n" \
                        "Index: [a, b, c]\n\n", repr(TDF)

    #       time_points is a Collection of time points
    #           index is None
    TDF = TemporalDataFrame(data=None, time_list=['*', '*', '*'], time_col=None, time_points=['0h', '5h'], index=None)
    assert repr(TDF) == "TemporalDataFrame 'No_Name'\n" \
                        "\033[4mTime point : 0 hours\033[0m\n" \
                        "Empty DataFrame\n" \
                        "Columns: []\n" \
                        "Index: [0, 1, 2]\n" \
                        "\n" \
                        "\033[4mTime point : 5 hours\033[0m\n" \
                        "Empty DataFrame\n" \
                        "Columns: []\n" \
                        "Index: [0, 1, 2]\n\n", repr(TDF)

    #           index is a Collection of values
    TDF = TemporalDataFrame(data=None, time_list=['*', '*', '*'], time_col=None, time_points=['0h', '5h'],
                            index=['a', 'b', 'c'])
    assert repr(TDF) == "TemporalDataFrame 'No_Name'\n" \
                        "\033[4mTime point : 0 hours\033[0m\n" \
                        "Empty DataFrame\n" \
                        "Columns: []\n" \
                        "Index: [a, b, c]\n" \
                        "\n" \
                        "\033[4mTime point : 5 hours\033[0m\n" \
                        "Empty DataFrame\n" \
                        "Columns: []\n" \
                        "Index: [a, b, c]\n\n", repr(TDF)

    # data is a dictionary
    data = {'col1': [1, 2, 3, 4, 5, 6, 7, 8, 9]}

    #   time_list is None
    #       time_points is None
    #           index is None
    TDF = TemporalDataFrame(data=data, time_list=None, time_col=None, time_points=None, index=None)
    assert repr(TDF) == "TemporalDataFrame 'No_Name'\n" \
                        "\033[4mTime point : 0 (no unit)\033[0m\n" \
                        "   col1\n" \
                        "0   1.0\n" \
                        "1   2.0\n" \
                        "2   3.0\n" \
                        "3   4.0\n" \
                        "4   5.0\n" \
                        "5   6.0\n" \
                        "6   7.0\n" \
                        "7   8.0\n" \
                        "8   9.0\n\n", repr(TDF)

    #           index is a Collection of values, same length as data
    TDF = TemporalDataFrame(data=data, time_list=None, time_col=None, time_points=None,
                            index=['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i'])
    assert repr(TDF) == "TemporalDataFrame 'No_Name'\n" \
                        "\033[4mTime point : 0 (no unit)\033[0m\n" \
                        "   col1\n" \
                        "a   1.0\n" \
                        "b   2.0\n" \
                        "c   3.0\n" \
                        "d   4.0\n" \
                        "e   5.0\n" \
                        "f   6.0\n" \
                        "g   7.0\n" \
                        "h   8.0\n" \
                        "i   9.0\n\n", repr(TDF)

    #       time_points is a Collection of time points
    #           index is None
    TDF = TemporalDataFrame(data=data, time_list=None, time_col=None, time_points=['0h', '5h', '10h'], index=None)
    assert repr(TDF) == "TemporalDataFrame 'No_Name'\n" \
                        "\033[4mTime point : 0 hours\033[0m\n" \
                        "   col1\n" \
                        "0   1.0\n" \
                        "1   2.0\n" \
                        "2   3.0\n" \
                        "3   4.0\n" \
                        "4   5.0\n" \
                        "5   6.0\n" \
                        "6   7.0\n" \
                        "7   8.0\n" \
                        "8   9.0\n\n" \
                        "\033[4mTime point : 5 hours\033[0m\n" \
                        "   col1\n" \
                        "0   1.0\n" \
                        "1   2.0\n" \
                        "2   3.0\n" \
                        "3   4.0\n" \
                        "4   5.0\n" \
                        "5   6.0\n" \
                        "6   7.0\n" \
                        "7   8.0\n" \
                        "8   9.0\n\n" \
                        "\033[4mTime point : 10 hours\033[0m\n" \
                        "   col1\n" \
                        "0   1.0\n" \
                        "1   2.0\n" \
                        "2   3.0\n" \
                        "3   4.0\n" \
                        "4   5.0\n" \
                        "5   6.0\n" \
                        "6   7.0\n" \
                        "7   8.0\n" \
                        "8   9.0\n\n", repr(TDF)

    #           index is a Collection of values, same length as data
    TDF = TemporalDataFrame(data=data, time_list=None, time_col=None, time_points=['0h', '5h', '10h'],
                            index=['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i'])
    assert repr(TDF) == "TemporalDataFrame 'No_Name'\n" \
                        "\033[4mTime point : 0 hours\033[0m\n" \
                        "   col1\n" \
                        "a   1.0\n" \
                        "b   2.0\n" \
                        "c   3.0\n" \
                        "d   4.0\n" \
                        "e   5.0\n" \
                        "f   6.0\n" \
                        "g   7.0\n" \
                        "h   8.0\n" \
                        "i   9.0\n\n" \
                        "\033[4mTime point : 5 hours\033[0m\n" \
                        "   col1\n" \
                        "a   1.0\n" \
                        "b   2.0\n" \
                        "c   3.0\n" \
                        "d   4.0\n" \
                        "e   5.0\n" \
                        "f   6.0\n" \
                        "g   7.0\n" \
                        "h   8.0\n" \
                        "i   9.0\n\n" \
                        "\033[4mTime point : 10 hours\033[0m\n" \
                        "   col1\n" \
                        "a   1.0\n" \
                        "b   2.0\n" \
                        "c   3.0\n" \
                        "d   4.0\n" \
                        "e   5.0\n" \
                        "f   6.0\n" \
                        "g   7.0\n" \
                        "h   8.0\n" \
                        "i   9.0\n\n", repr(TDF)

    #   time_list is a Collection of time points
    #       time_points is None
    #           index is None
    TDF = TemporalDataFrame(data=data, time_list=['0h', '0h', '0h', '5h', '5h', '5h', '10h', '10h', '10h'],
                            time_col=None, time_points=None, index=None)
    assert repr(TDF) == "TemporalDataFrame 'No_Name'\n" \
                        "\033[4mTime point : 0 hours\033[0m\n" \
                        "   col1\n" \
                        "0   1.0\n" \
                        "1   2.0\n" \
                        "2   3.0\n" \
                        "\n" \
                        "\033[4mTime point : 5 hours\033[0m\n" \
                        "   col1\n" \
                        "3   4.0\n" \
                        "4   5.0\n" \
                        "5   6.0\n" \
                        "\n" \
                        "\033[4mTime point : 10 hours\033[0m\n" \
                        "   col1\n" \
                        "6   7.0\n" \
                        "7   8.0\n" \
                        "8   9.0\n\n", repr(TDF)

    #           index is a Collection of values, same length as data
    TDF = TemporalDataFrame(data=data, time_list=['0h', '0h', '0h', '5h', '5h', '5h', '10h', '10h', '10h'],
                            time_col=None, time_points=None,
                            index=['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i'])
    assert repr(TDF) == "TemporalDataFrame 'No_Name'\n" \
                        "\033[4mTime point : 0 hours\033[0m\n" \
                        "   col1\n" \
                        "a   1.0\n" \
                        "b   2.0\n" \
                        "c   3.0\n" \
                        "\n" \
                        "\033[4mTime point : 5 hours\033[0m\n" \
                        "   col1\n" \
                        "d   4.0\n" \
                        "e   5.0\n" \
                        "f   6.0\n" \
                        "\n" \
                        "\033[4mTime point : 10 hours\033[0m\n" \
                        "   col1\n" \
                        "g   7.0\n" \
                        "h   8.0\n" \
                        "i   9.0\n\n", repr(TDF)

    #           index is a Collection of values, divides data
    TDF = TemporalDataFrame(data=data, time_list=['0h', '0h', '0h', '5h', '5h', '5h', '10h', '10h', '10h'],
                            time_col=None, time_points=None,
                            index=['a', 'b', 'c'])
    assert repr(TDF) == "TemporalDataFrame 'No_Name'\n" \
                        "\033[4mTime point : 0 hours\033[0m\n" \
                        "   col1\n" \
                        "a   1.0\n" \
                        "b   2.0\n" \
                        "c   3.0\n" \
                        "\n" \
                        "\033[4mTime point : 5 hours\033[0m\n" \
                        "   col1\n" \
                        "a   4.0\n" \
                        "b   5.0\n" \
                        "c   6.0\n" \
                        "\n" \
                        "\033[4mTime point : 10 hours\033[0m\n" \
                        "   col1\n" \
                        "a   7.0\n" \
                        "b   8.0\n" \
                        "c   9.0\n\n", repr(TDF)

    #       time_points is a Collection of time points
    #           index is None
    TDF = TemporalDataFrame(data=data, time_list=['0h', '0h', '0h', '5h', '5h', '5h', '10h', '10h', '10h'],
                            time_col=None, time_points=['0h', '5h', '10h', '15h'],
                            index=None)
    assert repr(TDF) == "TemporalDataFrame 'No_Name'\n" \
                        "\033[4mTime point : 0 hours\033[0m\n" \
                        "   col1\n" \
                        "0   1.0\n" \
                        "1   2.0\n" \
                        "2   3.0\n" \
                        "\n" \
                        "\033[4mTime point : 5 hours\033[0m\n" \
                        "   col1\n" \
                        "3   4.0\n" \
                        "4   5.0\n" \
                        "5   6.0\n" \
                        "\n" \
                        "\033[4mTime point : 10 hours\033[0m\n" \
                        "   col1\n" \
                        "6   7.0\n" \
                        "7   8.0\n" \
                        "8   9.0\n" \
                        "\n" \
                        "\033[4mTime point : 15 hours\033[0m\n" \
                        "Empty DataFrame\n" \
                        "Columns: [col1]\n" \
                        "Index: []\n\n", repr(TDF)

    #           index is a Collection of values, same length as data
    TDF = TemporalDataFrame(data=data, time_list=['0h', '0h', '0h', '5h', '5h', '5h', '10h', '10h', '10h'],
                            time_col=None, time_points=['0h', '5h', '10h', '15h'],
                            index=['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i'])
    assert repr(TDF) == "TemporalDataFrame 'No_Name'\n" \
                        "\033[4mTime point : 0 hours\033[0m\n" \
                        "   col1\n" \
                        "a   1.0\n" \
                        "b   2.0\n" \
                        "c   3.0\n" \
                        "\n" \
                        "\033[4mTime point : 5 hours\033[0m\n" \
                        "   col1\n" \
                        "d   4.0\n" \
                        "e   5.0\n" \
                        "f   6.0\n" \
                        "\n" \
                        "\033[4mTime point : 10 hours\033[0m\n" \
                        "   col1\n" \
                        "g   7.0\n" \
                        "h   8.0\n" \
                        "i   9.0\n" \
                        "\n" \
                        "\033[4mTime point : 15 hours\033[0m\n" \
                        "Empty DataFrame\n" \
                        "Columns: [col1]\n" \
                        "Index: []\n\n", repr(TDF)

    #           index is a Collection of values, divides data
    TDF = TemporalDataFrame(data=data, time_list=['0h', '0h', '0h', '5h', '5h', '5h', '10h', '10h', '10h'],
                            time_col=None, time_points=['0h', '5h', '10h'],
                            index=['a', 'b', 'c'])
    assert repr(TDF) == "TemporalDataFrame 'No_Name'\n" \
                        "\033[4mTime point : 0 hours\033[0m\n" \
                        "   col1\n" \
                        "a   1.0\n" \
                        "b   2.0\n" \
                        "c   3.0\n" \
                        "\n" \
                        "\033[4mTime point : 5 hours\033[0m\n" \
                        "   col1\n" \
                        "a   4.0\n" \
                        "b   5.0\n" \
                        "c   6.0\n" \
                        "\n" \
                        "\033[4mTime point : 10 hours\033[0m\n" \
                        "   col1\n" \
                        "a   7.0\n" \
                        "b   8.0\n" \
                        "c   9.0\n\n", repr(TDF)

    # data is a pandas DataFrame
    data = pd.DataFrame()


test_TDF_creation()
