# coding: utf-8
# Created on 05/04/2022 09:11
# Author : matteo

# ====================================================
# imports
import pytest
from pathlib import Path

import numpy as np

from .utils import get_TDF, get_backed_TDF, cleanup
from vdata.name_utils import H5Mode


# ====================================================
# code
def test_delete():
    # TDF is not backed
    TDF = get_TDF('1')

    #   column numerical
    del TDF.col1

    assert repr(TDF) == "TemporalDataFrame '1'\n" \
                        "" \
                        "\x1b[4mTime point : 0.0 hours\x1b[0m\n" \
                        "   Time-point      col2    col3 col4\n" \
                        "50       0.0h  |  150.0  |  250  350\n" \
                        "51       0.0h  |  151.0  |  251  351\n" \
                        "52       0.0h  |  152.0  |  252  352\n" \
                        "53       0.0h  |  153.0  |  253  353\n" \
                        "54       0.0h  |  154.0  |  254  354\n" \
                        "[50 x 3]\n" \
                        "\n" \
                        "\x1b[4mTime point : 1.0 hours\x1b[0m\n" \
                        "  Time-point      col2    col3 col4\n" \
                        "0       1.0h  |  100.0  |  200  300\n" \
                        "1       1.0h  |  101.0  |  201  301\n" \
                        "2       1.0h  |  102.0  |  202  302\n" \
                        "3       1.0h  |  103.0  |  203  303\n" \
                        "4       1.0h  |  104.0  |  204  304\n" \
                        "[50 x 3]\n\n"

    assert np.all(TDF.values_num == np.concatenate((np.arange(150, 200), np.arange(100, 150)))[:, None])
    assert np.all(TDF.values_str == np.vstack((
        np.concatenate((np.arange(250, 300).astype(str), np.arange(200, 250).astype(str))),
        np.concatenate((np.arange(350, 400).astype(str), np.arange(300, 350).astype(str)))
    )).T)

    #   column string
    del TDF.col4

    assert repr(TDF) == "TemporalDataFrame '1'\n" \
                        "\x1b[4mTime point : 0.0 hours\x1b[0m\n" \
                        "   Time-point      col2    col3\n" \
                        "50       0.0h  |  150.0  |  250\n" \
                        "51       0.0h  |  151.0  |  251\n" \
                        "52       0.0h  |  152.0  |  252\n" \
                        "53       0.0h  |  153.0  |  253\n" \
                        "54       0.0h  |  154.0  |  254\n" \
                        "[50 x 2]\n" \
                        "\n" \
                        "\x1b[4mTime point : 1.0 hours\x1b[0m\n" \
                        "  Time-point      col2    col3\n" \
                        "0       1.0h  |  100.0  |  200\n" \
                        "1       1.0h  |  101.0  |  201\n" \
                        "2       1.0h  |  102.0  |  202\n" \
                        "3       1.0h  |  103.0  |  203\n" \
                        "4       1.0h  |  104.0  |  204\n" \
                        "[50 x 2]\n\n"

    assert np.all(TDF.values_num == np.concatenate((np.arange(150, 200), np.arange(100, 150)))[:, None])
    assert np.all(TDF.values_str ==
                  np.concatenate((np.arange(250, 300).astype(str), np.arange(200, 250).astype(str)))[:, None])

    # TDF is backed
    input_file = Path(__file__).parent / 'test_modification_TDF'
    cleanup([input_file])

    TDF = get_backed_TDF(input_file, '2', mode=H5Mode.READ_WRITE)
    TDF.unlock_columns()
    #   column numerical
    del TDF.col2

    assert repr(TDF) == "Backed TemporalDataFrame '2'\n" \
                        "\x1b[4mTime point : 0.0 hours\x1b[0m\n" \
                        "  Time-point    col1    col3\n" \
                        "0       0.0h  |  0.0  |  100\n" \
                        "1       0.0h  |  2.0  |  101\n" \
                        "2       0.0h  |  4.0  |  102\n" \
                        "3       0.0h  |  6.0  |  103\n" \
                        "4       0.0h  |  8.0  |  104\n" \
                        "[25 x 2]\n" \
                        "\n" \
                        "\x1b[4mTime point : 1.0 hours\x1b[0m\n" \
                        "   Time-point     col1    col3\n" \
                        "25       1.0h  |  50.0  |  125\n" \
                        "26       1.0h  |  52.0  |  126\n" \
                        "27       1.0h  |  54.0  |  127\n" \
                        "28       1.0h  |  56.0  |  128\n" \
                        "29       1.0h  |  58.0  |  129\n" \
                        "[25 x 2]\n\n"

    assert np.all(TDF.values_num == np.arange(0, 100, 2)[:, None])
    assert np.all(TDF.values_str == np.arange(100, 150).astype(str)[:, None])

    #   column string
    del TDF.col3

    assert repr(TDF) == "Backed TemporalDataFrame '2'\n" \
                        "\x1b[4mTime point : 0.0 hours\x1b[0m\n" \
                        "  Time-point    col1\n" \
                        "0       0.0h  |  0.0\n" \
                        "1       0.0h  |  2.0\n" \
                        "2       0.0h  |  4.0\n" \
                        "3       0.0h  |  6.0\n" \
                        "4       0.0h  |  8.0\n" \
                        "[25 x 1]\n" \
                        "\n" \
                        "\x1b[4mTime point : 1.0 hours\x1b[0m\n" \
                        "   Time-point     col1\n" \
                        "25       1.0h  |  50.0\n" \
                        "26       1.0h  |  52.0\n" \
                        "27       1.0h  |  54.0\n" \
                        "28       1.0h  |  56.0\n" \
                        "29       1.0h  |  58.0\n" \
                        "[25 x 1]\n\n"

    assert np.all(TDF.values_num == np.arange(0, 100, 2)[:, None])
    assert TDF.values_str.size == 0

    cleanup([input_file])


def test_append():
    # TDF is not backed
    TDF = get_TDF('1')

    #   column numerical
    #       column exists
    TDF.col1 = np.arange(500, 600)

    assert repr(TDF) == "TemporalDataFrame '1'\n" \
                        "\x1b[4mTime point : 0.0 hours\x1b[0m\n" \
                        "   Time-point      col1   col2    col3 col4\n" \
                        "50       0.0h  |  500.0  150.0  |  250  350\n" \
                        "51       0.0h  |  501.0  151.0  |  251  351\n" \
                        "52       0.0h  |  502.0  152.0  |  252  352\n" \
                        "53       0.0h  |  503.0  153.0  |  253  353\n" \
                        "54       0.0h  |  504.0  154.0  |  254  354\n" \
                        "[50 x 4]\n" \
                        "\n" \
                        "\x1b[4mTime point : 1.0 hours\x1b[0m\n" \
                        "  Time-point      col1   col2    col3 col4\n" \
                        "0       1.0h  |  550.0  100.0  |  200  300\n" \
                        "1       1.0h  |  551.0  101.0  |  201  301\n" \
                        "2       1.0h  |  552.0  102.0  |  202  302\n" \
                        "3       1.0h  |  553.0  103.0  |  203  303\n" \
                        "4       1.0h  |  554.0  104.0  |  204  304\n" \
                        "[50 x 4]\n\n"

    #       column new
    TDF.col7 = np.arange(600, 700)

    assert repr(TDF) == "TemporalDataFrame '1'\n" \
                        "\x1b[4mTime point : 0.0 hours\x1b[0m\n" \
                        "   Time-point      col1   col2   col7    col3 col4\n" \
                        "50       0.0h  |  500.0  150.0  600.0  |  250  350\n" \
                        "51       0.0h  |  501.0  151.0  601.0  |  251  351\n" \
                        "52       0.0h  |  502.0  152.0  602.0  |  252  352\n" \
                        "53       0.0h  |  503.0  153.0  603.0  |  253  353\n" \
                        "54       0.0h  |  504.0  154.0  604.0  |  254  354\n" \
                        "[50 x 5]\n" \
                        "\n" \
                        "\x1b[4mTime point : 1.0 hours\x1b[0m\n" \
                        "  Time-point      col1   col2   col7    col3 col4\n" \
                        "0       1.0h  |  550.0  100.0  650.0  |  200  300\n" \
                        "1       1.0h  |  551.0  101.0  651.0  |  201  301\n" \
                        "2       1.0h  |  552.0  102.0  652.0  |  202  302\n" \
                        "3       1.0h  |  553.0  103.0  653.0  |  203  303\n" \
                        "4       1.0h  |  554.0  104.0  654.0  |  204  304\n" \
                        "[50 x 5]\n\n"

    #   column string
    #       column exists
    TDF.col4 = np.arange(700, 800).astype(str)

    assert repr(TDF) == "TemporalDataFrame '1'\n" \
                        "\x1b[4mTime point : 0.0 hours\x1b[0m\n" \
                        "   Time-point      col1   col2   col7    col3 col4\n" \
                        "50       0.0h  |  500.0  150.0  600.0  |  250  700\n" \
                        "51       0.0h  |  501.0  151.0  601.0  |  251  701\n" \
                        "52       0.0h  |  502.0  152.0  602.0  |  252  702\n" \
                        "53       0.0h  |  503.0  153.0  603.0  |  253  703\n" \
                        "54       0.0h  |  504.0  154.0  604.0  |  254  704\n" \
                        "[50 x 5]\n" \
                        "\n" \
                        "\x1b[4mTime point : 1.0 hours\x1b[0m\n" \
                        "  Time-point      col1   col2   col7    col3 col4\n" \
                        "0       1.0h  |  550.0  100.0  650.0  |  200  750\n" \
                        "1       1.0h  |  551.0  101.0  651.0  |  201  751\n" \
                        "2       1.0h  |  552.0  102.0  652.0  |  202  752\n" \
                        "3       1.0h  |  553.0  103.0  653.0  |  203  753\n" \
                        "4       1.0h  |  554.0  104.0  654.0  |  204  754\n" \
                        "[50 x 5]\n\n"

    #       column new
    TDF.col8 = np.arange(800, 900).astype(str)

    assert repr(TDF) == "TemporalDataFrame '1'\n" \
                        "\x1b[4mTime point : 0.0 hours\x1b[0m\n" \
                        "   Time-point      col1   col2   col7    col3 col4 col8\n" \
                        "50       0.0h  |  500.0  150.0  600.0  |  250  700  800\n" \
                        "51       0.0h  |  501.0  151.0  601.0  |  251  701  801\n" \
                        "52       0.0h  |  502.0  152.0  602.0  |  252  702  802\n" \
                        "53       0.0h  |  503.0  153.0  603.0  |  253  703  803\n" \
                        "54       0.0h  |  504.0  154.0  604.0  |  254  704  804\n" \
                        "[50 x 6]\n" \
                        "\n" \
                        "\x1b[4mTime point : 1.0 hours\x1b[0m\n" \
                        "  Time-point      col1   col2   col7    col3 col4 col8\n" \
                        "0       1.0h  |  550.0  100.0  650.0  |  200  750  850\n" \
                        "1       1.0h  |  551.0  101.0  651.0  |  201  751  851\n" \
                        "2       1.0h  |  552.0  102.0  652.0  |  202  752  852\n" \
                        "3       1.0h  |  553.0  103.0  653.0  |  203  753  853\n" \
                        "4       1.0h  |  554.0  104.0  654.0  |  204  754  854\n" \
                        "[50 x 6]\n\n"

    # TDF is backed
    input_file = Path(__file__).parent / 'test_modification_TDF'
    cleanup([input_file])

    TDF = get_backed_TDF(input_file, '2', mode=H5Mode.READ_WRITE)

    #   column numerical
    #       column exists
    TDF.col1 = np.arange(500, 550)

    assert repr(TDF) == "Backed TemporalDataFrame '2'\n" \
                        "\x1b[4mTime point : 0.0 hours\x1b[0m\n" \
                        "  Time-point      col1 col2    col3\n" \
                        "0       0.0h  |  500.0  1.0  |  100\n" \
                        "1       0.0h  |  501.0  3.0  |  101\n" \
                        "2       0.0h  |  502.0  5.0  |  102\n" \
                        "3       0.0h  |  503.0  7.0  |  103\n" \
                        "4       0.0h  |  504.0  9.0  |  104\n" \
                        "[25 x 3]\n" \
                        "\n" \
                        "\x1b[4mTime point : 1.0 hours\x1b[0m\n" \
                        "   Time-point      col1  col2    col3\n" \
                        "25       1.0h  |  525.0  51.0  |  125\n" \
                        "26       1.0h  |  526.0  53.0  |  126\n" \
                        "27       1.0h  |  527.0  55.0  |  127\n" \
                        "28       1.0h  |  528.0  57.0  |  128\n" \
                        "29       1.0h  |  529.0  59.0  |  129\n" \
                        "[25 x 3]\n\n"

    #       column new
    TDF.col4 = np.arange(600, 650)

    assert repr(TDF) == "Backed TemporalDataFrame '2'\n" \
                        "\x1b[4mTime point : 0.0 hours\x1b[0m\n" \
                        "  Time-point      col1 col2   col4    col3\n" \
                        "0       0.0h  |  500.0  1.0  600.0  |  100\n" \
                        "1       0.0h  |  501.0  3.0  601.0  |  101\n" \
                        "2       0.0h  |  502.0  5.0  602.0  |  102\n" \
                        "3       0.0h  |  503.0  7.0  603.0  |  103\n" \
                        "4       0.0h  |  504.0  9.0  604.0  |  104\n" \
                        "[25 x 4]\n" \
                        "\n" \
                        "\x1b[4mTime point : 1.0 hours\x1b[0m\n" \
                        "   Time-point      col1  col2   col4    col3\n" \
                        "25       1.0h  |  525.0  51.0  625.0  |  125\n" \
                        "26       1.0h  |  526.0  53.0  626.0  |  126\n" \
                        "27       1.0h  |  527.0  55.0  627.0  |  127\n" \
                        "28       1.0h  |  528.0  57.0  628.0  |  128\n" \
                        "29       1.0h  |  529.0  59.0  629.0  |  129\n" \
                        "[25 x 4]\n\n"

    #   column string
    #       column exists
    TDF.col3 = np.arange(700, 750).astype(str)

    assert repr(TDF) == "Backed TemporalDataFrame '2'\n" \
                        "\x1b[4mTime point : 0.0 hours\x1b[0m\n" \
                        "  Time-point      col1 col2   col4    col3\n" \
                        "0       0.0h  |  500.0  1.0  600.0  |  700\n" \
                        "1       0.0h  |  501.0  3.0  601.0  |  701\n" \
                        "2       0.0h  |  502.0  5.0  602.0  |  702\n" \
                        "3       0.0h  |  503.0  7.0  603.0  |  703\n" \
                        "4       0.0h  |  504.0  9.0  604.0  |  704\n" \
                        "[25 x 4]\n" \
                        "\n" \
                        "\x1b[4mTime point : 1.0 hours\x1b[0m\n" \
                        "   Time-point      col1  col2   col4    col3\n" \
                        "25       1.0h  |  525.0  51.0  625.0  |  725\n" \
                        "26       1.0h  |  526.0  53.0  626.0  |  726\n" \
                        "27       1.0h  |  527.0  55.0  627.0  |  727\n" \
                        "28       1.0h  |  528.0  57.0  628.0  |  728\n" \
                        "29       1.0h  |  529.0  59.0  629.0  |  729\n" \
                        "[25 x 4]\n\n"

    #       column new
    TDF.col5 = np.arange(800, 850).astype(str)

    assert repr(TDF) == "Backed TemporalDataFrame '2'\n" \
                        "\x1b[4mTime point : 0.0 hours\x1b[0m\n" \
                        "  Time-point      col1 col2   col4    col3 col5\n" \
                        "0       0.0h  |  500.0  1.0  600.0  |  700  800\n" \
                        "1       0.0h  |  501.0  3.0  601.0  |  701  801\n" \
                        "2       0.0h  |  502.0  5.0  602.0  |  702  802\n" \
                        "3       0.0h  |  503.0  7.0  603.0  |  703  803\n" \
                        "4       0.0h  |  504.0  9.0  604.0  |  704  804\n" \
                        "[25 x 5]\n" \
                        "\n" \
                        "\x1b[4mTime point : 1.0 hours\x1b[0m\n" \
                        "   Time-point      col1  col2   col4    col3 col5\n" \
                        "25       1.0h  |  525.0  51.0  625.0  |  725  825\n" \
                        "26       1.0h  |  526.0  53.0  626.0  |  726  826\n" \
                        "27       1.0h  |  527.0  55.0  627.0  |  727  827\n" \
                        "28       1.0h  |  528.0  57.0  628.0  |  728  828\n" \
                        "29       1.0h  |  529.0  59.0  629.0  |  729  829\n" \
                        "[25 x 5]\n\n"

    cleanup([input_file])


def test_insert():
    # TDF is not backed
    TDF = get_TDF('1')

    #   column numerical
    TDF.insert(1, 'col5', np.arange(500, 600))

    assert repr(TDF) == "TemporalDataFrame '1'\n" \
                        "\x1b[4mTime point : 0.0 hours\x1b[0m\n" \
                        "   Time-point     col1   col5   col2    col3 col4\n" \
                        "50       0.0h  |  50.0  500.0  150.0  |  250  350\n" \
                        "51       0.0h  |  51.0  501.0  151.0  |  251  351\n" \
                        "52       0.0h  |  52.0  502.0  152.0  |  252  352\n" \
                        "53       0.0h  |  53.0  503.0  153.0  |  253  353\n" \
                        "54       0.0h  |  54.0  504.0  154.0  |  254  354\n" \
                        "[50 x 5]\n" \
                        "\n" \
                        "\x1b[4mTime point : 1.0 hours\x1b[0m\n" \
                        "  Time-point    col1   col5   col2    col3 col4\n" \
                        "0       1.0h  |  0.0  550.0  100.0  |  200  300\n" \
                        "1       1.0h  |  1.0  551.0  101.0  |  201  301\n" \
                        "2       1.0h  |  2.0  552.0  102.0  |  202  302\n" \
                        "3       1.0h  |  3.0  553.0  103.0  |  203  303\n" \
                        "4       1.0h  |  4.0  554.0  104.0  |  204  304\n" \
                        "[50 x 5]\n\n"

    #   column string
    TDF.insert(2, 'col6', np.arange(700, 800).astype(str))

    assert repr(TDF) == "TemporalDataFrame '1'\n" \
                        "\x1b[4mTime point : 0.0 hours\x1b[0m\n" \
                        "   Time-point     col1   col5   col2    col3 col4 col6\n" \
                        "50       0.0h  |  50.0  500.0  150.0  |  250  350  700\n" \
                        "51       0.0h  |  51.0  501.0  151.0  |  251  351  701\n" \
                        "52       0.0h  |  52.0  502.0  152.0  |  252  352  702\n" \
                        "53       0.0h  |  53.0  503.0  153.0  |  253  353  703\n" \
                        "54       0.0h  |  54.0  504.0  154.0  |  254  354  704\n" \
                        "[50 x 6]\n" \
                        "\n" \
                        "\x1b[4mTime point : 1.0 hours\x1b[0m\n" \
                        "  Time-point    col1   col5   col2    col3 col4 col6\n" \
                        "0       1.0h  |  0.0  550.0  100.0  |  200  300  750\n" \
                        "1       1.0h  |  1.0  551.0  101.0  |  201  301  751\n" \
                        "2       1.0h  |  2.0  552.0  102.0  |  202  302  752\n" \
                        "3       1.0h  |  3.0  553.0  103.0  |  203  303  753\n" \
                        "4       1.0h  |  4.0  554.0  104.0  |  204  304  754\n" \
                        "[50 x 6]\n\n"

    #   array is empty
    del TDF.col1
    del TDF.col5
    del TDF.col2

    TDF.insert(0, 'col1', np.arange(800, 900))

    assert repr(TDF) == "TemporalDataFrame '1'\n" \
                        "\x1b[4mTime point : 0.0 hours\x1b[0m\n" \
                        "   Time-point      col1    col3 col4 col6\n" \
                        "50       0.0h  |  800.0  |  250  350  700\n" \
                        "51       0.0h  |  801.0  |  251  351  701\n" \
                        "52       0.0h  |  802.0  |  252  352  702\n" \
                        "53       0.0h  |  803.0  |  253  353  703\n" \
                        "54       0.0h  |  804.0  |  254  354  704\n" \
                        "[50 x 4]\n" \
                        "\n" \
                        "\x1b[4mTime point : 1.0 hours\x1b[0m\n" \
                        "  Time-point      col1    col3 col4 col6\n" \
                        "0       1.0h  |  850.0  |  200  300  750\n" \
                        "1       1.0h  |  851.0  |  201  301  751\n" \
                        "2       1.0h  |  852.0  |  202  302  752\n" \
                        "3       1.0h  |  853.0  |  203  303  753\n" \
                        "4       1.0h  |  854.0  |  204  304  754\n" \
                        "[50 x 4]\n\n"

    # TDF is backed
    input_file = Path(__file__).parent / 'test_insertion_TDF'
    cleanup([input_file])

    TDF = get_backed_TDF(input_file, '2', mode=H5Mode.READ_WRITE)
    TDF.unlock_columns()

    #   column numerical
    TDF.insert(0, 'col4', np.arange(500, 550))

    assert repr(TDF) == "Backed TemporalDataFrame '2'\n" \
                        "\x1b[4mTime point : 0.0 hours\x1b[0m\n" \
                        "  Time-point      col4 col1 col2    col3\n" \
                        "0       0.0h  |  500.0  0.0  1.0  |  100\n" \
                        "1       0.0h  |  501.0  2.0  3.0  |  101\n" \
                        "2       0.0h  |  502.0  4.0  5.0  |  102\n" \
                        "3       0.0h  |  503.0  6.0  7.0  |  103\n" \
                        "4       0.0h  |  504.0  8.0  9.0  |  104\n" \
                        "[25 x 4]\n" \
                        "\n" \
                        "\x1b[4mTime point : 1.0 hours\x1b[0m\n" \
                        "   Time-point      col4  col1  col2    col3\n" \
                        "25       1.0h  |  525.0  50.0  51.0  |  125\n" \
                        "26       1.0h  |  526.0  52.0  53.0  |  126\n" \
                        "27       1.0h  |  527.0  54.0  55.0  |  127\n" \
                        "28       1.0h  |  528.0  56.0  57.0  |  128\n" \
                        "29       1.0h  |  529.0  58.0  59.0  |  129\n" \
                        "[25 x 4]\n\n"

    #   column string
    TDF.insert(1, 'col5', np.arange(700, 750).astype(str))

    assert repr(TDF) == "Backed TemporalDataFrame '2'\n" \
                        "\x1b[4mTime point : 0.0 hours\x1b[0m\n" \
                        "  Time-point      col4 col1 col2    col3 col5\n" \
                        "0       0.0h  |  500.0  0.0  1.0  |  100  700\n" \
                        "1       0.0h  |  501.0  2.0  3.0  |  101  701\n" \
                        "2       0.0h  |  502.0  4.0  5.0  |  102  702\n" \
                        "3       0.0h  |  503.0  6.0  7.0  |  103  703\n" \
                        "4       0.0h  |  504.0  8.0  9.0  |  104  704\n" \
                        "[25 x 5]\n" \
                        "\n" \
                        "\x1b[4mTime point : 1.0 hours\x1b[0m\n" \
                        "   Time-point      col4  col1  col2    col3 col5\n" \
                        "25       1.0h  |  525.0  50.0  51.0  |  125  725\n" \
                        "26       1.0h  |  526.0  52.0  53.0  |  126  726\n" \
                        "27       1.0h  |  527.0  54.0  55.0  |  127  727\n" \
                        "28       1.0h  |  528.0  56.0  57.0  |  128  728\n" \
                        "29       1.0h  |  529.0  58.0  59.0  |  129  729\n" \
                        "[25 x 5]\n\n"

    #   array is empty
    del TDF.col1
    del TDF.col2
    del TDF.col4

    TDF.insert(0, 'col1', np.arange(800, 850))

    assert repr(TDF) == "Backed TemporalDataFrame '2'\n" \
                        "\x1b[4mTime point : 0.0 hours\x1b[0m\n" \
                        "  Time-point      col1    col3 col5\n" \
                        "0       0.0h  |  800.0  |  100  700\n" \
                        "1       0.0h  |  801.0  |  101  701\n" \
                        "2       0.0h  |  802.0  |  102  702\n" \
                        "3       0.0h  |  803.0  |  103  703\n" \
                        "4       0.0h  |  804.0  |  104  704\n" \
                        "[25 x 3]\n" \
                        "\n" \
                        "\x1b[4mTime point : 1.0 hours\x1b[0m\n" \
                        "   Time-point      col1    col3 col5\n" \
                        "25       1.0h  |  825.0  |  125  725\n" \
                        "26       1.0h  |  826.0  |  126  726\n" \
                        "27       1.0h  |  827.0  |  127  727\n" \
                        "28       1.0h  |  828.0  |  128  728\n" \
                        "29       1.0h  |  829.0  |  129  729\n" \
                        "[25 x 3]\n\n"

    cleanup([input_file])


def test_add_sub_mul_div():
    # TDF is not backed
    TDF = get_TDF('1')

    #   add
    #       numerical value
    assert repr(TDF + 1) == "TemporalDataFrame '1 + 1'\n" \
                            "\x1b[4mTime point : 0.0 hours\x1b[0m\n" \
                            "   Time-point     col1   col2    col3 col4\n" \
                            "50       0.0h  |  51.0  151.0  |  250  350\n" \
                            "51       0.0h  |  52.0  152.0  |  251  351\n" \
                            "52       0.0h  |  53.0  153.0  |  252  352\n" \
                            "53       0.0h  |  54.0  154.0  |  253  353\n" \
                            "54       0.0h  |  55.0  155.0  |  254  354\n" \
                            "[50 x 4]\n" \
                            "\n" \
                            "\x1b[4mTime point : 1.0 hours\x1b[0m\n" \
                            "  Time-point    col1   col2    col3 col4\n" \
                            "0       1.0h  |  1.0  101.0  |  200  300\n" \
                            "1       1.0h  |  2.0  102.0  |  201  301\n" \
                            "2       1.0h  |  3.0  103.0  |  202  302\n" \
                            "3       1.0h  |  4.0  104.0  |  203  303\n" \
                            "4       1.0h  |  5.0  105.0  |  204  304\n" \
                            "[50 x 4]\n\n"

    #       string value
    assert repr(TDF + '_1') == "TemporalDataFrame '1 + _1'\n" \
                               "\x1b[4mTime point : 0.0 hours\x1b[0m\n" \
                               "   Time-point     col1   col2      col3   col4\n" \
                               "50       0.0h  |  50.0  150.0  |  250_1  350_1\n" \
                               "51       0.0h  |  51.0  151.0  |  251_1  351_1\n" \
                               "52       0.0h  |  52.0  152.0  |  252_1  352_1\n" \
                               "53       0.0h  |  53.0  153.0  |  253_1  353_1\n" \
                               "54       0.0h  |  54.0  154.0  |  254_1  354_1\n" \
                               "[50 x 4]\n" \
                               "\n" \
                               "\x1b[4mTime point : 1.0 hours\x1b[0m\n" \
                               "  Time-point    col1   col2      col3   col4\n" \
                               "0       1.0h  |  0.0  100.0  |  200_1  300_1\n" \
                               "1       1.0h  |  1.0  101.0  |  201_1  301_1\n" \
                               "2       1.0h  |  2.0  102.0  |  202_1  302_1\n" \
                               "3       1.0h  |  3.0  103.0  |  203_1  303_1\n" \
                               "4       1.0h  |  4.0  104.0  |  204_1  304_1\n" \
                               "[50 x 4]\n\n"

    #   sub
    assert repr(TDF - 1) == "TemporalDataFrame '1 - 1'\n" \
                            "\x1b[4mTime point : 0.0 hours\x1b[0m\n" \
                            "   Time-point     col1   col2    col3 col4\n" \
                            "50       0.0h  |  49.0  149.0  |  250  350\n" \
                            "51       0.0h  |  50.0  150.0  |  251  351\n" \
                            "52       0.0h  |  51.0  151.0  |  252  352\n" \
                            "53       0.0h  |  52.0  152.0  |  253  353\n" \
                            "54       0.0h  |  53.0  153.0  |  254  354\n" \
                            "[50 x 4]\n" \
                            "\n" \
                            "\x1b[4mTime point : 1.0 hours\x1b[0m\n" \
                            "  Time-point    col1   col2    col3 col4\n" \
                            "0       1.0h  | -1.0   99.0  |  200  300\n" \
                            "1       1.0h  |  0.0  100.0  |  201  301\n" \
                            "2       1.0h  |  1.0  101.0  |  202  302\n" \
                            "3       1.0h  |  2.0  102.0  |  203  303\n" \
                            "4       1.0h  |  3.0  103.0  |  204  304\n" \
                            "[50 x 4]\n\n"

    #   mul
    assert repr(TDF * 2) == "TemporalDataFrame '1 * 2'\n" \
                            "\x1b[4mTime point : 0.0 hours\x1b[0m\n" \
                            "   Time-point      col1   col2    col3 col4\n" \
                            "50       0.0h  |  100.0  300.0  |  250  350\n" \
                            "51       0.0h  |  102.0  302.0  |  251  351\n" \
                            "52       0.0h  |  104.0  304.0  |  252  352\n" \
                            "53       0.0h  |  106.0  306.0  |  253  353\n" \
                            "54       0.0h  |  108.0  308.0  |  254  354\n" \
                            "[50 x 4]\n" \
                            "\n" \
                            "\x1b[4mTime point : 1.0 hours\x1b[0m\n" \
                            "  Time-point    col1   col2    col3 col4\n" \
                            "0       1.0h  |  0.0  200.0  |  200  300\n" \
                            "1       1.0h  |  2.0  202.0  |  201  301\n" \
                            "2       1.0h  |  4.0  204.0  |  202  302\n" \
                            "3       1.0h  |  6.0  206.0  |  203  303\n" \
                            "4       1.0h  |  8.0  208.0  |  204  304\n" \
                            "[50 x 4]\n\n"

    #   div
    assert repr(TDF / 2) == "TemporalDataFrame '1 / 2'\n" \
                            "\x1b[4mTime point : 0.0 hours\x1b[0m\n" \
                            "   Time-point     col1  col2    col3 col4\n" \
                            "50       0.0h  |  25.0  75.0  |  250  350\n" \
                            "51       0.0h  |  25.5  75.5  |  251  351\n" \
                            "52       0.0h  |  26.0  76.0  |  252  352\n" \
                            "53       0.0h  |  26.5  76.5  |  253  353\n" \
                            "54       0.0h  |  27.0  77.0  |  254  354\n" \
                            "[50 x 4]\n" \
                            "\n" \
                            "\x1b[4mTime point : 1.0 hours\x1b[0m\n" \
                            "  Time-point    col1  col2    col3 col4\n" \
                            "0       1.0h  |  0.0  50.0  |  200  300\n" \
                            "1       1.0h  |  0.5  50.5  |  201  301\n" \
                            "2       1.0h  |  1.0  51.0  |  202  302\n" \
                            "3       1.0h  |  1.5  51.5  |  203  303\n" \
                            "4       1.0h  |  2.0  52.0  |  204  304\n" \
                            "[50 x 4]\n\n"

    # add to empty array
    del TDF.col1
    del TDF.col2

    with pytest.raises(ValueError) as exc_info:
        TDF + 1

    assert str(exc_info.value) == "No numerical data to add."

    # sub in empty array
    with pytest.raises(ValueError) as exc_info:
        TDF - 1

    assert str(exc_info.value) == "No numerical data to subtract."

    # TDF is backed
    input_file = Path(__file__).parent / 'test_insertion_TDF'
    cleanup([input_file])

    TDF = get_backed_TDF(input_file, '2', mode=H5Mode.READ)

    #   add
    #       numerical value
    assert repr(TDF + 1) == "TemporalDataFrame '2 + 1'\n" \
                            "\x1b[4mTime point : 0.0 hours\x1b[0m\n" \
                            "  Time-point    col1  col2    col3\n" \
                            "0       0.0h  |  1.0   2.0  |  100\n" \
                            "1       0.0h  |  3.0   4.0  |  101\n" \
                            "2       0.0h  |  5.0   6.0  |  102\n" \
                            "3       0.0h  |  7.0   8.0  |  103\n" \
                            "4       0.0h  |  9.0  10.0  |  104\n" \
                            "[25 x 3]\n" \
                            "\n" \
                            "\x1b[4mTime point : 1.0 hours\x1b[0m\n" \
                            "   Time-point     col1  col2    col3\n" \
                            "25       1.0h  |  51.0  52.0  |  125\n" \
                            "26       1.0h  |  53.0  54.0  |  126\n" \
                            "27       1.0h  |  55.0  56.0  |  127\n" \
                            "28       1.0h  |  57.0  58.0  |  128\n" \
                            "29       1.0h  |  59.0  60.0  |  129\n" \
                            "[25 x 3]\n\n"

    #       string value
    assert repr(TDF + '_1') == "TemporalDataFrame '2 + _1'\n" \
                               "\x1b[4mTime point : 0.0 hours\x1b[0m\n" \
                               "  Time-point    col1 col2      col3\n" \
                               "0       0.0h  |  0.0  1.0  |  100_1\n" \
                               "1       0.0h  |  2.0  3.0  |  101_1\n" \
                               "2       0.0h  |  4.0  5.0  |  102_1\n" \
                               "3       0.0h  |  6.0  7.0  |  103_1\n" \
                               "4       0.0h  |  8.0  9.0  |  104_1\n" \
                               "[25 x 3]\n" \
                               "\n" \
                               "\x1b[4mTime point : 1.0 hours\x1b[0m\n" \
                               "   Time-point     col1  col2      col3\n" \
                               "25       1.0h  |  50.0  51.0  |  125_1\n" \
                               "26       1.0h  |  52.0  53.0  |  126_1\n" \
                               "27       1.0h  |  54.0  55.0  |  127_1\n" \
                               "28       1.0h  |  56.0  57.0  |  128_1\n" \
                               "29       1.0h  |  58.0  59.0  |  129_1\n" \
                               "[25 x 3]\n\n"

    # sub
    assert repr(TDF - 1) == "TemporalDataFrame '2 - 1'\n" \
                            "\x1b[4mTime point : 0.0 hours\x1b[0m\n" \
                            "  Time-point    col1 col2    col3\n" \
                            "0       0.0h  | -1.0  0.0  |  100\n" \
                            "1       0.0h  |  1.0  2.0  |  101\n" \
                            "2       0.0h  |  3.0  4.0  |  102\n" \
                            "3       0.0h  |  5.0  6.0  |  103\n" \
                            "4       0.0h  |  7.0  8.0  |  104\n" \
                            "[25 x 3]\n" \
                            "\n" \
                            "\x1b[4mTime point : 1.0 hours\x1b[0m\n" \
                            "   Time-point     col1  col2    col3\n" \
                            "25       1.0h  |  49.0  50.0  |  125\n" \
                            "26       1.0h  |  51.0  52.0  |  126\n" \
                            "27       1.0h  |  53.0  54.0  |  127\n" \
                            "28       1.0h  |  55.0  56.0  |  128\n" \
                            "29       1.0h  |  57.0  58.0  |  129\n" \
                            "[25 x 3]\n\n"

    # mul
    assert repr(TDF * 2) == "TemporalDataFrame '2 * 2'\n" \
                            "\x1b[4mTime point : 0.0 hours\x1b[0m\n" \
                            "  Time-point     col1  col2    col3\n" \
                            "0       0.0h  |   0.0   2.0  |  100\n" \
                            "1       0.0h  |   4.0   6.0  |  101\n" \
                            "2       0.0h  |   8.0  10.0  |  102\n" \
                            "3       0.0h  |  12.0  14.0  |  103\n" \
                            "4       0.0h  |  16.0  18.0  |  104\n" \
                            "[25 x 3]\n" \
                            "\n" \
                            "\x1b[4mTime point : 1.0 hours\x1b[0m\n" \
                            "   Time-point      col1   col2    col3\n" \
                            "25       1.0h  |  100.0  102.0  |  125\n" \
                            "26       1.0h  |  104.0  106.0  |  126\n" \
                            "27       1.0h  |  108.0  110.0  |  127\n" \
                            "28       1.0h  |  112.0  114.0  |  128\n" \
                            "29       1.0h  |  116.0  118.0  |  129\n" \
                            "[25 x 3]\n\n"

    # div
    assert repr(TDF / 2) == "TemporalDataFrame '2 / 2'\n" \
                            "\x1b[4mTime point : 0.0 hours\x1b[0m\n" \
                            "  Time-point    col1 col2    col3\n" \
                            "0       0.0h  |  0.0  0.5  |  100\n" \
                            "1       0.0h  |  1.0  1.5  |  101\n" \
                            "2       0.0h  |  2.0  2.5  |  102\n" \
                            "3       0.0h  |  3.0  3.5  |  103\n" \
                            "4       0.0h  |  4.0  4.5  |  104\n" \
                            "[25 x 3]\n" \
                            "\n" \
                            "\x1b[4mTime point : 1.0 hours\x1b[0m\n" \
                            "   Time-point     col1  col2    col3\n" \
                            "25       1.0h  |  25.0  25.5  |  125\n" \
                            "26       1.0h  |  26.0  26.5  |  126\n" \
                            "27       1.0h  |  27.0  27.5  |  127\n" \
                            "28       1.0h  |  28.0  28.5  |  128\n" \
                            "29       1.0h  |  29.0  29.5  |  129\n" \
                            "[25 x 3]\n\n"

    TDF.file.close()
    cleanup([input_file])

    # TDF is a view -----------------------------------------------------------
    # TDF is not backed
    TDF = get_TDF('3')
    view = TDF[:, range(10, 90), ['col1', 'col4']]

    #   add
    #       numerical value
    assert repr(view + 1) == "TemporalDataFrame 'view of 3 + 1'\n" \
                             "\x1b[4mTime point : 0.0 hours\x1b[0m\n" \
                             "   Time-point     col1    col4\n" \
                             "50       0.0h  |  51.0  |  350\n" \
                             "51       0.0h  |  52.0  |  351\n" \
                             "52       0.0h  |  53.0  |  352\n" \
                             "53       0.0h  |  54.0  |  353\n" \
                             "54       0.0h  |  55.0  |  354\n" \
                             "[40 x 2]\n" \
                             "\n" \
                             "\x1b[4mTime point : 1.0 hours\x1b[0m\n" \
                             "   Time-point     col1    col4\n" \
                             "10       1.0h  |  11.0  |  310\n" \
                             "11       1.0h  |  12.0  |  311\n" \
                             "12       1.0h  |  13.0  |  312\n" \
                             "13       1.0h  |  14.0  |  313\n" \
                             "14       1.0h  |  15.0  |  314\n" \
                             "[40 x 2]\n\n"

    #       string value
    assert repr(view + '_1') == "TemporalDataFrame 'view of 3 + _1'\n" \
                                "\x1b[4mTime point : 0.0 hours\x1b[0m\n" \
                                "   Time-point     col1      col4\n" \
                                "50       0.0h  |  50.0  |  350_1\n" \
                                "51       0.0h  |  51.0  |  351_1\n" \
                                "52       0.0h  |  52.0  |  352_1\n" \
                                "53       0.0h  |  53.0  |  353_1\n" \
                                "54       0.0h  |  54.0  |  354_1\n" \
                                "[40 x 2]\n" \
                                "\n" \
                                "\x1b[4mTime point : 1.0 hours\x1b[0m\n" \
                                "   Time-point     col1      col4\n" \
                                "10       1.0h  |  10.0  |  310_1\n" \
                                "11       1.0h  |  11.0  |  311_1\n" \
                                "12       1.0h  |  12.0  |  312_1\n" \
                                "13       1.0h  |  13.0  |  313_1\n" \
                                "14       1.0h  |  14.0  |  314_1\n" \
                                "[40 x 2]\n\n"

    # sub
    assert repr(view - 1) == "TemporalDataFrame 'view of 3 - 1'\n" \
                             "\x1b[4mTime point : 0.0 hours\x1b[0m\n" \
                             "   Time-point     col1    col4\n" \
                             "50       0.0h  |  49.0  |  350\n" \
                             "51       0.0h  |  50.0  |  351\n" \
                             "52       0.0h  |  51.0  |  352\n" \
                             "53       0.0h  |  52.0  |  353\n" \
                             "54       0.0h  |  53.0  |  354\n" \
                             "[40 x 2]\n" \
                             "\n" \
                             "\x1b[4mTime point : 1.0 hours\x1b[0m\n" \
                             "   Time-point     col1    col4\n" \
                             "10       1.0h  |   9.0  |  310\n" \
                             "11       1.0h  |  10.0  |  311\n" \
                             "12       1.0h  |  11.0  |  312\n" \
                             "13       1.0h  |  12.0  |  313\n" \
                             "14       1.0h  |  13.0  |  314\n" \
                             "[40 x 2]\n\n"

    # mul
    assert repr(view * 2) == "TemporalDataFrame 'view of 3 * 2'\n" \
                             "\x1b[4mTime point : 0.0 hours\x1b[0m\n" \
                             "   Time-point      col1    col4\n" \
                             "50       0.0h  |  100.0  |  350\n" \
                             "51       0.0h  |  102.0  |  351\n" \
                             "52       0.0h  |  104.0  |  352\n" \
                             "53       0.0h  |  106.0  |  353\n" \
                             "54       0.0h  |  108.0  |  354\n" \
                             "[40 x 2]\n" \
                             "\n" \
                             "\x1b[4mTime point : 1.0 hours\x1b[0m\n" \
                             "   Time-point     col1    col4\n" \
                             "10       1.0h  |  20.0  |  310\n" \
                             "11       1.0h  |  22.0  |  311\n" \
                             "12       1.0h  |  24.0  |  312\n" \
                             "13       1.0h  |  26.0  |  313\n" \
                             "14       1.0h  |  28.0  |  314\n" \
                             "[40 x 2]\n\n"

    # div
    assert repr(view / 2) == "TemporalDataFrame 'view of 3 / 2'\n" \
                             "\x1b[4mTime point : 0.0 hours\x1b[0m\n" \
                             "   Time-point     col1    col4\n" \
                             "50       0.0h  |  25.0  |  350\n" \
                             "51       0.0h  |  25.5  |  351\n" \
                             "52       0.0h  |  26.0  |  352\n" \
                             "53       0.0h  |  26.5  |  353\n" \
                             "54       0.0h  |  27.0  |  354\n" \
                             "[40 x 2]\n" \
                             "\n" \
                             "\x1b[4mTime point : 1.0 hours\x1b[0m\n" \
                             "   Time-point    col1    col4\n" \
                             "10       1.0h  |  5.0  |  310\n" \
                             "11       1.0h  |  5.5  |  311\n" \
                             "12       1.0h  |  6.0  |  312\n" \
                             "13       1.0h  |  6.5  |  313\n" \
                             "14       1.0h  |  7.0  |  314\n" \
                             "[40 x 2]\n\n"

    # add to empty array
    view = TDF[:, range(10, 90), ['col4']]

    with pytest.raises(ValueError) as exc_info:
        view + 1

    assert str(exc_info.value) == "No numerical data to add."

    # sub in empty array
    with pytest.raises(ValueError) as exc_info:
        view - 1

    assert str(exc_info.value) == "No numerical data to subtract."

    # TDF is a view of a backed TDF
    input_file = Path(__file__).parent / 'test_insertion_TDF'
    cleanup([input_file])

    TDF = get_backed_TDF(input_file, '4', mode=H5Mode.READ)

    view = TDF[:, range(10, 40), ['col1', 'col3']]

    #   add
    #       numerical value
    assert repr(view + 1) == "TemporalDataFrame 'view of 4 + 1'\n" \
                             "\x1b[4mTime point : 0.0 hours\x1b[0m\n" \
                             "   Time-point     col1    col3\n" \
                             "10       0.0h  |  21.0  |  110\n" \
                             "11       0.0h  |  23.0  |  111\n" \
                             "12       0.0h  |  25.0  |  112\n" \
                             "13       0.0h  |  27.0  |  113\n" \
                             "14       0.0h  |  29.0  |  114\n" \
                             "[15 x 2]\n" \
                             "\n" \
                             "\x1b[4mTime point : 1.0 hours\x1b[0m\n" \
                             "   Time-point     col1    col3\n" \
                             "25       1.0h  |  51.0  |  125\n" \
                             "26       1.0h  |  53.0  |  126\n" \
                             "27       1.0h  |  55.0  |  127\n" \
                             "28       1.0h  |  57.0  |  128\n" \
                             "29       1.0h  |  59.0  |  129\n" \
                             "[15 x 2]\n\n"

    #       string value
    assert repr(view + '_1') == "TemporalDataFrame 'view of 4 + _1'\n" \
                                "\x1b[4mTime point : 0.0 hours\x1b[0m\n" \
                                "   Time-point     col1      col3\n" \
                                "10       0.0h  |  20.0  |  110_1\n" \
                                "11       0.0h  |  22.0  |  111_1\n" \
                                "12       0.0h  |  24.0  |  112_1\n" \
                                "13       0.0h  |  26.0  |  113_1\n" \
                                "14       0.0h  |  28.0  |  114_1\n" \
                                "[15 x 2]\n" \
                                "\n" \
                                "\x1b[4mTime point : 1.0 hours\x1b[0m\n" \
                                "   Time-point     col1      col3\n" \
                                "25       1.0h  |  50.0  |  125_1\n" \
                                "26       1.0h  |  52.0  |  126_1\n" \
                                "27       1.0h  |  54.0  |  127_1\n" \
                                "28       1.0h  |  56.0  |  128_1\n" \
                                "29       1.0h  |  58.0  |  129_1\n" \
                                "[15 x 2]\n\n"

    # sub
    assert repr(view - 1) == "TemporalDataFrame 'view of 4 - 1'\n" \
                             "\x1b[4mTime point : 0.0 hours\x1b[0m\n" \
                             "   Time-point     col1    col3\n" \
                             "10       0.0h  |  19.0  |  110\n" \
                             "11       0.0h  |  21.0  |  111\n" \
                             "12       0.0h  |  23.0  |  112\n" \
                             "13       0.0h  |  25.0  |  113\n" \
                             "14       0.0h  |  27.0  |  114\n" \
                             "[15 x 2]\n" \
                             "\n" \
                             "\x1b[4mTime point : 1.0 hours\x1b[0m\n" \
                             "   Time-point     col1    col3\n" \
                             "25       1.0h  |  49.0  |  125\n" \
                             "26       1.0h  |  51.0  |  126\n" \
                             "27       1.0h  |  53.0  |  127\n" \
                             "28       1.0h  |  55.0  |  128\n" \
                             "29       1.0h  |  57.0  |  129\n" \
                             "[15 x 2]\n\n"

    # mul
    assert repr(view * 2) == "TemporalDataFrame 'view of 4 * 2'\n" \
                             "\x1b[4mTime point : 0.0 hours\x1b[0m\n" \
                             "   Time-point     col1    col3\n" \
                             "10       0.0h  |  40.0  |  110\n" \
                             "11       0.0h  |  44.0  |  111\n" \
                             "12       0.0h  |  48.0  |  112\n" \
                             "13       0.0h  |  52.0  |  113\n" \
                             "14       0.0h  |  56.0  |  114\n" \
                             "[15 x 2]\n" \
                             "\n" \
                             "\x1b[4mTime point : 1.0 hours\x1b[0m\n" \
                             "   Time-point      col1    col3\n" \
                             "25       1.0h  |  100.0  |  125\n" \
                             "26       1.0h  |  104.0  |  126\n" \
                             "27       1.0h  |  108.0  |  127\n" \
                             "28       1.0h  |  112.0  |  128\n" \
                             "29       1.0h  |  116.0  |  129\n" \
                             "[15 x 2]\n\n" \
                             ""

    # div
    assert repr(view / 2) == "TemporalDataFrame 'view of 4 / 2'\n" \
                             "\x1b[4mTime point : 0.0 hours\x1b[0m\n" \
                             "   Time-point     col1    col3\n" \
                             "10       0.0h  |  10.0  |  110\n" \
                             "11       0.0h  |  11.0  |  111\n" \
                             "12       0.0h  |  12.0  |  112\n" \
                             "13       0.0h  |  13.0  |  113\n" \
                             "14       0.0h  |  14.0  |  114\n" \
                             "[15 x 2]\n" \
                             "\n" \
                             "\x1b[4mTime point : 1.0 hours\x1b[0m\n" \
                             "   Time-point     col1    col3\n" \
                             "25       1.0h  |  25.0  |  125\n" \
                             "26       1.0h  |  26.0  |  126\n" \
                             "27       1.0h  |  27.0  |  127\n" \
                             "28       1.0h  |  28.0  |  128\n" \
                             "29       1.0h  |  29.0  |  129\n" \
                             "[15 x 2]\n\n"

    TDF.file.close()

    cleanup([input_file])


def test_inplace_add_sub_mul_div():
    # TDF is not backed

    #   add
    #       numerical value
    TDF = get_TDF('1')
    TDF += 1
    assert repr(TDF) == "TemporalDataFrame '1'\n" \
                        "\x1b[4mTime point : 0.0 hours\x1b[0m\n" \
                        "   Time-point     col1   col2    col3 col4\n" \
                        "50       0.0h  |  51.0  151.0  |  250  350\n" \
                        "51       0.0h  |  52.0  152.0  |  251  351\n" \
                        "52       0.0h  |  53.0  153.0  |  252  352\n" \
                        "53       0.0h  |  54.0  154.0  |  253  353\n" \
                        "54       0.0h  |  55.0  155.0  |  254  354\n" \
                        "[50 x 4]\n" \
                        "\n" \
                        "\x1b[4mTime point : 1.0 hours\x1b[0m\n" \
                        "  Time-point    col1   col2    col3 col4\n" \
                        "0       1.0h  |  1.0  101.0  |  200  300\n" \
                        "1       1.0h  |  2.0  102.0  |  201  301\n" \
                        "2       1.0h  |  3.0  103.0  |  202  302\n" \
                        "3       1.0h  |  4.0  104.0  |  203  303\n" \
                        "4       1.0h  |  5.0  105.0  |  204  304\n" \
                        "[50 x 4]\n\n"

    #       string value
    TDF = get_TDF('2')
    TDF += '_1'
    assert repr(TDF) == "TemporalDataFrame '2'\n" \
                        "\x1b[4mTime point : 0.0 hours\x1b[0m\n" \
                        "   Time-point     col1   col2      col3   col4\n" \
                        "50       0.0h  |  50.0  150.0  |  250_1  350_1\n" \
                        "51       0.0h  |  51.0  151.0  |  251_1  351_1\n" \
                        "52       0.0h  |  52.0  152.0  |  252_1  352_1\n" \
                        "53       0.0h  |  53.0  153.0  |  253_1  353_1\n" \
                        "54       0.0h  |  54.0  154.0  |  254_1  354_1\n" \
                        "[50 x 4]\n" \
                        "\n" \
                        "\x1b[4mTime point : 1.0 hours\x1b[0m\n" \
                        "  Time-point    col1   col2      col3   col4\n" \
                        "0       1.0h  |  0.0  100.0  |  200_1  300_1\n" \
                        "1       1.0h  |  1.0  101.0  |  201_1  301_1\n" \
                        "2       1.0h  |  2.0  102.0  |  202_1  302_1\n" \
                        "3       1.0h  |  3.0  103.0  |  203_1  303_1\n" \
                        "4       1.0h  |  4.0  104.0  |  204_1  304_1\n" \
                        "[50 x 4]\n\n"

    #   sub
    TDF = get_TDF('3')
    TDF -= 2
    assert repr(TDF) == "TemporalDataFrame '3'\n" \
                        "\x1b[4mTime point : 0.0 hours\x1b[0m\n" \
                        "   Time-point     col1   col2    col3 col4\n" \
                        "50       0.0h  |  48.0  148.0  |  250  350\n" \
                        "51       0.0h  |  49.0  149.0  |  251  351\n" \
                        "52       0.0h  |  50.0  150.0  |  252  352\n" \
                        "53       0.0h  |  51.0  151.0  |  253  353\n" \
                        "54       0.0h  |  52.0  152.0  |  254  354\n" \
                        "[50 x 4]\n" \
                        "\n" \
                        "\x1b[4mTime point : 1.0 hours\x1b[0m\n" \
                        "  Time-point    col1   col2    col3 col4\n" \
                        "0       1.0h  | -2.0   98.0  |  200  300\n" \
                        "1       1.0h  | -1.0   99.0  |  201  301\n" \
                        "2       1.0h  |  0.0  100.0  |  202  302\n" \
                        "3       1.0h  |  1.0  101.0  |  203  303\n" \
                        "4       1.0h  |  2.0  102.0  |  204  304\n" \
                        "[50 x 4]\n\n"

    #   mul
    TDF = get_TDF('4')
    TDF *= 2
    assert repr(TDF) == "TemporalDataFrame '4'\n" \
                        "\x1b[4mTime point : 0.0 hours\x1b[0m\n" \
                        "   Time-point      col1   col2    col3 col4\n" \
                        "50       0.0h  |  100.0  300.0  |  250  350\n" \
                        "51       0.0h  |  102.0  302.0  |  251  351\n" \
                        "52       0.0h  |  104.0  304.0  |  252  352\n" \
                        "53       0.0h  |  106.0  306.0  |  253  353\n" \
                        "54       0.0h  |  108.0  308.0  |  254  354\n" \
                        "[50 x 4]\n" \
                        "\n" \
                        "\x1b[4mTime point : 1.0 hours\x1b[0m\n" \
                        "  Time-point    col1   col2    col3 col4\n" \
                        "0       1.0h  |  0.0  200.0  |  200  300\n" \
                        "1       1.0h  |  2.0  202.0  |  201  301\n" \
                        "2       1.0h  |  4.0  204.0  |  202  302\n" \
                        "3       1.0h  |  6.0  206.0  |  203  303\n" \
                        "4       1.0h  |  8.0  208.0  |  204  304\n" \
                        "[50 x 4]\n\n"

    #   div
    TDF = get_TDF('5')
    TDF /= 4
    assert repr(TDF) == "TemporalDataFrame '5'\n" \
                        "\x1b[4mTime point : 0.0 hours\x1b[0m\n" \
                        "   Time-point      col1   col2    col3 col4\n" \
                        "50       0.0h  |   12.5   37.5  |  250  350\n" \
                        "51       0.0h  |  12.75  37.75  |  251  351\n" \
                        "52       0.0h  |   13.0   38.0  |  252  352\n" \
                        "53       0.0h  |  13.25  38.25  |  253  353\n" \
                        "54       0.0h  |   13.5   38.5  |  254  354\n" \
                        "[50 x 4]\n" \
                        "\n" \
                        "\x1b[4mTime point : 1.0 hours\x1b[0m\n" \
                        "  Time-point     col1   col2    col3 col4\n" \
                        "0       1.0h  |   0.0   25.0  |  200  300\n" \
                        "1       1.0h  |  0.25  25.25  |  201  301\n" \
                        "2       1.0h  |   0.5   25.5  |  202  302\n" \
                        "3       1.0h  |  0.75  25.75  |  203  303\n" \
                        "4       1.0h  |   1.0   26.0  |  204  304\n" \
                        "[50 x 4]\n\n"

    #   add to empty array
    TDF = get_TDF('6')
    del TDF.col1
    del TDF.col2

    with pytest.raises(ValueError) as exc_info:
        TDF += 1

    assert str(exc_info.value) == "No numerical data to add."

    # TDF is backed
    input_file = Path(__file__).parent / 'test_insertion_TDF'
    cleanup([input_file])

    TDF = get_backed_TDF(input_file, '7', H5Mode.READ_WRITE)

    #   add
    #       numerical value
    TDF += 1
    assert np.all(TDF.values_num == np.arange(1, 101).reshape(50, 2))
    assert np.all(TDF.values_str == np.arange(100, 150).astype(str)[:, None])

    #       string value
    # TODO : does not work yet
    # TDF += '_1'
    # assert np.all(TDF.values_num == np.arange(1, 101).reshape(50, 2))
    # assert np.all(TDF.values_str == np.arange(100, 150).astype(str)[:, None])

    #   sub
    TDF -= 2
    assert np.all(TDF.values_num == np.arange(-1, 99).reshape(50, 2))
    assert np.all(TDF.values_str == np.arange(100, 150).astype(str)[:, None])

    #   mul
    TDF *= 2
    assert np.all(TDF.values_num == np.arange(-2, 198, 2).reshape(50, 2))
    assert np.all(TDF.values_str == np.arange(100, 150).astype(str)[:, None])

    #   div
    TDF /= 4
    # noinspection PyTypeChecker
    assert np.all(TDF.values_num == np.arange(-0.5, 49.5, 0.5).reshape(50, 2))
    assert np.all(TDF.values_str == np.arange(100, 150).astype(str)[:, None])

    cleanup([input_file])

    # TDF is a view -----------------------------------------------------------
    # TDF is not backed
    TDF = get_TDF('8')

    view = TDF[:, range(10, 90), ['col1', 'col4']]

    #   add
    #       numerical value
    view += 1
    assert np.all(TDF.values_num == np.vstack((
        np.concatenate((np.arange(51, 91), np.arange(90, 100), np.arange(0, 10), np.arange(11, 51))),
        np.concatenate((np.arange(150, 200), np.arange(100, 150)))
    )).T)
    assert np.all(TDF.values_str == np.vstack((
        np.concatenate((np.arange(250, 300), np.arange(200, 250))).astype(str),
        np.concatenate((np.arange(350, 400), np.arange(300, 350))).astype(str)
    )).T)

    #       string value
    view += '_1'
    assert np.all(TDF.values_num == np.vstack((
        np.concatenate((np.arange(51, 91), np.arange(90, 100), np.arange(0, 10), np.arange(11, 51))),
        np.concatenate((np.arange(150, 200), np.arange(100, 150)))
    )).T)
    assert np.all(TDF.values_str == np.vstack((
        np.concatenate((np.arange(250, 300), np.arange(200, 250))).astype(str),
        np.concatenate((np.char.add(np.arange(350, 390).astype(str), '_1'),
                        np.arange(390, 400).astype(str),
                        np.arange(300, 310).astype(str),
                        np.char.add(np.arange(310, 350).astype(str), '_1')))
    )).T)

    #   sub
    view -= 2
    assert np.all(TDF.values_num == np.vstack((
        np.concatenate((np.arange(49, 89), np.arange(90, 100), np.arange(0, 10), np.arange(9, 49))),
        np.concatenate((np.arange(150, 200), np.arange(100, 150)))
    )).T)
    assert np.all(TDF.values_str == np.vstack((
        np.concatenate((np.arange(250, 300), np.arange(200, 250))).astype(str),
        np.concatenate((np.char.add(np.arange(350, 390).astype(str), '_1'),
                        np.arange(390, 400).astype(str),
                        np.arange(300, 310).astype(str),
                        np.char.add(np.arange(310, 350).astype(str), '_1')))
    )).T)

    #   mul
    view *= 2
    assert np.all(TDF.values_num == np.vstack((
        np.concatenate((np.arange(98, 178, 2), np.arange(90, 100), np.arange(0, 10), np.arange(18, 98, 2))),
        np.concatenate((np.arange(150, 200), np.arange(100, 150)))
    )).T)
    assert np.all(TDF.values_str == np.vstack((
        np.concatenate((np.arange(250, 300), np.arange(200, 250))).astype(str),
        np.concatenate((np.char.add(np.arange(350, 390).astype(str), '_1'),
                        np.arange(390, 400).astype(str),
                        np.arange(300, 310).astype(str),
                        np.char.add(np.arange(310, 350).astype(str), '_1')))
    )).T)

    #   div
    view /= 4
    # noinspection PyTypeChecker
    assert np.all(TDF.values_num == np.vstack((
        np.concatenate((np.arange(24.5, 44.5, 0.5), np.arange(90, 100), np.arange(0, 10), np.arange(4.5, 24.5, 0.5))),
        np.concatenate((np.arange(150, 200), np.arange(100, 150)))
    )).T)
    assert np.all(TDF.values_str == np.vstack((
        np.concatenate((np.arange(250, 300), np.arange(200, 250))).astype(str),
        np.concatenate((np.char.add(np.arange(350, 390).astype(str), '_1'),
                        np.arange(390, 400).astype(str),
                        np.arange(300, 310).astype(str),
                        np.char.add(np.arange(310, 350).astype(str), '_1')))
    )).T)

    #   add to empty array
    del TDF.col1
    del TDF.col2

    with pytest.raises(ValueError) as exc_info:
        TDF += 1

    assert str(exc_info.value) == "No numerical data to add."

    # TDF is backed
    input_file = Path(__file__).parent / 'test_insertion_TDF'
    cleanup([input_file])

    TDF = get_backed_TDF(input_file, '9', H5Mode.READ_WRITE)

    view = TDF[:, range(10, 40), ['col1', 'col3']]

    #   add
    #       numerical value
    view += 1
    assert np.all(TDF.values_num == np.vstack((
        np.arange(0, 20).reshape(10, 2),
        np.repeat(np.arange(21, 80, 2), 2).reshape(30, 2),
        np.arange(80, 100).reshape(10, 2)
    )))
    assert np.all(TDF.values_str == np.arange(100, 150).astype(str)[:, None])

    #       string value
    # TODO
    # view += '_1'

    #   sub
    view -= 2
    assert np.all(TDF.values_num == np.vstack((
        np.arange(0, 20).reshape(10, 2),
        np.hstack((np.arange(19, 78, 2)[:, None], np.arange(21, 80, 2)[:, None])),
        np.arange(80, 100).reshape(10, 2)
    )))
    assert np.all(TDF.values_str == np.arange(100, 150).astype(str)[:, None])

    #   mul
    view *= 2
    assert np.all(TDF.values_num == np.vstack((
        np.arange(0, 20).reshape(10, 2),
        np.hstack((np.arange(38, 156, 4)[:, None], np.arange(21, 80, 2)[:, None])),
        np.arange(80, 100).reshape(10, 2)
    )))
    assert np.all(TDF.values_str == np.arange(100, 150).astype(str)[:, None])

    #   div
    view /= 4
    # noinspection PyTypeChecker
    assert np.all(TDF.values_num == np.vstack((
        np.arange(0, 20).reshape(10, 2),
        np.hstack((np.arange(9.5, 39, 1)[:, None], np.arange(21, 80, 2)[:, None])),
        np.arange(80, 100).reshape(10, 2)
    )))
    assert np.all(TDF.values_str == np.arange(100, 150).astype(str)[:, None])

    cleanup([input_file])


if __name__ == '__main__':
    test_delete()
    test_append()
    test_insert()
    test_add_sub_mul_div()
    test_inplace_add_sub_mul_div()
