# coding: utf-8
# Created on 05/04/2022 09:11
# Author : matteo

# ====================================================
# imports
from pathlib import Path

import numpy as np

from .utils import get_TDF, get_backed_TDF
from ..name_utils import H5Mode


# ====================================================
# code
def cleanup(path: Path) -> None:
    if path.exists():
        path.unlink(missing_ok=True)


def test_delete():
    # TDF is not backed
    TDF = get_TDF('1')

    #   column numerical
    del TDF.col1

    assert repr(TDF) == "TemporalDataFrame '1'\n" \
                        "\x1b[4mTime point : 0.0 hours\x1b[0m\n" \
                        "   Time-point    col2    col3 col4\n" \
                        "50       0.0h  |  150  |  250  350\n" \
                        "51       0.0h  |  151  |  251  351\n" \
                        "52       0.0h  |  152  |  252  352\n" \
                        "53       0.0h  |  153  |  253  353\n" \
                        "54       0.0h  |  154  |  254  354\n" \
                        "[50 x 3]\n" \
                        "\n" \
                        "\x1b[4mTime point : 1.0 hours\x1b[0m\n" \
                        "  Time-point    col2    col3 col4\n" \
                        "0       1.0h  |  100  |  200  300\n" \
                        "1       1.0h  |  101  |  201  301\n" \
                        "2       1.0h  |  102  |  202  302\n" \
                        "3       1.0h  |  103  |  203  303\n" \
                        "4       1.0h  |  104  |  204  304\n" \
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
                        "   Time-point    col2    col3\n" \
                        "50       0.0h  |  150  |  250\n" \
                        "51       0.0h  |  151  |  251\n" \
                        "52       0.0h  |  152  |  252\n" \
                        "53       0.0h  |  153  |  253\n" \
                        "54       0.0h  |  154  |  254\n" \
                        "[50 x 2]\n" \
                        "\n" \
                        "\x1b[4mTime point : 1.0 hours\x1b[0m\n" \
                        "  Time-point    col2    col3\n" \
                        "0       1.0h  |  100  |  200\n" \
                        "1       1.0h  |  101  |  201\n" \
                        "2       1.0h  |  102  |  202\n" \
                        "3       1.0h  |  103  |  203\n" \
                        "4       1.0h  |  104  |  204\n" \
                        "[50 x 2]\n\n"

    assert np.all(TDF.values_num == np.concatenate((np.arange(150, 200), np.arange(100, 150)))[:, None])
    assert np.all(TDF.values_str ==
                  np.concatenate((np.arange(250, 300).astype(str), np.arange(200, 250).astype(str)))[:, None])

    # TDF is backed
    input_file = Path(__file__).parent / 'test_modification_TDF'
    cleanup(input_file)

    TDF = get_backed_TDF(input_file, '2', mode=H5Mode.READ_WRITE)
    #   column numerical
    del TDF.col2

    assert repr(TDF) == "Backed TemporalDataFrame '2'\n" \
                        "\x1b[4mTime point : 0.0 hours\x1b[0m\n" \
                        "  Time-point    col1    col3\n" \
                        "0       0.0h  |    0  |  100\n" \
                        "1       0.0h  |    2  |  101\n" \
                        "2       0.0h  |    4  |  102\n" \
                        "3       0.0h  |    6  |  103\n" \
                        "4       0.0h  |    8  |  104\n" \
                        "[25 x 2]\n" \
                        "\n" \
                        "\x1b[4mTime point : 1.0 hours\x1b[0m\n" \
                        "   Time-point    col1    col3\n" \
                        "25       1.0h  |   50  |  125\n" \
                        "26       1.0h  |   52  |  126\n" \
                        "27       1.0h  |   54  |  127\n" \
                        "28       1.0h  |   56  |  128\n" \
                        "29       1.0h  |   58  |  129\n" \
                        "[25 x 2]\n\n"

    assert np.all(TDF.values_num == np.arange(0, 100, 2)[:, None])
    assert np.all(TDF.values_str == np.arange(100, 150).astype(str)[:, None])

    #   column string
    del TDF.col3

    assert repr(TDF) == "Backed TemporalDataFrame '2'\n" \
                        "\x1b[4mTime point : 0.0 hours\x1b[0m\n" \
                        "  Time-point    col1\n" \
                        "0       0.0h  |    0\n" \
                        "1       0.0h  |    2\n" \
                        "2       0.0h  |    4\n" \
                        "3       0.0h  |    6\n" \
                        "4       0.0h  |    8\n" \
                        "[25 x 1]\n" \
                        "\n" \
                        "\x1b[4mTime point : 1.0 hours\x1b[0m\n" \
                        "   Time-point    col1\n" \
                        "25       1.0h  |   50\n" \
                        "26       1.0h  |   52\n" \
                        "27       1.0h  |   54\n" \
                        "28       1.0h  |   56\n" \
                        "29       1.0h  |   58\n" \
                        "[25 x 1]\n\n"

    assert np.all(TDF.values_num == np.arange(0, 100, 2)[:, None])
    assert TDF.values_str.size == 0

    cleanup(input_file)


def test_append():
    # TDF is not backed
    TDF = get_TDF('1')

    #   column numerical
    #       column exists
    TDF.col1 = np.arange(500, 600)

    assert repr(TDF) == "TemporalDataFrame '1'\n" \
                        "\x1b[4mTime point : 0.0 hours\x1b[0m\n" \
                        "   Time-point    col1 col2    col3 col4\n" \
                        "50       0.0h  |  500  150  |  250  350\n" \
                        "51       0.0h  |  501  151  |  251  351\n" \
                        "52       0.0h  |  502  152  |  252  352\n" \
                        "53       0.0h  |  503  153  |  253  353\n" \
                        "54       0.0h  |  504  154  |  254  354\n" \
                        "[50 x 4]\n" \
                        "\n" \
                        "\x1b[4mTime point : 1.0 hours\x1b[0m\n" \
                        "  Time-point    col1 col2    col3 col4\n" \
                        "0       1.0h  |  550  100  |  200  300\n" \
                        "1       1.0h  |  551  101  |  201  301\n" \
                        "2       1.0h  |  552  102  |  202  302\n" \
                        "3       1.0h  |  553  103  |  203  303\n" \
                        "4       1.0h  |  554  104  |  204  304\n" \
                        "[50 x 4]\n\n"

    #       column new
    TDF.col7 = np.arange(600, 700)

    assert repr(TDF) == "TemporalDataFrame '1'\n" \
                        "\x1b[4mTime point : 0.0 hours\x1b[0m\n" \
                        "   Time-point    col1 col2 col7    col3 col4\n" \
                        "50       0.0h  |  500  150  600  |  250  350\n" \
                        "51       0.0h  |  501  151  601  |  251  351\n" \
                        "52       0.0h  |  502  152  602  |  252  352\n" \
                        "53       0.0h  |  503  153  603  |  253  353\n" \
                        "54       0.0h  |  504  154  604  |  254  354\n" \
                        "[50 x 5]\n" \
                        "\n" \
                        "\x1b[4mTime point : 1.0 hours\x1b[0m\n" \
                        "  Time-point    col1 col2 col7    col3 col4\n" \
                        "0       1.0h  |  550  100  650  |  200  300\n" \
                        "1       1.0h  |  551  101  651  |  201  301\n" \
                        "2       1.0h  |  552  102  652  |  202  302\n" \
                        "3       1.0h  |  553  103  653  |  203  303\n" \
                        "4       1.0h  |  554  104  654  |  204  304\n" \
                        "[50 x 5]\n\n"

    #   column string
    #       column exists
    TDF.col4 = np.arange(700, 800).astype(str)

    assert repr(TDF) == "TemporalDataFrame '1'\n" \
                        "\x1b[4mTime point : 0.0 hours\x1b[0m\n" \
                        "   Time-point    col1 col2 col7    col3 col4\n" \
                        "50       0.0h  |  500  150  600  |  250  700\n" \
                        "51       0.0h  |  501  151  601  |  251  701\n" \
                        "52       0.0h  |  502  152  602  |  252  702\n" \
                        "53       0.0h  |  503  153  603  |  253  703\n" \
                        "54       0.0h  |  504  154  604  |  254  704\n" \
                        "[50 x 5]\n" \
                        "\n" \
                        "\x1b[4mTime point : 1.0 hours\x1b[0m\n" \
                        "  Time-point    col1 col2 col7    col3 col4\n" \
                        "0       1.0h  |  550  100  650  |  200  750\n" \
                        "1       1.0h  |  551  101  651  |  201  751\n" \
                        "2       1.0h  |  552  102  652  |  202  752\n" \
                        "3       1.0h  |  553  103  653  |  203  753\n" \
                        "4       1.0h  |  554  104  654  |  204  754\n" \
                        "[50 x 5]\n\n"

    #       column new
    TDF.col8 = np.arange(800, 900).astype(str)

    assert repr(TDF) == "TemporalDataFrame '1'\n" \
                        "\x1b[4mTime point : 0.0 hours\x1b[0m\n" \
                        "   Time-point    col1 col2 col7    col3 col4 col8\n" \
                        "50       0.0h  |  500  150  600  |  250  700  800\n" \
                        "51       0.0h  |  501  151  601  |  251  701  801\n" \
                        "52       0.0h  |  502  152  602  |  252  702  802\n" \
                        "53       0.0h  |  503  153  603  |  253  703  803\n" \
                        "54       0.0h  |  504  154  604  |  254  704  804\n" \
                        "[50 x 6]\n" \
                        "\n" \
                        "\x1b[4mTime point : 1.0 hours\x1b[0m\n" \
                        "  Time-point    col1 col2 col7    col3 col4 col8\n" \
                        "0       1.0h  |  550  100  650  |  200  750  850\n" \
                        "1       1.0h  |  551  101  651  |  201  751  851\n" \
                        "2       1.0h  |  552  102  652  |  202  752  852\n" \
                        "3       1.0h  |  553  103  653  |  203  753  853\n" \
                        "4       1.0h  |  554  104  654  |  204  754  854\n" \
                        "[50 x 6]\n\n"

    # TDF is backed
    input_file = Path(__file__).parent / 'test_modification_TDF'
    cleanup(input_file)

    TDF = get_backed_TDF(input_file, '2', mode=H5Mode.READ_WRITE)

    #   column numerical
    #       column exists
    TDF.col1 = np.arange(500, 550)

    assert repr(TDF) == "Backed TemporalDataFrame '2'\n" \
                        "\x1b[4mTime point : 0.0 hours\x1b[0m\n" \
                        "  Time-point    col1 col2    col3\n" \
                        "0       0.0h  |  500    1  |  100\n" \
                        "1       0.0h  |  501    3  |  101\n" \
                        "2       0.0h  |  502    5  |  102\n" \
                        "3       0.0h  |  503    7  |  103\n" \
                        "4       0.0h  |  504    9  |  104\n" \
                        "[25 x 3]\n" \
                        "\n" \
                        "\x1b[4mTime point : 1.0 hours\x1b[0m\n" \
                        "   Time-point    col1 col2    col3\n" \
                        "25       1.0h  |  525   51  |  125\n" \
                        "26       1.0h  |  526   53  |  126\n" \
                        "27       1.0h  |  527   55  |  127\n" \
                        "28       1.0h  |  528   57  |  128\n" \
                        "29       1.0h  |  529   59  |  129\n" \
                        "[25 x 3]\n\n"

    #       column new
    TDF.col4 = np.arange(600, 650)

    assert repr(TDF) == "Backed TemporalDataFrame '2'\n" \
                        "\x1b[4mTime point : 0.0 hours\x1b[0m\n" \
                        "  Time-point    col1 col2 col4    col3\n" \
                        "0       0.0h  |  500    1  600  |  100\n" \
                        "1       0.0h  |  501    3  601  |  101\n" \
                        "2       0.0h  |  502    5  602  |  102\n" \
                        "3       0.0h  |  503    7  603  |  103\n" \
                        "4       0.0h  |  504    9  604  |  104\n" \
                        "[25 x 4]\n" \
                        "\n" \
                        "\x1b[4mTime point : 1.0 hours\x1b[0m\n" \
                        "   Time-point    col1 col2 col4    col3\n" \
                        "25       1.0h  |  525   51  625  |  125\n" \
                        "26       1.0h  |  526   53  626  |  126\n" \
                        "27       1.0h  |  527   55  627  |  127\n" \
                        "28       1.0h  |  528   57  628  |  128\n" \
                        "29       1.0h  |  529   59  629  |  129\n" \
                        "[25 x 4]\n\n"

    #   column string
    #       column exists
    TDF.col3 = np.arange(700, 750).astype(str)

    assert repr(TDF) == "Backed TemporalDataFrame '2'\n" \
                        "\x1b[4mTime point : 0.0 hours\x1b[0m\n" \
                        "  Time-point    col1 col2 col4    col3\n" \
                        "0       0.0h  |  500    1  600  |  700\n" \
                        "1       0.0h  |  501    3  601  |  701\n" \
                        "2       0.0h  |  502    5  602  |  702\n" \
                        "3       0.0h  |  503    7  603  |  703\n" \
                        "4       0.0h  |  504    9  604  |  704\n" \
                        "[25 x 4]\n" \
                        "\n" \
                        "\x1b[4mTime point : 1.0 hours\x1b[0m\n" \
                        "   Time-point    col1 col2 col4    col3\n" \
                        "25       1.0h  |  525   51  625  |  725\n" \
                        "26       1.0h  |  526   53  626  |  726\n" \
                        "27       1.0h  |  527   55  627  |  727\n" \
                        "28       1.0h  |  528   57  628  |  728\n" \
                        "29       1.0h  |  529   59  629  |  729\n" \
                        "[25 x 4]\n\n"

    #       column new
    TDF.col5 = np.arange(800, 850).astype(str)

    assert repr(TDF) == "Backed TemporalDataFrame '2'\n" \
                        "\x1b[4mTime point : 0.0 hours\x1b[0m\n" \
                        "  Time-point    col1 col2 col4    col3 col5\n" \
                        "0       0.0h  |  500    1  600  |  700  800\n" \
                        "1       0.0h  |  501    3  601  |  701  801\n" \
                        "2       0.0h  |  502    5  602  |  702  802\n" \
                        "3       0.0h  |  503    7  603  |  703  803\n" \
                        "4       0.0h  |  504    9  604  |  704  804\n" \
                        "[25 x 5]\n" \
                        "\n" \
                        "\x1b[4mTime point : 1.0 hours\x1b[0m\n" \
                        "   Time-point    col1 col2 col4    col3 col5\n" \
                        "25       1.0h  |  525   51  625  |  725  825\n" \
                        "26       1.0h  |  526   53  626  |  726  826\n" \
                        "27       1.0h  |  527   55  627  |  727  827\n" \
                        "28       1.0h  |  528   57  628  |  728  828\n" \
                        "29       1.0h  |  529   59  629  |  729  829\n" \
                        "[25 x 5]\n\n"

    cleanup(input_file)


if __name__ == '__main__':
    test_delete()
    test_append()
