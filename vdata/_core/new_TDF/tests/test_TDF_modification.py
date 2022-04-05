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


def test_modification():
    # TDF is not backed
    TDF = get_TDF('1')

    #   delete column
    #       column numerical
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

    #       column string
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
    #   delete column
    #       column numerical
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

    #       column string
    del TDF.col3

    repr(TDF)

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


if __name__ == '__main__':
    test_modification()
