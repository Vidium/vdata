# coding: utf-8
# Created on 31/03/2022 11:24
# Author : matteo

# ====================================================
# imports
import numpy as np
from pathlib import Path

from .utils import get_backed_TDF, reference_backed_data
from ..name_utils import H5Mode
from ..dataframe import TemporalDataFrame


# ====================================================
# code
def check_TDF(TDF: TemporalDataFrame,
              name: str) -> None:
    """
    Check a TemporalDataFrame was correctly read from a H5 file.
    """
    assert TDF.name == name
    assert np.all(TDF.values_num == reference_backed_data['values_numerical'])
    assert np.all(TDF.values_str == reference_backed_data['values_string'])
    assert np.all(TDF.index == reference_backed_data['index'])
    assert np.all(TDF.columns_num == reference_backed_data['columns_numerical'])
    assert np.all(TDF.columns_str == reference_backed_data['columns_string'])
    assert np.all(TDF.timepoints_column == reference_backed_data['timepoints'])
    assert TDF.lock[0] == reference_backed_data['locked_indices']
    assert TDF.lock[1] == reference_backed_data['locked_columns']
    assert TDF.timepoints_column_name is None


def cleanup(path: Path) -> None:
    if path.exists():
        path.unlink(missing_ok=True)


def test_read():
    input_file = Path(__file__).parent / 'test_read_TDF'

    cleanup(input_file)

    TDF = get_backed_TDF(input_file, '1')

    assert TDF.is_backed
    assert TDF.file.mode == H5Mode.READ
    check_TDF(TDF, '1')

    assert repr(TDF) == "Backed TemporalDataFrame '1'\n" \
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

    TDF.file.close()

    cleanup(input_file)


if __name__ == '__main__':
    test_read()
