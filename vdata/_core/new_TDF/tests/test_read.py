# coding: utf-8
# Created on 31/03/2022 11:24
# Author : matteo

# ====================================================
# imports
import numpy as np
from h5py import File, string_dtype
from pathlib import Path

from ..name_utils import H5Mode
from .._read import read_TDF
from ..dataframe import TemporalDataFrame


# ====================================================
# code
reference_data = {
    'type': 'TDF',
    'name': '1',
    'locked_indices': False,
    'locked_columns': True,
    'time_points_column_name': '__TDF_None__',
    'index': np.array(range(50)),
    'columns_numerical': np.array(['col1', 'col2'], dtype=np.dtype('O')),
    'columns_string': np.array(['col3'], dtype=np.dtype('O')),
    'timepoints': np.array(['0.0h' for _ in range(25)] + ['1.0h' for _ in range(25)], dtype=np.dtype('O')),
    'values_numerical': np.array(range(100)).reshape((50, 2)),
    'values_string': np.array(list(map(str, range(100, 150))), dtype=np.dtype('O')).reshape((50, 1))
}


def check_TDF(TDF: TemporalDataFrame) -> None:
    """
    Check a TemporalDataFrame was correctly read from a H5 file.
    """
    assert TDF.name == reference_data['name']
    assert np.all(TDF.values_num == reference_data['values_numerical'])
    assert np.all(TDF.values_str == reference_data['values_string'])
    assert np.all(TDF.index == reference_data['index'])
    assert np.all(TDF.columns_num == reference_data['columns_numerical'])
    assert np.all(TDF.columns_str == reference_data['columns_string'])
    assert np.all(TDF.timepoints_column == reference_data['timepoints'])
    assert TDF.lock[0] == reference_data['locked_indices']
    assert TDF.lock[1] == reference_data['locked_columns']
    assert TDF.timepoints_column_name is None


def cleanup(path: Path) -> None:
    if path.exists():
        path.unlink(missing_ok=True)


def test_read():
    input_file = Path(__file__).parent / 'test_read_TDF'

    cleanup(input_file)

    with File(input_file, H5Mode.WRITE_TRUNCATE) as h5_file:
        # write data to h5 file directly
        h5_file.attrs['type'] = reference_data['type']
        h5_file.attrs['name'] = reference_data['name']
        h5_file.attrs['locked_indices'] = reference_data['locked_indices']
        h5_file.attrs['locked_columns'] = reference_data['locked_columns']
        h5_file.attrs['time_points_column_name'] = reference_data['time_points_column_name']

        h5_file.create_dataset('index', data=reference_data['index'])
        h5_file.create_dataset('columns_numerical', data=reference_data['columns_numerical'], dtype=string_dtype())
        h5_file.create_dataset('columns_string', data=reference_data['columns_string'], dtype=string_dtype())
        h5_file.create_dataset('timepoints', data=reference_data['timepoints'], dtype=string_dtype())

        h5_file.create_dataset('values_numerical', data=reference_data['values_numerical'],
                               chunks=True, maxshape=(None, None))

        h5_file.create_dataset('values_string', data=reference_data['values_string'], dtype=string_dtype(),
                               chunks=True, maxshape=(None, None))

    # read TDF from file
    TDF = read_TDF(input_file, mode=H5Mode.READ)
    assert TDF.is_backed
    assert TDF.file.mode == H5Mode.READ
    check_TDF(TDF)

    assert repr(TDF) == "Backed TemporalDataFrame '1'\n" \
                        "\x1b[4mTime point : 0.0 hours\x1b[0m\n" \
                        "  Time-point    col1 col2       col3\n" \
                        "0       0.0h  |    0    1  |  b'100'\n" \
                        "1       0.0h  |    2    3  |  b'101'\n" \
                        "2       0.0h  |    4    5  |  b'102'\n" \
                        "3       0.0h  |    6    7  |  b'103'\n" \
                        "4       0.0h  |    8    9  |  b'104'\n" \
                        "[25 x 3]\n" \
                        "\n" \
                        "\x1b[4mTime point : 1.0 hours\x1b[0m\n" \
                        "   Time-point    col1 col2       col3\n" \
                        "25       1.0h  |   50   51  |  b'125'\n" \
                        "26       1.0h  |   52   53  |  b'126'\n" \
                        "27       1.0h  |   54   55  |  b'127'\n" \
                        "28       1.0h  |   56   57  |  b'128'\n" \
                        "29       1.0h  |   58   59  |  b'129'\n" \
                        "[25 x 3]\n\n"

    TDF.file.close()

    cleanup(input_file)


if __name__ == '__main__':
    test_read()
