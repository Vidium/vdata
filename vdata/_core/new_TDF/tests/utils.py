# coding: utf-8
# Created on 31/03/2022 16:35
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
reference_backed_data = {
    'type': 'TDF',
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


def get_TDF(name: str) -> TemporalDataFrame:
    return TemporalDataFrame({'col1': [i for i in range(100)],
                              'col2': [i for i in range(100, 200)],
                              'col3': [str(i) for i in range(200, 300)],
                              'col4': [str(i) for i in range(300, 400)]},
                             name=name,
                             time_list=['1h' for _ in range(50)] + ['0h' for _ in range(50)])


def get_backed_TDF(input_file: Path,
                   name: str) -> TemporalDataFrame:
    with File(input_file, H5Mode.WRITE_TRUNCATE) as h5_file:
        # write data to h5 file directly
        h5_file.attrs['type'] = reference_backed_data['type']
        h5_file.attrs['name'] = name
        h5_file.attrs['locked_indices'] = reference_backed_data['locked_indices']
        h5_file.attrs['locked_columns'] = reference_backed_data['locked_columns']
        h5_file.attrs['time_points_column_name'] = reference_backed_data['time_points_column_name']

        h5_file.create_dataset('index', data=reference_backed_data['index'])
        h5_file.create_dataset('columns_numerical', data=reference_backed_data['columns_numerical'],
                               dtype=string_dtype())
        h5_file.create_dataset('columns_string', data=reference_backed_data['columns_string'], dtype=string_dtype())
        h5_file.create_dataset('timepoints', data=reference_backed_data['timepoints'], dtype=string_dtype())

        h5_file.create_dataset('values_numerical', data=reference_backed_data['values_numerical'],
                               chunks=True, maxshape=(None, None))

        h5_file.create_dataset('values_string', data=reference_backed_data['values_string'], dtype=string_dtype(),
                               chunks=True, maxshape=(None, None))

    # read TDF from file
    return read_TDF(input_file, mode=H5Mode.READ)
