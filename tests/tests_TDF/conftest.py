# coding: utf-8
# Created on 16/10/2022 10:36
# Author : matteo
import pytest
import numpy as np
from vdata.h5pickle import File
from h5py import string_dtype
from pathlib import Path

from vdata import TemporalDataFrame, BackedTemporalDataFrame
from vdata.core.attribute_proxy.attribute import NONE_VALUE
from vdata.h5pickle.name_utils import H5Mode
from vdata.read_write import read_TDF


# ====================================================
# imports

# ====================================================
# code
REFERENCE_BACKED_DATA = {
    'type': 'tdf',
    'locked_indices': False,
    'locked_columns': False,
    'timepoints_column_name': NONE_VALUE,
    'index': np.concatenate((np.arange(50, 100), np.arange(0, 50))),
    'repeating_index': False,
    'columns_numerical': np.array(['col1', 'col2'], dtype=np.dtype('O')),
    'columns_string': np.array(['col3', 'col4'], dtype=np.dtype('O')),
    'timepoints': np.array(['0.0h' for _ in range(50)] + ['1.0h' for _ in range(50)], dtype=np.dtype('O')),
    'values_numerical': np.vstack((np.concatenate((np.arange(50, 100), np.arange(0, 50))),
                                   np.concatenate((np.arange(150, 200), np.arange(100, 150))))).T.astype(float),
    'values_string': np.vstack((np.concatenate((np.arange(250, 300), np.arange(200, 250))),
                                np.concatenate((np.arange(350, 400), np.arange(300, 350))))).T.astype(str).astype('O')
}


def get_TDF(name: str = '1') -> TemporalDataFrame:
    return TemporalDataFrame({'col1': [i for i in range(100)],
                              'col2': [i for i in range(100, 200)],
                              'col3': [str(i) for i in range(200, 300)],
                              'col4': [str(i) for i in range(300, 400)]},
                             name=name,
                             time_list=['1h' for _ in range(50)] + ['0h' for _ in range(50)])


def get_backed_TDF(name: str = '1') -> BackedTemporalDataFrame:
    with File('backed_TDF_' + name, H5Mode.WRITE_TRUNCATE) as h5_file:
        # write data to h5 file directly
        h5_file.attrs['type'] = REFERENCE_BACKED_DATA['type']
        h5_file.attrs['name'] = name
        h5_file.attrs['locked_indices'] = REFERENCE_BACKED_DATA['locked_indices']
        h5_file.attrs['locked_columns'] = REFERENCE_BACKED_DATA['locked_columns']
        h5_file.attrs['timepoints_column_name'] = REFERENCE_BACKED_DATA['timepoints_column_name']
        h5_file.attrs['repeating_index'] = REFERENCE_BACKED_DATA['repeating_index']

        h5_file.create_dataset('index', data=REFERENCE_BACKED_DATA['index'])
        h5_file.create_dataset('columns_numerical', data=REFERENCE_BACKED_DATA['columns_numerical'],
                               chunks=True, maxshape=(None,), dtype=string_dtype())
        h5_file.create_dataset('columns_string', data=REFERENCE_BACKED_DATA['columns_string'],
                               chunks=True, maxshape=(None,), dtype=string_dtype())
        h5_file.create_dataset('timepoints', data=REFERENCE_BACKED_DATA['timepoints'], dtype=string_dtype())

        h5_file.create_dataset('values_numerical', data=REFERENCE_BACKED_DATA['values_numerical'],
                               chunks=True, maxshape=(None, None))

        h5_file.create_dataset('values_string', data=REFERENCE_BACKED_DATA['values_string'], dtype=string_dtype(),
                               chunks=True, maxshape=(None, None))

    # read tdf from file
    return read_TDF('backed_TDF_' + name, mode=H5Mode.READ_WRITE)


@pytest.fixture
def TDF(request) -> TemporalDataFrame:
    if hasattr(request, 'param'):
        which = request.param

    else:
        which = 'plain'

    if 'backed' in which:
        if 'view' in which:
            yield get_backed_TDF()[:, np.arange(10, 90), ['col1', 'col4']]
            Path('backed_TDF_' + '1').unlink()

        else:
            yield get_backed_TDF()
            Path('backed_TDF_' + '1').unlink()

    else:
        if 'view' in which:
            yield get_TDF()[:, np.arange(10, 90), ['col1', 'col4']]

        else:
            yield get_TDF()


@pytest.fixture(scope='class')
def class_TDF_backed(request):
    request.cls.TDF = get_backed_TDF()
    yield
    Path('backed_TDF_' + '1').unlink()


@pytest.fixture
def TDF1(request) -> TemporalDataFrame:
    if hasattr(request, 'param'):
        which = request.param

    else:
        which = 'plain'

    if 'backed' in which:
        if 'view' in which:
            yield get_backed_TDF()[:]
            Path('backed_TDF_' + '1').unlink()

        else:
            yield get_backed_TDF()
            Path('backed_TDF_' + '1').unlink()

    else:
        if 'view' in which:
            yield get_TDF()[:]

        else:
            yield get_TDF()


@pytest.fixture(scope='class')
def class_TDF1(request) -> TemporalDataFrame:
    if hasattr(request, 'param'):
        which = request.param

    else:
        which = 'plain'

    if 'backed' in which:
        if 'view' in which:
            request.cls.TDF = get_backed_TDF()[:]
            yield

        else:
            request.cls.TDF = get_backed_TDF()
            yield

        Path('backed_TDF_' + '1').unlink()

    else:
        if 'view' in which:
            request.cls.TDF = get_TDF()[:]
            yield

        else:
            request.cls.TDF = get_TDF()
            yield


@pytest.fixture
def TDF2(request) -> TemporalDataFrame:
    if hasattr(request, 'param'):
        which = request.param

    else:
        which = 'plain'

    if 'backed' in which:
        if 'view' in which:
            yield get_backed_TDF('2')[:]
            Path('backed_TDF_' + '2').unlink()

        else:
            yield get_backed_TDF('2')
            Path('backed_TDF_' + '2').unlink()

    else:
        if 'view' in which:
            yield get_TDF('2')[:]

        else:
            yield get_TDF('2')


@pytest.fixture(scope='class')
def h5_file(request):
    if hasattr(request, 'param'):
        which = request.param

    else:
        which = 'plain'

    if 'backed' in which:
        if 'view' in which:
            tdf = get_backed_TDF()[:, np.arange(10, 90), ['col1', 'col4']]

        else:
            tdf = get_backed_TDF()

        tdf.write('TDF_write.vd')
        tdf.close()
        Path('backed_TDF_' + '1').unlink()

    else:
        if 'view' in which:
            tdf = get_TDF()[:, np.arange(10, 90), ['col1', 'col4']]

        else:
            tdf = get_TDF()

        tdf.write('TDF_write.vd')

    request.cls.h5_file = File('TDF_write.vd', H5Mode.READ)
    yield

    Path('TDF_write.vd').unlink()
