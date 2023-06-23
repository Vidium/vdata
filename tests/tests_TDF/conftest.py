import pickle
from pathlib import Path
from typing import Any, Generator

import numpy as np
import pytest
from ch5mpy import File, H5Mode
from h5py import string_dtype

from vdata.tdf import TemporalDataFrame, TemporalDataFrameBase
from vdata.timepoint import TimePointArray

REFERENCE_BACKED_DATA = {
    'type': 'tdf',
    'locked_indices': False,
    'locked_columns': False,
    'timepoints_column_name': None,
    'index': np.concatenate((np.arange(50, 100), np.arange(0, 50))),
    'repeating_index': False,
    'columns_numerical': np.array(['col1', 'col2'], dtype=np.dtype('O')),
    'columns_string': np.array(['col3', 'col4'], dtype=np.dtype('O')),
    'timepoints': np.array([0 for _ in range(50)] + [1 for _ in range(50)], dtype=float),
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


def get_backed_TDF(name: str,
                   view: tuple | None = None) -> TemporalDataFrameBase:
    with File('backed_TDF_' + name, H5Mode.WRITE_TRUNCATE) as h5_file:
        h5_file.attrs['name'] = name
        h5_file.attrs['locked_indices'] = REFERENCE_BACKED_DATA['locked_indices']
        h5_file.attrs['locked_columns'] = REFERENCE_BACKED_DATA['locked_columns']
        h5_file.attrs['timepoints_column_name'] = REFERENCE_BACKED_DATA['timepoints_column_name']
        h5_file.attrs['repeating_index'] = REFERENCE_BACKED_DATA['repeating_index']

        h5_file.create_dataset('index', data=REFERENCE_BACKED_DATA['index'])
        
        h5_file.create_dataset('columns_numerical', data=REFERENCE_BACKED_DATA['columns_numerical'],
                               chunks=True, maxshape=(None,), dtype=string_dtype())
        h5_file['columns_numerical'].attrs['dtype'] = '<U4'
        
        h5_file.create_dataset('columns_string', data=REFERENCE_BACKED_DATA['columns_string'],
                               chunks=True, maxshape=(None,), dtype=string_dtype())
        h5_file['columns_string'].attrs['dtype'] = '<U4'

        h5_file.create_dataset('numerical_array', data=REFERENCE_BACKED_DATA['values_numerical'],
                               chunks=True, maxshape=(None, None))

        h5_file.create_dataset('string_array', data=REFERENCE_BACKED_DATA['values_string'], dtype=string_dtype(),
                               chunks=True, maxshape=(None, None))
        h5_file['string_array'].attrs['dtype'] = '<U4'
        
        h5_file.create_group('timepoints_array')
        h5_file['timepoints_array'].attrs['__h5_type__'] = 'object'
        h5_file['timepoints_array'].attrs['__h5_class__'] = np.void(pickle.dumps(TimePointArray, 
                                                                                 protocol=pickle.HIGHEST_PROTOCOL))
        h5_file['timepoints_array'].attrs['unit'] = 'h'
        h5_file['timepoints_array'].create_dataset('array', data=REFERENCE_BACKED_DATA['timepoints'])
 
    # read tdf from file
    tdf = TemporalDataFrame.read('backed_TDF_' + name, mode=H5Mode.READ_WRITE)
    
    if view is not None:
        return tdf[view]
    
    return tdf
    
    
def clean(tdf: TemporalDataFrameBase) -> None:
    filename = tdf.data.file.filename
    
    if tdf.is_view:
        tdf.parent.close()
    else:
        tdf.close()
        
    Path(filename).unlink()


@pytest.fixture(scope='function')
def TDF(request: Any) -> Generator[TemporalDataFrameBase, None, None]:
    if hasattr(request, 'param'):
        which = request.param

    else:
        which = 'plain'

    if 'backed' in which:
        if 'view' in which:
            tdf = get_backed_TDF('1', (slice(None), np.arange(10, 90), ['col1', 'col4']))

        else:
            tdf = get_backed_TDF('1')
            
        yield tdf
        clean(tdf)

    else:
        if 'view' in which:
            yield get_TDF()[:, np.arange(10, 90), ['col1', 'col4']]

        else:
            yield get_TDF()


@pytest.fixture(scope='class')
def class_TDF_backed(request: Any) -> Generator[None, None, None]:    
    tdf = get_backed_TDF('1')
    request.cls.TDF = tdf
    yield
    clean(tdf)


@pytest.fixture
def TDF1(request: Any) -> Generator[TemporalDataFrameBase, None, None]:
    if hasattr(request, 'param'):
        which = request.param

    else:
        which = 'plain'

    if 'backed' in which:
        if 'view' in which:
            tdf = get_backed_TDF('1', (slice(None),))

        else:
            tdf = get_backed_TDF('1')
            
        yield tdf
        clean(tdf)

    else:
        if 'view' in which:
            yield get_TDF()[:]

        else:
            yield get_TDF()


@pytest.fixture(scope='class')
def class_TDF1(request: Any) -> Generator[None, None, None]:
    if hasattr(request, 'param'):
        which = request.param

    else:
        which = 'plain'

    if 'backed' in which:
        if 'view' in which:
            tdf = get_backed_TDF('1', (slice(None),))

        else:
            tdf = get_backed_TDF('1', (slice(None),))
        
        request.cls.TDF = tdf
        yield
        clean(tdf)

    else:
        if 'view' in which:
            request.cls.TDF = get_TDF()[:]
            yield

        else:
            request.cls.TDF = get_TDF()
            yield


@pytest.fixture
def TDF2(request: Any) -> Generator[TemporalDataFrameBase, None, None]:
    if hasattr(request, 'param'):
        which = request.param

    else:
        which = 'plain'

    if 'backed' in which:
        if 'view' in which:
            tdf = get_backed_TDF('2', (slice(None),))

        else:
            tdf = get_backed_TDF('2')
            
        yield tdf
        clean(tdf)

    else:
        if 'view' in which:
            yield get_TDF('2')[:]

        else:
            yield get_TDF('2')


@pytest.fixture(scope='class')
def h5_file(request: Any) -> Generator[None, None, None]:
    if hasattr(request, 'param'):
        which = request.param

    else:
        which = 'plain'

    if 'backed' in which:
        if 'view' in which:
            tdfv = get_backed_TDF('1', (slice(None), np.arange(10, 90), ['col1', 'col4']))

        else:
            tdfv = get_backed_TDF('1')
        
        tdfv.write('TDF_write.vd')
        clean(tdfv)

    else:
        if 'view' in which:
            tdf: TemporalDataFrameBase = get_TDF()[:, np.arange(10, 90), ['col1', 'col4']]

        else:
            tdf = get_TDF()

        tdf.write('TDF_write.vd')

    request.cls.h5_file = File('TDF_write.vd', H5Mode.READ)
    yield

    Path('TDF_write.vd').unlink()
