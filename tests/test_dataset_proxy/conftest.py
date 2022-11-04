# coding: utf-8
# Created on 24/10/2022 10:40
# Author : matteo
from pathlib import Path

import numpy as np
# ====================================================
# imports
import pytest
from vdata.h5pickle import File
from h5py import string_dtype

from vdata.core.dataset_proxy.dataset import DatasetProxy
from vdata.core.dataset_proxy.dataset_1D import TPDatasetProxy1D
from vdata.core.dataset_proxy.dataset_2D import NumDatasetProxy2D, StrDatasetProxy2D
from vdata.h5pickle.name_utils import H5Mode


# ====================================================
# code
@pytest.fixture
def num_dataset(request) -> NumDatasetProxy2D:
    if hasattr(request, 'param'):
        which = request.param

    else:
        which = 'plain'

    h5_file = File('NumDatasetProxy2D', H5Mode.WRITE_TRUNCATE)
    h5_file.create_dataset('data', data=np.arange(10*5).reshape(10, 5))

    if 'view' in which:
        dataset = NumDatasetProxy2D(h5_file['data'], view_on=(np.arange(2, 8),
                                                              np.array([0, 2, 4])))

    else:
        dataset = NumDatasetProxy2D(h5_file['data'])

    yield dataset

    h5_file.close()
    Path('NumDatasetProxy2D').unlink()


@pytest.fixture
def str_dataset() -> StrDatasetProxy2D:
    h5_file = File('StrDatasetProxy2D', H5Mode.WRITE_TRUNCATE)
    h5_file.create_dataset('data', data=np.arange(10*5).astype(str).astype('O').reshape(10, 5), dtype=string_dtype())

    yield StrDatasetProxy2D(h5_file['data'])

    h5_file.close()
    Path('StrDatasetProxy2D').unlink()


@pytest.fixture
def tp_dataset() -> TPDatasetProxy1D:
    h5_file = File('TPDatasetProxy1D', H5Mode.WRITE_TRUNCATE)
    h5_file.create_dataset('data', data=[b'0.0h', b'1.0h', b'2.0h'])

    yield TPDatasetProxy1D(h5_file['data'])

    h5_file.close()
    Path('TPDatasetProxy1D').unlink()


@pytest.fixture
def dataset_proxy(request, num_dataset, str_dataset) -> DatasetProxy:
    if hasattr(request, 'param'):
        typ = request.param

    else:
        typ = 'num'

    if typ == 'num':
        return DatasetProxy(num_dataset)

    elif typ == 'str':
        return DatasetProxy(str_dataset)

    raise TypeError


@pytest.fixture
def dataset(request, num_dataset, str_dataset, tp_dataset):
    if hasattr(request, 'param'):
        which = request.param

    else:
        which = 'num'

    if which == 'num':
        return DatasetProxy(num_dataset)

    elif which == 'str':
        return DatasetProxy(str_dataset)

    elif which == 'tp':
        return DatasetProxy(tp_dataset)

    raise ValueError
