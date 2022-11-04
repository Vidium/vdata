# coding: utf-8
# Created on 04/11/2022 13:49
# Author : matteo

# ====================================================
# imports
import pytest
import pickle
import numpy as np
from vdata import TimePoint

from vdata.core.dataset_proxy import DatasetProxy


# ====================================================
# code
@pytest.fixture
def pickled_dataset(request, num_dataset, str_dataset, tp_dataset):
    if hasattr(request, 'param'):
        which = request.param

    else:
        which = 'num'

    if 'num' in which:
        dataset = DatasetProxy(num_dataset)

    elif 'str' in which:
        dataset = DatasetProxy(str_dataset)

    elif 'tp' in which:
        dataset = DatasetProxy(tp_dataset)

    else:
        raise ValueError

    if 'view' in which:
        dataset._proxy._view_on = (np.array([2, 3, 4, 5, 6, 7]), np.array([1, 2, 3]))

    pickled = pickle.dumps(dataset)
    data = dataset[:]

    dataset.close()

    return pickled, data


@pytest.mark.parametrize(
    'dataset',
    ['num', 'str', 'tp'],
    indirect=True
)
def test_should_pickle(dataset):
    _ = pickle.dumps(dataset)


@pytest.mark.parametrize(
    'pickled_dataset',
    ['num', 'str', 'tp'],
    indirect=True
)
def test_should_unpickle(pickled_dataset):
    un_pickled = pickle.loads(pickled_dataset[0])

    assert np.all(un_pickled[:] == pickled_dataset[1])


@pytest.mark.parametrize(
    'pickled_dataset',
    ['num view'],
    indirect=True
)
def test_view_should_unpickle(pickled_dataset):
    un_pickled = pickle.loads(pickled_dataset[0])

    assert np.all(un_pickled[:] == pickled_dataset[1])


@pytest.mark.parametrize(
    'pickled_dataset,dtype',
    [
        ('num', np.int64),
        ('str', np.dtype('<U2')),
        ('tp', TimePoint)
    ],
    indirect=['pickled_dataset']
)
def test_should_keep_dtype(pickled_dataset, dtype):
    un_pickled = pickle.loads(pickled_dataset[0])

    assert un_pickled.dtype == dtype
