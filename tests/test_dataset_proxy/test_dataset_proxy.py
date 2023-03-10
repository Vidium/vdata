# coding: utf-8
# Created on 24/10/2022 10:39
# Author : matteo

# ====================================================
# imports
import pytest
import numpy as np
from h5py import File
from tempfile import TemporaryFile

from vdata.core.dataset_proxy import DatasetProxy
from vdata.core.dataset_proxy.utils import auto_DatasetProxy
from vdata.h5pickle.name_utils import H5Mode
from vdata.time_point import TimePoint


# ====================================================
# code
@pytest.mark.parametrize(
    'num_dataset,shape',
    [
        ('plain', (10, 5)),
        ('view', (6, 3))
    ],
    indirect=['num_dataset']
)
def test_dataset_gives_correct_shape(num_dataset, shape):
    assert num_dataset.shape == shape


@pytest.mark.parametrize(
    'num_dataset,size',
    [
        ('plain', 50),
        ('view', 18)
    ],
    indirect=['num_dataset']
)
def test_dataset_gives_correct_size(num_dataset, size):
    assert num_dataset.size == size


def test_num_dataset_gives_correct_dtype(num_dataset):
    assert num_dataset.dtype == np.int64


def test_str_dataset_gives_correct_dtype(str_dataset):
    assert str_dataset.dtype == np.dtype('<U2')


def test_str_dataset_should_return_unique_values(str_dataset):
    str_dataset[:, -1] = ['a', 'b', 'c', 'a', 'b', 'c', 'a', 'b', 'c', 'a']

    assert np.all(str_dataset.unique() == ['0', '1', '10', '11', '12', '13', '15', '16', '17', '18', '2',
                                           '20', '21', '22', '23', '25', '26', '27', '28', '3', '30', '31',
                                           '32', '33', '35', '36', '37', '38', '40', '41', '42', '43', '45',
                                           '46', '47', '48', '5', '6', '7', '8', 'a', 'b', 'c'])


def test_str_dataset_should_return_unique_values_with_numpy(str_dataset):
    np.unique(str_dataset)

    assert np.all(np.unique(str_dataset) == ['0', '1', '10', '11', '12', '13', '14', '15', '16', '17', '18',
                                             '19', '2', '20', '21', '22', '23', '24', '25', '26', '27', '28',
                                             '29', '3', '30', '31', '32', '33', '34', '35', '36', '37', '38',
                                             '39', '4', '40', '41', '42', '43', '44', '45', '46', '47', '48',
                                             '49', '5', '6', '7', '8', '9'])


def test_tp_dataset_gives_correct_dtype(tp_dataset):
    assert tp_dataset.dtype == TimePoint


def test_dataset_proxy_creation_from_num_dataset_gives_correct_dtype(dataset_proxy):
    assert dataset_proxy.dtype == np.int64


def test_dataset_proxy_creation_from_num_dataset_gives_correct_shape(dataset_proxy):
    assert dataset_proxy.shape == (10, 5)


def test_num_dataset_proxy_can_be_cast_to_str(dataset_proxy):
    dataset_proxy.astype(str)

    assert dataset_proxy.dtype == np.dtype('<U2')


@pytest.mark.parametrize(
    'dataset_proxy',
    ['str'],
    indirect=True
)
def test_str_dataset_proxy_can_be_cast_to_num(dataset_proxy):
    dataset_proxy.astype(float)

    assert dataset_proxy.dtype == np.float64


def test_can_add_inplace(dataset_proxy):
    dataset_proxy += 1.5

    assert dataset_proxy[0, 0] == 1.5


def test_can_add(dataset_proxy):
    arr = dataset_proxy + 1.5

    assert arr[0, 0] == 1.5


def test_can_sub_inplace(dataset_proxy):
    dataset_proxy -= 1.5

    assert dataset_proxy[0, 0] == -1.5


def test_can_sub(dataset_proxy):
    arr = dataset_proxy - 1.5

    assert arr[0, 0] == -1.5


def test_can_mul_inplace(dataset_proxy):
    dataset_proxy *= 2.5

    assert dataset_proxy[0, 1] == 2.5


def test_can_mul(dataset_proxy):
    arr = dataset_proxy * 2.5

    assert arr[0, 1] == 2.5


def test_can_div_inplace(dataset_proxy):
    dataset_proxy /= 2

    assert dataset_proxy[0, 1] == 0.5


def test_can_div(dataset_proxy):
    arr = dataset_proxy / 2

    assert arr[0, 1] == 0.5


def test_should_subset_index_in_disorder(dataset_proxy):
    view = dataset_proxy[[2, 4, 1]]

    assert isinstance(view, np.ndarray)


def test_subset_index_in_disorder_should_return_index_in_disorder(dataset_proxy):
    view = dataset_proxy[[2, 4, 1]]

    assert np.all(view[:, 0] == [10, 20, 5])


def test_should_create_dataset_3D():
    temp = TemporaryFile()

    with File(temp, H5Mode.WRITE_TRUNCATE) as h5_file:
        h5_file.create_dataset('data', data=np.arange(10 * 5 * 5).reshape((10, 5, 5)))
        dataset = auto_DatasetProxy(h5_file['data'])

        assert dataset.ndim == 3


def test_should_create_dataset_proxy_from_3D_array(dataset_proxy_3d):
    assert dataset_proxy_3d.ndim == 3


def test_should_subset_dataset_proxy_from_3D_array(dataset_proxy_3d):
    assert np.all(dataset_proxy_3d[1, 2:4, 2:4] == np.array([[37, 38],
                                                             [42, 43]]))
