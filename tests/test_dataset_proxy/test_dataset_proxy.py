# coding: utf-8
# Created on 24/10/2022 10:39
# Author : matteo

# ====================================================
# imports
import pytest

from vdata.core.dataset_proxy import int_, float_, num_, str_, tp_


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
    assert num_dataset.dtype == int_


def test_str_dataset_gives_correct_dtype(str_dataset):
    assert str_dataset.dtype == str_


def test_tp_dataset_gives_correct_dtype(tp_dataset):
    assert tp_dataset.dtype == tp_


def test_dataset_proxy_creation_from_num_dataset_gives_correct_dtype(dataset):
    assert dataset.dtype == int_


def test_dataset_proxy_creation_from_num_dataset_gives_correct_shape(dataset):
    assert dataset.shape == (10, 5)


def test_num_dataset_proxy_can_be_cast_to_str(dataset):
    dataset.astype(str_)

    assert dataset.dtype == str_


@pytest.mark.parametrize(
    'dataset',
    ['str'],
    indirect=True
)
def test_str_dataset_proxy_can_be_cast_to_num(dataset):
    dataset.astype(num_)

    assert dataset.dtype == float_
