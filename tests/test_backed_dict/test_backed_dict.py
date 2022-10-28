# coding: utf-8
# Created on 16/10/2022 12:06
# Author : matteo

# ====================================================
# imports
import pytest
import numpy as np
from pathlib import Path

from h5py import File

from vdata.core.backed_dict import BackedDict
from vdata.name_utils import H5Mode
from vdata.read_write.write import write_Dict


# ====================================================
# code
@pytest.fixture(scope='module')
def backed_dict() -> BackedDict:
    data = {'a': 1,
            'b': [1, 2, 3],
            'c': {'d': 'test',
                  'e': np.arange(100)}}

    with File('backed_dict', H5Mode.WRITE_TRUNCATE) as h5_file:
        write_Dict(data, h5_file, key='uns')

    yield BackedDict(File('backed_dict', H5Mode.READ_WRITE)['uns'])

    Path('backed_dict').unlink()


def test_backed_dict_creation(backed_dict):
    assert isinstance(backed_dict, BackedDict)


def test_backed_dict_can_iterate_through_keys(backed_dict):
    assert list(iter(backed_dict)) == ['a', 'b', 'c']


def test_backed_dict_has_correct_keys(backed_dict):
    assert list(backed_dict.keys()) == ['a', 'b', 'c']


def test_backed_dict_can_get_regular_values(backed_dict):
    assert backed_dict['a'] == 1


def test_backed_dict_should_return_string(backed_dict):
    assert isinstance(backed_dict['c']['d'], str)


def test_backed_dict_gets_nested_backed_dicts(backed_dict):
    assert isinstance(backed_dict['c'], BackedDict)


def test_backed_dict_has_correct_values(backed_dict):
    values_list = list(backed_dict.values())

    assert values_list[0] == 1 and \
           np.all(values_list[1] == [1, 2, 3]) and \
           list(values_list[2].keys()) == ['d', 'e']


def test_backed_dict_has_correct_items(backed_dict):
    assert list(backed_dict.items())[0] == ('a', 1)


def test_backed_dict_can_set_regular_value(backed_dict):
    backed_dict['a'] = 5

    assert np.all(backed_dict['a'] == 5)


def test_backed_dict_can_set_array_value(backed_dict):
    backed_dict['b'][1] = 6

    assert np.all(backed_dict['b'] == [1, 6, 3])


def test_backed_dict_can_set_new_regular_value(backed_dict):
    backed_dict['x'] = 9

    assert backed_dict['x'] == 9


def test_backed_dict_can_set_new_array(backed_dict):
    backed_dict['y'] = np.array([1, 2, 3])

    assert np.all(backed_dict['y'] == [1, 2, 3])


def test_backed_dict_can_set_new_dict(backed_dict):
    backed_dict['z'] = {'l': 10,
                        'm': [10, 11, 12],
                        'n': {'o': 13}}

    assert isinstance(backed_dict['z'], BackedDict) and \
           backed_dict['z']['l'] == 10 and \
           np.all(backed_dict['z']['m'] == [10, 11, 12]) and \
           isinstance(backed_dict['z']['n'], BackedDict) and \
           backed_dict['z']['n']['o'] == 13


def test_backed_dict_can_delete_regular_value(backed_dict):
    del backed_dict['a']

    assert 'a' not in backed_dict.keys()


def test_backed_dict_can_delete_array(backed_dict):
    del backed_dict['b']

    assert 'b' not in backed_dict.keys()


def test_backed_dict_can_delete_dict(backed_dict):
    del backed_dict['c']

    assert 'c' not in backed_dict.keys()


def test_backed_dict_can_close_file(backed_dict):
    backed_dict.close()

    with pytest.raises(ValueError):
        _ = backed_dict['x']
