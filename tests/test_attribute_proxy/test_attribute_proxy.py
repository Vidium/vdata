# coding: utf-8
# Created on 23/10/2022 10:09
# Author : matteo

# ====================================================
# imports
import pytest


# ====================================================
# code
def test_gets_correct_value(attr):
    assert attr['name'] == 'test_name'


def test_raises_error_if_item_not_in_attributes(attr):
    with pytest.raises(KeyError):
        _ = attr['no attr']


def test_can_set_new_value(attr):
    attr['test_new'] = 'test_new'
    assert attr['test_new'] == 'test_new'


def test_can_modify_existing_value(attr):
    attr['type'] = 'new_type'
    assert attr['type'] == 'new_type'


def test_returns_None_when_attribute_is_None(attr):
    assert attr['none'] is None


def test_can_set_none_value(attr):
    attr['test_none'] = None
    assert attr['test_none'] is None
