# coding: utf-8
# Created on 03/05/2022 09:40
# Author : matteo

# ====================================================
# imports
import pytest

from vdata.tdf import TemporalDataFrameBase


# ====================================================
@pytest.mark.parametrize("TDF1", ["plain", "backed"], indirect=True)
@pytest.mark.parametrize("TDF2", ["plain", "backed"], indirect=True)
def test_are_equal(TDF1: TemporalDataFrameBase, TDF2: TemporalDataFrameBase) -> None:
    assert id(TDF1) != id(TDF2)
    assert TDF1 == TDF2


@pytest.mark.parametrize("TDF1", ["view", "backed view"], indirect=True)
@pytest.mark.parametrize("TDF2", ["view", "backed view"], indirect=True)
def test_are_equal_views(TDF1: TemporalDataFrameBase, TDF2: TemporalDataFrameBase) -> None:
    assert id(TDF1) != id(TDF2)
    assert TDF1 == TDF2


@pytest.mark.parametrize("TDF1", ["plain", "view", "backed", "backed view"], indirect=True)
@pytest.mark.parametrize("TDF2", ["plain", "view", "backed", "backed view"], indirect=True)
def test_are_different_when_modified(TDF1: TemporalDataFrameBase, TDF2: TemporalDataFrameBase) -> None:
    TDF2.iloc[0, 0] = -1
    assert TDF1 != TDF2
