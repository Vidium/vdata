# coding: utf-8
# Created on 04/04/2022 17:24
# Author : matteo

# ====================================================
# imports
import pytest
import numpy as np

from vdata import TemporalDataFrame


# ====================================================
# code
@pytest.mark.parametrize(
    'TDF',
    ['plain', 'view', 'backed', 'backed view'],
    indirect=True
)
def test_convert_without_timepoints(TDF):
    df = TDF.to_pandas()

    nb_col = 1 if TDF.is_view else 2

    assert np.all(df.values[:, :nb_col] == TDF.values_num) and np.all(TDF.values_str == df.values[:, nb_col:])


@pytest.mark.parametrize(
    'TDF',
    ['plain', 'view', 'backed', 'backed view'],
    indirect=True
)
def test_convert_with_timepoints_as_str(TDF):
    df = TDF.to_pandas(with_timepoints='timepoints', timepoints_type='string')

    assert df.columns[0] == 'timepoints' and np.all(df.values[:, 0] == TDF.timepoints_column_str)


@pytest.mark.parametrize(
    'TDF',
    ['plain', 'view', 'backed', 'backed view'],
    indirect=True
)
def test_convert_with_timepoints_as_numbers(TDF):
    df = TDF.to_pandas(with_timepoints='timepoints', timepoints_type='numerical')

    assert df.columns[0] == 'timepoints' and np.all(df.values[:, 0] == TDF.timepoints_column_numerical)


@pytest.mark.parametrize(
    'TDF',
    ['plain', 'view', 'backed', 'backed view'],
    indirect=True
)
def test_create_copy_TDF(TDF):
    copy = TDF.copy()

    assert isinstance(copy, TemporalDataFrame)


@pytest.mark.parametrize(
    'TDF',
    ['plain', 'view', 'backed', 'backed view'],
    indirect=True
)
def test_modifying_copy_does_not_modify_original(TDF):
    copy = TDF.copy()

    copy[:, :, 'col1'] = -1

    assert np.sum(copy.values_num[:, 0] == TDF.values_num[:, 0]) == 0
