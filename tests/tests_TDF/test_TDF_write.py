# coding: utf-8
# Created on 30/03/2022 17:35
# Author : matteo
from pathlib import Path

# ====================================================
# imports
import pytest
from h5py import File

from vdata import read_TDF, BackedTemporalDataFrame
from vdata.core.tdf.base import BaseTemporalDataFrame
from vdata.name_utils import H5Mode


# ====================================================
# code
@pytest.mark.parametrize(
    'TDF',
    ['plain', 'view', 'backed', 'backed view'],
    indirect=True
)
def test_can_read_written_TDF(TDF):
    TDF.write('TDF_write.vd')

    tdf = read_TDF('TDF_write.vd')
    assert isinstance(tdf, BaseTemporalDataFrame)

    tdf.close()
    Path('TDF_write.vd').unlink()


@pytest.mark.parametrize(
    'TDF',
    ['backed'],
    indirect=True
)
def test_write_backed_TDF_to_same_file(TDF):
    TDF.write()


@pytest.mark.parametrize(
    'TDF',
    ['backed view'],
    indirect=True
)
def test_write_backed_view_with_no_file_should_fail(TDF):
    with pytest.raises(ValueError):
        TDF.write()


@pytest.mark.usefixtures('h5_file')
class TestWriteTDF:

    def test_wrote_correct_type(self):
        assert self.h5_file.attrs['type'] == 'tdf'

    def test_wrote_correct_attributes(self):
        assert sorted(list(self.h5_file.attrs.keys())) == ['locked_columns', 'locked_indices', 'name',
                                                           'repeating_index', 'timepoints_column_name', 'type']

    def test_wrote_datasets(self):
        assert sorted(list(self.h5_file.keys())) == ['columns_numerical', 'columns_string', 'index', 'timepoints',
                                                     'values_numerical', 'values_string']

    @pytest.mark.parametrize('TDF', ['plain'], indirect=True)
    def test_wrote_all_data_correctly(self, TDF):
        written_tdf = BackedTemporalDataFrame(self.h5_file)

        assert written_tdf == TDF
