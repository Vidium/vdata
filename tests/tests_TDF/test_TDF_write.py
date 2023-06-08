# coding: utf-8
# Created on 30/03/2022 17:35
# Author : matteo

# ====================================================
# imports
from pathlib import Path

import ch5mpy as ch
import pytest

from vdata import TemporalDataFrame
from vdata.tdf import TemporalDataFrameBase


# ====================================================
# code
@pytest.mark.parametrize(
    'TDF',
    ['plain', 'view', 'backed', 'backed view'],
    indirect=True
)
def test_can_read_written_TDF(TDF: TemporalDataFrameBase) -> None:
    TDF.write('TDF_write.vd')

    tdf = TemporalDataFrame.read('TDF_write.vd')
    assert isinstance(tdf, TemporalDataFrameBase)

    tdf.close()
    Path('TDF_write.vd').unlink()


@pytest.mark.usefixtures('h5_file')
class TestWriteTDF:
    h5_file: ch.File

    def test_wrote_correct_attributes(self) -> None:
        assert sorted(list(self.h5_file.attrs.keys())) == ['__h5_class__', '__h5_type__', 
                                                           'locked_columns', 'locked_indices', 'name',
                                                           'repeating_index', 'timepoints_column_name']

    def test_wrote_datasets(self) -> None:
        assert sorted(list(self.h5_file.keys())) == ['columns_numerical', 'columns_string', 'index',
                                                     'numerical_array', 'string_array', 'timepoints_array']

    @pytest.mark.parametrize('TDF', ['plain'], indirect=True)
    def test_wrote_all_data_correctly(self, TDF: TemporalDataFrame) -> None:
        written_tdf = TemporalDataFrame.read(self.h5_file)

        assert written_tdf == TDF
