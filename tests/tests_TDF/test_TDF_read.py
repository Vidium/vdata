# coding: utf-8
# Created on 31/03/2022 11:24
# Author : matteo

# ====================================================
# imports
import numpy as np
import pytest

from vdata.tdf import TemporalDataFrameBase


# ====================================================
# code
@pytest.mark.usefixtures('class_TDF_backed')
class TestReadTDF:
    TDF: TemporalDataFrameBase

    def test_can_read_TDF(self) -> None:
        assert isinstance(self.TDF, TemporalDataFrameBase) and self.TDF.is_backed

    def test_read_TDF_attributes(self) -> None:
        assert self.TDF.name == '1'

    def test_read_index(self) -> None:
        assert np.all(self.TDF.index == np.concatenate((np.arange(50, 100), np.arange(0, 50))))

    def test_read_timepoints(self) -> None:
        assert np.all(self.TDF.timepoints_column == ['0.0h' for _ in range(50)] + ['1.0h' for _ in range(50)])

    def test_read_values(self) -> None:
        assert np.all(self.TDF.values_num == np.vstack((
            np.concatenate((np.arange(50, 100), np.arange(0, 50))),
            np.concatenate((np.arange(150, 200), np.arange(100, 150)))
        )).T.astype(float))

    def test_get_correct_shape(self) -> None:
        assert self.TDF.shape == (2, [50, 50], 4)
