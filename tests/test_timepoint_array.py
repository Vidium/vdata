from tempfile import TemporaryDirectory

import ch5mpy as ch
import numpy as np
import pytest
from pytest import fixture

from vdata.timepoint import TimePoint, TimePointArray, TimePointIndex


@fixture
def tpi():
    return TimePointIndex.from_array(TimePointArray([1, 1, 2, 2, 2, 2, 3, 3, 3]))


def test_timepointarray_equality_check():
    tpa = TimePointArray([1, 1, 2, 3, 4, 5, 1])
    assert np.array_equal(tpa == "1h", [True, True, False, False, False, False, True])


def test_timepointindex_disordered():
    tpi = TimePointIndex.from_array(TimePointArray([2, 2, 2, 2, 1, 1, 3, 3, 3]))
    assert np.array_equal(tpi.ranges, [4, 6, 9])
    assert np.array_equal(tpi.timepoints, TimePointArray([2, 1, 3]))


def test_timepointindex_len():
    tpi = TimePointIndex.from_array(TimePointArray([1, 1, 2, 2, 2, 2, 3, 3, 3]))
    assert len(tpi) == 9


def test_timepointindex_can_convert_to_tparray(tpi):
    tpa = tpi.as_array()
    assert isinstance(tpa, TimePointArray)
    assert np.array_equal(tpa, TimePointArray([1, 1, 2, 2, 2, 2, 3, 3, 3]))


def test_timepointindex_should_get_indices_at(tpi):
    assert np.array_equal(tpi.at(TimePoint("2h")), [2, 3, 4, 5])


def test_timepointindex_should_get_indices_where(tpi):
    assert np.array_equal(
        tpi.where(TimePoint("2h")), np.array([False, False, True, True, True, True, False, False, False])
    )


def test_timpointindex_should_getitem(tpi):
    assert tpi[0] == TimePoint("1h")
    assert tpi[3] == TimePoint("2h")
    assert tpi[-1] == TimePoint("3h")

    with pytest.raises(IndexError):
        tpi[9]

    with pytest.raises(IndexError):
        tpi[-10]


@pytest.mark.parametrize(
    "slicer, result",
    [
        [slice(None), TimePointIndex.from_array(TimePointArray([1, 1, 2, 2, 2, 2, 3, 3, 3]))],
        [slice(None, 7), TimePointIndex.from_array(TimePointArray([1, 1, 2, 2, 2, 2, 3]))],
        [slice(3, None), TimePointIndex.from_array(TimePointArray([2, 2, 2, 3, 3, 3]))],
        [slice(3, 7), TimePointIndex.from_array(TimePointArray([2, 2, 2, 3]))],
        [slice(3, 6), TimePointIndex.from_array(TimePointArray([2, 2, 2]))],
    ],
)
def test_timepointindex_should_getitem_from_slice(tpi, slicer, result):
    assert tpi[slicer] == result


def test_timepointindex_should_write_read(tpi):
    with TemporaryDirectory() as dir:
        with ch.File(dir + "/tpi.h5", "w") as ch_file:
            ch.write_object(tpi, ch_file, "tpi")

            assert tuple(ch_file["tpi"].keys()) == ("timepoints", "ranges")
            assert np.array_equal(ch_file["tpi"]["timepoints"]["array"][:], [1, 2, 3])
            assert np.array_equal(ch_file["tpi"]["ranges"][:], [2, 6, 9])

            tpi2 = TimePointIndex.read(ch.H5Dict.read(dir + "/tpi.h5", "tpi", mode=ch.H5Mode.READ))
            assert tpi == tpi2
