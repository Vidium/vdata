import numpy as np

from vdata.timepoint import TimePointArray


def test_timepointarray_equality_check():
    tpa = TimePointArray([1, 1, 2, 3, 4, 5, 1])
    assert np.array_equal(tpa == '1h', [True, True, False, False, False, False, True])
    