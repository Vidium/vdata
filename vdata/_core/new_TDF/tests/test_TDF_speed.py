# coding: utf-8
# Created on 06/04/2022 12:04
# Author : matteo

# ====================================================
# imports
from time import time

from ..dataframe import TemporalDataFrame


# ====================================================
# code
MAX_ELAPSED_TIME = 5


def test_speed():
    TDF = TemporalDataFrame({'col1': [i for i in range(20_000)],
                             'col2': [i for i in range(20_000, 40_000)],
                             'col3': [str(i) for i in range(40_000, 60_000)],
                             'col4': [str(i) for i in range(60_000, 80_000)]},
                            name='1',
                            time_list=['0h' for _ in range(2500)] +
                                      ['1h' for _ in range(2500)] +
                                      ['2h' for _ in range(2500)] +
                                      ['3h' for _ in range(2500)] +
                                      ['4h' for _ in range(2500)] +
                                      ['5h' for _ in range(2500)] +
                                      ['6h' for _ in range(2500)] +
                                      ['7h' for _ in range(2500)])

    # TDF representation
    start = time()
    repr(TDF)

    assert time() - start < MAX_ELAPSED_TIME

    # TDF sub-setting
    start = time()
    view = TDF[['2h', '0h', '4h'], range(5_000, 15_000)]

    assert time() - start < MAX_ELAPSED_TIME

    # view TDF representation
    start = time()
    repr(view)

    assert time() - start < MAX_ELAPSED_TIME

    # TDF view sub

    # backed TDF
    # TODO


if __name__ == '__main__':
    test_speed()
