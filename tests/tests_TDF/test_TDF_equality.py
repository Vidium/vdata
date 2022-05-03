# coding: utf-8
# Created on 03/05/2022 09:40
# Author : matteo

# ====================================================
# imports
import numpy as np

from .utils import get_TDF
from vdata import TemporalDataFrame


# ====================================================
# code
def test_equality():
    # TDF is not a view
    #   other is TDF
    TDF1 = get_TDF('1')
    TDF2 = get_TDF('2')

    assert id(TDF1) != id(TDF2)
    assert TDF1 == TDF2

    TDF2.iloc[0, 0] = -50
    assert TDF1 != TDF2

    #   other is single value
    assert np.array_equal(TDF1 == 0, np.array([[False, False],
                                               [False, False],
                                               [False, False],
                                               [False, False],
                                               [False, False],
                                               [False, False],
                                               [False, False],
                                               [False, False],
                                               [False, False],
                                               [False, False],
                                               [False, False],
                                               [False, False],
                                               [False, False],
                                               [False, False],
                                               [False, False],
                                               [False, False],
                                               [False, False],
                                               [False, False],
                                               [False, False],
                                               [False, False],
                                               [False, False],
                                               [False, False],
                                               [False, False],
                                               [False, False],
                                               [False, False],
                                               [False, False],
                                               [False, False],
                                               [False, False],
                                               [False, False],
                                               [False, False],
                                               [False, False],
                                               [False, False],
                                               [False, False],
                                               [False, False],
                                               [False, False],
                                               [False, False],
                                               [False, False],
                                               [False, False],
                                               [False, False],
                                               [False, False],
                                               [False, False],
                                               [False, False],
                                               [False, False],
                                               [False, False],
                                               [False, False],
                                               [False, False],
                                               [False, False],
                                               [False, False],
                                               [False, False],
                                               [False, False],
                                               [True, False],
                                               [False, False],
                                               [False, False],
                                               [False, False],
                                               [False, False],
                                               [False, False],
                                               [False, False],
                                               [False, False],
                                               [False, False],
                                               [False, False],
                                               [False, False],
                                               [False, False],
                                               [False, False],
                                               [False, False],
                                               [False, False],
                                               [False, False],
                                               [False, False],
                                               [False, False],
                                               [False, False],
                                               [False, False],
                                               [False, False],
                                               [False, False],
                                               [False, False],
                                               [False, False],
                                               [False, False],
                                               [False, False],
                                               [False, False],
                                               [False, False],
                                               [False, False],
                                               [False, False],
                                               [False, False],
                                               [False, False],
                                               [False, False],
                                               [False, False],
                                               [False, False],
                                               [False, False],
                                               [False, False],
                                               [False, False],
                                               [False, False],
                                               [False, False],
                                               [False, False],
                                               [False, False],
                                               [False, False],
                                               [False, False],
                                               [False, False],
                                               [False, False],
                                               [False, False],
                                               [False, False],
                                               [False, False],
                                               [False, False]]))

    # TDF is a view
    vTDF1 = TDF1['0h', 50:60:2, 'col1']

    #   other is a TDF
    TDF2 = TemporalDataFrame({'col1': [50, 52, 54, 56, 58]},
                             index=[50, 52, 54, 56, 58],
                             name='3',
                             time_list=['0h' for _ in range(5)])

    assert vTDF1 == TDF2

    #   other is a view
    vTDF2 = get_TDF('4')['0h', 50:60:2, 'col1']

    assert vTDF1 == vTDF2

    #   other is a single value
    assert np.array_equal(vTDF1 == 52, np.array([[False],
                                                 [True],
                                                 [False],
                                                 [False],
                                                 [False]]))
