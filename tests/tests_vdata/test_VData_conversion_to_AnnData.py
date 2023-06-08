# coding: utf-8
# Created on 20/01/2021 16:58
# Author : matteo

# ====================================================
# imports
import numpy as np
import pandas as pd
import pytest

import vdata


# ====================================================
# code
@pytest.fixture
def v() -> vdata.VData:
    _tp = ['0h' for _ in range(7)] + ['1h' for _ in range(3)] + ['2h' for _ in range(10)]
    _RNA = pd.DataFrame(np.vstack((np.zeros((7, 4)),
                        np.ones((3, 4)),
                        2 * np.ones((10, 4)))))
    _Protein = _RNA * 10
    
    _RNA['tp'] = _tp
    _Protein['tp'] = _tp
    
    return vdata.VData({'RNA': _RNA,
                        'Protein': _Protein},
                       time_col_name='tp')


def test_VData_conversion_to_AnnData_single_timepoint(v: vdata.VData) -> None:
    assert repr(v.to_AnnData('0h', into_one=False)) == "[AnnData object with n_obs × n_vars = 7 × 4\n" \
                                                       "    layers: 'RNA', 'Protein']"


def test_VData_conversion_into_multiple_anndatas(v: vdata.VData) -> None:
    assert repr(v.to_AnnData(into_one=False)) == "[AnnData object with n_obs × n_vars = 7 × 4\n" \
                                                 "    layers: 'RNA', 'Protein', " \
                                                 "AnnData object with n_obs × n_vars = 3 × 4\n" \
                                                 "    layers: 'RNA', 'Protein', " \
                                                 "AnnData object with n_obs × n_vars = 10 × 4\n" \
                                                 "    layers: 'RNA', 'Protein']"


def test_VData_conversion_into_one_anndata(v: vdata.VData) -> None:
    assert repr(v.to_AnnData(into_one=True)) == "AnnData object with n_obs × n_vars = 20 × 4\n" \
                                                "    obs: 'Time-point'\n" \
                                                "    layers: 'RNA', 'Protein'"
