# coding: utf-8
# Created on 10/02/2021 15:18
# Author : matteo

# ====================================================
# imports
import pandas as pd
import numpy as np
import pytest

import vdata
from vdata import VData

from . import expr_data_simple


# ====================================================
# code
def test_VData_copy(VData_uns):
    v2 = VData_uns.copy()

    assert id(VData_uns) != id(v2)
    assert repr(VData_uns) == "VData '1' with n_obs x n_var = 6 x 3 over 1 time point.\n\t" \
                              "layers: 'data'\n\t" \
                              "obs: 'data', 'data_bis'\n\t" \
                              "var: 'gene_name'\n\t" \
                              "timepoints: 'value'\n\t" \
                              "uns: 'colors', 'date'", repr(VData_uns)

    # repeating index
    timepoints = pd.DataFrame({"value": ["0h", "1h"]})
    var = pd.DataFrame({"gene_name": ["g1", "g2", "g3"]})
    obs = vdata.TemporalDataFrame({'data': np.random.randint(0, 20, 6),
                                   'data_bis': np.random.randint(0, 20, 6)},
                                  index=['a', 'b', 'c', 'a', 'b', 'c'],
                                  repeating_index=True,
                                  time_list=["0h", "0h", "0h", "1h", "1h", "1h"])
    uns = {"colors": ['blue', 'red', 'yellow'],
           "date": '25/01/2021'}

    data = vdata.TemporalDataFrame(pd.DataFrame(expr_data_simple),
                                   index=['a', 'b', 'c', 'a', 'b', 'c'],
                                   repeating_index=True,
                                   time_list=["0h", "0h", "0h", "1h", "1h", "1h"])

    v3 = vdata.VData(data, timepoints=timepoints, obs=obs, var=var, uns=uns, name=3)

    v4 = v3.copy()
    assert repr(v4) == "VData '3_copy' with n_obs x n_var = [3, 3] x 3 over 2 time points.\n" \
                       "\tlayers: 'data'\n" \
                       "\tobs: 'data', 'data_bis'\n" \
                       "\tvar: 'gene_name'\n" \
                       "\ttimepoints: 'value'\n" \
                       "\tuns: 'colors', 'date'"


def test_VData_copy_subset(VData_uns):
    v_subset = VData_uns[:, 0:4, 0:2]
    v2 = v_subset.copy()

    assert id(v_subset) != id(v2)
    assert repr(v2) == "VData '1_view_copy' with n_obs x n_var = 4 x 2 over 1 time point.\n\t" \
                       "layers: 'data'\n\t" \
                       "obs: 'data', 'data_bis'\n\t" \
                       "var: 'gene_name'\n\t" \
                       "timepoints: 'value'\n\t" \
                       "uns: 'colors', 'date'", repr(v2)


@pytest.mark.parametrize(
    'VData_uns',
    ['backed'],
    indirect=True
)
def test_copy_of_backed_VData_should_not_be_backed(VData_uns: VData):
    v_copy = VData_uns.copy()

    assert not v_copy.is_backed


@pytest.mark.parametrize(
    'VData_uns',
    ['backed'],
    indirect=True
)
def test_copy_of_backed_VData_layers_should_not_be_backed(VData_uns: VData):
    v_copy = VData_uns.copy()

    assert not v_copy.layers['data'].is_backed
