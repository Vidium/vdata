# coding: utf-8
# Created on 10/02/2021 16:54
# Author : matteo

# ====================================================
# imports
import os
import pandas as pd
import numpy as np
from pathlib import Path

import vdata
from . import expr_data_complex, obs_index_data
from .test_VData_write import out_test_VData_write


# ====================================================
# code
def test_VData_concatenate():
    timepoints = pd.DataFrame({"value": ['0h', '5h']})
    var = pd.DataFrame({"gene_name": ["g1", "g2", "g3"]}, index=["g1", "g2", "g3"])
    obs = vdata.TemporalDataFrame({'data': np.random.randint(0, 20, 6),
                                   'data_bis': np.random.randint(0, 20, 6)},
                                  time_list=["0h", "0h", "0h", "0h", "5h", "5h"],
                                  index=obs_index_data)
    uns = {"colors": ['blue', 'red', 'yellow'],
           "date": '25/01/2021'}

    v1 = vdata.VData(expr_data_complex, timepoints=timepoints, obs=obs, var=var, uns=uns, name=1)

    expr_data_complex_modif = {key: TDF * -1 for key, TDF in expr_data_complex.items()}
    obs = vdata.TemporalDataFrame({'data': np.random.randint(0, 20, 6),
                                   'data_bis': np.random.randint(0, 20, 6)},
                                  time_list=["0h", "0h", "0h", "0h", "5h", "5h"],
                                  index=obs_index_data)
    uns = {"colors": ['blue', 'red', 'pink'],
           "date": '24/01/2021'}

    v2 = vdata.VData(expr_data_complex_modif, timepoints=timepoints, obs=obs, var=var, uns=uns, name=2)

    v2.set_obs_index([f"C_{i}" for i in range(6, 12)])

    v_merged = vdata.concatenate((v1, v2))

    assert repr(v_merged) == "VData 'No_Name' with n_obs x n_var = [8, 4] x 3 over 2 time points.\n\t" \
                             "layers: 'spliced', 'unspliced'\n\t" \
                             "obs: 'data', 'data_bis'\n\t" \
                             "var: 'gene_name'\n\t" \
                             "timepoints: 'value'\n\t" \
                             "uns: 'colors', 'date'", repr(v_merged)

    assert np.all(v_merged.obs.index == ['C_0', 'C_1', 'C_2', 'C_3', 'C_6', 'C_7', 'C_8', 'C_9',
                                         'C_4', 'C_5', 'C_10', 'C_11']), v_merged.obs.index

    assert np.all(v_merged.layers['spliced'].index == ['C_0', 'C_1', 'C_2', 'C_3', 'C_6', 'C_7', 'C_8',
                                                       'C_9', 'C_4', 'C_5', 'C_10', 'C_11']), \
        v_merged.layers['spliced'].index


# TODO : implement back in TDFs
# def test_VData_concatenate_mean():
#     output_dir = Path(__file__).parent.parent / 'ref'
#
#     if not os.path.exists(output_dir / 'vdata.vd'):
#         # first write data
#         out_test_VData_write()
#
#     v3 = vdata.read(output_dir / 'vdata.vd', name=3)
#
#     v4 = v3.copy()
#
#     vm3 = v3.mean()
#
#     vm4 = v4.mean()
#
#     vm4.set_obs_index(vm3.obs.index + '_2')
#
#     v_merged = vdata.concatenate((vm3, vm4))
#
#     assert repr(v_merged) == "VData 'No_Name' with n_obs x n_var = [2, 2, 2, 2, 2, 2, 2, 2, 2, 2] x 1000 " \
#                              "over 10 time points.\n\t" \
#                              "layers: 'data'\n\t" \
#                              "timepoints: 'value'", repr(v_merged)
#
#     v3.file.close()


if __name__ == '__main__':
    vdata.setLoggingLevel('DEBUG')

    test_VData_concatenate()
    # test_VData_concatenate_mean()
