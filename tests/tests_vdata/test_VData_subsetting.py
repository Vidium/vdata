# coding: utf-8
# Created on 11/25/20 3:41 PM
# Author : matteo

# ====================================================
# imports
import numpy as np
import pandas as pd

import vdata


# ====================================================
# code
def test_VData_sub_setting():
    genes = list(map(lambda x: "g_" + str(x), range(50)))
    cells = list(map(lambda x: "c_" + str(x), range(300)))

    mask_genes = ['g_10', 'g_2', 'g_5', 'g_25', 'g_49']
    mask_cells = ['c_20', 'c_1', 'c_199', 'c_100', 'c_30', 'c_10', 'c_150', 'c_151']

    v = vdata.VData(data={'data': pd.DataFrame(np.array(range(300 * 50)).reshape((300, 50)),
                                               index=cells,
                                               columns=genes)},
                    obs=pd.DataFrame({'col1': range(300)}, index=cells),
                    var=pd.DataFrame({'col1': range(50)}, index=genes),
                    time_list=['0h' for _ in range(100)] + ['1h' for _ in range(100)] + ['2h' for _ in range(100)])

    v.obsm['test_obsm'] = vdata.TemporalDataFrame(
        pd.DataFrame(np.array(range(v.n_obs_total * 2)).reshape((v.n_obs_total, 2)), columns=['X1', 'X2']),
        index=v.obs.index,
        time_list=v.obs.time_points_column
    )

    v.obsp['test_obsp'] = pd.DataFrame(
        np.array(range(v.n_obs_total * v.n_obs_total)).reshape((v.n_obs_total, v.n_obs_total)),
        columns=v.obs.index,
        index=v.obs.index
    )

    v.varm['test_varm'] = pd.DataFrame(
        np.array(range(v.n_var * 2)).reshape((v.n_var, 2)),
        index=v.var.index,
        columns=['X1', 'X2']
    )

    v.varp['test_varp'] = pd.DataFrame(
        np.array(range(v.n_var * v.n_var)).reshape((v.n_var, v.n_var)),
        columns=v.var.index,
        index=v.var.index
    )

    assert repr(v) == "VData 'No_Name' with n_obs x n_var = [100, 100, 100] x 50 over 3 time points.\n" \
                      "\tlayers: 'data'\n" \
                      "\tobs: 'col1'\n" \
                      "\tvar: 'col1'\n" \
                      "\ttime_points: 'value'\n" \
                      "\tobsm: 'test_obsm'\n" \
                      "\tvarm: 'test_varm'\n" \
                      "\tobsp: 'test_obsp'\n" \
                      "\tvarp: 'test_varp'", repr(v)

    sub_vdata = v[:, mask_cells, mask_genes]

    assert repr(sub_vdata) == "View of VData 'No_Name' with n_obs x n_var = [4, 4] x 5 over 2 time points\n" \
                              "\tlayers: 'data'\n" \
                              "\tobs: 'col1'\n" \
                              "\tvar: 'col1'\n" \
                              "\ttime_points: 'value'\n" \
                              "\tobsm: 'test_obsm'\n" \
                              "\tvarm: 'test_varm'\n" \
                              "\tobsp: 'test_obsp'\n" \
                              "\tvarp: 'test_varp'", repr(sub_vdata)

    index_cells = ['c_20', 'c_1', 'c_30', 'c_10', 'c_199', 'c_100', 'c_150', 'c_151']

    assert sub_vdata.layers['data'].index.equals(pd.Index(index_cells)), sub_vdata.layers['data'].index
    assert sub_vdata.layers['data'].columns.equals(pd.Index(mask_genes)), sub_vdata.layers['data'].columns
    assert np.array_equal(sub_vdata.layers['data'].values.flatten(),
                          np.array([int(c[2:]) * 50 + int(g[2:]) for c in index_cells for g in mask_genes]))

    assert sub_vdata.obs.index.equals(pd.Index(index_cells)), sub_vdata.obs.index
    assert sub_vdata.obs.columns.equals(v.obs.columns), sub_vdata.obs.columns
    assert np.array_equal(sub_vdata.obs.values.flatten(),
                          np.array([int(c[2:]) for c in index_cells]))

    assert sub_vdata.var.index.equals(pd.Index(mask_genes)), sub_vdata.var.index
    assert sub_vdata.var.columns.equals(v.var.columns), sub_vdata.var.columns
    assert np.array_equal(sub_vdata.var.values.flatten(),
                          np.array([int(g[2:]) for g in mask_genes]))

    assert sub_vdata.obsm['test_obsm'].index.equals(pd.Index(index_cells)), sub_vdata.obsm['test_obsm'].index
    assert sub_vdata.obsm['test_obsm'].columns.equals(v.obsm['test_obsm'].columns), sub_vdata.obsm['test_obsm'].columns
    assert np.array_equal(sub_vdata.obsm['test_obsm'].values.flatten(),
                          np.array([int(c[2:]) * 2 + col for c in index_cells for col in (0, 1)]))

    assert sub_vdata.obsp['test_obsp'].index.equals(pd.Index(index_cells)), sub_vdata.obsp['test_obsp'].index
    assert sub_vdata.obsp['test_obsp'].columns.equals(pd.Index(index_cells)), sub_vdata.obsp['test_obsp'].columns
    assert np.array_equal(sub_vdata.obsp['test_obsp'].values.flatten(),
                          np.array([int(c[2:]) * 300 + int(c2[2:]) for c in index_cells for c2 in index_cells]))

    assert sub_vdata.varm['test_varm'].index.equals(pd.Index(mask_genes)), sub_vdata.varm['test_varm'].index
    assert sub_vdata.varm['test_varm'].columns.equals(v.varm['test_varm'].columns), sub_vdata.varm['test_varm'].columns
    assert np.array_equal(sub_vdata.varm['test_varm'].values.flatten(),
                          np.array([int(g[2:]) * 2 + col for g in mask_genes for col in (0, 1)]))

    assert sub_vdata.varp['test_varp'].index.equals(pd.Index(mask_genes)), sub_vdata.varp['test_varp'].index
    assert sub_vdata.varp['test_varp'].columns.equals(pd.Index(mask_genes)), sub_vdata.varp['test_varp'].columns
    assert np.array_equal(sub_vdata.varp['test_varp'].values.flatten(),
                          np.array([int(g[2:]) * 50 + int(g2[2:]) for g in mask_genes for g2 in mask_genes]))

    assert sub_vdata.n_obs_total == len(index_cells), sub_vdata.n_obs_total
    assert sub_vdata.n_var == len(mask_genes), sub_vdata.n_var

    assert sub_vdata.layers.shape == (1, 2, [4, 4], 5)
    assert sub_vdata.layers.shape == (1, 2, sub_vdata.n_obs, sub_vdata.n_var)


if __name__ == "__main__":
    vdata.setLoggingLevel('DEBUG')

    test_VData_sub_setting()
