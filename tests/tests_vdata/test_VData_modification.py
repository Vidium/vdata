# coding: utf-8
# Created on 04/05/2022 17:22
# Author : matteo

# ====================================================
# imports
import numpy as np
import pandas as pd
from pathlib import Path

import vdata

# ====================================================
# code
REF_DIR = Path(__file__).parent.parent / 'ref'


def test_VData_modification():
    v1 = vdata.read(REF_DIR / "vdata.vd", mode='r+')

    # set once
    v1.obsm['X'] = vdata.TemporalDataFrame(data=pd.DataFrame({'col1': range(v1.n_obs_total)}),
                                           time_list=v1.obs.timepoints_column,
                                           index=v1.obs.index)

    # set a second time
    v1.obsm['X'] = vdata.TemporalDataFrame(data=pd.DataFrame({'col1': 2*np.arange(v1.n_obs_total)}),
                                           time_list=v1.obs.timepoints_column,
                                           index=v1.obs.index)

    del v1.obsm['X']
    v1.write()              # TODO : should not have to do this, should be written automatically

    v1.file.close()


def test_VData_set_index():
    v1 = vdata.read(REF_DIR / "vdata.vd", mode='r+')

    # not repeating ==> not repeating
    v1.set_obs_index(values=range(v1.n_obs_total))

    assert np.array_equal(v1.obs.index, range(v1.n_obs_total))
    assert np.array_equal(v1.layers['data'].index, range(v1.n_obs_total))

    # subset first 10 indices of each time-point
    index = np.concatenate([v1.obs.index_at(tp)[:10] for tp in v1.timepoints.value])
    v2 = v1[:, index].copy()
    v1.file.close()

    # not repeating ==> repeating
    new_index = np.concatenate([range(10) for _ in range(v2.n_timepoints)])

    v2.set_obs_index(values=new_index,
                     repeating_index=True)

    assert np.array_equal(v2.obs.index, new_index)
    assert np.array_equal(v2.layers['data'].index, new_index)

    # repeating ==> repeating
    new_index = np.concatenate([range(10) for _ in range(v2.n_timepoints)]) * 2

    v2.set_obs_index(values=new_index,
                     repeating_index=True)

    assert np.array_equal(v2.obs.index, new_index)
    assert np.array_equal(v2.layers['data'].index, new_index)

    # repeating ==> not repeating
    new_index = range(10 * v2.n_timepoints)

    v2.set_obs_index(values=new_index)

    assert np.array_equal(v2.obs.index, new_index)
    assert np.array_equal(v2.layers['data'].index, new_index)


if __name__ == '__main__':
    test_VData_modification()
    test_VData_set_index()
