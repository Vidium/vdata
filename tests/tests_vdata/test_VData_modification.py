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
    v1 = vdata.read(REF_DIR / "vdata.vd", name=1, mode='r+')

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


if __name__ == '__main__':
    test_VData_modification()
