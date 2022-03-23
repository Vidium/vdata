# coding: utf-8
# Created on 20/01/2021 16:58
# Author : matteo

# ====================================================
# imports
import vdata
from . import data


# ====================================================
# code
def test_VData_conversion_to_AnnData():
    v = vdata.read_from_dict(data)

    assert repr(v.to_AnnData('0h', into_one=False)) == "[AnnData object with n_obs × n_vars = 7 × 4\n" \
                                                       "    layers: 'RNA', 'Protein']", \
        repr(v.to_AnnData('0h', into_one=False))

    assert repr(v.to_AnnData(into_one=False)) == "[AnnData object with n_obs × n_vars = 7 × 4\n" \
                                                 "    layers: 'RNA', 'Protein', " \
                                                 "AnnData object with n_obs × n_vars = 3 × 4\n" \
                                                 "    layers: 'RNA', 'Protein', " \
                                                 "AnnData object with n_obs × n_vars = 10 × 4\n" \
                                                 "    layers: 'RNA', 'Protein']", repr(v.to_AnnData(into_one=False))

    print('test')

    assert repr(v.to_AnnData(into_one=True)) == "AnnData object with n_obs × n_vars = 20 × 4\n" \
                                                "    obs: 'Time_Point'\n" \
                                                "    layers: 'RNA', 'Protein'", repr(v.to_AnnData(into_one=True))


if __name__ == '__main__':
    vdata.setLoggingLevel('DEBUG')

    test_VData_conversion_to_AnnData()
