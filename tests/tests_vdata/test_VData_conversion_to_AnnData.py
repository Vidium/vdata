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

    print(v)
    print(v['0h'])

    print(v.to_AnnData('0h', into_one=False))
    print(v.to_AnnData(into_one=False))

    print(v.to_AnnData(into_one=True))


if __name__ == '__main__':
    vdata.setLoggingLevel('DEBUG')

    test_VData_conversion_to_AnnData()
