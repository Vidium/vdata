# coding: utf-8
# Created on 20/01/2021 16:58
# Author : matteo

# ====================================================
# imports
import numpy as np

from vdata import read_from_dict, setLoggingLevel


# ====================================================
# code
setLoggingLevel('INFO')

data = {
    'RNA': {
        '0h': np.zeros((7, 3)),
        '5h': np.ones((3, 3)),
        '10h': 2 * np.ones((10, 3))
    },
    'Protein': {
        '0h': 10 * np.ones((7, 3)),
        '5h': 20 * np.ones((3, 3)),
        '10h': 30 * np.ones((10, 3))
    }
}

v = read_from_dict(data)

print(v)
print(v['0h'])

print(v.to_AnnData('0h', into_one=False))
print(v.to_AnnData(into_one=False))

print(v.to_AnnData(into_one=True))
