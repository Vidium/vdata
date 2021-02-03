# coding: utf-8
# Created on 29/01/2021 09:07
# Author : matteo

# ====================================================
# imports
import numpy as np

# ====================================================
# code
data = {
    'RNA': {
        '0h': np.zeros((7, 4)),
        '5h': np.ones((3, 4)),
        '10h': 2 * np.ones((10, 4))
    },
    'Protein': {
        '0h': 10 * np.ones((7, 4)),
        '5h': 20 * np.ones((3, 4)),
        '10h': 30 * np.ones((10, 4))
    }
}