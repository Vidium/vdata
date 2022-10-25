# coding: utf-8
# Created on 07/10/2022 15:12
# Author : matteo

# ====================================================
# imports
import vdata


# ====================================================
# code
data = vdata.read('/home/matteo/git/Real_platform/storage/trace/uploads/test_vdata.vd', mode='r+')
data.uns['cell_subsets']['test'] = ['titi', 'toto']
data.write()
data.file.close()
del data
