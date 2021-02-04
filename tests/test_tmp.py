# coding: utf-8
# Created on 04/02/2021 11:11
# Author : matteo

# ====================================================
# imports
import vdata


# ====================================================
# code
vdata.setLoggingLevel('DEBUG')
v = vdata.read("/home/matteo/git/WASABI/simulations/simu_553_6_1601/vsimu.vdata")
print(v)
