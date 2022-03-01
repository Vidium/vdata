# coding: utf-8
# Created on 28/02/2022 16:19
# Author : matteo

# ====================================================
# imports
import vdata
import cProfile

# ====================================================
# code
# vdata.setLoggingLevel('INFO')

# with cProfile.Profile() as prof:
v = vdata.read("/home/matteo/git/Real_platform/storage/mbouvier/EI52/uploads/combinedData_filtered.vd", mode='r+')
# prof.dump_stats(file='/home/matteo/Desktop/vdata_read.prof')

v.write('/home/matteo/git/Real_platform/storage/mbouvier/EI52/uploads/combinedData_filtered2.vd')

v2 = vdata.read('/home/matteo/git/Real_platform/storage/mbouvier/EI52/uploads/combinedData_filtered2.vd')
print(v2)
