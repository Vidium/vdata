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

with cProfile.Profile() as prof:
    vdata.read("/home/matteo/git/Real_platform/storage/mbouvier/EI52/uploads/combinedData_filtered.vd")
prof.dump_stats(file='/home/matteo/Desktop/vdata_read.prof')
