# coding: utf-8
# Created on 11/23/20 9:45 AM
# Author : matteo

# ====================================================
# imports
import scanpy

from .._IO.read import read_from_csv, read
from vdata._core import VData


# ====================================================
# code
print('---- read VData from csv')
vdata = read_from_csv("/home/matteo/Desktop/vdata", log_level='DEBUG')
print(vdata)

print('-----------------------------------------------------')
print('---- read AnnData')
folder = '/home/matteo/Desktop/NAS/Project/DMD/2-Genetic_Dynamic_Characterization/1-Dynamic_analysis/'
adata_filename = folder + 'sel_JB_scRNAseq.h5ad'

adata = scanpy.read(adata_filename)

print('---- convert to VData')
vdata = VData(adata)

vdata_filename = folder + 'sel_JB_scRNAseq.vdata'

print('---- write VData')
vdata.write(vdata_filename)

print('---- read VData')
vdata2 = read(vdata_filename)
print(vdata2)
