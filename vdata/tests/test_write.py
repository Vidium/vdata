# coding: utf-8
# Created on 11/20/20 11:04 AM
# Author : matteo

# ====================================================
# imports
import os
import scanpy as sc

from vdata import VData
from vdata import setLoggingLevel


# ====================================================
# code
def test_write():
    os.system('rm -rf ~/vdata')

    # create vdata
    source_vdata_path = \
        "/home/matteo/Desktop/JN/Project/DMD/2-Genetic_Dynamic_Characterization/1-Dynamic_analysis/sel_JB_scRNAseq.h5ad"

    adata = sc.read(source_vdata_path)

    v = VData(adata, time_col='Time_hour')
    print(v)

    # write vdata in h5 file format
    v.write("~/vdata.h5")

    print("------------------------------------------")

    # write vdata in csv files
    v.write_to_csv("~/vdata")


if __name__ == "__main__":
    setLoggingLevel('DEBUG')

    test_write()




# expr_matrix = np.array([[[0, 10, 20], [10, 0, 15], [0, 9, 16], [15, 2, 16]]])
# obs = pd.DataFrame({"cell_name": ["c1", "c2", "c3", "c4"],
#                     "batch": [1, 1, 2, 2],
#                     "cat": pd.Series(["a", "b", "c", 1], dtype="category", index=[1, 2, 3, 4])}, index=[1, 2, 3, 4])
# obsm = {'umap': np.zeros((1, 4, 2))}
# obsp = {'connect': np.zeros((4, 4))}
# var = pd.DataFrame({"gene_name": ["g1", "g2", "g3"]}, index=["a", "b", "c"])
# varm = None
# varp = {'correlation': np.zeros((3, 3))}
# layers = {'spliced': np.zeros((1, 4, 3))}
# uns = {'color': ["#c1c1c1"], 'str': "test string", "int": 2}
# time_points = pd.DataFrame({"value": [5], "unit": ["hour"]})
#
#
# a = VData(data=expr_matrix,
#           obs=obs, obsm=obsm, obsp=obsp,
#           var=var, varm=varm, varp=varp,
#           uns=uns, time_points=time_points, )
# print(a)
#
# a.write_to_csv("/home/matteo/Desktop/vdata")
