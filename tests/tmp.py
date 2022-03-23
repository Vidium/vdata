import vdata
import cProfile
from pathlib import Path

current_file_name = Path('/home/matteo/git/Real_platform/storage/mbouvier/EI52/uploads/combinedData_filtered.vd')
current_path = Path(current_file_name)
data = vdata.read(current_path, mode='r')

list_cells = []
cell_group = 'batch'
list_cell_category = ['E0']
for cat in list_cell_category:
    list_cells += list(data.obs.index[data.obs[:, :, cell_group] == cat])

# juste le print de la ligne ci-dessous prend des plombes
with cProfile.Profile() as prof:
    subset = data[:, list_cells]

prof.dump_stats('/home/matteo/Desktop/vdata.prof')
