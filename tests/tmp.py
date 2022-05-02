import random
import vdata
import cProfile
from pathlib import Path

current_file_name = Path('/home/matteo/')
current_path = Path(current_file_name)
data = vdata.read(current_path, mode='r+')

# list_cells = []
# cell_group = 'batch'
# list_cell_category = ['E0']
# for cat in list_cell_category:
#     list_cells += list(data.obs.index[data.obs[:, :, cell_group] == cat])
#
# # juste le print de la ligne ci-dessous prend des plombes
# with cProfile.Profile() as prof:
#     subset = data[:, list_cells]
#
# prof.dump_stats('/home/matteo/Desktop/vdata.prof')

list_id = random.choices(data.obs.index, k=100)
group = 'new_col'
name = 'cluster1'

# TODO : raise error if col not in TDF !?
data.obs[:, list_id, group]

with cProfile.Profile() as prof:
    # if group in data.obs.columns:
    #     print('profiling del')
    #     del data.obs[:, :, group]
    #
    # else:
    #     print('profiling insert')
    #     data.obs.insert(1, group, 'Undef')

    # data.obs[:, list_id, group] = name

    repr(data.obs.new_col)

prof.dump_stats('/home/matteo/Desktop/vdata.prof')
