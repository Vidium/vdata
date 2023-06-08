import pickle
from pathlib import Path

import ch5mpy as ch
import numpy as np

from vdata import TemporalDataFrame, VDataFrame


def update_tdf(data: ch.File) -> None:
    data.attrs["__h5_type__"] = "object"
    data.attrs["__h5_class__"] = np.void(pickle.dumps(TemporalDataFrame, protocol=pickle.HIGHEST_PROTOCOL))
    del data.attrs['type']
    
    if data.attrs['timepoints_column_name'] == '__ATTRIBUTE_None__':
        data.attrs['timepoints_column_name'] = '__h5_NONE__'
        
    data.move('timepoints', 'timepoints_array')
    data.move('values_numerical', 'numerical_array')
    data.move('values_string', 'string_array')
    
    
def _update_vdf(data: ch.File) -> None:
    data.attrs["__h5_type__"] = "object"
    data.attrs["__h5_class__"] = np.void(pickle.dumps(VDataFrame, protocol=pickle.HIGHEST_PROTOCOL))
    del data.attrs['type']
    
    columns = []

    if 'data_numeric' in data.keys():
        data.move('data_numeric/data', 'data_numeric')
        columns += list(data['data_numeric']['columns'].asstr())
        del data['data_numeric']
        
    else:
        data.create_dataset('data_numeric', data=np.empty((len(data['index']), 0)))
    
    if 'data_str' in data.keys():
        data.move('data_str/data', 'data_string')
        columns += list(data['data_str']['columns'].asstr())
        del data['data_str']
        
    else:
        data.create_dataset('data_string', data=np.empty((len(data['index']), 0)))
        
    data['columns'][()] = columns
    data.create_dataset('columns_stored_order', data=data['columns'])


def update_vdata(path: Path | str) -> None:
    data = ch.File(path, mode=ch.H5Mode.READ_WRITE)
    
    for layer in data['layers'].values():
        update_tdf(layer)
        
    if 'obs' not in data.keys():        
        first_layer = data['layers'][list(data['layers'].keys())[0]]
        
        obs = TemporalDataFrame(index=ch.read_object(first_layer['index']),
                                repeating_index=first_layer.attrs['repeating_index'],
                                time_list=ch.read_object(first_layer['timepoints_array']))
        ch.write_object(data, 'obs', obs)
    else:
        update_tdf(data['obs'])
    
    if 'var' not in data.keys():
        first_layer = data['layers'][list(data['layers'].keys())[0]]
        
        var = VDataFrame(index=np.concatenate((
            ch.read_object(first_layer['columns_numerical']),
            ch.read_object(first_layer['columns_string'])
        )))
        ch.write_object(data, 'var', var)
    else:
        _update_vdf(data['var'])
        
    if 'timepoints' not in data.keys():
        first_layer = data['layers'][list(data['layers'].keys())[0]]
        
        timepoints = VDataFrame({'value': np.unique(ch.read_object(first_layer['timepoints_array']))})
        ch.write_object(data, 'timepoints', timepoints)
    else:
        _update_vdf(data['timepoints'])
    
    data.close()

# from vdata.update import update_vdata
# update_vdata(output_dir / "vdata.vd")
