from vdata.data._parse.data import ParsingDataIn, ParsingDataOut
from vdata.data._parse.objects.layers import parse_layers
from vdata.data._parse.objects.obs import parse_obs, parse_obsm, parse_obsp
from vdata.data._parse.objects.uns import parse_uns
from vdata.data._parse.objects.var import parse_var, parse_varm, parse_varp
from vdata.data._parse.utils import log_timepoints
from vdata.IO.logger import generalLogger
from vdata.tdf import TemporalDataFrameBase
from vdata.vdataframe import VDataFrame


def _valid_timepoints(data: ParsingDataIn, obs: TemporalDataFrameBase) -> VDataFrame:
    if data.timepoints.empty:        
        generalLogger.debug("Default empty DataFrame for time points.")
        data.timepoints['value'] = list(obs.timepoints)
    
    log_timepoints(data.timepoints)
    return data.timepoints
                      
            
def parse_objects(data: ParsingDataIn) -> ParsingDataOut:
    generalLogger.debug('  VData creation from scratch.')
    
    parse_layers(data)
    _obs = parse_obs(data)
    
    return ParsingDataOut(data.layers, 
                          _obs, 
                          parse_obsm(data),
                          parse_obsp(data),
                          parse_var(data),
                          parse_varm(data),
                          parse_varp(data),
                          _valid_timepoints(data, _obs),
                          parse_uns(data))
