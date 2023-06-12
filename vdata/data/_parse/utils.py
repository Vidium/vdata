from vdata.IO.logger import generalLogger
from vdata.utils import repr_array
from vdata.vdataframe import VDataFrame


def log_timepoints(timepoints: VDataFrame) -> None:    
    generalLogger.debug(f"  {len(timepoints)} time point"
                        f"{' was' if len(timepoints) == 1 else 's were'} found finally.")
    generalLogger.debug(f"    \u21B3 Time point{' is' if len(timepoints) == 1 else 's are'} : "
                        f"{repr_array(list(timepoints.value)) if len(timepoints) else '[]'}")
