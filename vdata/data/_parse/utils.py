import pandas as pd

from vdata.IO.logger import generalLogger
from vdata.utils import repr_array


def log_timepoints(timepoints: pd.DataFrame) -> None:
    generalLogger.debug(
        lambda: f"  {len(timepoints)} time point" f"{' was' if len(timepoints) == 1 else 's were'} found finally."
    )
    generalLogger.debug(
        lambda: f"    \u21B3 Time point{' is' if len(timepoints) == 1 else 's are'} : "
        f"{repr_array(list(timepoints.value)) if len(timepoints) else '[]'}"
    )
