from typing import Any

from vdata.data._parse.data import ParsingDataIn
from vdata.IO.logger import generalLogger


def parse_uns(data: ParsingDataIn) -> dict[str, Any]:
    if data.uns is None:
        generalLogger.debug("    8. \u2717 'uns' was not given.")
        return {}
        
    if not isinstance(data.uns, dict):
        raise TypeError("'uns' must be a dictionary.")
    
    generalLogger.debug("    8. \u2713 'uns' is a dictionary.")
    
    return data.uns
