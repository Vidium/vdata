# coding: utf-8
# Created on 19/01/2021 15:36
# Author : matteo

# ====================================================
# imports
import os
from pathlib import Path
from typing import Union


# ====================================================
# code
def parse_path(path: Union[str, Path]) -> Path:
    """
    Convert a given path to a valid path. The '~' character is replaced by the $HOME variable.

    :param path: a path to parse.
    :return: a valid path.
    """
    # make sure directory is a path
    if not isinstance(path, Path):
        path = Path(path)

    if path.parts[0] == '~':
        path = Path(os.environ['HOME'] / Path("/".join(path.parts[1:])))

    return path
