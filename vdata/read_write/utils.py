# coding: utf-8
# Created on 19/01/2021 15:36
# Author : matteo

# ====================================================
# imports
from __future__ import annotations

import os
from pathlib import Path


# ====================================================
# code
def parse_path(path: None | str | Path) -> None | Path:
    """
    Convert a given path to a valid path. The '~' character is replaced by the $HOME variable.

    Args:
        path: a path to parse.

    Returns:
        A valid path.
    """
    if path is None:
        return None

    # make sure directory is a path
    if not isinstance(path, Path):
        path = Path(path)


    if path.parts[0] == '~':
        path = Path(os.environ['HOME'] / Path("/".join(path.parts[1:])))

    return path
