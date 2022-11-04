# coding: utf-8
# Created on 18/05/2022 11:00
# Author : matteo

# ====================================================
# imports
from vdata.h5pickle import File
from pathlib import Path

from typing import Union

from vdata import read


# ====================================================
# code
def update_to_latest_vdata(vdata_path: Union[Path, str]) -> None:
    with File(vdata_path, 'r+') as vdata_file:
        if 'name' not in vdata_file.attrs.keys():
            vdata_file.attrs['name'] = 'No_name'

        if 'dtype' not in vdata_file.attrs.keys():
            vdata_file.attrs['dtype'] = 'float'

    try:
        _ = read(vdata_path)
        print('\u2714 Successfully updated vdata to latest version !')

    except Exception:
        raise IOError('\u2718 Could not update vdata to latest version !')
