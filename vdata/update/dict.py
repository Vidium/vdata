from __future__ import annotations

from typing import Any

import ch5mpy as ch

from vdata.update.array import update_array


def _update_dict_v0_to_v1(obj: ch.H5Dict[Any]) -> None:
    for key in obj.keys():
        if isinstance(obj @ key, ch.H5Array):
            update_array[0](obj @ key)

        elif isinstance(obj @ key, ch.H5Dict):
            update_dict[0](obj @ key)


def _update_dict_v1_to_v2(obj: ch.H5Dict[Any]) -> None:
    pass


update_dict = {
    0: _update_dict_v0_to_v1,
    1: _update_dict_v1_to_v2,
}
