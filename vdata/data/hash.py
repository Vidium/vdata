from __future__ import annotations

import numpy as np

import vdata


def _hash_tp(data: vdata.VData) -> int:
    return hash(np.asanyarray(data.timepoints.value.values).data.tobytes())


def _hash_obs(data: vdata.VData) -> int:
    return hash(data.obs.index)


def _hash_var(data: vdata.VData) -> int:
    return hash(data.var.index.values.data.tobytes())


class VDataHash:

    __slots__ = "_data", "_tp_hash", "_obs_hash", "_var_hash"

    # region magic methods
    def __init__(
        self,
        data: vdata.VData,
        timepoints: bool = False,
        obs: bool = False,
        var: bool = False,
    ):
        assert any((timepoints, obs, var))

        self._data = data
        self._tp_hash = _hash_tp(data) if timepoints else None
        self._obs_hash = _hash_obs(data) if obs else None
        self._var_hash = _hash_var(data) if var else None

    # endregion

    # region methods
    def assert_unchanged(self) -> None:
        new_tp_hash = None if self._tp_hash is None else _hash_tp(self._data)
        new_obs_hash = None if self._obs_hash is None else _hash_obs(self._data)
        new_var_hash = None if self._var_hash is None else _hash_var(self._data)

        if (self._tp_hash, self._obs_hash, self._var_hash) != (new_tp_hash, new_obs_hash, new_var_hash):
            raise AssertionError

    # endregion
