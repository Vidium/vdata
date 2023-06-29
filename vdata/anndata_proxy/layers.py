from typing import Any

import numpy.typing as npt

from vdata.data.arrays import VLayersArrayContainer, VLayersArrayContainerView


class LayersProxy:

    __slots__ = ("_layers",)

    # region magic methods
    def __init__(self, layers: VLayersArrayContainer | VLayersArrayContainerView) -> None:
        self._layers = layers

    def __repr__(self) -> str:
        return f"Layers with keys: {', '.join(self._layers.keys())}"

    def __getitem__(self, key: str) -> npt.NDArray[Any]:
        return self._layers[str(key)].values

    # endregion
