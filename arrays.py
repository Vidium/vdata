# coding: utf-8
# Created on 11/4/20 10:40 AM
# Author : matteo

# ====================================================
# imports
import abc
from abc import ABC
from typing import Optional, Dict, TypeVar
from typing_extensions import Literal

from . import vdata
from .NameUtils import ArrayLike_2D, ArrayLike_3D
from .IO.errors import ShapeError

ArrayLike = TypeVar('ArrayLike')


# ====================================================
# code
class VBaseArrayContainer(ABC):
    """
    TODO
    """

    def __init__(self, parent: "vdata.VData", data: Optional[Dict[str, ArrayLike]]):
        self._parent = parent
        self._data = data

    def __getattr__(self, item: str) -> ArrayLike:
        if item in self._data.keys():
            return self._data[item]

        else:
            raise AttributeError(f"{self._name} array has no attribute '{item}'")

    def __setattr__(self, key: str, value: ArrayLike) -> None:
        # first check that value has the correct shape
        if value.shape[0] == getattr(self._parent, f"n_{self._axis}"):
            self._data[key] = value

        else:
            raise ShapeError(f"The supplied array-like object has incorrect shape {value.shape}, expected ({self._parent.__getattribute__(f'n_{self.name}')}, ...)")

    @property
    @abc.abstractmethod
    def name(self):
        pass


class VAxisArray(VBaseArrayContainer):
    """
    Class for obsm and varm.
    These objects contain any number of array-like objects, with shape (<n_obs or n_var>, any).
    The arrays-like objects can be accessed from the parent VData object by :
        VData.obs.<array_name>  (VData.obs['<array_name>'])
    """

    def __init__(self, parent: "vdata.VData", axis: Literal['obs', 'var'], data: Dict[str, ArrayLike_3D]):
        super().__init__(parent, data)
        self._axis = axis

    @property
    def name(self):
        return self._axis

    @property
    def shape(self):
        return self._data[list(self._data.keys())[0]].shape


class VPairwiseArray:
    """
    Class for obsp and varp
    """

    def __init__(self, parent: "vdata.VData", axis: Literal['obs', 'var'], data: ArrayLike_2D):
        self._parent = parent
        self._axis = axis
        self._data = data

    @property
    def shape(self):
        return self._data.shape


class VLayersArrays(VBaseArrayContainer):
    """
    Class for layers
    """

    def __init__(self, parent: "vdata.VData", data: Dict[str, ArrayLike_3D]):
        super().__init__(parent, data)

    @property
    def keys(self):
        return self._data.keys()

    @property
    def values(self):
        return self._data.values()

    @property
    def items(self):
        return self._data.items()

    @property
    def name(self):
        return "layers"
