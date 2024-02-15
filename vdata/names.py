from typing import Protocol, runtime_checkable
import numpy as np

Number = int, float, np.int_, np.float_

NO_NAME = "No_Name"
DEFAULT_TIME_COL_NAME = "Time-point"


@runtime_checkable
class Unpickleable(Protocol):
    def read(self, n: int) -> bytes:
        ...

    def readline(self) -> bytes:
        ...
