from typing import Any


def skip_time_axis(slicer: Any) -> tuple[Any, ...]:
    if not isinstance(slicer, tuple):
        slicer = (slicer,)
    return (slice(None),) + slicer
