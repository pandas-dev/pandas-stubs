from pandas.core.window.rolling import (
    BaseWindowGroupby,
    RollingAndExpandingMixin,
)

from pandas._typing import (
    Axis,
    IndexLabel,
    NDFrameT,
)

class Expanding(RollingAndExpandingMixin[NDFrameT]):
    def __init__(
        self,
        obj: NDFrameT,
        min_periods: int = ...,
        axis: Axis = ...,
        method: str = ...,
        selection: IndexLabel | None = ...,
    ) -> None: ...

class ExpandingGroupby(BaseWindowGroupby[NDFrameT], Expanding[NDFrameT]): ...
