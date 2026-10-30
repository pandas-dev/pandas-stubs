from pandas.core.window.rolling import (
    BaseWindowGroupBy,
    RollingAndExpandingMixin,
)

from pandas._typing import NDFrameT

class Expanding(RollingAndExpandingMixin[NDFrameT]): ...
class ExpandingGroupBy(BaseWindowGroupBy[NDFrameT], Expanding[NDFrameT]): ...
