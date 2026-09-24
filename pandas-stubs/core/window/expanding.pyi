from pandas.core.window.rolling import (
    BaseWindowGroupby,
    RollingAndExpandingMixin,
)

from pandas._typing import NDFrameT

class Expanding(RollingAndExpandingMixin[NDFrameT]): ...

# TODO: rename to ExpandingGroupBy (pandas-dev/pandas-stubs#1948)
class ExpandingGroupby(BaseWindowGroupby[NDFrameT], Expanding[NDFrameT]): ...
