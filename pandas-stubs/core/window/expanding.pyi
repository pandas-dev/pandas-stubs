from typing import (
    Any,
    Callable,
    Dict,
    Optional,
    Tuple,
)

from pandas.core.window.common import WindowGroupByMixin as WindowGroupByMixin
from pandas.core.window.rolling import _Rolling_and_Expanding

from pandas._typing import FrameOrSeriesUnion as FrameOrSeries

class Expanding(_Rolling_and_Expanding):
    def __init__(
        self, obj, min_periods: int = ..., center: bool = ..., axis: int = ..., **kwargs
    ) -> None: ...
    def count(self, **kwargs) -> FrameOrSeries: ...
    def apply(
        self,
        func: Callable[..., Any],
        raw: bool = ...,
        engine: Optional[str] = ...,
        engine_kwargs: Optional[Dict[str, bool]] = ...,
        args: Optional[Tuple[Any, ...]] = ...,
        kwargs: Optional[Dict[str, Any]] = ...,
    ): ...

class ExpandingGroupby(WindowGroupByMixin, Expanding): ...
