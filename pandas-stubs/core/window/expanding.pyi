from typing import (
    Any,
    Callable,
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
        engine: str | None = ...,
        engine_kwargs: dict[str, bool] | None = ...,
        args: tuple[Any, ...] | None = ...,
        kwargs: dict[str, Any] | None = ...,
    ): ...

class ExpandingGroupby(WindowGroupByMixin, Expanding): ...
