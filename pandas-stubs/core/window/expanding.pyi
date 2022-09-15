from typing import (
    Any,
    Callable,
)

from pandas import (
    DataFrame,
    Series,
)
from pandas.core.window.rolling import _Rolling_and_Expanding

class Expanding(_Rolling_and_Expanding):
    def __init__(
        self, obj, min_periods: int = ..., center: bool = ..., axis: int = ..., **kwargs
    ) -> None: ...
    def count(self, **kwargs) -> DataFrame | Series: ...
    def apply(
        self,
        func: Callable[..., Any],
        raw: bool = ...,
        engine: str | None = ...,
        engine_kwargs: dict[str, bool] | None = ...,
        args: tuple[Any, ...] | None = ...,
        kwargs: dict[str, Any] | None = ...,
    ): ...

class ExpandingGroupby(Expanding): ...
