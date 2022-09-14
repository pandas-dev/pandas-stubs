from typing import (
    Any,
    Callable,
)

from pandas import (
    DataFrame,
    Series,
)
from pandas.core.generic import NDFrame
from pandas.core.window.rolling import (
    BaseWindowGroupby,
    RollingAndExpandingMixin,
    _NumbaKwargs,
)

from pandas._typing import (
    Axis,
    NDFrameT,
    WindowingRankType,
)

class Expanding(RollingAndExpandingMixin[NDFrameT]):
    def __init__(
        self,
        obj: NDFrame,
        min_periods: int = ...,
        center: Any | None = ...,  # Incomplete
        axis: Axis = ...,
        method: str = ...,
        selection: Any | None = ...,  # Incomplete
    ) -> None: ...
    def aggregate(self, func, *args, **kwargs) -> NDFrameT: ...
    agg = aggregate
    def count(self) -> NDFrameT: ...
    def apply(
        self,
        func: Callable[..., Any],
        raw: bool = ...,
        engine: str | None = ...,
        engine_kwargs: _NumbaKwargs | None = ...,
        args: tuple[Any, ...] | None = ...,
        kwargs: dict[str, Any] | None = ...,
    ) -> NDFrameT: ...
    def sum(
        self,
        *args,
        engine: str | None = ...,
        engine_kwargs: dict[str, bool] | None = ...,
        **kwargs,
    ) -> NDFrameT: ...
    def max(
        self,
        *args,
        engine: str | None = ...,
        engine_kwargs: dict[str, bool] | None = ...,
        **kwargs,
    ) -> NDFrameT: ...
    def min(
        self,
        *args,
        engine: str | None = ...,
        engine_kwargs: dict[str, bool] | None = ...,
        **kwargs,
    ) -> NDFrameT: ...
    def mean(
        self,
        *args,
        engine: str | None = ...,
        engine_kwargs: dict[str, bool] | None = ...,
        **kwargs,
    ) -> NDFrameT: ...
    def median(
        self,
        engine: str | None = ...,
        engine_kwargs: dict[str, bool] | None = ...,
        **kwargs,
    ) -> NDFrameT: ...
    def std(
        self,
        ddof: int = ...,
        *args,
        engine: str | None = ...,
        engine_kwargs: dict[str, bool] | None = ...,
        **kwargs,
    ) -> NDFrameT: ...
    def var(
        self,
        ddof: int = ...,
        *args,
        engine: str | None = ...,
        engine_kwargs: dict[str, bool] | None = ...,
        **kwargs,
    ) -> NDFrameT: ...
    def sem(self, ddof: int = ..., *args, **kwargs) -> NDFrameT: ...
    def skew(self, **kwargs) -> NDFrameT: ...
    def kurt(self, **kwargs) -> NDFrameT: ...
    def quantile(
        self, quantile: float, interpolation: str = ..., **kwargs
    ) -> NDFrameT: ...
    def rank(
        self,
        method: WindowingRankType = ...,
        ascending: bool = ...,
        pct: bool = ...,
        **kwargs,
    ) -> NDFrameT: ...
    def cov(
        self,
        other: DataFrame | Series | None = ...,
        pairwise: bool | None = ...,
        ddof: int = ...,
        **kwargs,
    ) -> NDFrameT: ...
    def corr(
        self,
        other: DataFrame | Series | None = ...,
        pairwise: bool | None = ...,
        ddof: int = ...,
        **kwargs,
    ) -> NDFrameT: ...

class ExpandingGroupby(BaseWindowGroupby, Expanding): ...
