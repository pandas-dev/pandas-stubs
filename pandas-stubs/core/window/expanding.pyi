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
)

from pandas._typing import (
    Axis as Axis,
    WindowingRankType as WindowingRankType,
)

class Expanding(RollingAndExpandingMixin):
    def __init__(
        self,
        obj: NDFrame,
        min_periods: int = ...,
        center: Any | None = ...,  # Incomplete
        axis: Axis = ...,
        method: str = ...,
        selection: Any | None = ...,  # Incomplete
    ) -> None: ...
    def aggregate(self, func, *args, **kwargs): ...
    agg = aggregate
    def count(self): ...
    def apply(
        self,
        func: Callable[..., Any],
        raw: bool = ...,
        engine: str | None = ...,
        engine_kwargs: dict[str, bool] | None = ...,
        args: tuple[Any, ...] | None = ...,
        kwargs: dict[str, Any] | None = ...,
    ): ...
    def sum(
        self,
        *args,
        engine: str | None = ...,
        engine_kwargs: dict[str, bool] | None = ...,
        **kwargs,
    ) -> DataFrame | Series: ...
    def max(
        self,
        *args,
        engine: str | None = ...,
        engine_kwargs: dict[str, bool] | None = ...,
        **kwargs,
    ) -> DataFrame | Series: ...
    def min(
        self,
        *args,
        engine: str | None = ...,
        engine_kwargs: dict[str, bool] | None = ...,
        **kwargs,
    ) -> DataFrame | Series: ...
    def mean(
        self,
        *args,
        engine: str | None = ...,
        engine_kwargs: dict[str, bool] | None = ...,
        **kwargs,
    ) -> DataFrame | Series: ...
    def median(
        self,
        engine: str | None = ...,
        engine_kwargs: dict[str, bool] | None = ...,
        **kwargs,
    ) -> DataFrame | Series: ...
    def std(
        self,
        ddof: int = ...,
        *args,
        engine: str | None = ...,
        engine_kwargs: dict[str, bool] | None = ...,
        **kwargs,
    ) -> DataFrame | Series: ...
    def var(
        self,
        ddof: int = ...,
        *args,
        engine: str | None = ...,
        engine_kwargs: dict[str, bool] | None = ...,
        **kwargs,
    ) -> DataFrame | Series: ...
    def sem(self, ddof: int = ..., *args, **kwargs) -> DataFrame | Series: ...
    def skew(self, **kwargs) -> DataFrame | Series: ...
    def kurt(self, **kwargs) -> DataFrame | Series: ...
    def quantile(
        self, quantile: float, interpolation: str = ..., **kwargs
    ) -> DataFrame | Series: ...
    def rank(
        self,
        method: WindowingRankType = ...,
        ascending: bool = ...,
        pct: bool = ...,
        **kwargs,
    ) -> DataFrame | Series: ...
    def cov(
        self,
        other: DataFrame | Series | None = ...,
        pairwise: bool | None = ...,
        ddof: int = ...,
        **kwargs,
    ) -> DataFrame | Series: ...
    def corr(
        self,
        other: DataFrame | Series | None = ...,
        pairwise: bool | None = ...,
        ddof: int = ...,
        **kwargs,
    ) -> DataFrame | Series: ...

class ExpandingGroupby(BaseWindowGroupby, Expanding): ...
