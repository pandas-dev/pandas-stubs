from typing import (
    Any,
    Callable,
    Generic,
    overload,
)

import numpy as np
from pandas import (
    DataFrame,
    Series,
)
from pandas.core.base import SelectionMixin

from pandas._typing import (
    AggFuncTypeBase,
    AggFuncTypeFrame,
    AggFuncTypeSeriesToFrame,
    NDFrameT,
    QuantileInterpolation,
    WindowingEngine,
    WindowingEngineKwargs,
    WindowingRankType,
)

class BaseWindow(SelectionMixin[NDFrameT], Generic[NDFrameT]):
    def __getattr__(self, attr: str): ...
    def __iter__(self): ...
    @overload
    def aggregate(
        self: BaseWindow[Series], func: AggFuncTypeBase, *args: Any, **kwargs: Any
    ) -> Series: ...
    @overload
    def aggregate(
        self: BaseWindow[Series],
        func: AggFuncTypeSeriesToFrame,
        *args: Any,
        **kwargs: Any,
    ) -> DataFrame: ...
    @overload
    def aggregate(
        self: BaseWindow[DataFrame],
        func: AggFuncTypeFrame,
        *args: Any,
        **kwargs: Any,
    ) -> DataFrame: ...
    agg = aggregate

class BaseWindowGroupby(BaseWindow[NDFrameT]): ...

class Window(BaseWindow[NDFrameT]):
    @overload
    def aggregate(
        self: Window[Series], func: AggFuncTypeBase, *args: Any, **kwargs: Any
    ) -> Series: ...
    @overload
    def aggregate(
        self: Window[Series],
        func: AggFuncTypeSeriesToFrame,
        *args: Any,
        **kwargs: Any,
    ) -> DataFrame: ...
    @overload
    def aggregate(
        self: Window[DataFrame],
        func: AggFuncTypeFrame,
        *args: Any,
        **kwargs: Any,
    ) -> DataFrame: ...
    def sum(self, **kwargs: Any) -> NDFrameT: ...
    def mean(self, **kwargs: Any) -> NDFrameT: ...
    def var(self, ddof: int = ..., **kwargs: Any) -> NDFrameT: ...
    def std(self, ddof: int = ..., **kwargs: Any) -> NDFrameT: ...

class RollingAndExpandingMixin(BaseWindow[NDFrameT], Generic[NDFrameT]):
    def count(self) -> NDFrameT: ...
    def apply(
        self,
        func: Callable[..., Any],
        raw: bool = ...,
        engine: WindowingEngine = ...,
        engine_kwargs: WindowingEngineKwargs = ...,
        args: tuple[Any, ...] | None = ...,
        kwargs: dict[str, Any] | None = ...,
    ) -> NDFrameT: ...
    def sum(
        self,
        *,
        engine: WindowingEngine = ...,
        engine_kwargs: WindowingEngineKwargs = ...,
    ) -> NDFrameT: ...
    def max(
        self,
        *,
        engine: WindowingEngine = ...,
        engine_kwargs: WindowingEngineKwargs = ...,
    ) -> NDFrameT: ...
    def min(
        self,
        *,
        engine: WindowingEngine = ...,
        engine_kwargs: WindowingEngineKwargs = ...,
    ) -> NDFrameT: ...
    def mean(
        self,
        *,
        engine: WindowingEngine = ...,
        engine_kwargs: WindowingEngineKwargs = ...,
    ) -> NDFrameT: ...
    def median(
        self,
        engine: WindowingEngine = ...,
        engine_kwargs: WindowingEngineKwargs = ...,
    ) -> NDFrameT: ...
    def std(
        self,
        ddof: int = ...,
        *,
        engine: WindowingEngine = ...,
        engine_kwargs: WindowingEngineKwargs = ...,
    ) -> NDFrameT: ...
    def var(
        self,
        ddof: int = ...,
        *,
        engine: WindowingEngine = ...,
        engine_kwargs: WindowingEngineKwargs = ...,
    ) -> NDFrameT: ...
    def skew(self) -> NDFrameT: ...
    def sem(self, ddof: int = ...) -> NDFrameT: ...
    def kurt(self) -> NDFrameT: ...
    def quantile(
        self,
        quantile: float,
        interpolation: QuantileInterpolation = ...,
    ) -> NDFrameT: ...
    def rank(
        self,
        method: WindowingRankType = ...,
        ascending: bool = ...,
        pct: bool = ...,
    ) -> NDFrameT: ...
    def cov(
        self,
        other: DataFrame | Series | None = ...,
        pairwise: bool | None = ...,
        ddof: int = ...,
    ) -> NDFrameT: ...
    def corr(
        self,
        other: DataFrame | Series | None = ...,
        pairwise: bool | None = ...,
        ddof: int = ...,
    ) -> NDFrameT: ...

class Rolling(RollingAndExpandingMixin[NDFrameT]):
    @overload
    def aggregate(
        self: Rolling[Series], func: AggFuncTypeBase, *args: Any, **kwargs: Any
    ) -> Series: ...
    @overload
    def aggregate(
        self: Rolling[Series],
        func: AggFuncTypeSeriesToFrame,
        *args: Any,
        **kwargs: Any,
    ) -> DataFrame: ...
    @overload
    def aggregate(
        self: Rolling[DataFrame],
        func: AggFuncTypeFrame,
        *args: Any,
        **kwargs: Any,
    ) -> DataFrame: ...
    def apply(
        self,
        func: Callable[..., Any],
        raw: bool = ...,
        engine: WindowingEngine = ...,
        engine_kwargs: WindowingEngineKwargs | None = ...,
        args: tuple[Any, ...] | None = ...,
        kwargs: dict[str, Any] | None = ...,
    ) -> NDFrameT: ...

class RollingGroupby(BaseWindowGroupby[NDFrameT], Rolling): ...
