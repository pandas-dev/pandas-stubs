from typing import (
    Any,
    Callable,
    Generic,
    Hashable,
)

import numpy as np
from pandas import (
    DataFrame,
    Series,
)
from pandas.core.base import SelectionMixin
from pandas.core.groupby.ops import BaseGrouper
from pandas.core.indexes.api import Index

from pandas._typing import (
    AggFuncType,
    Axis,
    NDFrameT,
    Scalar,
    WindowingRankType,
)

class BaseWindow(SelectionMixin[NDFrameT], Generic[NDFrameT]):
    exclusions: frozenset[Hashable]
    obj: Any = ...  # Incomplete
    on: Any = ...  # Incomplete
    closed: Any = ...  # Incomplete
    window: Any = ...  # Incomplete
    min_periods: Any = ...  # Incomplete
    center: Any = ...  # Incomplete
    axis: Any = ...  # Incomplete
    method: Any = ...  # Incomplete
    def __init__(
        self,
        obj: NDFrameT,
        window: Any | None = ...,
        min_periods: int | None = ...,
        center: bool = ...,
        win_type: str | None = ...,
        axis: Axis = ...,
        on: str | Index | None = ...,
        closed: str | None = ...,
        method: str = ...,
        *,
        selection: Any | None = ...,
    ) -> None: ...
    @property
    def win_type(self): ...
    @property
    def is_datetimelike(self) -> bool: ...
    def validate(self) -> None: ...
    def __getattr__(self, attr: str): ...
    def __iter__(self): ...
    def aggregate(
        self, func: AggFuncType, *args, **kwargs
    ) -> Scalar | DataFrame | Series: ...
    agg = aggregate

class BaseWindowGroupby(BaseWindow[NDFrameT]):
    def __init__(
        self,
        obj: NDFrameT,
        *args,
        _grouper: BaseGrouper,
        _as_index: bool = ...,
        **kwargs,
    ) -> None: ...

class Window(BaseWindow[NDFrameT]):
    def aggregate(
        self, func: AggFuncType, *args, **kwargs
    ) -> Scalar | Series | DataFrame: ...
    agg = aggregate
    def sum(self, *args, **kwargs) -> NDFrameT: ...
    def mean(self, *args, **kwargs) -> NDFrameT: ...
    def var(self, ddof: int = ..., *args, **kwargs) -> NDFrameT: ...
    def std(self, ddof: int = ..., *args, **kwargs) -> NDFrameT: ...

class RollingAndExpandingMixin(BaseWindow[NDFrameT], Generic[NDFrameT]):
    def count(self) -> NDFrameT: ...
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
    def skew(self, **kwargs) -> NDFrameT: ...
    def sem(self, ddof: int = ..., *args, **kwargs) -> NDFrameT: ...
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

class Rolling(RollingAndExpandingMixin[NDFrameT]):
    min_periods: int
    def aggregate(self, func, *args, **kwargs) -> Scalar | Series | DataFrame: ...
    agg = aggregate
    def count(self) -> NDFrameT: ...
    def apply(
        self,
        func: Callable[..., Any],
        raw: bool = ...,
        engine: str | None = ...,
        engine_kwargs: dict[str, bool] | None = ...,
        args: tuple[Any, ...] | None = ...,
        kwargs: dict[str, Any] | None = ...,
    ) -> Scalar | Series | DataFrame: ...

class RollingGroupby(BaseWindowGroupby[NDFrameT], Rolling): ...
