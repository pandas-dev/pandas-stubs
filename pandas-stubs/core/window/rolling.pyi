from typing import (
    Any,
    Callable,
    Generic,
    Hashable,
    Literal,
    TypedDict,
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

class _NumbaKwargs(TypedDict, total=False):
    nopython: bool
    nogil: bool
    parallel: bool

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
    def aggregate(self, func: AggFuncType, *args, **kwargs) -> NDFrameT: ...
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
    def aggregate(self, func: AggFuncType, *args, **kwargs) -> NDFrameT: ...
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
        engine: Literal["cython", "numba"] | None = ...,
        engine_kwargs: _NumbaKwargs = ...,
        args: tuple[Any, ...] | None = ...,
        kwargs: dict[str, Any] | None = ...,
    ) -> NDFrameT: ...
    def sum(
        self,
        *,
        engine: str | None = ...,
        engine_kwargs: dict[str, bool] | None = ...,
    ) -> NDFrameT: ...
    def max(
        self,
        *,
        engine: str | None = ...,
        engine_kwargs: dict[str, bool] | None = ...,
    ) -> NDFrameT: ...
    def min(
        self,
        *,
        engine: str | None = ...,
        engine_kwargs: dict[str, bool] | None = ...,
    ) -> NDFrameT: ...
    def mean(
        self,
        *,
        engine: str | None = ...,
        engine_kwargs: dict[str, bool] | None = ...,
    ) -> NDFrameT: ...
    def median(
        self,
        engine: str | None = ...,
        engine_kwargs: dict[str, bool] | None = ...,
    ) -> NDFrameT: ...
    def std(
        self,
        ddof: int = ...,
        *,
        engine: str | None = ...,
        engine_kwargs: dict[str, bool] | None = ...,
    ) -> NDFrameT: ...
    def var(
        self,
        ddof: int = ...,
        *,
        engine: str | None = ...,
        engine_kwargs: dict[str, bool] | None = ...,
    ) -> NDFrameT: ...
    def skew(self) -> NDFrameT: ...
    def sem(self, ddof: int = ...) -> NDFrameT: ...
    def kurt(self) -> NDFrameT: ...
    def quantile(
        self,
        quantile: float,
        interpolation: str = ...,
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
    def aggregate(self, func, *args, **kwargs) -> NDFrameT: ...
    agg = aggregate
    def apply(
        self,
        func: Callable[..., Any],
        raw: bool = ...,
        engine: Literal["cython", "numba"] | None = ...,
        engine_kwargs: _NumbaKwargs = ...,
        args: tuple[Any, ...] | None = ...,
        kwargs: dict[str, Any] | None = ...,
    ) -> NDFrameT: ...

class RollingGroupby(BaseWindowGroupby[NDFrameT], Rolling): ...
