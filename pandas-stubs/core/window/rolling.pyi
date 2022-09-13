from typing import (
    Any,
    Callable,
    Hashable,
)

import numpy as np
from pandas import (
    DataFrame,
    Series,
)
from pandas.core.base import SelectionMixin
from pandas.core.generic import NDFrame
from pandas.core.groupby.ops import BaseGrouper
from pandas.core.indexes.api import Index

from pandas._typing import (
    AggFuncType,
    Axis,
    Scalar,
    WindowingRankType,
)

class BaseWindow(SelectionMixin):
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
        obj: NDFrame,
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

class BaseWindowGroupby(BaseWindow):
    def __init__(
        self,
        obj: DataFrame | Series,
        *args,
        _grouper: BaseGrouper,
        _as_index: bool = ...,
        **kwargs,
    ) -> None: ...

class Window(BaseWindow):
    def aggregate(
        self, func: AggFuncType, *args, **kwargs
    ) -> Scalar | Series | DataFrame: ...
    agg = aggregate
    def sum(self, *args, **kwargs) -> DataFrame | Series: ...
    def mean(self, *args, **kwargs) -> DataFrame | Series: ...
    def var(self, ddof: int = ..., *args, **kwargs) -> DataFrame | Series: ...
    def std(self, ddof: int = ..., *args, **kwargs) -> DataFrame | Series: ...

class RollingAndExpandingMixin(BaseWindow):
    def count(self) -> DataFrame | Series: ...
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
    def skew(self, **kwargs) -> DataFrame | Series: ...
    def sem(self, ddof: int = ..., *args, **kwargs) -> DataFrame | Series: ...
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
    ): ...
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

class Rolling(RollingAndExpandingMixin):
    min_periods: int
    def aggregate(self, func, *args, **kwargs) -> Scalar | Series | DataFrame: ...
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
    ) -> Scalar | Series | DataFrame: ...

class RollingGroupby(BaseWindowGroupby, Rolling): ...
