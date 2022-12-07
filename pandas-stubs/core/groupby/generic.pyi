from typing import (
    Any,
    Callable,
    Generic,
    Iterable,
    Iterator,
    Literal,
    NamedTuple,
    Sequence,
    Union,
    overload,
)

from matplotlib.axes import (
    Axes as PlotAxes,
    SubplotBase as AxesSubplot,
)
from pandas.core.frame import DataFrame
from pandas.core.generic import NDFrame
from pandas.core.groupby.groupby import (  # , get_groupby as get_groupby
    GroupBy as GroupBy,
)
from pandas.core.groupby.grouper import Grouper
from pandas.core.series import Series
from typing_extensions import TypeAlias

from pandas._typing import (
    S1,
    AggFuncTypeBase,
    AggFuncTypeFrame,
    AxisType,
    Level,
    ListLike,
    Scalar,
)

AggScalar: TypeAlias = Union[str, Callable[..., Any]]
ScalarResult = ...

class NamedAgg(NamedTuple):
    column: str = ...
    aggfunc: AggScalar = ...

def generate_property(name: str, klass: type[NDFrame]): ...

class _SeriesGroupByScalar(SeriesGroupBy[S1]):
    def __iter__(self) -> Iterator[tuple[Scalar, Series]]: ...

class _SeriesGroupByNonScalar(SeriesGroupBy[S1]):
    def __iter__(self) -> Iterator[tuple[tuple, Series]]: ...

class SeriesGroupBy(GroupBy, Generic[S1]):
    def any(self, skipna: bool = ...) -> Series[bool]: ...
    def all(self, skipna: bool = ...) -> Series[bool]: ...
    def apply(self, func, *args, **kwargs) -> Series: ...
    @overload
    def aggregate(self, func: list[AggFuncTypeBase], *args, **kwargs) -> DataFrame: ...
    @overload
    def aggregate(self, func: AggFuncTypeBase, *args, **kwargs) -> Series: ...
    @overload
    def agg(self, func: list[AggFuncTypeBase], *args, **kwargs) -> DataFrame: ...
    @overload
    def agg(self, func: AggFuncTypeBase, *args, **kwargs) -> Series: ...
    def transform(self, func: Callable | str, *args, **kwargs) -> Series: ...
    def filter(self, func, dropna: bool = ..., *args, **kwargs): ...
    def nunique(self, dropna: bool = ...) -> Series: ...
    def describe(self, **kwargs) -> DataFrame: ...
    @overload
    def value_counts(
        self,
        normalize: Literal[False] = ...,
        sort: bool = ...,
        ascending: bool = ...,
        bins=...,
        dropna: bool = ...,
    ) -> Series[int]: ...
    @overload
    def value_counts(
        self,
        normalize: Literal[True],
        sort: bool = ...,
        ascending: bool = ...,
        bins=...,
        dropna: bool = ...,
    ) -> Series[float]: ...
    def count(self) -> Series[int]: ...
    def pct_change(
        self,
        periods: int = ...,
        fill_method: str = ...,
        limit=...,
        freq=...,
        axis: AxisType = ...,
    ) -> Series[float]: ...
    # Overrides and others from original pylance stubs
    @property
    def is_monotonic_increasing(self) -> bool: ...
    @property
    def is_monotonic_decreasing(self) -> bool: ...
    def bfill(self, limit: int | None = ...) -> Series[S1]: ...
    def cummax(self, axis: AxisType = ..., **kwargs) -> Series[S1]: ...
    def cummin(self, axis: AxisType = ..., **kwargs) -> Series[S1]: ...
    def cumprod(self, axis: AxisType = ..., **kwargs) -> Series[S1]: ...
    def cumsum(self, axis: AxisType = ..., **kwargs) -> Series[S1]: ...
    def ffill(self, limit: int | None = ...) -> Series[S1]: ...
    def first(self, **kwargs) -> Series[S1]: ...
    def head(self, n: int = ...) -> Series[S1]: ...
    def last(self, **kwargs) -> Series[S1]: ...
    def max(self, **kwargs) -> Series[S1]: ...
    def mean(self, **kwargs) -> Series[S1]: ...
    def median(self, **kwargs) -> Series[S1]: ...
    def min(self, **kwargs) -> Series[S1]: ...
    def nlargest(self, n: int = ..., keep: str = ...) -> Series[S1]: ...
    def nsmallest(self, n: int = ..., keep: str = ...) -> Series[S1]: ...
    def nth(self, n: int | Sequence[int], dropna: str | None = ...) -> Series[S1]: ...
    def sum(
        self,
        numeric_only: bool = ...,
        min_count: int = ...,
        engine=...,
        engine_kwargs=...,
    ) -> Series[S1]: ...
    def prod(self, numeric_only: bool = ..., min_count: int = ...) -> Series[S1]: ...
    def sem(self, ddof: int = ..., numeric_only: bool = ...) -> Series[float]: ...
    def std(self, ddof: int = ..., numeric_only: bool = ...) -> Series[float]: ...
    def var(self, ddof: int = ..., numeric_only: bool = ...) -> Series[float]: ...
    def tail(self, n: int = ...) -> Series[S1]: ...
    def unique(self) -> Series: ...
    def hist(
        self,
        by=...,
        ax: PlotAxes | None = ...,
        grid: bool = ...,
        xlabelsize: int | None = ...,
        xrot: float | None = ...,
        ylabelsize: int | None = ...,
        yrot: float | None = ...,
        figsize: tuple[float, float] | None = ...,
        bins: int | Sequence = ...,
        backend: str | None = ...,
        legend: bool = ...,
        **kwargs,
    ) -> AxesSubplot: ...

class _DataFrameGroupByScalar(DataFrameGroupBy):
    def __iter__(self) -> Iterator[tuple[Scalar, DataFrame]]: ...

class _DataFrameGroupByNonScalar(DataFrameGroupBy):
    def __iter__(self) -> Iterator[tuple[tuple, DataFrame]]: ...

class DataFrameGroupBy(GroupBy):
    def any(self, skipna: bool = ...) -> DataFrame: ...
    def all(self, skipna: bool = ...) -> DataFrame: ...
    # error: Overload 3 for "apply" will never be used because its parameters overlap overload 1
    @overload
    def apply(  # type: ignore[misc]
        self, func: Callable[[DataFrame], Scalar | list | dict], *args, **kwargs
    ) -> Series: ...
    @overload
    def apply(
        self, func: Callable[[DataFrame], Series | DataFrame], *args, **kwargs
    ) -> DataFrame: ...
    @overload
    def apply(  # pyright: ignore[reportOverlappingOverload]
        self, func: Callable[[Iterable], float], *args, **kwargs
    ) -> DataFrame: ...
    def aggregate(self, arg: AggFuncTypeFrame = ..., *args, **kwargs) -> DataFrame: ...
    agg = aggregate
    def transform(self, func: Callable | str, *args, **kwargs) -> DataFrame: ...
    def filter(
        self, func: Callable, dropna: bool = ..., *args, **kwargs
    ) -> DataFrame: ...
    def nunique(self, dropna: bool = ...) -> DataFrame: ...
    @overload
    def __getitem__(self, item: str) -> SeriesGroupBy: ...
    @overload
    def __getitem__(self, item: list[str]) -> DataFrameGroupBy: ...
    def count(self) -> DataFrame: ...
    def boxplot(
        self,
        grouped: DataFrame,
        subplots: bool = ...,
        column: str | Sequence | None = ...,
        fontsize: float | str = ...,
        rot: float = ...,
        grid: bool = ...,
        ax: PlotAxes | None = ...,
        figsize: tuple[float, float] | None = ...,
        layout: tuple[int, int] | None = ...,
        sharex: bool = ...,
        sharey: bool = ...,
        bins: int | Sequence = ...,
        backend: str | None = ...,
        **kwargs,
    ) -> AxesSubplot | Sequence[AxesSubplot]: ...
    # Overrides and others from original pylance stubs
    # These are "properties" but properties can't have all these arguments?!
    def corr(self, method: str | Callable, min_periods: int = ...) -> DataFrame: ...
    def cov(self, min_periods: int = ...) -> DataFrame: ...
    def diff(self, periods: int = ..., axis: AxisType = ...) -> DataFrame: ...
    def bfill(self, limit: int | None = ...) -> DataFrame: ...
    def corrwith(
        self,
        other: DataFrame,
        axis: AxisType = ...,
        drop: bool = ...,
        method: str = ...,
    ) -> Series: ...
    def cummax(
        self, axis: AxisType = ..., numeric_only: bool = ..., **kwargs
    ) -> DataFrame: ...
    def cummin(
        self, axis: AxisType = ..., numeric_only: bool = ..., **kwargs
    ) -> DataFrame: ...
    def cumprod(self, axis: AxisType = ..., **kwargs) -> DataFrame: ...
    def cumsum(self, axis: AxisType = ..., **kwargs) -> DataFrame: ...
    def describe(self, **kwargs) -> DataFrame: ...
    def ffill(self, limit: int | None = ...) -> DataFrame: ...
    @overload
    def fillna(
        self,
        value,
        method: str | None = ...,
        axis: AxisType = ...,
        limit: int | None = ...,
        downcast: dict | None = ...,
        *,
        inplace: Literal[True],
    ) -> None: ...
    @overload
    def fillna(
        self,
        value,
        method: str | None = ...,
        axis: AxisType = ...,
        limit: int | None = ...,
        downcast: dict | None = ...,
        *,
        inplace: Literal[False],
    ) -> DataFrame: ...
    @overload
    def fillna(
        self,
        value,
        method: str | None = ...,
        axis: AxisType = ...,
        inplace: bool = ...,
        limit: int | None = ...,
        downcast: dict | None = ...,
    ) -> DataFrame | None: ...
    def first(self, **kwargs) -> DataFrame: ...
    def head(self, n: int = ...) -> DataFrame: ...
    def hist(
        self,
        data: DataFrame,
        column: str | Sequence | None = ...,
        by=...,
        grid: bool = ...,
        xlabelsize: int | None = ...,
        xrot: float | None = ...,
        ylabelsize: int | None = ...,
        yrot: float | None = ...,
        ax: PlotAxes | None = ...,
        sharex: bool = ...,
        sharey: bool = ...,
        figsize: tuple[float, float] | None = ...,
        layout: tuple[int, int] | None = ...,
        bins: int | Sequence = ...,
        backend: str | None = ...,
        **kwargs,
    ) -> AxesSubplot | Sequence[AxesSubplot]: ...
    def idxmax(
        self, axis: AxisType = ..., skipna: bool = ..., numeric_only: bool = ...
    ) -> Series: ...
    def idxmin(
        self, axis: AxisType = ..., skipna: bool = ..., numeric_only: bool = ...
    ) -> Series: ...
    def last(self, **kwargs) -> DataFrame: ...
    def max(self, **kwargs) -> DataFrame: ...
    def mean(self, **kwargs) -> DataFrame: ...
    def median(self, **kwargs) -> DataFrame: ...
    def min(self, **kwargs) -> DataFrame: ...
    def nth(self, n: int | Sequence[int], dropna: str | None = ...) -> DataFrame: ...
    def pct_change(
        self,
        periods: int = ...,
        fill_method: str = ...,
        limit=...,
        freq=...,
        axis: AxisType = ...,
    ) -> DataFrame: ...
    def prod(self, numeric_only: bool = ..., min_count: int = ...) -> DataFrame: ...
    def quantile(
        self, q: float = ..., interpolation: str = ..., numeric_only: bool = ...
    ) -> DataFrame: ...
    def resample(self, rule, *args, **kwargs) -> Grouper: ...
    def sample(
        self,
        n: int | None = ...,
        frac: float | None = ...,
        replace: bool = ...,
        weights: ListLike | None = ...,
        random_state: int | None = ...,
    ) -> DataFrame: ...
    def sem(self, ddof: int = ..., numeric_only: bool = ...) -> DataFrame: ...
    def shift(
        self,
        periods: int = ...,
        freq: str = ...,
        axis: AxisType = ...,
        fill_value=...,
    ) -> DataFrame: ...
    @overload
    def skew(
        self,
        axis: AxisType = ...,
        skipna: bool = ...,
        numeric_only: bool = ...,
        *,
        level: Level,
        **kwargs,
    ) -> DataFrame: ...
    @overload
    def skew(
        self,
        axis: AxisType = ...,
        skipna: bool = ...,
        level: None = ...,
        numeric_only: bool = ...,
        **kwargs,
    ) -> Series: ...
    def std(self, ddof: int = ..., numeric_only: bool = ...) -> DataFrame: ...
    def sum(
        self,
        numeric_only: bool = ...,
        min_count: int = ...,
        engine=...,
        engine_kwargs=...,
    ) -> DataFrame: ...
    def tail(self, n: int = ...) -> DataFrame: ...
    def take(self, indices: Sequence, axis: AxisType = ..., **kwargs) -> DataFrame: ...
    def tshift(self, periods: int, freq=..., axis: AxisType = ...) -> DataFrame: ...
    def var(self, ddof: int = ..., numeric_only: bool = ...) -> DataFrame: ...
    @overload
    def value_counts(
        self,
        subset: ListLike | None = ...,
        normalize: Literal[False] = ...,
        sort: bool = ...,
        ascending: bool = ...,
        dropna: bool = ...,
    ) -> Series[int]: ...
    @overload
    def value_counts(
        self,
        subset: ListLike | None,
        normalize: Literal[True],
        sort: bool = ...,
        ascending: bool = ...,
        dropna: bool = ...,
    ) -> Series[float]: ...
    def __getattr__(self, name: str) -> SeriesGroupBy: ...
