from collections.abc import (
    Callable,
    Hashable,
    Iterable,
    Iterator,
    Mapping,
    Sequence,
)
from typing import (
    Any,
    Generic,
    Literal,
    NamedTuple,
    final,
    overload,
)

from matplotlib.axes import Axes as PlotAxes
import numpy as np
from pandas.core.frame import DataFrame
from pandas.core.groupby.groupby import (
    GroupBy,
    GroupByPlot,
)
from pandas.core.series import Series
from typing_extensions import (
    Self,
    TypeAlias,
)

from pandas._libs.lib import NoDefault
from pandas._typing import (
    S1,
    AggFuncTypeBase,
    AggFuncTypeFrame,
    ArrayLike,
    Axis,
    ByT,
    CorrelationMethod,
    Dtype,
    IndexLabel,
    Level,
    ListLike,
    Scalar,
    TakeIndexer,
    WindowingEngine,
    WindowingEngineKwargs,
)

AggScalar: TypeAlias = str | Callable[..., Any]

class NamedAgg(NamedTuple):
    column: str
    aggfunc: AggScalar

class SeriesGroupBy(GroupBy[Series[S1]], Generic[S1, ByT]):
    @overload
    def aggregate(
        self,
        func: list[AggFuncTypeBase],
        *args,
        engine: WindowingEngine = ...,
        engine_kwargs: WindowingEngineKwargs = ...,
        **kwargs,
    ) -> DataFrame: ...
    @overload
    def aggregate(
        self,
        func: AggFuncTypeBase | None = ...,
        *args,
        engine: WindowingEngine = ...,
        engine_kwargs: WindowingEngineKwargs = ...,
        **kwargs,
    ) -> Series: ...
    agg = aggregate
    def transform(
        self,
        func: Callable | str,
        *args,
        engine: WindowingEngine = ...,
        engine_kwargs: WindowingEngineKwargs = ...,
        **kwargs,
    ) -> Series: ...
    def filter(
        self, func: Callable | str, dropna: bool = ..., *args, **kwargs
    ) -> Series: ...
    def nunique(self, dropna: bool = ...) -> Series[int]: ...
    # describe delegates to super() method but here it has keyword-only parameters
    def describe(  # type: ignore[override] # pyright: ignore[reportIncompatibleMethodOverride]
        self,
        *,
        percentiles: Iterable[float] | None = ...,
        include: Literal["all"] | list[Dtype] | None = ...,
        exclude: list[Dtype] | None = ...,
    ) -> DataFrame: ...
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
    def fillna(
        self,
        value: (
            Scalar | ArrayLike | Series | DataFrame | Mapping[Hashable, Scalar] | None
        ) = ...,
        method: Literal["bfill", "ffill"] | None = ...,
        axis: Axis | None | NoDefault = ...,
        inplace: bool = ...,
        limit: int | None = ...,
        downcast: dict | None | NoDefault = ...,
    ) -> Series[S1] | None: ...
    def take(
        self,
        indices: TakeIndexer,
        axis: Axis | NoDefault = ...,
        **kwargs,
    ) -> Series[S1]: ...
    def skew(
        self,
        axis: Axis | NoDefault = ...,
        skipna: bool = ...,
        numeric_only: bool = ...,
        **kwargs,
    ) -> Series: ...
    @property
    def plot(self) -> GroupByPlot[Self]: ...
    def nlargest(
        self, n: int = ..., keep: Literal["first", "last", "all"] = ...
    ) -> Series[S1]: ...
    def nsmallest(
        self, n: int = ..., keep: Literal["first", "last", "all"] = ...
    ) -> Series[S1]: ...
    def idxmin(self, axis: Axis | NoDefault = ..., skipna: bool = ...) -> Series: ...
    def idxmax(self, axis: Axis | NoDefault = ..., skipna: bool = ...) -> Series: ...
    def corr(
        self,
        other: Series,
        method: CorrelationMethod = ...,
        min_periods: int | None = ...,
    ) -> Series: ...
    def cov(
        self, other: Series, min_periods: int | None = ..., ddof: int | None = ...
    ) -> Series: ...
    @property
    def is_monotonic_increasing(self) -> Series[bool]: ...
    @property
    def is_monotonic_decreasing(self) -> Series[bool]: ...
    def hist(
        self,
        by: IndexLabel | None = ...,
        ax: PlotAxes | None = ...,
        grid: bool = ...,
        xlabelsize: int | None = ...,
        xrot: float | None = ...,
        ylabelsize: int | None = ...,
        yrot: float | None = ...,
        figsize: tuple[float, float] | None = ...,
        bins: int | Sequence[int] = ...,
        backend: str | None = ...,
        legend: bool = ...,
        **kwargs,
    ) -> Series: ...  # Series[Axes] but this is not allowed
    @property
    def dtype(self) -> Series: ...
    def unique(self) -> Series: ...
    # Overrides that provide more precise return types over the GroupBy class
    @final  # type: ignore[misc]
    def __iter__(  # pyright: ignore[reportIncompatibleMethodOverride]
        self,
    ) -> Iterator[tuple[ByT, Series[S1]]]: ...

class DataFrameGroupBy(GroupBy[DataFrame], Generic[ByT]):
    # error: Overload 3 for "apply" will never be used because its parameters overlap overload 1
    @overload  # type: ignore[override]
    def apply(  # type: ignore[overload-overlap]
        self,
        func: Callable[[DataFrame], Scalar | list | dict],
        *args,
        **kwargs,
    ) -> Series: ...
    @overload
    def apply(
        self,
        func: Callable[[DataFrame], Series | DataFrame],
        *args,
        **kwargs,
    ) -> DataFrame: ...
    @overload
    def apply(  # pyright: ignore[reportOverlappingOverload,reportIncompatibleMethodOverride]
        self,
        func: Callable[[Iterable], float],
        *args,
        **kwargs,
    ) -> DataFrame: ...
    # error: overload 1 overlaps overload 2 because of different return types
    @overload
    def aggregate(self, func: Literal["size"]) -> Series: ...  # type: ignore[overload-overlap]  # pyright: ignore[reportOverlappingOverload]
    @overload
    def aggregate(
        self,
        func: AggFuncTypeFrame | None = ...,
        *args,
        engine: WindowingEngine = ...,
        engine_kwargs: WindowingEngineKwargs = ...,
        **kwargs,
    ) -> DataFrame: ...
    agg = aggregate
    def transform(
        self,
        func: Callable | str,
        *args,
        engine: WindowingEngine = ...,
        engine_kwargs: WindowingEngineKwargs = ...,
        **kwargs,
    ) -> DataFrame: ...
    def filter(
        self, func: Callable, dropna: bool = ..., *args, **kwargs
    ) -> DataFrame: ...
    @overload
    def __getitem__(  # type: ignore[overload-overlap]
        self, key: Scalar | Hashable | tuple[Hashable, ...]
    ) -> SeriesGroupBy[Any, ByT]: ...
    @overload
    def __getitem__(  # pyright: ignore[reportIncompatibleMethodOverride]
        self, key: Iterable[Hashable] | slice
    ) -> DataFrameGroupBy[ByT]: ...
    def nunique(self, dropna: bool = ...) -> DataFrame: ...
    def idxmax(
        self,
        axis: Axis | None | NoDefault = ...,
        skipna: bool = ...,
        numeric_only: bool = ...,
    ) -> DataFrame: ...
    def idxmin(
        self,
        axis: Axis | None | NoDefault = ...,
        skipna: bool = ...,
        numeric_only: bool = ...,
    ) -> DataFrame: ...
    @overload
    def boxplot(
        self,
        subplots: Literal[True] = ...,
        column: IndexLabel | None = ...,
        fontsize: float | str | None = ...,
        rot: float = ...,
        grid: bool = ...,
        ax: PlotAxes | None = ...,
        figsize: tuple[float, float] | None = ...,
        layout: tuple[int, int] | None = ...,
        sharex: bool = ...,
        sharey: bool = ...,
        backend: str | None = ...,
        **kwargs,
    ) -> Series: ...  # Series[PlotAxes] but this is not allowed
    @overload
    def boxplot(
        self,
        subplots: Literal[False],
        column: IndexLabel | None = ...,
        fontsize: float | str | None = ...,
        rot: float = ...,
        grid: bool = ...,
        ax: PlotAxes | None = ...,
        figsize: tuple[float, float] | None = ...,
        layout: tuple[int, int] | None = ...,
        sharex: bool = ...,
        sharey: bool = ...,
        backend: str | None = ...,
        **kwargs,
    ) -> PlotAxes: ...
    @overload
    def boxplot(
        self,
        subplots: bool,
        column: IndexLabel | None = ...,
        fontsize: float | str | None = ...,
        rot: float = ...,
        grid: bool = ...,
        ax: PlotAxes | None = ...,
        figsize: tuple[float, float] | None = ...,
        layout: tuple[int, int] | None = ...,
        sharex: bool = ...,
        sharey: bool = ...,
        backend: str | None = ...,
        **kwargs,
    ) -> PlotAxes | Series: ...  # Series[PlotAxes]
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
    def take(
        self, indices: TakeIndexer, axis: Axis | None | NoDefault = ..., **kwargs
    ) -> DataFrame: ...
    @overload
    def skew(  # type: ignore[overload-overlap]
        self,
        axis: Axis | None | NoDefault = ...,
        skipna: bool = ...,
        numeric_only: bool = ...,
        *,
        level: Level,
        **kwargs,
    ) -> DataFrame: ...
    @overload
    def skew(
        self,
        axis: Axis | None | NoDefault = ...,
        skipna: bool = ...,
        numeric_only: bool = ...,
        *,
        level: None = ...,
        **kwargs,
    ) -> Series: ...
    @property
    def plot(self) -> GroupByPlot[Self]: ...
    def corr(
        self,
        method: str | Callable[[np.ndarray, np.ndarray], float] = ...,
        min_periods: int = ...,
        numeric_only: bool = ...,
    ) -> DataFrame: ...
    def cov(
        self,
        min_periods: int | None = ...,
        ddof: int | None = ...,
        numeric_only: bool = ...,
    ) -> DataFrame: ...
    def hist(
        self,
        column: IndexLabel | None = ...,
        by: IndexLabel | None = ...,
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
        bins: int | Sequence[int] = ...,
        backend: str | None = ...,
        legend: bool = ...,
        **kwargs,
    ) -> Series: ...  # Series[Axes] but this is not allowed
    @property
    def dtypes(self) -> Series: ...
    def corrwith(
        self,
        other: DataFrame | Series,
        axis: Axis | NoDefault = ...,
        drop: bool = ...,
        method: CorrelationMethod = ...,
        numeric_only: bool = ...,
    ) -> DataFrame: ...
    def __getattr__(self, name: str) -> SeriesGroupBy[Any, ByT]: ...
    # Overrides that provide more precise return types over the GroupBy class
    @final  # type: ignore[misc]
    def __iter__(  # pyright: ignore[reportIncompatibleMethodOverride]
        self,
    ) -> Iterator[tuple[ByT, DataFrame]]: ...
