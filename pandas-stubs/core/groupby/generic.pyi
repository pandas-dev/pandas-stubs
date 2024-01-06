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
    TypeVar,
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
    FillnaOptions,
    IndexLabel,
    Level,
    ListLike,
    Scalar,
    TakeIndexer,
    WindowingEngine,
    WindowingEngineKwargs,
)

from pandas.plotting import boxplot_frame_groupby

AggScalar: TypeAlias = str | Callable[..., Any]
ScalarResult = TypeVar("ScalarResult")  # noqa: PYI001

class NamedAgg(NamedTuple):
    column: str
    aggfunc: AggScalar

class SeriesGroupBy(GroupBy[Series[S1]], Generic[S1, ByT]):
    @overload
    def aggregate(
        self,
        func: list[AggFuncTypeBase],
        *args,
        engine: WindowingEngine = None,
        engine_kwargs: WindowingEngineKwargs = None,
        **kwargs,
    ) -> DataFrame: ...
    @overload
    def aggregate(
        self,
        func: AggFuncTypeBase | None = None,
        *args,
        engine: WindowingEngine = None,
        engine_kwargs: WindowingEngineKwargs = None,
        **kwargs,
    ) -> Series: ...
    agg = aggregate
    def transform(
        self,
        func: Callable | str,
        *args,
        engine: WindowingEngine = None,
        engine_kwargs: WindowingEngineKwargs = None,
        **kwargs,
    ) -> Series: ...
    def filter(
        self, func: Callable | str, dropna: bool = True, *args, **kwargs
    ) -> Series: ...
    def nunique(self, dropna: bool = True) -> Series[int]: ...
    # describe delegates to super() method but here it has keyword-only parameters
    def describe(  # type: ignore[override] # pyright: ignore[reportIncompatibleMethodOverride]
        self,
        *,
        percentiles: Iterable[float] | None = None,
        include: Literal["all"] | list[Dtype] | None = None,
        exclude: list[Dtype] | None = None,
    ) -> DataFrame: ...
    @overload
    def value_counts(
        self,
        normalize: Literal[False] = False,
        sort: bool = True,
        ascending: bool = False,
        bins=None,
        dropna: bool = True,
    ) -> Series[int]: ...
    @overload
    def value_counts(
        self,
        normalize: Literal[True],
        sort: bool = True,
        ascending: bool = False,
        bins=None,
        dropna: bool = True,
    ) -> Series[float]: ...
    def fillna(
        self,
        value: object | ArrayLike | None = None,
        method: FillnaOptions | None = None,
        axis: Axis | None | NoDefault = ...,
        inplace: bool = False,
        limit: int | None = None,
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
        skipna: bool = True,
        numeric_only: bool = False,
        **kwargs,
    ) -> Series: ...
    @property
    def plot(self) -> GroupByPlot[Self]: ...
    def nlargest(
        self, n: int = 5, keep: Literal["first", "last", "all"] = "first"
    ) -> Series[S1]: ...
    def nsmallest(
        self, n: int = 5, keep: Literal["first", "last", "all"] = "first"
    ) -> Series[S1]: ...
    def idxmin(self, axis: Axis | NoDefault = ..., skipna: bool = True) -> Series: ...
    def idxmax(self, axis: Axis | NoDefault = ..., skipna: bool = True) -> Series: ...
    def corr(
        self,
        other: Series,
        method: CorrelationMethod = "pearson",
        min_periods: int | None = None,
    ) -> Series: ...
    def cov(
        self, other: Series, min_periods: int | None = None, ddof: int | None = 1
    ) -> Series: ...
    @property
    def is_monotonic_increasing(self) -> Series[bool]: ...
    @property
    def is_monotonic_decreasing(self) -> Series[bool]: ...
    def hist(
        self,
        by: IndexLabel | None = None,
        ax: PlotAxes | None = None,
        grid: bool = True,
        xlabelsize: float | None = None,
        xrot: float | None = None,
        ylabelsize: float | None = None,
        yrot: float | None = None,
        figsize: tuple[float, float] | None = None,
        bins: int | Sequence[int] = 10,
        backend: str | None = None,
        legend: bool = False,
        **kwargs,
    ) -> Series: ...  # Series[Axes] but this is not allowed
    @property
    def dtype(self) -> Series: ...
    def unique(self) -> Series: ...
    # Overrides that provide more precise return types over the GroupBy class
    @final  # type: ignore[misc]
    def __iter__(self) -> Iterator[tuple[ByT, Series[S1]]]: ...  # pyright: ignore

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
        func: AggFuncTypeFrame | None = None,
        *args,
        engine: WindowingEngine = None,
        engine_kwargs: WindowingEngineKwargs = None,
        **kwargs,
    ) -> DataFrame: ...
    agg = aggregate
    def transform(
        self,
        func: Callable | str,
        *args,
        engine: WindowingEngine = None,
        engine_kwargs: WindowingEngineKwargs = None,
        **kwargs,
    ) -> DataFrame: ...
    def filter(
        self, func: Callable, dropna: bool = True, *args, **kwargs
    ) -> DataFrame: ...
    @overload
    def __getitem__(  # type: ignore[overload-overlap]
        self, key: Scalar | Hashable | tuple[Hashable, ...]
    ) -> SeriesGroupBy[Any, ByT]: ...
    @overload
    def __getitem__(  # pyright: ignore[reportIncompatibleMethodOverride]
        self, key: Iterable[Hashable] | slice
    ) -> DataFrameGroupBy[ByT]: ...
    def nunique(self, dropna: bool = True) -> DataFrame: ...
    def idxmax(
        self,
        axis: Axis | None | NoDefault = ...,
        skipna: bool = True,
        numeric_only: bool = False,
    ) -> DataFrame: ...
    def idxmin(
        self,
        axis: Axis | None | NoDefault = ...,
        skipna: bool = True,
        numeric_only: bool = False,
    ) -> DataFrame: ...
    boxplot = boxplot_frame_groupby
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
    def fillna(
        self,
        value: Hashable | Mapping | Series | DataFrame | None = None,
        method: FillnaOptions | None = None,
        axis: Axis | None | NoDefault = ...,
        inplace: Literal[False] = False,
        limit: int | None = None,
        downcast: dict | None | NoDefault = ...,
    ) -> DataFrame: ...
    def take(
        self, indices: TakeIndexer, axis: Axis | None | NoDefault = ..., **kwargs
    ) -> DataFrame: ...
    @overload
    def skew(  # type: ignore[overload-overlap]
        self,
        axis: Axis | None | NoDefault = ...,
        skipna: bool = True,
        numeric_only: bool = False,
        *,
        level: Level,
        **kwargs,
    ) -> DataFrame: ...
    @overload
    def skew(
        self,
        axis: Axis | None | NoDefault = ...,
        skipna: bool = True,
        numeric_only: bool = False,
        *,
        level: None = None,
        **kwargs,
    ) -> Series: ...
    @property
    def plot(self) -> GroupByPlot[Self]: ...
    def corr(
        self,
        method: str | Callable[[np.ndarray, np.ndarray], float] = "pearson",
        min_periods: int = 1,
        numeric_only: bool = False,
    ) -> DataFrame: ...
    def cov(
        self,
        min_periods: int | None = None,
        ddof: int | None = 1,
        numeric_only: bool = False,
    ) -> DataFrame: ...
    def hist(
        self,
        column: IndexLabel | None = None,
        by: IndexLabel | None = None,
        grid: bool = True,
        xlabelsize: float | None = None,
        xrot: float | None = None,
        ylabelsize: float | None = None,
        yrot: float | None = None,
        ax: PlotAxes | None = None,
        sharex: bool = False,
        sharey: bool = False,
        figsize: tuple[float, float] | None = None,
        layout: tuple[int, int] | None = None,
        bins: int | Sequence[int] = 10,
        backend: str | None = None,
        legend: bool = False,
        **kwargs,
    ) -> Series: ...  # Series[Axes] but this is not allowed
    @property
    def dtypes(self) -> Series: ...
    def corrwith(
        self,
        other: DataFrame | Series,
        axis: Axis | NoDefault = ...,
        drop: bool = False,
        method: CorrelationMethod = "pearson",
        numeric_only: bool = False,
    ) -> DataFrame: ...
    def __getattr__(self, name: str) -> SeriesGroupBy[Any, ByT]: ...
    # Overrides that provide more precise return types over the GroupBy class
    @final  # type: ignore[misc]
    def __iter__(self) -> Iterator[tuple[ByT, DataFrame]]: ...  # pyright: ignore
