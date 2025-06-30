from collections.abc import (
    Callable,
    Hashable,
    Iterable,
    Iterator,
    Sequence,
)
from typing import (
    Any,
    Concatenate,
    Generic,
    Literal,
    NamedTuple,
    Protocol,
    TypeVar,
    final,
    overload,
)

from matplotlib.axes import Axes as PlotAxes
import numpy as np
from pandas.core.frame import DataFrame
from pandas.core.groupby.base import TransformReductionListType
from pandas.core.groupby.groupby import (
    GroupBy,
    GroupByPlot,
)
from pandas.core.series import Series
from typing_extensions import (
    Self,
    TypeAlias,
)

from pandas._libs.tslibs.timestamps import Timestamp
from pandas._typing import (
    S2,
    S3,
    AggFuncTypeBase,
    AggFuncTypeFrame,
    ByT,
    CorrelationMethod,
    Dtype,
    IndexLabel,
    Level,
    ListLike,
    NsmallestNlargestKeep,
    P,
    Scalar,
    TakeIndexer,
    WindowingEngine,
    WindowingEngineKwargs,
)

AggScalar: TypeAlias = str | Callable[..., Any]

class NamedAgg(NamedTuple):
    column: str
    aggfunc: AggScalar

class SeriesGroupBy(GroupBy[Series[S2]], Generic[S2, ByT]):
    @overload
    def aggregate(  # pyrefly: ignore
        self,
        func: Callable[Concatenate[Series[S2], P], S3],
        /,
        *args,
        engine: WindowingEngine = ...,
        engine_kwargs: WindowingEngineKwargs = ...,
        **kwargs,
    ) -> Series[S3]: ...
    @overload
    def aggregate(
        self,
        func: Callable[[Series], S3],
        *args,
        engine: WindowingEngine = ...,
        engine_kwargs: WindowingEngineKwargs = ...,
        **kwargs,
    ) -> Series[S3]: ...
    @overload
    def aggregate(
        self,
        func: list[AggFuncTypeBase],
        /,
        *args,
        engine: WindowingEngine = ...,
        engine_kwargs: WindowingEngineKwargs = ...,
        **kwargs,
    ) -> DataFrame: ...
    @overload
    def aggregate(
        self,
        func: AggFuncTypeBase | None = ...,
        /,
        *args,
        engine: WindowingEngine = ...,
        engine_kwargs: WindowingEngineKwargs = ...,
        **kwargs,
    ) -> Series: ...
    agg = aggregate
    @overload
    def transform(
        self,
        func: Callable[Concatenate[Series[S2], P], Series[S3]],
        /,
        *args: Any,
        engine: WindowingEngine = ...,
        engine_kwargs: WindowingEngineKwargs = ...,
        **kwargs: Any,
    ) -> Series[S3]: ...
    @overload
    def transform(
        self,
        func: Callable,
        *args: Any,
        **kwargs: Any,
    ) -> Series: ...
    @overload
    def transform(
        self, func: TransformReductionListType, *args, **kwargs
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
        bins: int | Sequence[int] | None = ...,
        dropna: bool = ...,
    ) -> Series[int]: ...
    @overload
    def value_counts(
        self,
        normalize: Literal[True],
        sort: bool = ...,
        ascending: bool = ...,
        bins: int | Sequence[int] | None = ...,
        dropna: bool = ...,
    ) -> Series[float]: ...
    def take(
        self,
        indices: TakeIndexer,
        **kwargs,
    ) -> Series[S2]: ...
    def skew(
        self,
        skipna: bool = ...,
        numeric_only: bool = ...,
        **kwargs,
    ) -> Series: ...
    @property
    def plot(self) -> GroupByPlot[Self]: ...
    def nlargest(
        self, n: int = ..., keep: NsmallestNlargestKeep = ...
    ) -> Series[S2]: ...
    def nsmallest(
        self, n: int = ..., keep: NsmallestNlargestKeep = ...
    ) -> Series[S2]: ...
    def idxmin(self, skipna: bool = ...) -> Series: ...
    def idxmax(self, skipna: bool = ...) -> Series: ...
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
        xlabelsize: float | str | None = ...,
        xrot: float | None = ...,
        ylabelsize: float | str | None = ...,
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
    ) -> Iterator[tuple[ByT, Series[S2]]]: ...

_TT = TypeVar("_TT", bound=Literal[True, False])

# ty ignore needed because of https://github.com/astral-sh/ty/issues/157#issuecomment-3017337945
class DFCallable1(Protocol[P]):  # ty: ignore[invalid-argument-type]
    def __call__(
        self, df: DataFrame, /, *args: P.args, **kwargs: P.kwargs
    ) -> Scalar | list | dict: ...

class DFCallable2(Protocol[P]):  # ty: ignore[invalid-argument-type]
    def __call__(
        self, df: DataFrame, /, *args: P.args, **kwargs: P.kwargs
    ) -> DataFrame | Series: ...

class DFCallable3(Protocol[P]):  # ty: ignore[invalid-argument-type]
    def __call__(self, df: Iterable, /, *args: P.args, **kwargs: P.kwargs) -> float: ...

class DataFrameGroupBy(GroupBy[DataFrame], Generic[ByT, _TT]):
    # error: Overload 3 for "apply" will never be used because its parameters overlap overload 1
    @overload  # type: ignore[override]
    def apply(
        self,
        func: DFCallable1[P],
        /,
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> Series: ...
    @overload
    def apply(
        self,
        func: DFCallable2[P],
        /,
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> DataFrame: ...
    @overload
    def apply(
        self,
        func: DFCallable3[P],
        /,
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> DataFrame: ...
    # error: overload 1 overlaps overload 2 because of different return types
    @overload
    def aggregate(self, func: Literal["size"]) -> Series: ...  # type: ignore[overload-overlap]
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
    @overload
    def transform(
        self,
        func: Callable[Concatenate[DataFrame, P], DataFrame],
        *args: Any,
        engine: WindowingEngine = ...,
        engine_kwargs: WindowingEngineKwargs = ...,
        **kwargs: Any,
    ) -> DataFrame: ...
    @overload
    def transform(
        self,
        func: Callable,
        *args: Any,
        **kwargs: Any,
    ) -> DataFrame: ...
    @overload
    def transform(
        self, func: TransformReductionListType, *args, **kwargs
    ) -> DataFrame: ...
    def filter(
        self, func: Callable, dropna: bool = ..., *args, **kwargs
    ) -> DataFrame: ...
    @overload
    def __getitem__(self, key: Scalar) -> SeriesGroupBy[Any, ByT]: ...  # type: ignore[overload-overlap] # pyright: ignore[reportOverlappingOverload]
    @overload
    def __getitem__(  # pyright: ignore[reportIncompatibleMethodOverride]
        self, key: Iterable[Hashable]
    ) -> DataFrameGroupBy[ByT, _TT]: ...
    def nunique(self, dropna: bool = ...) -> DataFrame: ...
    def idxmax(
        self,
        skipna: bool = ...,
        numeric_only: bool = ...,
    ) -> DataFrame: ...
    def idxmin(
        self,
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
        self: DataFrameGroupBy[ByT, Literal[True]],
        subset: ListLike | None = ...,
        normalize: Literal[False] = ...,
        sort: bool = ...,
        ascending: bool = ...,
        dropna: bool = ...,
    ) -> Series[int]: ...
    @overload
    def value_counts(
        self: DataFrameGroupBy[ByT, Literal[True]],
        subset: ListLike | None,
        normalize: Literal[True],
        sort: bool = ...,
        ascending: bool = ...,
        dropna: bool = ...,
    ) -> Series[float]: ...
    @overload
    def value_counts(
        self: DataFrameGroupBy[ByT, Literal[False]],
        subset: ListLike | None = ...,
        normalize: Literal[False] = ...,
        sort: bool = ...,
        ascending: bool = ...,
        dropna: bool = ...,
    ) -> DataFrame: ...
    @overload
    def value_counts(
        self: DataFrameGroupBy[ByT, Literal[False]],
        subset: ListLike | None,
        normalize: Literal[True],
        sort: bool = ...,
        ascending: bool = ...,
        dropna: bool = ...,
    ) -> DataFrame: ...
    def take(self, indices: TakeIndexer, **kwargs) -> DataFrame: ...
    @overload
    def skew(
        self,
        skipna: bool = ...,
        numeric_only: bool = ...,
        *,
        level: Level,
        **kwargs,
    ) -> DataFrame: ...
    @overload
    def skew(
        self,
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
        xlabelsize: float | str | None = ...,
        xrot: float | None = ...,
        ylabelsize: float | str | None = ...,
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
    def __getattr__(self, name: str) -> SeriesGroupBy[Any, ByT]: ...
    # Overrides that provide more precise return types over the GroupBy class
    @final  # type: ignore[misc]
    def __iter__(  # pyright: ignore[reportIncompatibleMethodOverride]
        self,
    ) -> Iterator[tuple[ByT, DataFrame]]: ...
    @overload
    def size(self: DataFrameGroupBy[ByT, Literal[True]]) -> Series[int]: ...
    @overload
    def size(self: DataFrameGroupBy[ByT, Literal[False]]) -> DataFrame: ...
    @overload
    def size(self: DataFrameGroupBy[Timestamp, Literal[True]]) -> Series[int]: ...
    @overload
    def size(self: DataFrameGroupBy[Timestamp, Literal[False]]) -> DataFrame: ...
