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
    Never,
    Self,
    TypeAlias,
    TypeVar,
    final,
    overload,
)

from matplotlib.axes import Axes as PlotAxes
from pandas.core.frame import DataFrame
from pandas.core.groupby.base import TransformReductionListType
from pandas.core.groupby.groupby import (
    GroupBy,
    GroupByPlot,
)
from pandas.core.series import Series

from pandas._libs.tslibs.timestamps import Timestamp
from pandas._typing import (
    S2,
    S3,
    AggFuncTypeBase,
    AggFuncTypeFrame,
    ByT,
    CorrelationMethod,
    CovariantList,
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
    np_ndarray,
)

AggScalar: TypeAlias = str | Callable[..., Any]

class NamedAgg:
    def __init__(
        self,
        column: Hashable,
        aggfunc: Callable[..., Any] | str,
        *args: Any,
        **kwargs: Any,
    ) -> None: ...
    @property
    def column(self) -> str: ...
    @property
    def aggfunc(self) -> AggScalar: ...

class SeriesGroupBy(GroupBy[Series[S2]], Generic[S2, ByT]):
    @overload
    def aggregate(
        self,
        func: Callable[[Series], S3],
        /,
        *args: Any,
        engine: WindowingEngine = None,
        engine_kwargs: WindowingEngineKwargs = None,
        **kwargs: Any,
    ) -> Series[S3]: ...
    @overload
    def aggregate(
        self,
        func: list[AggFuncTypeBase[...]],
        /,
        *args: Any,
        engine: WindowingEngine = None,
        engine_kwargs: WindowingEngineKwargs = None,
        **kwargs: Any,
    ) -> DataFrame: ...
    @overload
    def aggregate(
        self,
        func: AggFuncTypeBase[...] | None = ...,
        /,
        *args: Any,
        engine: WindowingEngine = None,
        engine_kwargs: WindowingEngineKwargs = None,
        **kwargs: Any,
    ) -> Series: ...
    agg = aggregate
    @overload
    def transform(
        self,
        func: Callable[Concatenate[Series[S2], P], Series[S3]],
        /,
        *args: Any,
        engine: WindowingEngine = None,
        engine_kwargs: WindowingEngineKwargs = None,
        **kwargs: Any,
    ) -> Series[S3]: ...
    @overload
    def transform(
        self,
        func: Callable[Concatenate[Series, P], Any],
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> Series: ...
    @overload
    def transform(
        self, func: TransformReductionListType, *args: Any, **kwargs: Any
    ) -> Series: ...
    def filter(
        self,
        func: Callable[Concatenate[Series, P], Any] | str,
        dropna: bool = ...,
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> Series: ...
    def nunique(self, dropna: bool = ...) -> Series[int]: ...
    # describe delegates to super() method but here it has keyword-only parameters
    def describe(  # type: ignore[override] # pyright: ignore[reportIncompatibleMethodOverride] # pyrefly: ignore[bad-override] # ty: ignore[invalid-method-override]
        self,
        *,
        percentiles: Iterable[float] | None = ...,
        include: Literal["all"] | list[Dtype] | None = ...,
        exclude: list[Dtype] | None = ...,
    ) -> DataFrame: ...
    @overload
    def value_counts(
        self,
        normalize: Literal[False] = False,
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
        **kwargs: Any,
    ) -> Series[S2]: ...
    def skew(
        self,
        skipna: bool = True,
        numeric_only: bool = False,
        **kwargs: Any,
    ) -> Series: ...
    @property
    def plot(self) -> GroupByPlot[Self]: ...
    def nlargest(
        self, n: int = 5, keep: NsmallestNlargestKeep = "first"
    ) -> Series[S2]: ...
    def nsmallest(
        self, n: int = 5, keep: NsmallestNlargestKeep = "first"
    ) -> Series[S2]: ...
    def idxmin(self, skipna: bool = True) -> Series: ...
    def idxmax(self, skipna: bool = True) -> Series: ...
    def corr(
        self,
        other: Series,
        method: CorrelationMethod = ...,
        min_periods: int | None = ...,
    ) -> Series: ...
    def cov(
        self,
        other: Series,
        min_periods: int | None = None,
        ddof: int | None = 1,
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
        xlabelsize: float | str | None = None,
        xrot: float | None = None,
        ylabelsize: float | str | None = None,
        yrot: float | None = None,
        figsize: tuple[float, float] | None = None,
        bins: int | Sequence[int] = 10,
        backend: str | None = None,
        legend: bool = False,
        **kwargs: Any,
    ) -> Series: ...  # Series[Axes] but this is not allowed
    @property
    def dtype(self) -> Series: ...
    def unique(self) -> Series: ...
    # Overrides that provide more precise return types over the GroupBy class
    @final  # type: ignore[misc]
    def __iter__(  # pyright: ignore[reportIncompatibleMethodOverride] # pyrefly: ignore[bad-override] # ty: ignore[override-of-final-method]
        self,
    ) -> Iterator[tuple[ByT, Series[S2]]]: ...

_TT = TypeVar("_TT", bound=Literal[True, False])

class DataFrameGroupBy(GroupBy[DataFrame], Generic[ByT, _TT]):
    @overload
    def apply(
        self,
        func: Callable[
            Concatenate[DataFrame, P], Scalar | list[Any] | dict[Hashable, Any]
        ],
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> Series: ...
    @overload
    def apply(
        self,
        func: Callable[Concatenate[DataFrame, P], Series],
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> Series | DataFrame: ...
    @overload
    def apply(
        self,
        func: Callable[Concatenate[DataFrame, P], DataFrame] | str,
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> DataFrame: ...
    # error: overload 1 overlaps overload 2 because of different return types
    @overload
    def aggregate(  # pyright: ignore[reportOverlappingOverload]
        self,
        func: Literal["size"],
        *args: Any,
        engine: WindowingEngine = None,
        engine_kwargs: WindowingEngineKwargs = None,
        **kwargs: Any,
    ) -> Series: ...
    @overload
    def aggregate(
        self,
        func: AggFuncTypeFrame[..., Any] | None = ...,
        *args: Any,
        engine: WindowingEngine = None,
        engine_kwargs: WindowingEngineKwargs = None,
        **kwargs: Any,
    ) -> DataFrame: ...
    @overload
    def aggregate(
        self,
        func: AggFuncTypeFrame[..., Any] | None = None,
        /,
        **kwargs: Any,
    ) -> DataFrame: ...
    agg = aggregate
    @overload
    def transform(
        self,
        func: Callable[Concatenate[DataFrame, P], DataFrame],
        *args: Any,
        engine: WindowingEngine = None,
        engine_kwargs: WindowingEngineKwargs = None,
        **kwargs: Any,
    ) -> DataFrame: ...
    @overload
    def transform(
        self,
        func: Callable[Concatenate[DataFrame, P], Any],
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> DataFrame: ...
    @overload
    def transform(
        self, func: TransformReductionListType, *args: Any, **kwargs: Any
    ) -> DataFrame: ...
    def filter(
        self,
        func: Callable[Concatenate[DataFrame, P], Any],
        dropna: bool = ...,
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> DataFrame: ...
    @overload
    def __getitem__(self, key: Scalar) -> SeriesGroupBy[Any, ByT]: ...
    @overload
    def __getitem__(  # pyright: ignore[reportIncompatibleMethodOverride]
        self, key: CovariantList[Hashable]
    ) -> DataFrameGroupBy[ByT, _TT]: ...
    def nunique(self, dropna: bool = True) -> DataFrame: ...
    def idxmax(
        self,
        skipna: bool = True,
        numeric_only: bool = False,
    ) -> DataFrame: ...
    def idxmin(
        self,
        skipna: bool = True,
        numeric_only: bool = False,
    ) -> DataFrame: ...
    @overload
    def boxplot(
        self,
        subplots: Literal[True] = True,
        column: IndexLabel | None = None,
        fontsize: float | str | None = None,
        rot: float = 0,
        grid: bool = True,
        ax: PlotAxes | None = None,
        figsize: tuple[float, float] | None = None,
        layout: tuple[int, int] | None = None,
        sharex: bool = False,
        sharey: bool = True,
        backend: str | None = None,
        **kwargs: Any,
    ) -> Series: ...  # Series[PlotAxes] but this is not allowed
    @overload
    def boxplot(
        self,
        subplots: Literal[False],
        column: IndexLabel | None = None,
        fontsize: float | str | None = None,
        rot: float = 0,
        grid: bool = True,
        ax: PlotAxes | None = None,
        figsize: tuple[float, float] | None = None,
        layout: tuple[int, int] | None = None,
        sharex: bool = False,
        sharey: bool = True,
        backend: str | None = None,
        **kwargs: Any,
    ) -> PlotAxes: ...
    @overload
    def boxplot(
        self,
        subplots: bool,
        column: IndexLabel | None = None,
        fontsize: float | str | None = None,
        rot: float = 0,
        grid: bool = True,
        ax: PlotAxes | None = None,
        figsize: tuple[float, float] | None = None,
        layout: tuple[int, int] | None = None,
        sharex: bool = False,
        sharey: bool = True,
        backend: str | None = None,
        **kwargs: Any,
    ) -> PlotAxes | Series: ...  # Series[PlotAxes]
    @overload
    def value_counts(
        self: DataFrameGroupBy[ByT, Literal[True]],
        subset: ListLike | None = ...,
        normalize: Literal[False] = False,
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
        normalize: Literal[False] = False,
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
    def take(self, indices: TakeIndexer, **kwargs: Any) -> DataFrame: ...
    @overload
    def skew(
        self,
        skipna: bool = ...,
        numeric_only: bool = ...,
        *,
        level: Level,
        **kwargs: Any,
    ) -> DataFrame: ...
    @overload
    def skew(
        self,
        skipna: bool = ...,
        numeric_only: bool = ...,
        *,
        level: None = None,
        **kwargs: Any,
    ) -> Series: ...
    @property
    def plot(self) -> GroupByPlot[Self]: ...
    def corr(
        self,
        method: str | Callable[[np_ndarray, np_ndarray], float] = ...,
        min_periods: int = ...,
        numeric_only: bool = False,
    ) -> DataFrame: ...
    def cov(
        self,
        min_periods: int | None = ...,
        ddof: int | None = 1,
        numeric_only: bool = False,
    ) -> DataFrame: ...
    def hist(
        self,
        column: IndexLabel | None = None,
        by: IndexLabel | None = None,
        grid: bool = True,
        xlabelsize: float | str | None = None,
        xrot: float | None = None,
        ylabelsize: float | str | None = None,
        yrot: float | None = None,
        ax: PlotAxes | None = None,
        sharex: bool = False,
        sharey: bool = False,
        figsize: tuple[float, float] | None = None,
        layout: tuple[int, int] | None = None,
        bins: int | Sequence[int] = 10,
        backend: str | None = None,
        legend: bool = False,
        **kwargs: Any,
    ) -> Series: ...  # Series[Axes] but this is not allowed
    @property
    def dtypes(self) -> Never: ...
    def __getattr__(self, attr: str) -> SeriesGroupBy[Any, ByT]: ...
    # Overrides that provide more precise return types over the GroupBy class
    @final  # type: ignore[misc]
    def __iter__(  # pyright: ignore[reportIncompatibleMethodOverride] # ty: ignore[override-of-final-method] # pyrefly: ignore[bad-override]
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
