from collections.abc import (
    Callable,
    Hashable,
    Iterable,
    Iterator,
    Mapping,
    Sequence,
)
import datetime as dt
from typing import (
    Any,
    Generic,
    Literal,
    TypeVar,
    final,
    overload,
)

import numpy as np
from pandas.core.base import (
    PandasObject,
    SelectionMixin,
)
from pandas.core.frame import DataFrame
from pandas.core.groupby import (
    generic,
    ops,
)
from pandas.core.groupby.indexing import (
    GroupByIndexingMixin,
    GroupByNthSelector,
)
from pandas.core.indexers import BaseIndexer
from pandas.core.indexes.api import Index
from pandas.core.resample import _ResamplerGroupBy
from pandas.core.series import Series
from pandas.core.window import (
    ExpandingGroupby,
    ExponentialMovingWindowGroupby,
    RollingGroupby,
)
from typing_extensions import (
    Concatenate,
    Self,
    TypeAlias,
)

from pandas._libs.lib import NoDefault
from pandas._libs.tslibs import BaseOffset
from pandas._typing import (
    S1,
    AnyArrayLike,
    Axis,
    AxisInt,
    CalculationMethod,
    Dtype,
    FillnaOptions,
    Frequency,
    IndexLabel,
    IntervalClosedType,
    KeysArgType,
    MaskType,
    NDFrameT,
    P,
    RandomState,
    Scalar,
    T,
    TimedeltaConvertibleTypes,
    TimeGrouperOrigin,
    TimestampConvention,
    TimestampConvertibleTypes,
    WindowingEngine,
    WindowingEngineKwargs,
    npt,
)

from pandas.plotting import PlotAccessor

_GroupByT = TypeVar("_GroupByT", bound=GroupBy)

_KeysArgType: TypeAlias = (
    Hashable
    | list[Hashable]
    | Callable[[Hashable], Hashable]
    | list[Callable[[Hashable], Hashable]]
    | Mapping[Hashable, Hashable]
)

# GroupByPlot does not really inherit from PlotAccessor but it delegates
# to it using __call__ and __getattr__. We lie here to avoid repeating the
# whole stub of PlotAccessor
@final
class GroupByPlot(PandasObject, PlotAccessor, Generic[_GroupByT]):
    def __init__(self, groupby: _GroupByT) -> None: ...
    # The following methods are inherited from the fake parent class PlotAccessor
    # def __call__(self, *args, **kwargs): ...
    # def __getattr__(self, name: str): ...

class BaseGroupBy(PandasObject, SelectionMixin[NDFrameT], GroupByIndexingMixin):
    axis: AxisInt
    grouper: ops.BaseGrouper
    keys: _KeysArgType | None
    level: IndexLabel | None
    group_keys: bool
    @final
    def __len__(self) -> int: ...
    @final
    def __repr__(self) -> str: ...  # noqa: PYI029 __repr__ here is final
    @final
    @property
    def groups(self) -> dict[Hashable, Index]: ...
    @final
    @property
    def ngroups(self) -> int: ...
    @final
    @property
    def indices(self) -> dict[Hashable, Index | npt.NDArray[np.int_] | list[int]]: ...
    @overload
    def pipe(
        self,
        func: Callable[Concatenate[Self, P], T],
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> T: ...
    @overload
    def pipe(
        self,
        func: tuple[Callable[..., T], str],
        *args: Any,
        **kwargs: Any,
    ) -> T: ...
    @final
    def get_group(self, name, obj: NDFrameT | None = ...) -> NDFrameT: ...
    @final
    def __iter__(self) -> Iterator[tuple[Hashable, NDFrameT]]: ...
    @overload
    def __getitem__(self: BaseGroupBy[DataFrame], key: Scalar | Hashable | tuple[Hashable, ...]) -> generic.SeriesGroupBy: ...  # type: ignore[overload-overlap]
    @overload
    def __getitem__(
        self: BaseGroupBy[DataFrame], key: Iterable[Hashable] | slice
    ) -> generic.DataFrameGroupBy: ...
    @overload
    def __getitem__(
        self: BaseGroupBy[Series[S1]],
        idx: (
            list[str]
            | Index
            | Series[S1]
            | slice
            | MaskType
            | tuple[Hashable | slice, ...]
        ),
    ) -> generic.SeriesGroupBy: ...
    @overload
    def __getitem__(self: BaseGroupBy[Series[S1]], idx: Scalar) -> S1: ...

class GroupBy(BaseGroupBy[NDFrameT]):
    as_index: bool
    sort: bool
    observed: bool
    @final
    def __init__(
        self,
        obj: NDFrameT,
        keys: _KeysArgType | None = None,
        axis: Axis = 0,
        level: IndexLabel | None = None,
        grouper: ops.BaseGrouper | None = None,
        exclusions: frozenset[Hashable] | None = None,
        selection: IndexLabel | None = None,
        as_index: bool = True,
        sort: bool = True,
        group_keys: bool = True,
        observed: bool | NoDefault = ...,
        dropna: bool = True,
    ) -> None: ...
    def __getattr__(self, attr: str) -> Any: ...
    def apply(self, func: Callable | str, *args, **kwargs) -> NDFrameT: ...
    @final
    @overload
    def any(self: GroupBy[Series], skipna: bool = True) -> Series[bool]: ...
    @overload
    def any(self: GroupBy[DataFrame], skipna: bool = True) -> DataFrame: ...
    @final
    @overload
    def all(self: GroupBy[Series], skipna: bool = True) -> Series[bool]: ...
    @overload
    def all(self: GroupBy[DataFrame], skipna: bool = True) -> DataFrame: ...
    @final
    def count(self) -> NDFrameT: ...
    @final
    def mean(
        self,
        numeric_only: bool = False,
        engine: WindowingEngine = None,
        engine_kwargs: WindowingEngineKwargs = None,
    ) -> NDFrameT: ...
    @final
    def median(self, numeric_only: bool = False) -> NDFrameT: ...
    @final
    @overload
    def std(
        self: GroupBy[Series],
        ddof: int = 1,
        engine: WindowingEngine = None,
        engine_kwargs: WindowingEngineKwargs = None,
        numeric_only: bool = False,
    ) -> Series[float]: ...
    @overload
    def std(
        self: GroupBy[DataFrame],
        ddof: int = 1,
        engine: WindowingEngine = None,
        engine_kwargs: WindowingEngineKwargs = None,
        numeric_only: bool = False,
    ) -> DataFrame: ...
    @final
    @overload
    def var(
        self: GroupBy[Series],
        ddof: int = 1,
        engine: WindowingEngine = None,
        engine_kwargs: WindowingEngineKwargs = None,
        numeric_only: bool = False,
    ) -> Series[float]: ...
    @overload
    def var(
        self: GroupBy[DataFrame],
        ddof: int = 1,
        engine: WindowingEngine = None,
        engine_kwargs: WindowingEngineKwargs = None,
        numeric_only: bool = False,
    ) -> DataFrame: ...
    @final
    @overload
    def sem(
        self: GroupBy[Series], ddof: int = 1, numeric_only: bool = False
    ) -> Series[float]: ...
    @overload
    def sem(
        self: GroupBy[DataFrame], ddof: int = 1, numeric_only: bool = False
    ) -> DataFrame: ...
    @final
    @overload
    def size(self: GroupBy[Series]) -> Series[int]: ...
    @overload  # return type depends on `as_index` for dataframe groupby
    def size(self: GroupBy[DataFrame]) -> DataFrame | Series[int]: ...
    @final
    def sum(
        self,
        numeric_only: bool = False,
        min_count: int = 0,
        engine: WindowingEngine = None,
        engine_kwargs: WindowingEngineKwargs = None,
    ) -> NDFrameT: ...
    @final
    def prod(self, numeric_only: bool = False, min_count: int = 0) -> NDFrameT: ...
    @final
    def min(
        self,
        numeric_only: bool = False,
        min_count: int = -1,
        engine: WindowingEngine = None,
        engine_kwargs: WindowingEngineKwargs = None,
    ) -> NDFrameT: ...
    @final
    def max(
        self,
        numeric_only: bool = False,
        min_count: int = -1,
        engine: WindowingEngine = None,
        engine_kwargs: WindowingEngineKwargs = None,
    ) -> NDFrameT: ...
    @final
    def first(self, numeric_only: bool = False, min_count: int = -1) -> NDFrameT: ...
    @final
    def last(self, numeric_only: bool = False, min_count: int = -1) -> NDFrameT: ...
    @final
    def ohlc(self) -> DataFrame: ...
    def describe(
        self,
        percentiles: Iterable[float] | None = None,
        include: Literal["all"] | list[Dtype] | None = None,
        exclude: list[Dtype] | None = None,
    ) -> DataFrame: ...
    @final
    def resample(
        self,
        rule: Frequency,
        # Arguments must be kept roughly inline with pandas.core.resample.get_resampler_for_grouping
        how: str | None = None,
        fill_method: str | None = None,
        limit: int | None = None,
        kind: str | None = None,
        on: Hashable | None = None,
        *,
        closed: Literal["left", "right"] | None = None,
        label: Literal["left", "right"] | None = None,
        axis: Axis = 0,
        convention: TimestampConvention | None = None,
        origin: TimeGrouperOrigin | TimestampConvertibleTypes = "start_day",
        offset: TimedeltaConvertibleTypes | None = None,
        group_keys: bool = False,
        **kwargs,
    ) -> _ResamplerGroupBy[NDFrameT]: ...
    @final
    def rolling(
        self,
        # Arguments must be kept roughly inline with pandas.core.window.RollingGroupby
        window: int | dt.timedelta | str | BaseOffset | BaseIndexer | None = None,
        min_periods: int | None = None,
        center: bool | None = False,
        win_type: str | None = None,
        axis: Axis = 0,
        on: str | Index | None = None,
        closed: IntervalClosedType | None = None,
        step: int | None = None,
        method: str = "single",
        *,
        selection: IndexLabel | None = None,
    ) -> RollingGroupby[NDFrameT]: ...
    @final
    def expanding(
        self,
        # Arguments must be kept roughly inline with pandas.core.window.ExpandingGroupby
        min_periods: int = 1,
        axis: Axis = 0,
        method: str = "single",
        selection: IndexLabel | None = None,
    ) -> ExpandingGroupby[NDFrameT]: ...
    @final
    def ewm(
        self,
        # Arguments must be kept roughly inline with pandas.core.window.ExponentialMovingWindowGroupby
        com: float | None = None,
        span: float | None = None,
        halflife: TimedeltaConvertibleTypes | None = None,
        alpha: float | None = None,
        min_periods: int | None = 0,
        adjust: bool = True,
        ignore_na: bool = False,
        axis: Axis = 0,
        times: str | np.ndarray | Series | np.timedelta64 | None = None,
        method: CalculationMethod = "single",
        *,
        selection: IndexLabel | None = None,
    ) -> ExponentialMovingWindowGroupby[NDFrameT]: ...
    @final
    def ffill(self, limit: int | None = None) -> NDFrameT: ...
    @final
    def bfill(self, limit: int | None = None) -> NDFrameT: ...
    @final
    @property
    def nth(self) -> GroupByNthSelector[Self]: ...
    @final
    def quantile(
        self,
        q: float | AnyArrayLike = 0.5,
        interpolation: str = "linear",
        numeric_only: bool = False,
    ) -> NDFrameT: ...
    @final
    def ngroup(self, ascending: bool = True) -> Series[int]: ...
    @final
    def cumcount(self, ascending: bool = True) -> Series[int]: ...
    @final
    def rank(
        self,
        method: str = "average",
        ascending: bool = True,
        na_option: str = "keep",
        pct: bool = False,
        axis: AxisInt | NoDefault = ...,
    ) -> NDFrameT: ...
    @final
    def cumprod(self, axis: Axis | NoDefault = ..., *args, **kwargs) -> NDFrameT: ...
    @final
    def cumsum(self, axis: Axis | NoDefault = ..., *args, **kwargs) -> NDFrameT: ...
    @final
    def cummin(
        self, axis: AxisInt | NoDefault = ..., numeric_only: bool = False, **kwargs
    ) -> NDFrameT: ...
    @final
    def cummax(
        self, axis: AxisInt | NoDefault = ..., numeric_only: bool = False, **kwargs
    ) -> NDFrameT: ...
    @final
    def shift(
        self,
        periods: int | Sequence[int] = 1,
        freq: Frequency | None = None,
        axis: Axis | NoDefault = ...,
        fill_value=...,
        suffix: str | None = None,
    ) -> NDFrameT: ...
    @final
    def diff(self, periods: int = 1, axis: AxisInt | NoDefault = ...) -> NDFrameT: ...
    @final
    def pct_change(
        self,
        periods: int = 1,
        fill_method: FillnaOptions | None | NoDefault = ...,
        limit: int | None | NoDefault = ...,
        freq=None,
        axis: Axis | NoDefault = ...,
    ) -> NDFrameT: ...
    @final
    def head(self, n: int = 5) -> NDFrameT: ...
    @final
    def tail(self, n: int = 5) -> NDFrameT: ...
    @final
    def sample(
        self,
        n: int | None = None,
        frac: float | None = None,
        replace: bool = False,
        weights: Sequence | Series | None = None,
        random_state: RandomState | None = None,
    ) -> NDFrameT: ...

@overload
def get_groupby(
    obj: Series,
    by: KeysArgType | None = None,
    axis: int = 0,
    grouper: ops.BaseGrouper | None = None,
    group_keys: bool = True,
) -> generic.SeriesGroupBy: ...
@overload
def get_groupby(
    obj: DataFrame,
    by: KeysArgType | None = None,
    axis: int = 0,
    grouper: ops.BaseGrouper | None = None,
    group_keys: bool = True,
) -> generic.DataFrameGroupBy: ...
