from collections.abc import (
    Callable,
    Hashable,
    Mapping,
)
from typing import (
    Literal,
    overload,
)

import numpy as np
from pandas import (
    DataFrame,
    DatetimeIndex,
    Index,
    PeriodIndex,
    Series,
    Timedelta,
    TimedeltaIndex,
)
from pandas.core.groupby.generic import SeriesGroupBy
from pandas.core.groupby.groupby import (
    BaseGroupBy,
    GroupBy,
)
from pandas.core.groupby.grouper import Grouper
from pandas.core.groupby.ops import BinGrouper
from typing_extensions import (
    Self,
    TypeAlias,
)

from pandas._libs.lib import NoDefault
from pandas._typing import (
    S1,
    Axis,
    FillnaOptions,
    Frequency,
    IndexLabel,
    InterpolateOptions,
    NDFrameT,
    Scalar,
    TimedeltaConvertibleTypes,
    TimeGrouperOrigin,
    TimestampConvention,
    TimestampConvertibleTypes,
    npt,
)

_FrameGroupByFunc: TypeAlias = (
    Callable[[DataFrame], Scalar]
    | Callable[[DataFrame], Series]
    | Callable[[DataFrame], DataFrame]
    | np.ufunc
)
_FrameGroupByFuncTypes: TypeAlias = (
    _FrameGroupByFunc | str | list[_FrameGroupByFunc | str]
)
_FrameGroupByFuncArgs: TypeAlias = (
    _FrameGroupByFuncTypes | Mapping[Hashable, _FrameGroupByFuncTypes]
)

_SeriesGroupByFunc: TypeAlias = (
    Callable[[Series], Scalar] | Callable[[Series], Series] | np.ufunc
)
_SeriesGroupByFuncTypes: TypeAlias = (
    _SeriesGroupByFunc | str | list[_SeriesGroupByFunc | str]
)
_SeriesGroupByFuncArgs: TypeAlias = (
    _SeriesGroupByFuncTypes | Mapping[Hashable, _SeriesGroupByFunc | str]
)

class Resampler(BaseGroupBy[NDFrameT]):
    grouper: BinGrouper  # pyright: ignore[reportIncompatibleVariableOverride]  # variance incompatibility
    binner: DatetimeIndex | TimedeltaIndex | PeriodIndex
    exclusions: frozenset[Hashable]
    ax: Index
    def __init__(
        self,
        obj: NDFrameT,
        timegrouper: TimeGrouper,
        axis: Axis = 0,
        kind: str | None = None,
        *,
        gpr_index: Index,
        group_keys: bool = False,
        selection: IndexLabel | None = None,
    ) -> None: ...
    def __getattr__(self, attr: str) -> SeriesGroupBy: ...
    @overload
    def aggregate(
        self: Resampler[DataFrame],
        func: _FrameGroupByFuncArgs | None = None,
        *args,
        **kwargs,
    ) -> DataFrame: ...
    @overload
    def aggregate(
        self: Resampler[Series],
        func: _SeriesGroupByFuncArgs | None = None,
        *args,
        **kwargs,
    ) -> Series | DataFrame: ...
    agg = aggregate
    apply = aggregate
    @overload
    def transform(
        self: Resampler[Series], arg: Callable[[Series], Series[S1]], *args, **kwargs
    ) -> Series[S1]: ...
    @overload
    def transform(
        self: Resampler[DataFrame], arg: Callable[[Series], Series[S1]], *args, **kwargs
    ) -> DataFrame: ...
    def ffill(self, limit: int | None = None) -> NDFrameT: ...
    def nearest(self, limit: int | None = None) -> NDFrameT: ...
    def bfill(self, limit: int | None = None) -> NDFrameT: ...
    def fillna(
        self, method: Literal[FillnaOptions, "nearest"], limit: int | None = None
    ) -> NDFrameT: ...
    @overload
    def interpolate(
        self,
        method: InterpolateOptions = "linear",
        *,
        axis: Axis = 0,
        limit: int | None = None,
        inplace: Literal[True],
        limit_direction: Literal["forward", "backward", "both"] = "forward",
        limit_area: Literal["inside", "outside"] | None = None,
        downcast: Literal["infer"] | None | NoDefault = ...,
        **kwargs,
    ) -> None: ...
    @overload
    def interpolate(
        self,
        method: InterpolateOptions = "linear",
        *,
        axis: Axis = 0,
        limit: int | None = None,
        inplace: Literal[False] = False,
        limit_direction: Literal["forward", "backward", "both"] = "forward",
        limit_area: Literal["inside", "outside"] | None = None,
        downcast: Literal["infer"] | None | NoDefault = ...,
        **kwargs,
    ) -> NDFrameT: ...
    def asfreq(self, fill_value: Scalar | None = None) -> NDFrameT: ...
    def sum(
        self, numeric_only: bool = False, min_count: int = 0, *args, **kwargs
    ) -> NDFrameT: ...
    def prod(
        self, numeric_only: bool = False, min_count: int = 0, *args, **kwargs
    ) -> NDFrameT: ...
    def min(
        self, numeric_only: bool = False, min_count: int = 0, *args, **kwargs
    ) -> NDFrameT: ...
    def max(
        self, numeric_only: bool = False, min_count: int = 0, *args, **kwargs
    ) -> NDFrameT: ...
    def first(
        self, numeric_only: bool = False, min_count: int = 0, *args, **kwargs
    ) -> NDFrameT: ...
    def last(
        self, numeric_only: bool = False, min_count: int = 0, *args, **kwargs
    ) -> NDFrameT: ...
    def median(self, numeric_only: bool = False, *args, **kwargs) -> NDFrameT: ...
    def mean(self, numeric_only: bool = False, *args, **kwargs) -> NDFrameT: ...
    def std(
        self, ddof: int = 1, numeric_only: bool = False, *args, **kwargs
    ) -> NDFrameT: ...
    def var(
        self, ddof: int = 1, numeric_only: bool = False, *args, **kwargs
    ) -> NDFrameT: ...
    def sem(
        self, ddof: int = 1, numeric_only: bool = False, *args, **kwargs
    ) -> NDFrameT: ...
    def ohlc(self, *args, **kwargs) -> DataFrame: ...
    @overload
    def nunique(self: Resampler[Series], *args, **kwargs) -> Series[int]: ...
    @overload
    def nunique(self: Resampler[DataFrame], *args, **kwargs) -> DataFrame: ...
    def size(self) -> Series[int]: ...
    @overload
    def count(self: Resampler[Series]) -> Series[int]: ...
    @overload
    def count(self: Resampler[DataFrame]) -> DataFrame: ...
    def quantile(
        self,
        q: float | list[float] | npt.NDArray[np.float_] | Series[float] = 0.5,
        **kwargs,
    ) -> NDFrameT: ...

# We lie about inheriting from Resampler because at runtime inherits all Resampler
# attributes via setattr
class _GroupByMixin(Resampler[NDFrameT]):
    k: str | list[str] | None
    def __init__(
        self,
        *,
        parent: Resampler,
        groupby: GroupBy,
        key=None,
        selection: IndexLabel | None = None,
    ) -> None: ...
    def __getitem__(self, key) -> Self: ...  # type: ignore[override] # pyright: ignore[reportIncompatibleMethodOverride]

class DatetimeIndexResampler(Resampler[NDFrameT]): ...

class DatetimeIndexResamplerGroupby(
    _GroupByMixin[NDFrameT], DatetimeIndexResampler[NDFrameT]
):
    def __getattr__(self, attr: str) -> Self: ...  # type: ignore[override] # pyright: ignore[reportIncompatibleMethodOverride]

class PeriodIndexResampler(DatetimeIndexResampler[NDFrameT]): ...

class PeriodIndexResamplerGroupby(
    _GroupByMixin[NDFrameT], PeriodIndexResampler[NDFrameT]
):
    def __getattr__(self, attr: str) -> Self: ...  # type: ignore[override] # pyright: ignore[reportIncompatibleMethodOverride]

class TimedeltaIndexResampler(DatetimeIndexResampler[NDFrameT]): ...

class TimedeltaIndexResamplerGroupby(
    _GroupByMixin[NDFrameT], TimedeltaIndexResampler[NDFrameT]
):
    def __getattr__(self, attr: str) -> Self: ...  # type: ignore[override] # pyright: ignore[reportIncompatibleMethodOverride]

_ResamplerGroupBy: TypeAlias = (
    DatetimeIndexResamplerGroupby[NDFrameT]
    | PeriodIndexResamplerGroupby[NDFrameT]
    | TimedeltaIndexResamplerGroupby[NDFrameT]
)

def get_resampler(
    obj: NDFrameT,
    kind: Literal["period", "timestamp", "timedelta"] | None = None,
    *,
    # Keyword argumentsmust be kept roughly inline with TimeGrouper.__init__ below
    freq: Frequency = "Min",
    closed: Literal["left", "right"] | None = None,
    label: Literal["left", "right"] | None = None,
    how: str = "mean",
    axis: Axis = 0,
    fill_method: str | None = None,
    limit: int | None = None,
    convention: TimestampConvention | None = None,
    origin: TimeGrouperOrigin | TimestampConvertibleTypes = "start_day",
    offset: TimedeltaConvertibleTypes | None = None,
    group_keys: bool = False,
    **kwds,
) -> DatetimeIndexResampler[NDFrameT]: ...
def get_resampler_for_grouping(
    groupby: GroupBy[NDFrameT],
    rule: Frequency,
    how: str | None = None,
    fill_method: str | None = None,
    limit: int | None = None,
    kind: Literal["period", "timestamp", "timedelta"] | None = None,
    on: Hashable | None = None,
    *,
    # Keyword argumentsmust be kept roughly inline with TimeGrouper.__init__ below
    closed: Literal["left", "right"] | None = None,
    label: Literal["left", "right"] | None = None,
    axis: Axis = 0,
    convention: TimestampConvention | None = None,
    origin: TimeGrouperOrigin | TimestampConvertibleTypes = "start_day",
    offset: TimedeltaConvertibleTypes | None = None,
    group_keys: bool = False,
    **kwargs,
) -> _ResamplerGroupBy[NDFrameT]: ...

class TimeGrouper(Grouper):
    closed: Literal["left", "right"]
    label: Literal["left", "right"]
    kind: str | None
    convention: TimestampConvention
    how: str
    fill_method: str | None
    limit: int | None
    group_keys: bool
    origin: TimeGrouperOrigin
    offset: Timedelta | None

    def __init__(
        self,
        freq: Frequency = "Min",
        closed: Literal["left", "right"] | None = None,
        label: Literal["left", "right"] | None = None,
        how: str = "mean",
        axis: Axis = 0,
        fill_method: str | None = None,
        limit: int | None = None,
        kind: Literal["period", "timestamp", "timedelta"] | None = None,
        convention: TimestampConvention | None = None,
        origin: TimeGrouperOrigin | TimestampConvertibleTypes = "start_day",
        offset: TimedeltaConvertibleTypes | None = None,
        group_keys: bool = False,
        **kwargs,
    ) -> None: ...

def asfreq(
    obj: NDFrameT,
    freq: Frequency,
    method: Literal[FillnaOptions, "nearest"] | None = None,
    how: str | None = None,
    normalize: bool = False,
    fill_value: Scalar | None = None,
) -> NDFrameT: ...
