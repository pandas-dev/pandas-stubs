from typing import (
    Callable,
    Generator,
    Generic,
    Hashable,
    Literal,
    overload,
)

import numpy as np
from pandas import (
    DataFrame,
    Series,
)
from pandas.core.groupby.groupby import BaseGroupBy

from pandas._typing import (
    Axis,
    NDFrameT,
    Scalar,
    T,
    npt,
)

_GroupByFunc = Callable[[DataFrame | Series], Scalar]
_GroupByFuncTypes = _GroupByFunc | str | list[_GroupByFunc | str]
_GroupByFuncArgs = _GroupByFuncTypes | dict[Hashable, _GroupByFuncTypes]
_Interpolation = Literal[
    "linear",
    "time",
    "index",
    "pad",
    "nearest",
    "zero",
    "slinear",
    "quadratic",
    "cubic",
    "spline",
    "barycentric",
    "polynomial",
    "krogh",
    "piecewise_polynomial",
    "spline",
    "pchip",
    "akima",
    "cubicspline",
    "from_derivatives",
]

class Resampler(BaseGroupBy, Generic[NDFrameT]):
    def __getattr__(self, attr: str): ...
    def __iter__(self) -> Generator[tuple[Hashable, NDFrameT], None, None]: ...
    @property
    def obj(self) -> NDFrameT: ...
    @property
    def ax(self): ...
    def pipe(
        self,
        func: Callable[..., T] | tuple[Callable[..., T], str],
        *args,
        **kwargs,
    ) -> T: ...
    def aggregate(
        self, func: _GroupByFuncArgs | None = ..., *args, **kwargs
    ) -> Scalar | Series | DataFrame: ...
    agg = aggregate
    apply = aggregate
    def transform(
        self, arg: Callable[[NDFrameT], NDFrameT], *args, **kwargs
    ) -> NDFrameT: ...
    def ffill(self, limit: int | None = ...) -> NDFrameT: ...
    def pad(self, limit: int | None = ...) -> NDFrameT: ...
    def nearest(self, limit: int | None = ...) -> NDFrameT: ...
    def backfill(self, limit: int | None = ...) -> NDFrameT: ...
    bfill = backfill
    def fillna(
        self,
        method: Literal["pad", "backfill", "ffill", "bfill", "nearest"],
        limit: int | None = ...,
    ) -> NDFrameT: ...
    @overload
    def interpolate(
        self,
        method: _Interpolation = ...,
        axis: Axis = ...,
        limit: int | None = ...,
        *,
        inplace: Literal[True],
        limit_direction: Literal["forward", "backward", "both"] = ...,
        limit_area: Literal["inside", "outside"] | None = ...,
        downcast: Literal["infer"] | None = ...,
        **kwargs,
    ) -> None: ...
    @overload
    def interpolate(
        self,
        method: _Interpolation = ...,
        axis: Axis = ...,
        limit: int | None = ...,
        inplace: Literal[False] = ...,
        limit_direction: Literal["forward", "backward", "both"] = ...,
        limit_area: Literal["inside", "outside"] | None = ...,
        downcast: Literal["infer"] | None = ...,
        **kwargs,
    ) -> NDFrameT: ...
    def asfreq(self, fill_value: Scalar | None = ...) -> NDFrameT: ...
    def std(self, ddof: int = ..., *args, **kwargs) -> NDFrameT: ...
    def var(self, ddof: int = ..., *args, **kwargs) -> NDFrameT: ...
    def size(self) -> NDFrameT: ...
    def count(self) -> NDFrameT: ...
    def quantile(
        self,
        q: float | list[float] | npt.NDArray[np.float_] | Series[float] = ...,
        **kwargs,
    ) -> NDFrameT: ...
    def sum(
        self, _method: Literal["sum"] = ..., min_count: int = ..., *args, **kwargs
    ) -> NDFrameT: ...
    def prod(
        self, _method: Literal["prod"] = ..., min_count: int = ..., *args, **kwargs
    ) -> NDFrameT: ...
    def min(
        self, _method: Literal["min"] = ..., min_count: int = ..., *args, **kwargs
    ) -> NDFrameT: ...
    def max(
        self, _method: Literal["max"] = ..., min_count: int = ..., *args, **kwargs
    ) -> NDFrameT: ...
    def first(
        self, _method: Literal["first"] = ..., min_count: int = ..., *args, **kwargs
    ) -> NDFrameT: ...
    def last(
        self, _method: Literal["last"] = ..., min_count: int = ..., *args, **kwargs
    ) -> NDFrameT: ...
    def mean(self, _method: Literal["mean"] = ..., *args, **kwargs) -> NDFrameT: ...
    def sem(self, _method: Literal["sem"] = ..., *args, **kwargs) -> NDFrameT: ...
    def median(self, _method: Literal["median"] = ..., *args, **kwargs) -> NDFrameT: ...
    def ohlc(self, _method: Literal["ohlc"] = ..., *args, **kwargs) -> DataFrame: ...
    def nunique(self, _method: Literal["first"] = ..., *args, **kwargs) -> NDFrameT: ...
