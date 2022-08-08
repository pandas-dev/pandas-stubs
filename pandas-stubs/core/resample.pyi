from typing import Generic

from pandas import DataFrame
from pandas.core.groupby.base import GroupByMixin
from pandas.core.groupby.groupby import BaseGroupBy
from pandas.core.groupby.grouper import Grouper

from pandas._typing import NDFrameT

class Resampler(BaseGroupBy, Generic[NDFrameT]):
    def __init__(self, obj, groupby, axis: int = ..., kind=..., **kwargs) -> None: ...
    def __getattr__(self, attr: str): ...
    def __iter__(self): ...
    @property
    def obj(self): ...
    @property
    def ax(self): ...
    def pipe(self, func, *args, **kwargs): ...
    def aggregate(self, func=..., *args, **kwargs): ...
    agg = aggregate
    def transform(self, arg, *args, **kwargs): ...
    def pad(self, limit=...): ...
    def nearest(self, limit=...): ...
    def backfill(self, limit=...): ...
    bfill = backfill
    def fillna(self, method, limit=...): ...
    def interpolate(
        self,
        method: str = ...,
        axis: int = ...,
        limit=...,
        inplace: bool = ...,
        limit_direction: str = ...,
        limit_area=...,
        downcast=...,
        **kwargs,
    ): ...
    def asfreq(self, fill_value=...): ...
    def std(self, ddof: int = ..., *args, **kwargs) -> NDFrameT: ...
    def var(self, ddof: int = ..., *args, **kwargs) -> NDFrameT: ...
    def size(self): ...
    def count(self): ...
    def quantile(self, q: float = ..., **kwargs) -> NDFrameT: ...
    def sum(self, _method=..., min_count: int = ..., *args, **kwargs) -> NDFrameT: ...
    def prod(self, _method=..., min_count: int = ..., *args, **kwargs) -> NDFrameT: ...
    def min(self, _method=..., min_count: int = ..., *args, **kwargs) -> NDFrameT: ...
    def max(self, _method=..., min_count: int = ..., *args, **kwargs) -> NDFrameT: ...
    def first(self, _method=..., min_count: int = ..., *args, **kwargs) -> NDFrameT: ...
    def last(self, _method=..., min_count: int = ..., *args, **kwargs) -> NDFrameT: ...
    def mean(self, _method=..., *args, **kwargs) -> NDFrameT: ...
    def sem(self, _method=..., *args, **kwargs) -> NDFrameT: ...
    def median(self, _method=..., *args, **kwargs) -> NDFrameT: ...
    def ohlc(self, _method=..., *args, **kwargs) -> DataFrame: ...

def get_resampler_for_grouping(
    groupby, rule, how=..., fill_method=..., limit=..., kind=..., **kwargs
): ...
def asfreq(obj, freq, method=..., how=..., normalize: bool = ..., fill_value=...): ...
