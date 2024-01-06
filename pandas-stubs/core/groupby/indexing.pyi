from typing import (
    Any,
    Generic,
    Literal,
    TypeVar,
)

from pandas import (
    DataFrame,
    Series,
)
from pandas.core.groupby import groupby

from pandas._typing import PositionalIndexer

_GroupByT = TypeVar("_GroupByT", bound=groupby.GroupBy[Any])

class GroupByIndexingMixin: ...

class GroupByPositionalSelector:
    groupby_object: groupby.GroupBy
    def __init__(self, groupby_object: groupby.GroupBy) -> None: ...
    def __getitem__(self, arg: PositionalIndexer | tuple) -> DataFrame | Series: ...

class GroupByNthSelector(Generic[_GroupByT]):
    groupby_object: _GroupByT

    def __init__(self, groupby_object: _GroupByT) -> None: ...
    def __call__(
        self,
        n: PositionalIndexer | tuple,
        dropna: Literal["any", "all", None] = None,
    ) -> DataFrame | Series: ...
    def __getitem__(self, n: PositionalIndexer | tuple) -> DataFrame | Series: ...
