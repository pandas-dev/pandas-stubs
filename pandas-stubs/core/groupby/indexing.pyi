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

class GroupByNthSelector(Generic[_GroupByT]):
    groupby_object: _GroupByT

    def __call__(
        self,
        n: PositionalIndexer | tuple[int, ...],
        dropna: Literal["any", "all"] | None = ...,
    ) -> DataFrame | Series: ...
    def __getitem__(
        self, n: PositionalIndexer | tuple[int, ...]
    ) -> DataFrame | Series: ...
