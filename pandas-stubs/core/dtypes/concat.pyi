from typing import (
    TypeVar,
    Union,
)

from pandas import (
    Categorical,
    CategoricalIndex,
    Series,
)

_CatT = TypeVar("_CatT", bound=Union[Categorical, CategoricalIndex, Series])

def union_categoricals(
    to_union: list[_CatT], sort_categories: bool = ..., ignore_order: bool = ...
) -> Categorical: ...
