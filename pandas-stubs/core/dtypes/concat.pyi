from typing import (
    TypeVar,
    overload,
)

from pandas import (
    Categorical,
    CategoricalIndex,
    Series,
)

_CatT = TypeVar("_CatT", bound=Categorical | CategoricalIndex | Series)

@overload
def union_categoricals(
    to_union: list[Categorical[str]] | list[Series[str]] | list[CategoricalIndex[str]],
    sort_categories: bool = False,
    ignore_order: bool = False,
) -> Categorical[str]: ...
@overload
def union_categoricals(
    to_union: list[Categorical[int]] | list[Series[int]] | list[CategoricalIndex[int]],
    sort_categories: bool = False,
    ignore_order: bool = False,
) -> Categorical[int]: ...
@overload
def union_categoricals(
    to_union: (
        list[Categorical[float]] | list[Series[float]] | list[CategoricalIndex[float]]
    ),
    sort_categories: bool = False,
    ignore_order: bool = False,
) -> Categorical[float]: ...
@overload
def union_categoricals(
    to_union: list[_CatT],
    sort_categories: bool = False,
    ignore_order: bool = False,
) -> Categorical: ...
