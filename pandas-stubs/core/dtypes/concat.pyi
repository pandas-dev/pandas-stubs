from typing import (
    TypeVar,
    overload,
)

from pandas import (
    Categorical,
    CategoricalIndex,
    Series,
)

from pandas.core.dtypes.dtypes import CategoricalValueT1

_CatT = TypeVar("_CatT", bound=Categorical | CategoricalIndex | Series)

@overload
def union_categoricals(
    to_union: (
        list[Categorical[CategoricalValueT1]]
        | list[Series[CategoricalValueT1]]
        | list[CategoricalIndex[CategoricalValueT1]]
    ),
    sort_categories: bool = False,
    ignore_order: bool = False,
) -> Categorical[CategoricalValueT1]: ...
@overload
def union_categoricals(
    to_union: list[_CatT],
    sort_categories: bool = False,
    ignore_order: bool = False,
) -> Categorical: ...
