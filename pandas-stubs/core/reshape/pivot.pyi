from typing import (
    Callable,
    Hashable,
    Literal,
    Sequence,
    TypeAlias,
    TypeVar,
)

import numpy as np
from pandas.core.frame import DataFrame
from pandas.core.groupby.grouper import Grouper
from pandas.core.series import Series

from pandas._typing import (
    Label,
    Scalar,
)

_HashableT1 = TypeVar("_HashableT1", bound=Hashable)
_HashableT2 = TypeVar("_HashableT2", bound=Hashable)
_HashableT3 = TypeVar("_HashableT3", bound=Hashable)

_PivotAggFunc: TypeAlias = Callable[[Series], Scalar]

def pivot_table(
    data: DataFrame,
    values: Label | None = ...,
    index: Label | list[_HashableT1] | Grouper | None = ...,
    columns: Label | list[_HashableT2] | Grouper | None = ...,
    aggfunc: _PivotAggFunc
    | list[_PivotAggFunc]
    | dict[_HashableT3, _PivotAggFunc]
    | Literal["mean", "sum", "count", "min", "max", "median", "std", "var"]
    | np.ufunc = ...,
    fill_value: Scalar | None = ...,
    margins: bool = ...,
    dropna: bool = ...,
    margins_name: str = ...,
    observed: bool = ...,
    sort: bool = ...,
) -> DataFrame: ...
def pivot(
    data: DataFrame,
    *,
    index: Hashable | list[_HashableT1] = ...,
    columns: Hashable | list[_HashableT2] = ...,
    values: Hashable | list[_HashableT3] = ...,
) -> DataFrame: ...
def crosstab(
    index: Sequence | Series,
    columns: Sequence | Series,
    values: Sequence | None = ...,
    rownames: Sequence | None = ...,
    colnames: Sequence | None = ...,
    aggfunc: Callable | None = ...,
    margins: bool = ...,
    margins_name: str = ...,
    dropna: bool = ...,
    normalize: bool | Literal[0, 1, "all", "index", "columns"] = ...,
) -> DataFrame: ...
