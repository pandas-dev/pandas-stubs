from typing import (
    Callable,
    Hashable,
    Literal,
    Sequence,
    TypeVar,
    Union,
    overload,
)

import numpy as np
import pandas as pd
from pandas.core.frame import DataFrame
from pandas.core.groupby.grouper import Grouper
from pandas.core.series import Series
from typing_extensions import TypeAlias

from pandas._typing import (
    AnyArrayLike,
    ArrayLike,
    HashableT,
    IndexLabel,
    Scalar,
)

_ExtendedAnyArrayLike: TypeAlias = Union[AnyArrayLike, ArrayLike]

_HashableT2 = TypeVar("_HashableT2", bound=Hashable)

def pivot_table(
    data: DataFrame,
    values: str | None = ...,
    index: str | Sequence | Grouper | None = ...,
    columns: str | Sequence | Grouper | None = ...,
    aggfunc=...,
    fill_value: Scalar | None = ...,
    margins: bool = ...,
    dropna: bool = ...,
    margins_name: str = ...,
    observed: bool = ...,
) -> DataFrame: ...
def pivot(
    data: DataFrame,
    *,
    index: IndexLabel = ...,
    columns: IndexLabel = ...,
    values: IndexLabel = ...,
) -> DataFrame: ...
@overload
def crosstab(
    index: list | _ExtendedAnyArrayLike | list[Sequence | _ExtendedAnyArrayLike],
    columns: list | _ExtendedAnyArrayLike | list[Sequence | _ExtendedAnyArrayLike],
    values: list | _ExtendedAnyArrayLike,
    rownames: list[HashableT] | None = ...,
    colnames: list[_HashableT2] | None = ...,
    *,
    aggfunc: str | np.ufunc | Callable[[Series], float],
    margins: bool = ...,
    margins_name: str = ...,
    dropna: bool = ...,
    normalize: bool | Literal[0, 1, "all", "index", "columns"] = ...,
) -> DataFrame: ...
@overload
def crosstab(
    index: list | _ExtendedAnyArrayLike | list[Sequence | _ExtendedAnyArrayLike],
    columns: list | _ExtendedAnyArrayLike | list[Sequence | _ExtendedAnyArrayLike],
    values: None = ...,
    rownames: list[HashableT] | None = ...,
    colnames: list[_HashableT2] | None = ...,
    aggfunc: None = ...,
    margins: bool = ...,
    margins_name: str = ...,
    dropna: bool = ...,
    normalize: bool | Literal[0, 1, "all", "index", "columns"] = ...,
) -> DataFrame: ...
