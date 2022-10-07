from typing import (
    Callable,
    Hashable,
    Literal,
    Sequence,
    TypeVar,
)

import numpy as np
from pandas.core.frame import DataFrame
from pandas.core.groupby.grouper import Grouper
from pandas.core.series import Series

from pandas._typing import (
    HashableT,
    IndexLabel,
    Scalar,
    npt,
)

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
def crosstab(
    index: list | npt.NDArray | Series | list[Sequence | npt.NDArray | Series],
    columns: list | npt.NDArray | Series | list[Sequence | npt.NDArray | Series],
    values: list | npt.NDArray | Series | None = ...,
    rownames: list[HashableT] | None = ...,
    colnames: list[_HashableT2] | None = ...,
    aggfunc: str | np.ufunc | Callable[[Series], float] | None = ...,
    margins: bool = ...,
    margins_name: str = ...,
    dropna: bool = ...,
    normalize: bool | Literal[0, 1, "all", "index", "columns"] = ...,
) -> DataFrame: ...
