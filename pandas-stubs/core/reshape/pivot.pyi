from typing import (
    Callable,
    Literal,
    Sequence,
)

from pandas.core.frame import DataFrame
from pandas.core.groupby.grouper import Grouper
from pandas.core.series import Series

from pandas._typing import (
    IndexLabel,
    Scalar,
)

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
    index: IndexLabel = ...,
    columns: IndexLabel = ...,
    values: IndexLabel = ...,
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
