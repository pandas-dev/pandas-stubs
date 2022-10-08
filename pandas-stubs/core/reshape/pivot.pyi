import datetime
from typing import (
    Callable,
    Hashable,
    Literal,
    Sequence,
    TypeVar,
    Union,
)

import numpy as np
import pandas as pd
from pandas.core.frame import DataFrame
from pandas.core.groupby.grouper import Grouper
from pandas.core.series import Series
from typing_extensions import TypeAlias

from pandas._typing import (
    Label,
    Scalar,
)

_HashableT1 = TypeVar("_HashableT1", bound=Hashable)
_HashableT2 = TypeVar("_HashableT2", bound=Hashable)
_HashableT3 = TypeVar("_HashableT3", bound=Hashable)

_PivotAggCallable: TypeAlias = Union[
    Callable[[Series], str],
    Callable[[Series], datetime.date],
    Callable[[Series], datetime.datetime],
    Callable[[Series], datetime.timedelta],
    Callable[[Series], bool],
    Callable[[Series], int],
    Callable[[Series], float],
    Callable[[Series], complex],
    Callable[[Series], pd.Timestamp],
    Callable[[Series], pd.Timedelta],
]

_PivotAggFunc: TypeAlias = Union[
    _PivotAggCallable,
    np.ufunc,
    Literal["mean", "sum", "count", "min", "max", "median", "std", "var"],
]

_NonIterableHashable: TypeAlias = Union[
    str,
    datetime.date,
    datetime.datetime,
    datetime.timedelta,
    bool,
    int,
    float,
    complex,
    pd.Timestamp,
    pd.Timedelta,
]

def pivot_table(
    data: DataFrame,
    values: Label | None = ...,
    index: Label | list[_HashableT1] | Grouper | None = ...,
    columns: Label | list[_HashableT2] | Grouper | None = ...,
    aggfunc: _PivotAggFunc
    | list[_PivotAggFunc]
    | dict[_HashableT3, _PivotAggFunc] = ...,
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
    index: _NonIterableHashable | list[_HashableT1] = ...,
    columns: _NonIterableHashable | list[_HashableT2] = ...,
    values: _NonIterableHashable | list[_HashableT3] = ...,
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
