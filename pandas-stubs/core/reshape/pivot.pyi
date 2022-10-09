import datetime
from typing import (
    Callable,
    Hashable,
    Literal,
    Mapping,
    Sequence,
    TypeVar,
    Union,
    overload,
)

import numpy as np
import pandas as pd
from pandas.core.frame import DataFrame
from pandas.core.groupby.grouper import Grouper
from pandas.core.indexes.base import Index
from pandas.core.series import Series
from typing_extensions import TypeAlias

from pandas._typing import (
    Label,
    Scalar,
    npt,
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

_PivotTableIndexTypes: TypeAlias = Union[
    Label, list[_HashableT1], Series, Grouper, None
]
_PivotTableColumnsTypes: TypeAlias = Union[
    Label, list[_HashableT2], Series, Grouper, None
]

@overload
def pivot_table(
    data: DataFrame,
    values: Label | None = ...,
    index: _PivotTableIndexTypes = ...,
    columns: _PivotTableColumnsTypes = ...,
    aggfunc: _PivotAggFunc
    | list[_PivotAggFunc]
    | Mapping[Hashable, _PivotAggFunc] = ...,
    fill_value: Scalar | None = ...,
    margins: bool = ...,
    dropna: bool = ...,
    margins_name: str = ...,
    observed: bool = ...,
    sort: bool = ...,
) -> DataFrame: ...

# Can only use Index or ndarray when index or columns is a Grouper
@overload
def pivot_table(
    data: DataFrame,
    values: Label | None = ...,
    *,
    index: Grouper,
    columns: _PivotTableColumnsTypes | Index | npt.NDArray = ...,
    aggfunc: _PivotAggFunc
    | list[_PivotAggFunc]
    | Mapping[Hashable, _PivotAggFunc] = ...,
    fill_value: Scalar | None = ...,
    margins: bool = ...,
    dropna: bool = ...,
    margins_name: str = ...,
    observed: bool = ...,
    sort: bool = ...,
) -> DataFrame: ...
@overload
def pivot_table(
    data: DataFrame,
    values: Label | None = ...,
    index: _PivotTableIndexTypes | Index | npt.NDArray = ...,
    *,
    columns: Grouper,
    aggfunc: _PivotAggFunc
    | list[_PivotAggFunc]
    | Mapping[Hashable, _PivotAggFunc] = ...,
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
