from collections.abc import (
    Callable,
    Hashable,
    Mapping,
    Sequence,
)
import datetime
from typing import (
    Any,
    Literal,
    TypeAlias,
    overload,
)

import numpy as np
import pandas as pd
from pandas.core.frame import DataFrame
from pandas.core.groupby.grouper import Grouper
from pandas.core.indexes.base import Index
from pandas.core.series import Series

from pandas._typing import (
    AnyArrayLike,
    ArrayLike,
    Label,
    Scalar,
    ScalarT,
    SequenceNotStr,
    np_ndarray,
)

_PivotAggCallable: TypeAlias = Callable[[Series], ScalarT]
_PivotAggFunc: TypeAlias = (
    _PivotAggCallable[ScalarT]
    | np.ufunc
    | Literal["mean", "sum", "count", "min", "max", "median", "std", "var"]
)
_PivotAggFuncTypes: TypeAlias = (
    _PivotAggFunc[ScalarT]
    | Sequence[_PivotAggFunc[ScalarT]]
    | Mapping[Any, _PivotAggFunc[ScalarT]]
)

_NonIterableHashable: TypeAlias = (
    str
    | datetime.date
    | datetime.datetime
    | datetime.timedelta
    | bool
    | int
    | float
    | complex
    | pd.Timestamp
    | pd.Timedelta
)

_PivotTableIndexTypes: TypeAlias = Label | Sequence[Hashable] | Series | Grouper | None
_PivotTableColumnsTypes: TypeAlias = (
    Label | Sequence[Hashable] | Series | Grouper | None
)
_PivotTableValuesTypes: TypeAlias = Label | Sequence[Hashable] | None

_ExtendedAnyArrayLike: TypeAlias = AnyArrayLike | ArrayLike
_Values: TypeAlias = SequenceNotStr[Any] | _ExtendedAnyArrayLike

@overload
def pivot_table(
    data: DataFrame,
    values: _PivotTableValuesTypes = None,
    index: _PivotTableIndexTypes = None,
    columns: _PivotTableColumnsTypes = None,
    aggfunc: _PivotAggFuncTypes[Scalar] = "mean",
    fill_value: Scalar | None = None,
    margins: bool = False,
    dropna: bool = True,
    margins_name: Hashable = "All",
    observed: bool = True,
    sort: bool = True,
) -> DataFrame: ...

# Can only use Index or ndarray when index or columns is a Grouper
@overload
def pivot_table(
    data: DataFrame,
    values: _PivotTableValuesTypes = None,
    *,
    index: Grouper,
    columns: _PivotTableColumnsTypes | np_ndarray | Index[Any] = None,
    aggfunc: _PivotAggFuncTypes[Scalar] = "mean",
    fill_value: Scalar | None = None,
    margins: bool = False,
    dropna: bool = True,
    margins_name: Hashable = "All",
    observed: bool = True,
    sort: bool = True,
) -> DataFrame: ...
@overload
def pivot_table(
    data: DataFrame,
    values: _PivotTableValuesTypes = None,
    index: _PivotTableIndexTypes | np_ndarray | Index[Any] = None,
    *,
    columns: Grouper,
    aggfunc: _PivotAggFuncTypes[Scalar] = "mean",
    fill_value: Scalar | None = None,
    margins: bool = False,
    dropna: bool = True,
    margins_name: Hashable = "All",
    observed: bool = True,
    sort: bool = True,
) -> DataFrame: ...
def pivot(
    data: DataFrame,
    *,
    index: _NonIterableHashable | Sequence[Hashable] = ...,
    columns: _NonIterableHashable | Sequence[Hashable] = ...,
    values: _NonIterableHashable | Sequence[Hashable] = ...,
) -> DataFrame: ...
@overload
def crosstab(
    index: _Values | list[_Values],
    columns: _Values | list[_Values],
    values: _Values,
    rownames: SequenceNotStr[Hashable] | None = None,
    colnames: SequenceNotStr[Hashable] | None = None,
    *,
    aggfunc: str | np.ufunc | Callable[[Series], float],
    margins: bool = False,
    margins_name: str = "All",
    dropna: bool = True,
    normalize: bool | Literal[0, 1, "all", "index", "columns"] = False,
) -> DataFrame: ...
@overload
def crosstab(
    index: _Values | list[_Values],
    columns: _Values | list[_Values],
    values: None = None,
    rownames: SequenceNotStr[Hashable] | None = None,
    colnames: SequenceNotStr[Hashable] | None = None,
    aggfunc: None = None,
    margins: bool = False,
    margins_name: str = "All",
    dropna: bool = True,
    normalize: bool | Literal[0, 1, "all", "index", "columns"] = False,
) -> DataFrame: ...
