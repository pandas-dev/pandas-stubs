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
    HashableT1,
    HashableT2,
    HashableT3,
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

_PivotTableIndexTypes: TypeAlias = (
    Label | Sequence[HashableT1] | Series | Grouper | None
)
_PivotTableColumnsTypes: TypeAlias = (
    Label | Sequence[HashableT2] | Series | Grouper | None
)
_PivotTableValuesTypes: TypeAlias = Label | Sequence[HashableT3] | None

_ExtendedAnyArrayLike: TypeAlias = AnyArrayLike | ArrayLike
_Values: TypeAlias = SequenceNotStr[Any] | _ExtendedAnyArrayLike

@overload
def pivot_table(
    data: DataFrame,
    values: _PivotTableValuesTypes[
        Hashable  # ty: ignore[invalid-type-arguments]
    ] = None,
    index: _PivotTableIndexTypes[Hashable] = None,  # ty: ignore[invalid-type-arguments]
    columns: _PivotTableColumnsTypes[
        Hashable  # ty: ignore[invalid-type-arguments]
    ] = None,
    aggfunc: (
        _PivotAggFunc[Scalar]
        | Sequence[_PivotAggFunc[Scalar]]
        | Mapping[Any, _PivotAggFunc[Scalar]]
    ) = "mean",
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
    values: _PivotTableValuesTypes[
        Hashable  # ty: ignore[invalid-type-arguments]
    ] = None,
    *,
    index: Grouper,
    columns: (
        _PivotTableColumnsTypes[Hashable]  # ty: ignore[invalid-type-arguments]
        | np_ndarray
        | Index[Any]
    ) = None,
    aggfunc: (
        _PivotAggFunc[Scalar]
        | Sequence[_PivotAggFunc[Scalar]]
        | Mapping[Any, _PivotAggFunc[Scalar]]
    ) = "mean",
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
    values: _PivotTableValuesTypes[
        Hashable  # ty: ignore[invalid-type-arguments]
    ] = None,
    index: (
        _PivotTableIndexTypes[Hashable]  # ty: ignore[invalid-type-arguments]
        | np_ndarray
        | Index[Any]
    ) = None,
    *,
    columns: Grouper,
    aggfunc: (
        _PivotAggFunc[Scalar]
        | Sequence[_PivotAggFunc[Scalar]]
        | Mapping[Any, _PivotAggFunc[Scalar]]
    ) = "mean",
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
    index: _NonIterableHashable | Sequence[HashableT1] = ...,
    columns: _NonIterableHashable | Sequence[HashableT2] = ...,
    values: _NonIterableHashable | Sequence[HashableT3] = ...,
) -> DataFrame: ...
@overload
def crosstab(
    index: _Values | list[_Values],
    columns: _Values | list[_Values],
    values: _Values,
    rownames: list[HashableT1] | None = ...,
    colnames: list[HashableT2] | None = ...,
    *,
    aggfunc: str | np.ufunc | Callable[[Series], float],
    margins: bool = ...,
    margins_name: str = ...,
    dropna: bool = ...,
    normalize: bool | Literal[0, 1, "all", "index", "columns"] = ...,
) -> DataFrame: ...
@overload
def crosstab(
    index: _Values | list[_Values],
    columns: _Values | list[_Values],
    values: None = None,
    rownames: list[HashableT1] | None = ...,
    colnames: list[HashableT2] | None = ...,
    aggfunc: None = None,
    margins: bool = ...,
    margins_name: str = ...,
    dropna: bool = ...,
    normalize: bool | Literal[0, 1, "all", "index", "columns"] = ...,
) -> DataFrame: ...
