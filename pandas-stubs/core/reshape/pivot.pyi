from collections.abc import (
    Callable,
    Hashable,
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
from pandas._stubs_only import (
    PivotAggFuncTypes,
    PivotTableColumnsTypes,
    PivotTableIndexTypes,
    PivotTableValuesTypes,
)
from pandas.core.frame import DataFrame
from pandas.core.groupby.grouper import Grouper
from pandas.core.indexes.base import Index
from pandas.core.series import Series

from pandas._typing import (
    AnyArrayLike,
    ArrayLike,
    Scalar,
    SequenceNotStr,
    np_ndarray,
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

_ExtendedAnyArrayLike: TypeAlias = AnyArrayLike | ArrayLike
_CrossTabValues: TypeAlias = SequenceNotStr[Any] | _ExtendedAnyArrayLike

@overload
def pivot_table(
    data: DataFrame,
    values: PivotTableValuesTypes = None,
    index: PivotTableIndexTypes = None,
    columns: PivotTableColumnsTypes = None,
    aggfunc: PivotAggFuncTypes[Scalar] = "mean",
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
    values: PivotTableValuesTypes = None,
    *,
    index: Grouper,
    columns: PivotTableColumnsTypes | np_ndarray | Index[Any] = None,
    aggfunc: PivotAggFuncTypes[Scalar] = "mean",
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
    values: PivotTableValuesTypes = None,
    index: PivotTableIndexTypes | np_ndarray | Index[Any] = None,
    *,
    columns: Grouper,
    aggfunc: PivotAggFuncTypes[Scalar] = "mean",
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
    index: _CrossTabValues | list[_CrossTabValues],
    columns: _CrossTabValues | list[_CrossTabValues],
    values: _CrossTabValues,
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
    index: _CrossTabValues | list[_CrossTabValues],
    columns: _CrossTabValues | list[_CrossTabValues],
    values: None = None,
    rownames: SequenceNotStr[Hashable] | None = None,
    colnames: SequenceNotStr[Hashable] | None = None,
    aggfunc: None = None,
    margins: bool = False,
    margins_name: str = "All",
    dropna: bool = True,
    normalize: bool | Literal[0, 1, "all", "index", "columns"] = False,
) -> DataFrame: ...
