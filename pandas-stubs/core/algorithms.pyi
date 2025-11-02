from typing import (
    Any,
    Literal,
    overload,
)

import numpy as np
from numpy import typing as npt
from pandas.api.extensions import ExtensionArray
from pandas.core.arrays.categorical import Categorical
from pandas.core.indexes.base import Index
from pandas.core.indexes.category import CategoricalIndex
from pandas.core.indexes.interval import IntervalIndex
from pandas.core.indexes.period import PeriodIndex
from pandas.core.series import Series

from pandas._typing import (
    S1,
    AnyArrayLike,
    GenericT_co,
    IntervalT,
    TakeIndexer,
    np_1darray,
)

# These are type: ignored because the Index types overlap due to inheritance but indices
# with extension types return the same type while standard type return ndarray

@overload
def unique(  # type: ignore[overload-overlap] # pyright: ignore[reportOverlappingOverload]
    values: PeriodIndex,
) -> PeriodIndex: ...
@overload
def unique(
    values: CategoricalIndex[S1, GenericT_co],
) -> CategoricalIndex[S1, GenericT_co]: ...
@overload
def unique(values: IntervalIndex[IntervalT]) -> IntervalIndex[IntervalT]: ...
@overload
def unique(values: Index[S1, np_1darray, GenericT_co]) -> np_1darray[GenericT_co]: ...
@overload
def unique(values: Categorical) -> Categorical: ...
@overload
def unique(values: Series) -> np.ndarray | ExtensionArray: ...
@overload
def unique(values: npt.NDArray[GenericT_co]) -> np_1darray[GenericT_co]: ...
@overload
def unique(values: ExtensionArray) -> ExtensionArray: ...
@overload
def factorize(
    values: np.ndarray,
    sort: bool = ...,
    use_na_sentinel: bool = ...,
    size_hint: int | None = ...,
) -> tuple[np.ndarray, np.ndarray]: ...
@overload
def factorize(
    values: Index | Series,
    sort: bool = ...,
    use_na_sentinel: bool = ...,
    size_hint: int | None = ...,
) -> tuple[np_1darray, Index]: ...
@overload
def factorize(
    values: Categorical,
    sort: bool = ...,
    use_na_sentinel: bool = ...,
    size_hint: int | None = ...,
) -> tuple[np_1darray, Categorical]: ...
def value_counts(
    values: AnyArrayLike | list | tuple,
    sort: bool = True,
    ascending: bool = False,
    normalize: bool = False,
    bins: int | None = None,
    dropna: bool = True,
) -> Series: ...
def take(
    arr: np.ndarray | ExtensionArray | Index | Series,
    indices: TakeIndexer,
    axis: Literal[0, 1] = 0,
    allow_fill: bool = False,
    fill_value: Any = None,
) -> np_1darray | ExtensionArray: ...
