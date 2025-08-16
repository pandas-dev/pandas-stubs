from typing import (
    Literal,
    overload,
)

import numpy as np
from pandas import (
    Categorical,
    CategoricalIndex,
    Index,
    IntervalIndex,
    PeriodIndex,
    Series,
)
from pandas.api.extensions import ExtensionArray

from pandas._typing import (
    AnyArrayLike,
    IntervalT,
    TakeIndexer,
    np_1darray,
)

# These are type: ignored because the Index types overlap due to inheritance but indices
# with extension types return the same type while standard type return ndarray

@overload
def unique(  # pyright: ignore[reportOverlappingOverload]
    values: PeriodIndex,
) -> PeriodIndex: ...
@overload
def unique(values: CategoricalIndex) -> CategoricalIndex: ...  # type: ignore[overload-overlap]
@overload
def unique(values: IntervalIndex[IntervalT]) -> IntervalIndex[IntervalT]: ...
@overload
def unique(values: Index) -> np.ndarray: ...
@overload
def unique(values: Categorical) -> Categorical: ...
@overload
def unique(values: Series) -> np.ndarray | ExtensionArray: ...
@overload
def unique(values: np.ndarray) -> np.ndarray: ...
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
    arr,
    indices: TakeIndexer,
    axis: Literal[0, 1] = 0,
    allow_fill: bool = False,
    fill_value=None,
): ...
