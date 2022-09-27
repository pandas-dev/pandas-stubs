from typing import (
    Hashable,
    Iterable,
)

from pandas import (
    DataFrame,
    Series,
)

from pandas._typing import (
    ArrayLike,
    Dtype,
    HashableT,
)

def get_dummies(
    data: ArrayLike | DataFrame | Series,
    prefix: str | Iterable[str] | dict[HashableT, str] | None = ...,
    prefix_sep: str = ...,
    dummy_na: bool = ...,
    columns: list[HashableT] | None = ...,
    sparse: bool = ...,
    drop_first: bool = ...,
    dtype: Dtype | None = ...,
) -> DataFrame: ...
def from_dummies(
    data: DataFrame,
    sep: str | None = ...,
    default_category: Hashable | dict[str, Hashable] | None = ...,
) -> DataFrame: ...
