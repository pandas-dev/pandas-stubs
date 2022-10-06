from typing import Literal

from pandas import (
    DataFrame,
    Series,
)

from pandas._libs.tslibs import Timedelta
from pandas._typing import (
    AnyArrayLike,
    HashableT,
    Label,
    ValidationOptions,
)

def merge(
    # TODO: Verify Series is accepted and correct in docs
    left: DataFrame | Series,
    right: DataFrame | Series,
    how: Literal["left", "right", "outer", "inner", "cross"] = ...,
    on: Label | list[HashableT] | AnyArrayLike | None = ...,
    left_on: Label | list[HashableT] | AnyArrayLike | None = ...,
    right_on: Label | list[HashableT] | AnyArrayLike | None = ...,
    left_index: bool = ...,
    right_index: bool = ...,
    sort: bool = ...,
    suffixes: list[str | None]
    | tuple[str, str]
    | tuple[None, str]
    | tuple[str, None] = ...,
    copy: bool = ...,
    indicator: bool | str = ...,
    validate: ValidationOptions = ...,
) -> DataFrame: ...
def merge_ordered(
    # TODO: Verify Series is accepted and correct in docs
    left: DataFrame | Series,
    # TODO: Verify Series is accepted and correct in docs
    right: DataFrame | Series,
    on: Label | list[HashableT] | AnyArrayLike | None = ...,
    left_on: Label | list[HashableT] | AnyArrayLike | None = ...,
    right_on: Label | list[HashableT] | AnyArrayLike | None = ...,
    left_by: Label | list[HashableT] | None = ...,
    right_by: Label | list[HashableT] | None = ...,
    fill_method: Literal["ffill"] | None = ...,
    suffixes: list[str | None]
    | tuple[str, str]
    | tuple[None, str]
    | tuple[str, None] = ...,
    how: Literal["left", "right", "outer", "inner"] = ...,
) -> DataFrame: ...
def merge_asof(
    left: DataFrame | Series,
    right: DataFrame | Series,
    on: Label | None = ...,
    # TODO: Is AnyArrayLike accepted?  Not in docs
    left_on: Label | AnyArrayLike | None = ...,
    # TODO: Is AnyArrayLike accepted?  Not in docs
    right_on: Label | AnyArrayLike | None = ...,
    left_index: bool = ...,
    right_index: bool = ...,
    by: Label | list[HashableT] | None = ...,
    left_by: Label | None = ...,
    right_by: Label | None = ...,
    suffixes: list[str | None]
    | tuple[str, str]
    | tuple[None, str]
    | tuple[str, None] = ...,
    tolerance: int | Timedelta | None = ...,
    allow_exact_matches: bool = ...,
    direction: Literal["backward", "forward", "nearest"] = ...,
) -> DataFrame: ...
