from typing import Sequence

from pandas import (
    DataFrame,
    Series,
)

from pandas._libs.tslibs import Timedelta
from pandas._typing import (
    AnyArrayLike,
    Label,
)

def merge(
    left: DataFrame | Series,
    right: DataFrame | Series,
    how: str = ...,
    on: Label | Sequence | AnyArrayLike | None = ...,
    left_on: Label | Sequence | AnyArrayLike | None = ...,
    right_on: Label | Sequence | AnyArrayLike | None = ...,
    left_index: bool = ...,
    right_index: bool = ...,
    sort: bool = ...,
    suffixes: Sequence[str | None] = ...,
    copy: bool = ...,
    indicator: bool | str = ...,
    validate: str = ...,
) -> DataFrame: ...
def merge_ordered(
    left: DataFrame | Series,
    right: DataFrame | Series,
    on: Label | Sequence | AnyArrayLike | None = ...,
    left_on: Label | Sequence | AnyArrayLike | None = ...,
    right_on: Label | Sequence | AnyArrayLike | None = ...,
    left_by: str | list[str] | None = ...,
    right_by: str | list[str] | None = ...,
    fill_method: str | None = ...,
    suffixes: Sequence[str | None] = ...,
    how: str = ...,
) -> DataFrame: ...
def merge_asof(
    left: DataFrame | Series,
    right: DataFrame | Series,
    on: Label | None = ...,
    left_on: Label | AnyArrayLike | None = ...,
    right_on: Label | AnyArrayLike | None = ...,
    left_index: bool = ...,
    right_index: bool = ...,
    by: str | list[str] | None = ...,
    left_by: str | None = ...,
    right_by: str | None = ...,
    suffixes: Sequence[str | None] = ...,
    tolerance: int | Timedelta | None = ...,
    allow_exact_matches: bool = ...,
    direction: str = ...,
) -> DataFrame: ...

class _MergeOperation:
    left = ...
    right = ...
    how = ...
    axis = ...
    on = ...
    left_on = ...
    right_on = ...
    copy = ...
    suffixes = ...
    sort = ...
    left_index = ...
    right_index = ...
    indicator = ...
    indicator_name = ...
    def __init__(
        self,
        left: Series | DataFrame,
        right: Series | DataFrame,
        how: str = ...,
        on=...,
        left_on=...,
        right_on=...,
        axis=...,
        left_index: bool = ...,
        right_index: bool = ...,
        sort: bool = ...,
        suffixes=...,
        copy: bool = ...,
        indicator: bool = ...,
        validate=...,
    ) -> None: ...
    def get_result(self): ...

class _OrderedMerge(_MergeOperation):
    fill_method = ...
    def __init__(
        self,
        left,
        right,
        on=...,
        left_on=...,
        right_on=...,
        left_index: bool = ...,
        right_index: bool = ...,
        axis=...,
        suffixes=...,
        copy: bool = ...,
        fill_method=...,
        how: str = ...,
    ) -> None: ...
    def get_result(self): ...

class _AsOfMerge(_OrderedMerge):
    by = ...
    left_by = ...
    right_by = ...
    tolerance = ...
    allow_exact_matches = ...
    direction = ...
    def __init__(
        self,
        left,
        right,
        on=...,
        left_on=...,
        right_on=...,
        left_index: bool = ...,
        right_index: bool = ...,
        by=...,
        left_by=...,
        right_by=...,
        axis=...,
        suffixes=...,
        copy: bool = ...,
        fill_method=...,
        how: str = ...,
        tolerance=...,
        allow_exact_matches: bool = ...,
        direction: str = ...,
    ) -> None: ...
