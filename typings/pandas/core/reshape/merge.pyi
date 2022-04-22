from pandas._libs.tslibs import Timedelta
from pandas import DataFrame as DataFrame, Series as Series
from pandas._typing import Label
from typing import Optional, Sequence, Union

def merge(
    left: Union[DataFrame, Series],
    right: Union[DataFrame, Series],
    how: str = ...,
    on: Optional[Union[Label, Sequence]] = ...,
    left_on: Optional[Union[Label, Sequence]] = ...,
    right_on: Optional[Union[Label, Sequence]] = ...,
    left_index: bool = ...,
    right_index: bool = ...,
    sort: bool = ...,
    suffixes: Sequence[Union[str, None]] = ...,
    copy: bool = ...,
    indicator: Union[bool, str] = ...,
    validate: str = ...,
) -> DataFrame: ...
def merge_ordered(
    left: Union[DataFrame, Series],
    right: Union[DataFrame, Series],
    on: Optional[Union[Label, Sequence]] = ...,
    left_on: Optional[Union[Label, Sequence]] = ...,
    right_on: Optional[Union[Label, Sequence]] = ...,
    left_by: Optional[Union[str, Sequence[str]]] = ...,
    right_by: Optional[Union[str, Sequence[str]]] = ...,
    fill_method: Optional[str] = ...,
    suffixes: Sequence[Union[str, None]] = ...,
    how: str = ...,
) -> DataFrame: ...
def merge_asof(
    left: Union[DataFrame, Series],
    right: Union[DataFrame, Series],
    on: Optional[Label] = ...,
    left_on: Optional[Label] = ...,
    right_on: Optional[Label] = ...,
    left_index: bool = ...,
    right_index: bool = ...,
    by: Optional[Union[str, Sequence[str]]] = ...,
    left_by: Optional[str] = ...,
    right_by: Optional[str] = ...,
    suffixes: Sequence[Union[str, None]] = ...,
    tolerance: Optional[Union[int, Timedelta]] = ...,
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
        left: Union[Series, DataFrame],
        right: Union[Series, DataFrame],
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
