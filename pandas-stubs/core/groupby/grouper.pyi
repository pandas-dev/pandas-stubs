from collections.abc import (
    Hashable,
    Iterator,
)
from typing import (
    final,
    overload,
)

import numpy as np
from pandas import (
    DataFrame,
    Index,
    Series,
)
from pandas.core.groupby.ops import BaseGrouper
from pandas.core.resample import TimeGrouper
from typing_extensions import Self

from pandas._libs.lib import NoDefault
from pandas._typing import (
    ArrayLike,
    Axis,
    Frequency,
    Incomplete,
    KeysArgType,
    Level,
    ListLikeHashable,
    NDFrameT,
    npt,
)
from pandas.util._decorators import cache_readonly

class Grouper:
    key: KeysArgType | None
    level: Level | ListLikeHashable[Level] | None
    freq: Frequency | None
    axis: Axis
    sort: bool
    dropna: bool
    binner: Incomplete
    @overload
    def __new__(
        cls,
        key: KeysArgType | None = ...,
        level: Level | ListLikeHashable[Level] | None = ...,
        axis: Axis | NoDefault = ...,
        sort: bool = ...,
        dropna: bool = ...,
    ) -> Self: ...
    @overload
    def __new__(cls, *args, freq: Frequency, **kwargs) -> TimeGrouper: ...
    def __init__(
        self,
        key: KeysArgType | None = ...,
        level: Level | ListLikeHashable[Level] | None = ...,
        freq: Frequency | None = ...,
        axis: Axis | NoDefault = ...,
        sort: bool = ...,
        dropna: bool = ...,
    ) -> None: ...
    @final
    def __repr__(self) -> str: ...  # noqa: PYI029 __repr__ here is final

@final
class Grouping:
    level: Level | None
    obj: DataFrame | Series | None
    in_axis: bool
    grouping_vector: Incomplete
    def __init__(
        self,
        index: Index,
        grouper=...,
        obj: DataFrame | Series | None = ...,
        level: Level | None = ...,
        sort: bool = ...,
        observed: bool = ...,
        in_axis: bool = ...,
        dropna: bool = ...,
        uniques: ArrayLike | None = ...,
    ) -> None: ...
    def __iter__(self) -> Iterator[Hashable]: ...
    @cache_readonly
    def name(self) -> Hashable: ...
    @cache_readonly
    def ngroups(self) -> int: ...
    @cache_readonly
    def indices(self) -> dict[Hashable, npt.NDArray[np.intp]]: ...
    @property
    def codes(self) -> npt.NDArray[np.signedinteger]: ...
    @cache_readonly
    def group_arraylike(self) -> ArrayLike: ...
    @cache_readonly
    def result_index(self) -> Index: ...
    @cache_readonly
    def group_index(self) -> Index: ...
    @cache_readonly
    def groups(self) -> dict[Hashable, np.ndarray]: ...

def get_grouper(
    obj: NDFrameT,
    key: KeysArgType | None = ...,
    axis: Axis = ...,
    level: Level | ListLikeHashable[Level] | None = ...,
    sort: bool = ...,
    observed: bool = ...,
    validate: bool = ...,
    dropna: bool = ...,
) -> tuple[BaseGrouper, frozenset[Hashable], NDFrameT]: ...
