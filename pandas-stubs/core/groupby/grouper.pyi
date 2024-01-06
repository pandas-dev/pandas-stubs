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
from typing_extensions import (
    Self,
    deprecated,
)

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
        key: KeysArgType | None = None,
        level: Level | ListLikeHashable[Level] | None = None,
        axis: Axis | NoDefault = ...,
        sort: bool = False,
        dropna: bool = True,
    ) -> Self: ...
    @overload
    def __new__(cls, *args, freq: Frequency, **kwargs) -> TimeGrouper: ...
    def __init__(
        self,
        key: KeysArgType | None = None,
        level: Level | ListLikeHashable[Level] | None = None,
        freq: Frequency | None = None,
        axis: Axis | NoDefault = ...,
        sort: bool = False,
        dropna: bool = True,
    ) -> None: ...
    @property
    @deprecated("Grouper.ax is deprecated. Use Resampler.ax instead.")
    def ax(self): ...
    @property
    @deprecated("Grouper.indexer is deprecated. Use Resampler.indexer instead.")
    def indexer(self): ...
    @property
    @deprecated("Grouper.obj is deprecated. Use GroupBy.obj instead.")
    def obj(self): ...
    @property
    @deprecated("Grouper.grouper is deprecated. Use GroupBy.grouper instead.")
    def grouper(self): ...
    @property
    @deprecated("Grouper.groups is deprecated. Use GroupBy.groups instead.")
    def groups(self): ...
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
        grouper=None,
        obj: DataFrame | Series | None = None,
        level: Level | None = None,
        sort: bool = True,
        observed: bool = False,
        in_axis: bool = False,
        dropna: bool = True,
        uniques: ArrayLike | None = None,
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
    key: KeysArgType | None = None,
    axis: Axis = 0,
    level: Level | ListLikeHashable[Level] | None = None,
    sort: bool = True,
    observed: bool = False,
    validate: bool = True,
    dropna: bool = True,
) -> tuple[BaseGrouper, frozenset[Hashable], NDFrameT]: ...
