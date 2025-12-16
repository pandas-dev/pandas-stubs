from collections.abc import (
    Hashable,
    Sequence,
)
from typing import (
    TypeAlias,
    TypeVar,
)

from pandas.core.base import IndexOpsMixin
from typing_extensions import Self

from pandas._libs.indexing import _NDFrameIndexerBase
from pandas._libs.missing import NAType
from pandas._libs.tslibs.nattype import NaTType
from pandas._typing import (
    Axis,
    AxisInt,
    MaskType,
    Scalar,
)

_IndexSliceTuple: TypeAlias = tuple[
    IndexOpsMixin | MaskType | Scalar | Sequence[Scalar] | slice, ...
]

_IndexSliceUnion: TypeAlias = slice | _IndexSliceTuple

_IndexSliceUnionT = TypeVar(
    "_IndexSliceUnionT", bound=_IndexSliceUnion  # pyrefly: ignore
)

class _IndexSlice:
    def __getitem__(self, arg: _IndexSliceUnionT) -> _IndexSliceUnionT: ...

IndexSlice: _IndexSlice

class _NDFrameIndexer(_NDFrameIndexerBase):
    axis: AxisInt | None = None
    def __call__(self, axis: Axis | None = None) -> Self: ...

class _LocIndexer(_NDFrameIndexer): ...
class _iLocIndexer(_NDFrameIndexer): ...

class _AtIndexer(_NDFrameIndexerBase):
    def __getitem__(self, key: Hashable) -> Scalar: ...
    def __setitem__(
        self, key: Hashable, value: Scalar | NAType | NaTType | None
    ) -> None: ...

class _iAtIndexer(_NDFrameIndexerBase):
    def __getitem__(self, key: int) -> Scalar: ...
    def __setitem__(
        self, key: int, value: Scalar | NAType | NaTType | None
    ) -> None: ...
