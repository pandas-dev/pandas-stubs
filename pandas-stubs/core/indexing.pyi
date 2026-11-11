from collections.abc import (
    Hashable,
    Sequence,
)
from typing import (
    Self,
    TypeAlias,
    TypeVar,
)

from pandas.core.base import IndexOpsMixin

from pandas._libs.indexing import NDFrameIndexerBase
from pandas._typing import (
    Axis,
    AxisInt,
    ExtScalarOrNA,
    MaskType,
    Scalar,
)

_IndexSliceTuple: TypeAlias = tuple[
    IndexOpsMixin | MaskType | Scalar | Sequence[Scalar] | slice, ...
]

_IndexSliceUnion: TypeAlias = slice | _IndexSliceTuple

_IndexSliceUnionT = TypeVar("_IndexSliceUnionT", bound=_IndexSliceUnion)

class _IndexSlice:
    def __getitem__(self, arg: _IndexSliceUnionT, /) -> _IndexSliceUnionT: ...

IndexSlice: _IndexSlice

class _NDFrameIndexer(NDFrameIndexerBase):
    axis: AxisInt | None = None
    def __call__(self, axis: Axis | None = None) -> Self: ...

class _LocIndexer(_NDFrameIndexer): ...
class _iLocIndexer(_NDFrameIndexer): ...

class _AtIndexer(NDFrameIndexerBase):
    def __getitem__(self, key: Hashable, /) -> ExtScalarOrNA: ...
    def __setitem__(self, key: Hashable, value: ExtScalarOrNA, /) -> None: ...

class _iAtIndexer(NDFrameIndexerBase):
    def __getitem__(self, key: int, /) -> ExtScalarOrNA: ...
    def __setitem__(self, key: int, value: ExtScalarOrNA, /) -> None: ...
