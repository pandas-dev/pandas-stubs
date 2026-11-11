from collections.abc import (
    Hashable,
    Sequence,
)
from typing import (
    Generic,
    Self,
    TypeAlias,
)

from pandas.core.base import IndexOpsMixin
from typing_extensions import TypeVar

from pandas._libs.indexing import NDFrameIndexerBase
from pandas._libs.missing import NAType
from pandas._libs.tslibs.nattype import NaTType
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

_ScalarT_co = TypeVar(
    "_ScalarT_co",
    covariant=True,
    default=ExtScalarOrNA,
)

class _AtIndexer(NDFrameIndexerBase, Generic[_ScalarT_co]):
    def __getitem__(self, key: Hashable, /) -> _ScalarT_co: ...
    def __setitem__(
        self, key: Hashable, value: Scalar | NAType | NaTType | None, /
    ) -> None: ...

class _iAtIndexer(NDFrameIndexerBase, Generic[_ScalarT_co]):
    def __getitem__(self, key: int, /) -> _ScalarT_co: ...
    def __setitem__(
        self, key: int, value: Scalar | NAType | NaTType | None, /
    ) -> None: ...
