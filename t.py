from typing import (
    TypeAlias,
    TypeVar,
    reveal_type,
)


class Index: ...


_IndexSliceUnion: TypeAlias = tuple[Index, slice]
_IndexSliceUnionT = TypeVar("_IndexSliceUnionT", bound=_IndexSliceUnion)


class IndexSlice:
    def __getitem__(self, arg: _IndexSliceUnionT) -> _IndexSliceUnionT: ...


def main(ind: Index) -> None:
    reveal_type(IndexSlice()[ind, :])
