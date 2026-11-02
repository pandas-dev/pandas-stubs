from collections.abc import (
    Callable,
    Hashable,
    Iterable,
    Mapping,
)
from typing import (
    Any,
    Literal,
    Never,
    Self,
    overload,
)

from pandas.core.accessor import PandasDelegate
from pandas.core.arrays.categorical import (
    Categorical,
    CategoricalMethods,
)
from pandas.core.indexes.base import Index
from pandas.core.indexes.extension import ExtensionIndex
from typing_extensions import override

from pandas._typing import (
    S1,
    S2,
    Dtype,
    ListLike,
    SequenceNotStr,
    np_1darray_intp,
)

from pandas.core.dtypes.dtypes import CategoricalValueT1

class CategoricalIndex(
    ExtensionIndex[S1],
    PandasDelegate,
    CategoricalMethods[CategoricalIndex[S1], Any],
):
    codes: np_1darray_intp = ...
    @property
    @override
    def categories(self) -> Index[S1]: ...
    @overload
    def set_categories(
        self,
        new_categories: SequenceNotStr[S2],
        ordered: bool | None = None,
        rename: Literal[True] = True,
    ) -> CategoricalIndex[S2]: ...
    @overload
    def set_categories(
        self,
        new_categories: SequenceNotStr[S1] | ListLike,
        ordered: bool | None = None,
        rename: Literal[False] = False,
    ) -> CategoricalIndex[S1]: ...
    @overload
    def set_categories(
        self,
        new_categories: SequenceNotStr[S1] | ListLike,
        ordered: bool | None = None,
        rename: bool = False,
    ) -> CategoricalIndex[S1]: ...
    @overload
    def rename_categories(
        self: CategoricalIndex[Hashable],
        new_categories: (
            SequenceNotStr[Any]
            | Mapping[Any, Any]
            | Callable[[Hashable], Any]
        ),
    ) -> CategoricalIndex[Any]: ...
    @overload
    def rename_categories(
        self,
        new_categories: (
            SequenceNotStr[S2]
            | Mapping[Any, S2]
            | Callable[[S1], S2]
        ),
    ) -> CategoricalIndex[S2]: ...
    @override
    def any(
        self, axis: None = None, out: None = None, keepdims: Literal[False] = False
    ) -> Never: ...
    @override
    def all(
        self, axis: None = None, out: None = None, keepdims: Literal[False] = False
    ) -> Never: ...
    @property
    @override
    def array(self) -> Categorical: ...  # type: ignore[override] # pyrefly: ignore[bad-override]
    def __new__(
        cls,
        data: Iterable[S1],
        categories: ListLike | None = None,
        ordered: bool | None = None,
        dtype: Dtype | None = None,
        copy: bool = False,
        name: Hashable | None = None,
    ) -> Self: ...
    # `item` might be `S1` but not one of the categories, thus changing
    # the return type from `CategoricalIndex` to `Index`.
    @override
    def insert(self, loc: int, item: object) -> Index: ...  # type: ignore[override] # pyrefly: ignore[bad-override] # ty: ignore[invalid-method-override]
