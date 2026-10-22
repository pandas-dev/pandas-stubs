from collections.abc import (
    Hashable,
    Iterable,
)
from typing import (
    Any,
    Literal,
    Never,
    Self,
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
    Dtype,
    ListLike,
    np_1darray_intp,
)

class CategoricalIndex(
    ExtensionIndex[S1],
    PandasDelegate,
    CategoricalMethods[CategoricalIndex[S1], Any],
):
    codes: np_1darray_intp = ...
    @property
    @override
    def categories(self) -> Index[S1]: ...
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
