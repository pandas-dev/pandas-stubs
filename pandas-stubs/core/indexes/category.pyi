from collections.abc import (
    Hashable,
    Iterable,
)
from typing import Self

from pandas.core.accessor import PandasDelegate
from pandas.core.arrays.categorical import Categorical
from pandas.core.indexes.base import Index
from pandas.core.indexes.extension import ExtensionIndex
from typing_extensions import override

from pandas._typing import (
    S1,
    Dtype,
    ListLike,
    np_1darray_intp,
)

class CategoricalIndex(ExtensionIndex[S1], PandasDelegate):
    codes: np_1darray_intp = ...
    categories: Index[S1] = ...
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
    def insert(self, loc: int, item: object) -> Index: ...  # type: ignore[override] # pyrefly: ignore[bad-override]
