from collections.abc import Sequence
from typing import (
    Any,
    TypeAlias,
    overload,
)

from pandas.core.arrays.base import ExtensionArray as ExtensionArray
from pandas.core.indexes.base import Index
from pandas.core.series import Series
import pyarrow as pa
from typing_extensions import Self

from pandas._libs.interval import (
    Interval as Interval,
    IntervalMixin as IntervalMixin,
    _OrderableT,
)
from pandas._typing import (
    AnyArrayLike,
    DtypeArg,
    IntervalClosedType,
    NpDtype,
    Scalar,
    ScalarIndexer,
    SequenceIndexer,
    TakeIndexer,
    np_1darray_bool,
    np_1darray_object,
    np_ndarray,
)

from pandas.core.dtypes.dtypes import IntervalDtype

IntervalOrNA: TypeAlias = Interval | float

class IntervalArray(IntervalMixin, ExtensionArray):
    can_hold_na: bool = True
    def __new__(
        cls,
        data: Sequence[Interval[_OrderableT]] | AnyArrayLike,
        closed: IntervalClosedType | None = None,
        dtype: DtypeArg | None = None,
        copy: bool = False,
        verify_integrity: bool = True,
    ) -> Self: ...
    @classmethod
    def from_breaks(
        cls,
        breaks: (
            Sequence[_OrderableT]
            | np_ndarray
            | ExtensionArray
            | Index[_OrderableT]
            | Series[_OrderableT]
        ),
        closed: str = "right",
        copy: bool = False,
        dtype: DtypeArg | None = None,
    ) -> Self: ...
    @classmethod
    def from_arrays(
        cls,
        left: (
            Sequence[_OrderableT]
            | np_ndarray
            | ExtensionArray
            | Index[_OrderableT]
            | Series[_OrderableT]
        ),
        right: (
            Sequence[_OrderableT]
            | np_ndarray
            | ExtensionArray
            | Index[_OrderableT]
            | Series[_OrderableT]
        ),
        closed: IntervalClosedType = "right",
        copy: bool = False,
        dtype: DtypeArg | None = None,
    ) -> Self: ...
    @classmethod
    def from_tuples(
        cls,
        data: Sequence[tuple[_OrderableT, _OrderableT]] | np_ndarray,
        closed: IntervalClosedType = "right",
        copy: bool = False,
        dtype: DtypeArg | None = None,
    ) -> Self: ...
    def __array__(
        self, dtype: NpDtype | None = None, copy: bool | None = None
    ) -> np_1darray_object: ...
    @overload
    def __getitem__(self, item: ScalarIndexer) -> IntervalOrNA: ...
    @overload
    def __getitem__(self, item: SequenceIndexer) -> Self: ...
    def __eq__(self, other: object) -> np_1darray_bool: ...  # type: ignore[override]  # pyright: ignore[reportIncompatibleMethodOverride]  # pyrefly: ignore[bad-override]  # ty: ignore[invalid-method-override]
    def __ne__(self, other: object) -> np_1darray_bool: ...  # type: ignore[override]  # pyright: ignore[reportIncompatibleMethodOverride]  # pyrefly: ignore[bad-override]  # ty: ignore[invalid-method-override]
    @property
    def dtype(self) -> IntervalDtype: ...
    @property
    def nbytes(self) -> int: ...
    @property
    def size(self) -> int: ...
    def shift(self, periods: int = 1, fill_value: object = ...) -> IntervalArray: ...
    def take(  # type: ignore[override] # pyright: ignore[reportIncompatibleMethodOverride] # pyrefly: ignore[bad-param-name-override] # ty: ignore[invalid-method-override]
        self,
        indices: TakeIndexer,
        *,
        allow_fill: bool = False,
        fill_value: Interval | None = None,
        axis: None = None,  # only for compatibility, does nothing
        **kwargs: Any,
    ) -> Self: ...
    @property
    def left(self) -> Index: ...
    @property
    def right(self) -> Index: ...
    @property
    def closed(self) -> bool: ...
    def set_closed(self, closed: IntervalClosedType) -> Self: ...
    @property
    def length(self) -> Index: ...
    @property
    def mid(self) -> Index: ...
    @property
    def is_non_overlapping_monotonic(self) -> bool: ...
    def __arrow_array__(
        self, type: DtypeArg | None = None
    ) -> pa.ExtensionArray[Any]: ...
    def to_tuples(self, na_tuple: bool = True) -> np_1darray_object: ...
    @overload
    def contains(self, other: Series) -> Series[bool]: ...
    @overload
    def contains(
        self, other: Scalar | ExtensionArray | Index | np_ndarray
    ) -> np_1darray_bool: ...
    def overlaps(self, other: Interval) -> bool: ...
