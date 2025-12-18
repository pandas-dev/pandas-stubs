from typing import (
    Any,
    TypeAlias,
    overload,
)

from pandas.core.arrays.base import ExtensionArray as ExtensionArray
from pandas.core.indexes.base import Index
from pandas.core.series import Series
from typing_extensions import Self

from pandas._libs.interval import (
    Interval as Interval,
    IntervalMixin as IntervalMixin,
)
from pandas._typing import (
    AnyArrayLike,
    Axis,
    DtypeArg,
    IntervalClosedType,
    NpDtype,
    Scalar,
    ScalarIndexer,
    SequenceIndexer,
    TakeIndexer,
    np_1darray,
    np_1darray_bool,
    np_ndarray,
)

IntervalOrNA: TypeAlias = Interval | float

class IntervalArray(IntervalMixin, ExtensionArray):
    can_hold_na: bool = ...
    def __new__(
        cls,
        data: AnyArrayLike,
        closed: IntervalClosedType = "right",
        dtype: DtypeArg | None = None,
        copy: bool = False,
        verify_integrity: bool = True,
    ) -> Self: ...
    @classmethod
    def from_breaks(
        cls,
        breaks: AnyArrayLike,
        closed: str = "right",
        copy: bool = False,
        dtype: DtypeArg | None = None,
    ) -> Self: ...
    @classmethod
    def from_arrays(
        cls,
        left: AnyArrayLike,
        right: AnyArrayLike,
        closed: str = "right",
        copy: bool = False,
        dtype: DtypeArg | None = None,
    ) -> Self: ...
    @classmethod
    def from_tuples(
        cls,
        data: AnyArrayLike,
        closed: str = "right",
        copy: bool = False,
        dtype: DtypeArg | None = None,
    ) -> Self: ...
    def __array__(
        self, dtype: NpDtype | None = None, copy: bool | None = None
    ) -> np_1darray: ...
    @overload
    def __getitem__(self, item: ScalarIndexer) -> IntervalOrNA: ...
    @overload
    def __getitem__(self, item: SequenceIndexer) -> Self: ...
    @property
    def nbytes(self) -> int: ...
    @property
    def size(self) -> int: ...
    def shift(self, periods: int = 1, fill_value: object = ...) -> IntervalArray: ...
    def take(  # type: ignore[override] # pyright: ignore[reportIncompatibleMethodOverride]
        self: Self,
        indices: TakeIndexer,
        *,
        allow_fill: bool = ...,
        fill_value: Interval | None = None,
        axis: Axis | None = None,
        **kwargs: Any,
    ) -> Self: ...
    @property
    def left(self) -> Index: ...
    @property
    def right(self) -> Index: ...
    @property
    def closed(self) -> bool: ...
    @property
    def length(self) -> Index: ...
    @property
    def mid(self) -> Index: ...
    @property
    def is_non_overlapping_monotonic(self) -> bool: ...
    @overload
    def contains(self, other: Series) -> Series[bool]: ...
    @overload
    def contains(
        self, other: Scalar | ExtensionArray | Index | np_ndarray
    ) -> np_1darray_bool: ...
    def overlaps(self, other: Interval) -> bool: ...
