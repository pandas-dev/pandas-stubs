from typing import (
    Any,
    Generic,
    Literal,
    TypeVar,
    overload,
)

import numpy as np
from pandas import (
    IntervalIndex,
    Series,
    Timedelta,
    Timestamp,
)

from pandas._typing import (
    IntervalClosedType,
    IntervalT,
    np_ndarray_bool,
    npt,
)

VALID_CLOSED: frozenset[str]

_OrderableScalarT = TypeVar("_OrderableScalarT", int, float)
_OrderableTimesT = TypeVar("_OrderableTimesT", Timestamp, Timedelta)
_OrderableT = TypeVar("_OrderableT", int, float, Timestamp, Timedelta)

class _LengthDescriptor:
    @overload
    def __get__(
        self, instance: Interval[_OrderableScalarT], owner: Any
    ) -> _OrderableScalarT: ...
    @overload
    def __get__(
        self, instance: Interval[_OrderableTimesT], owner: Any
    ) -> Timedelta: ...
    @overload
    def __get__(self, instance: IntervalTree, owner: Any) -> np.ndarray: ...

class _MidDescriptor:
    @overload
    def __get__(self, instance: Interval[_OrderableScalarT], owner: Any) -> float: ...
    @overload
    def __get__(
        self, instance: Interval[_OrderableTimesT], owner: Any
    ) -> _OrderableTimesT: ...
    @overload
    def __get__(self, instance: IntervalTree, owner: Any) -> np.ndarray: ...

class IntervalMixin:
    @property
    def closed_left(self) -> bool: ...
    @property
    def closed_right(self) -> bool: ...
    @property
    def open_left(self) -> bool: ...
    @property
    def open_right(self) -> bool: ...
    @property
    def is_empty(self) -> bool: ...

class Interval(IntervalMixin, Generic[_OrderableT]):
    @property
    def left(self: Interval[_OrderableT]) -> _OrderableT: ...
    @property
    def right(self: Interval[_OrderableT]) -> _OrderableT: ...
    @property
    def closed(self) -> IntervalClosedType: ...
    mid: _MidDescriptor
    length: _LengthDescriptor
    def __init__(
        self,
        left: _OrderableT,
        right: _OrderableT,
        closed: IntervalClosedType = ...,
    ): ...
    def __hash__(self) -> int: ...
    @overload
    def __contains__(self: Interval[_OrderableTimesT], _OrderableTimesT) -> bool: ...
    @overload
    def __contains__(self: Interval[_OrderableScalarT], key: float) -> bool: ...
    @overload
    def __add__(
        self: Interval[_OrderableTimesT], y: Timedelta
    ) -> Interval[_OrderableTimesT]: ...
    @overload
    def __add__(
        self: Interval[int], y: _OrderableScalarT
    ) -> Interval[_OrderableScalarT]: ...
    @overload
    def __add__(self: Interval[float], y: float) -> Interval[float]: ...
    @overload
    def __radd__(
        self: Interval[_OrderableTimesT], y: Timedelta
    ) -> Interval[_OrderableTimesT]: ...
    @overload
    def __radd__(
        self: Interval[int], y: _OrderableScalarT
    ) -> Interval[_OrderableScalarT]: ...
    @overload
    def __radd__(self: Interval[float], y: float) -> Interval[float]: ...
    @overload
    def __sub__(
        self: Interval[_OrderableTimesT], y: Timedelta
    ) -> Interval[_OrderableTimesT]: ...
    @overload
    def __sub__(
        self: Interval[int], y: _OrderableScalarT
    ) -> Interval[_OrderableScalarT]: ...
    @overload
    def __sub__(self: Interval[float], y: float) -> Interval[float]: ...
    @overload
    def __rsub__(
        self: Interval[_OrderableTimesT], y: Timedelta
    ) -> Interval[_OrderableTimesT]: ...
    @overload
    def __rsub__(
        self: Interval[int], y: _OrderableScalarT
    ) -> Interval[_OrderableScalarT]: ...
    @overload
    def __rsub__(self: Interval[float], y: float) -> Interval[float]: ...
    @overload
    def __mul__(
        self: Interval[int], y: _OrderableScalarT
    ) -> Interval[_OrderableScalarT]: ...
    @overload
    def __mul__(self: Interval[float], y: float) -> Interval[float]: ...
    @overload
    def __mul__(self: Interval[Timedelta], y: float) -> Interval[Timedelta]: ...
    @overload
    def __rmul__(
        self: Interval[int], y: _OrderableScalarT
    ) -> Interval[_OrderableScalarT]: ...
    @overload
    def __rmul__(self: Interval[float], y: float) -> Interval[float]: ...
    @overload
    def __rmul__(self: Interval[Timedelta], y: float) -> Interval[Timedelta]: ...
    @overload
    def __truediv__(self: Interval[int], y: _OrderableScalarT) -> Interval[float]: ...
    @overload
    def __truediv__(self: Interval[float], y: float) -> Interval[float]: ...
    @overload
    def __truediv__(self: Interval[Timedelta], y: float) -> Interval[Timedelta]: ...
    @overload
    def __floordiv__(
        self: Interval[int], y: _OrderableScalarT
    ) -> Interval[_OrderableScalarT]: ...
    @overload
    def __floordiv__(self: Interval[float], y: float) -> Interval[float]: ...
    @overload
    def __floordiv__(self: Interval[Timedelta], y: float) -> Interval[Timedelta]: ...
    @overload
    def overlaps(self: Interval[_OrderableT], other: Interval[_OrderableT]) -> bool: ...
    @overload
    def overlaps(self: Interval[int], other: Interval[float]) -> bool: ...
    @overload
    def overlaps(self: Interval[float], other: Interval[int]) -> bool: ...
    @overload
    def __gt__(self, other: Interval[_OrderableT]) -> bool: ...
    @overload
    def __gt__(self: IntervalT, other: IntervalIndex[IntervalT]) -> np_ndarray_bool: ...
    @overload
    def __gt__(self, other: Series[_OrderableT]) -> Series[bool]: ...
    @overload
    def __lt__(self, other: Interval[_OrderableT]) -> bool: ...
    @overload
    def __lt__(self: IntervalT, other: IntervalIndex[IntervalT]) -> np_ndarray_bool: ...
    @overload
    def __lt__(self, other: Series[_OrderableT]) -> Series[bool]: ...
    @overload
    def __ge__(self, other: Interval[_OrderableT]) -> bool: ...
    @overload
    def __ge__(self: IntervalT, other: IntervalIndex[IntervalT]) -> np_ndarray_bool: ...
    @overload
    def __ge__(self, other: Series[_OrderableT]) -> Series[bool]: ...
    @overload
    def __le__(self, other: Interval[_OrderableT]) -> bool: ...
    @overload
    def __le__(self: IntervalT, other: IntervalIndex[IntervalT]) -> np_ndarray_bool: ...
    @overload
    def __eq__(self, other: Interval[_OrderableT]) -> bool: ...  # type: ignore[misc]
    @overload
    def __eq__(self: IntervalT, other: IntervalIndex[IntervalT]) -> np_ndarray_bool: ...  # type: ignore[misc]
    @overload
    def __eq__(self, other: Series[_OrderableT]) -> Series[bool]: ...  # type: ignore[misc]
    @overload
    def __eq__(self, other: object) -> Literal[False]: ...
    @overload
    def __ne__(self, other: Interval[_OrderableT]) -> bool: ...  # type: ignore[misc]
    @overload
    def __ne__(self: IntervalT, other: IntervalIndex[IntervalT]) -> np_ndarray_bool: ...  # type: ignore[misc]
    @overload
    def __ne__(self, other: Series[_OrderableT]) -> Series[bool]: ...  # type: ignore[misc]
    @overload
    def __ne__(self, other: object) -> Literal[True]: ...

class IntervalTree(IntervalMixin):
    def __init__(
        self,
        left: np.ndarray,
        right: np.ndarray,
        closed: IntervalClosedType = ...,
        leaf_size: int = ...,
    ): ...
    def get_indexer(self, target) -> npt.NDArray[np.intp]: ...
    def get_indexer_non_unique(
        self, target
    ) -> tuple[npt.NDArray[np.intp], npt.NDArray[np.intp]]: ...
    _na_count: int
    @property
    def is_overlapping(self) -> bool: ...
    @property
    def is_monotonic_increasing(self) -> bool: ...
    def clear_mapping(self) -> None: ...
