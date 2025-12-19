from collections.abc import Sequence
from datetime import timedelta

from pandas.core.arrays.datetimelike import (
    DatetimeLikeArrayMixin,
    TimelikeOps,
)
from typing_extensions import Self

from pandas._typing import (
    AnyArrayLike,
    DtypeArg,
    Frequency,
)

class TimedeltaArray(DatetimeLikeArrayMixin, TimelikeOps):
    __array_priority__: int = ...
    def __new__(
        cls,
        values: AnyArrayLike,
        dtype: DtypeArg | None = None,
        freq: Frequency | None = None,
        copy: bool = False,
    ) -> Self: ...
    def total_seconds(self) -> int: ...
    def to_pytimedelta(self) -> Sequence[timedelta]: ...
    days: int = ...
    seconds: int = ...
    microseconds: int = ...
    nanoseconds: int = ...
    @property
    def components(self) -> int: ...
