from datetime import timedelta
from typing import (
    ClassVar,
    Literal,
    NamedTuple,
    TypeVar,
    Union,
    overload,
)

import numpy as np
from typing_extensions import TypeAlias

from pandas._libs.tslibs import (
    BaseOffset,
    NaTType,
    Tick,
)
from pandas._typing import npt

class Components(NamedTuple):
    days: int
    hours: int
    minutes: int
    seconds: int
    milliseconds: int
    microseconds: int
    nanoseconds: int

# This should be kept consistent with the keys in the dict timedelta_abbrevs
# in pandas/_libs/tslibs/timedeltas.pyx
TimeDeltaUnitChoices: TypeAlias = Literal[
    "W",
    "w",
    "D",
    "d",
    "days",
    "day",
    "hours",
    "hour",
    "hr",
    "h",
    "m",
    "minute",
    "min",
    "minutes",
    "t",
    "s",
    "seconds",
    "sec",
    "second",
    "ms",
    "milliseconds",
    "millisecond",
    "milli",
    "millis",
    "l",
    "us",
    "microseconds",
    "microsecond",
    "µs",
    "micro",
    "micros",
    "u",
    "ns",
    "nanoseconds",
    "nano",
    "nanos",
    "nanosecond",
    "n",
]

UnitChoices: TypeAlias = Union[
    TimeDeltaUnitChoices,
    Literal[
        "Y",
        "y",
        "M",
    ],
]

_S = TypeVar("_S", bound=timedelta)

class Timedelta(timedelta):
    min: ClassVar[Timedelta]
    max: ClassVar[Timedelta]
    resolution: ClassVar[Timedelta]
    value: int
    def __new__(
        cls: type[_S],
        value: str | int | Timedelta | timedelta | np.timedelta64 = ...,
        unit: TimeDeltaUnitChoices = ...,
        *,
        days: float | np.integer | np.floating = ...,
        seconds: float | np.integer | np.floating = ...,
        microseconds: float | np.integer | np.floating = ...,
        milliseconds: float | np.integer | np.floating = ...,
        minutes: float | np.integer | np.floating = ...,
        hours: float | np.integer | np.floating = ...,
        weeks: float | np.integer | np.floating = ...,
    ) -> _S: ...
    # GH 46171
    # While Timedelta can return pd.NaT, having the constructor return
    # a Union with NaTType makes things awkward for users of pandas
    @property
    def days(self) -> int: ...
    @property
    def nanoseconds(self) -> int: ...
    @property
    def seconds(self) -> int: ...
    @property
    def microseconds(self) -> int: ...
    def total_seconds(self) -> float: ...
    def to_pytimedelta(self) -> timedelta: ...
    def to_timedelta64(self) -> np.timedelta64: ...
    @property
    def asm8(self) -> np.timedelta64: ...
    # TODO: round/floor/ceil could return NaT?
    def round(self: _S, freq: str | BaseOffset) -> _S: ...
    def floor(self: _S, freq: str | BaseOffset) -> _S: ...
    def ceil(self: _S, freq: str | BaseOffset) -> _S: ...
    @property
    def resolution_string(self) -> str: ...
    def __add__(self, other: timedelta) -> Timedelta: ...
    def __radd__(self, other: timedelta) -> Timedelta: ...
    def __sub__(self, other: timedelta) -> Timedelta: ...
    def __rsub__(self, other: timedelta) -> Timedelta: ...
    def __neg__(self) -> Timedelta: ...
    def __pos__(self) -> Timedelta: ...
    def __abs__(self) -> Timedelta: ...
    def __mul__(self, other: float) -> Timedelta: ...
    def __rmul__(self, other: float) -> Timedelta: ...
    # error: Signature of "__floordiv__" incompatible with supertype "timedelta"
    @overload  # type: ignore[override]
    def __floordiv__(self, other: timedelta) -> int: ...
    @overload
    def __floordiv__(self, other: float) -> Timedelta: ...
    @overload
    def __floordiv__(
        self, other: npt.NDArray[np.timedelta64]
    ) -> npt.NDArray[np.intp]: ...
    @overload
    def __floordiv__(
        self, other: npt.NDArray[np.number]
    ) -> npt.NDArray[np.timedelta64] | Timedelta: ...
    @overload
    def __rfloordiv__(self, other: timedelta | str) -> int: ...
    @overload
    def __rfloordiv__(self, other: NaTType | None) -> NaTType: ...
    @overload
    def __rfloordiv__(self, other: np.ndarray) -> npt.NDArray[np.timedelta64]: ...
    @overload
    def __truediv__(self, other: timedelta) -> float: ...
    @overload
    def __truediv__(self, other: float) -> Timedelta: ...
    def __mod__(self, other: timedelta) -> Timedelta: ...
    def __divmod__(self, other: timedelta) -> tuple[int, Timedelta]: ...
    def __le__(self, other: timedelta) -> bool: ...
    def __lt__(self, other: timedelta) -> bool: ...
    def __ge__(self, other: timedelta) -> bool: ...
    def __gt__(self, other: timedelta) -> bool: ...
    def __hash__(self) -> int: ...
    def isoformat(self) -> str: ...
    def to_numpy(self) -> np.timedelta64: ...
    @property
    def components(self) -> Components: ...
    def view(self, dtype: npt.DTypeLike = ...) -> object: ...
