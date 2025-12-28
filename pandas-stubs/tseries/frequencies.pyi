from typing import overload

from pandas import (
    DatetimeIndex,
    Series,
    TimedeltaIndex,
)

from pandas._typing import Frequency

from pandas.tseries.offsets import BaseOffset

@overload
def to_offset(freq: None, is_period: bool = False) -> None: ...
@overload
def to_offset(freq: Frequency, is_period: bool = False) -> BaseOffset: ...
def infer_freq(index: Series | DatetimeIndex | TimedeltaIndex) -> str | None: ...
