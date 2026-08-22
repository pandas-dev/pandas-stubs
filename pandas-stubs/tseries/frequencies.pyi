from pandas.core.indexes.datetimes import DatetimeIndex
from pandas.core.indexes.timedeltas import TimedeltaIndex
from pandas.core.series import Series

from pandas._libs.tslibs.offsets import to_offset as to_offset

def infer_freq(index: Series | DatetimeIndex | TimedeltaIndex) -> str | None: ...
