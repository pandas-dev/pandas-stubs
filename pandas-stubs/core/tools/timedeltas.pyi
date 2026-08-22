from datetime import timedelta
from typing import overload

from pandas import Index
from pandas.core.indexes.timedeltas import TimedeltaIndex
from pandas.core.series import Series

from pandas._libs.tslibs import Timedelta
from pandas._libs.tslibs.timedeltas import TimeDeltaUnitChoices
from pandas._typing import (
    ArrayLike,
    RaiseCoerce,
    SequenceNotStr,
)

@overload
def to_timedelta(
    arg: str | float | timedelta,
    unit: TimeDeltaUnitChoices | None = None,
    errors: RaiseCoerce = "raise",
) -> Timedelta: ...
@overload
def to_timedelta(
    arg: Series,
    unit: TimeDeltaUnitChoices | None = None,
    errors: RaiseCoerce = "raise",
) -> Series[Timedelta]: ...
@overload
def to_timedelta(
    arg: SequenceNotStr[str | float | timedelta] | range | ArrayLike | Index,
    unit: TimeDeltaUnitChoices | None = None,
    errors: RaiseCoerce = "raise",
) -> TimedeltaIndex: ...
