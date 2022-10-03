from datetime import timedelta
from typing import (
    Sequence,
    overload,
)

from pandas import Index
from pandas.core.indexes.timedeltas import TimedeltaIndex
from pandas.core.series import (
    Series,
    TimedeltaSeries,
)

from pandas._libs.tslibs import Timedelta
from pandas._libs.tslibs.timedeltas import UnitChoices
from pandas._typing import (
    ArrayLike,
    DateTimeErrorChoices,
)

@overload
def to_timedelta(
    arg: str | float | timedelta,
    # TODO: Check all UnitChoices are valid
    unit: UnitChoices | None = ...,
    errors: DateTimeErrorChoices = ...,
) -> Timedelta: ...
@overload
def to_timedelta(
    arg: Series,
    unit: UnitChoices | None = ...,
    errors: DateTimeErrorChoices = ...,
) -> TimedeltaSeries: ...
@overload
def to_timedelta(
    arg: Sequence[float | timedelta]
    | list[str | float | timedelta]
    | tuple[str | float | timedelta, ...]
    | range
    | ArrayLike
    | Index,
    unit: UnitChoices | None = ...,
    errors: DateTimeErrorChoices = ...,
) -> TimedeltaIndex: ...
