# def to_timedelta(arg, unit: str = ..., errors: str = ...): ...
from datetime import timedelta
from typing import overload

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

# Copied from pandas/_libs/tslibs/timedeltas.pyx

@overload
def to_timedelta(
    arg: str | float | timedelta,
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
    arg: list | tuple | range | ArrayLike | Index,
    unit: UnitChoices | None = ...,
    errors: DateTimeErrorChoices = ...,
) -> TimedeltaIndex: ...
