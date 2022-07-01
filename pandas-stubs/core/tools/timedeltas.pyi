# def to_timedelta(arg, unit: str = ..., errors: str = ...): ...
from datetime import timedelta
from typing import (
    Literal,
    Optional,
    Union,
    overload,
)

from pandas.core.indexes.timedeltas import TimedeltaIndex
from pandas.core.series import (
    Series as Series,
    TimedeltaSeries,
)

from pandas._libs.tslibs import Timedelta
from pandas._libs.tslibs.timedeltas import UnitChoices
from pandas._typing import (
    ArrayLike,
    DateTimeErrorChoices,
    Index as Index,
)

# Copied from pandas/_libs/tslibs/timedeltas.pyx

@overload
def to_timedelta(
    arg: Union[str, int, float, timedelta],
    unit: Optional[UnitChoices] = ...,
    errors: DateTimeErrorChoices = ...,
) -> Timedelta: ...
@overload
def to_timedelta(
    arg: Series,
    unit: Optional[UnitChoices] = ...,
    errors: DateTimeErrorChoices = ...,
) -> TimedeltaSeries: ...
@overload
def to_timedelta(
    arg: Union[list, tuple, range, ArrayLike, Index],
    unit: Optional[UnitChoices] = ...,
    errors: DateTimeErrorChoices = ...,
) -> TimedeltaIndex: ...
