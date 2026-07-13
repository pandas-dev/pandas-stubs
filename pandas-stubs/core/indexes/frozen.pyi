from typing import (
    Generic,
    TypeVar,
)

from pandas.core.base import PandasObject

_T = TypeVar("_T")

class FrozenList(PandasObject, list[_T], Generic[_T]): ...
