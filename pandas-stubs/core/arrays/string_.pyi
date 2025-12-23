from typing import (
    Any,
    Literal,
)

from pandas.core.arrays import PandasArray
from pandas.core.series import Series

from pandas._libs.missing import NAType
from pandas._typing import (
    AnyArrayLike,
    DtypeArg,
)

from pandas.core.dtypes.base import ExtensionDtype

class StringDtype(ExtensionDtype):
    def __init__(self, storage: Literal["python", "pyarrow"] | None = None) -> None: ...
    @property
    def na_value(self) -> NAType: ...

class StringArray(PandasArray):
    def __init__(self, values: AnyArrayLike, copy: bool = False) -> None: ...
    def __arrow_array__(self, type: DtypeArg | None = None) -> Any: ...
    def __setitem__(self, key: Any, value: Any) -> None: ...
    def value_counts(self, dropna: bool = True) -> Series[int]: ...
