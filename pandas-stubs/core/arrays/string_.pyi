from typing import Literal

from pandas.core.arrays import PandasArray

from pandas._libs.missing import NAType
from pandas._typing import AnyArrayLike

from pandas.core.dtypes.base import ExtensionDtype

class StringDtype(ExtensionDtype):
    def __init__(self, storage: Literal["python", "pyarrow"] | None = None) -> None: ...
    @property
    def na_value(self) -> NAType: ...

class StringArray(PandasArray):
    def __init__(self, values: AnyArrayLike, copy: bool = False) -> None: ...
