from typing import (
    Literal,
    Sequence,
)

import numpy as np
import pandas as pd
from pandas.core.arrays import PandasArray

from pandas._typing import (
    AnyArrayLike,
    type_t,
)

from pandas.core.dtypes.base import ExtensionDtype

class StringDtype(ExtensionDtype):
    def __init__(self, storage: Literal["python", "pyarrow"] | None) -> None: ...
    @property
    def type(self) -> type_t: ...
    def __from_arrow__(self, array): ...

class StringArray(PandasArray):
    def __init__(
        self,
        # Also pd.NA and np.nan but not possible it seems
        values: AnyArrayLike | Sequence[str | None],
        copy: bool = ...,
    ) -> None: ...
    def __arrow_array__(self, type=...): ...
    def __setitem__(self, key, value) -> None: ...
    def fillna(self, value=..., method=..., limit=...): ...
    def astype(self, dtype, copy: bool = ...): ...
    def value_counts(self, dropna: bool = ...): ...
