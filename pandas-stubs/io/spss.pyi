from typing import Any

from pandas.core.frame import DataFrame

from pandas._libs.lib import NoDefault
from pandas._typing import (
    DtypeBackend,
    FilePath,
    HashableT,
)

def read_spss(
    path: FilePath,
    usecols: list[HashableT] | None = None,
    convert_categoricals: bool = True,
    dtype_backend: DtypeBackend | NoDefault = "numpy_nullable",
    **kwargs: Any,
) -> DataFrame: ...
