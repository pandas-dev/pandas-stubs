from pandas.core.frame import DataFrame

from pandas._libs.lib import NoDefaultDoNotUse
from pandas._typing import (
    DtypeBackend,
    FilePath,
    HashableT,
)

def read_spss(
    path: FilePath,
    usecols: list[HashableT] | None = None,
    convert_categoricals: bool = True,
    dtype_backend: DtypeBackend | NoDefaultDoNotUse = "numpy_nullable",
) -> DataFrame: ...
