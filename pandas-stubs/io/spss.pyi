from pandas.core.frame import DataFrame

from pandas._libs.lib import NoDefaultDoNotUse
from pandas._typing import (
    DtypeBackend,
    FilePath,
    HashableT,
)

def read_spss(
    path: FilePath,
    usecols: list[HashableT] | None = ...,
    convert_categoricals: bool = ...,
    dtype_backend: DtypeBackend | NoDefaultDoNotUse = ...,
) -> DataFrame: ...
