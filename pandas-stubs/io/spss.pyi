from typing import (
    Optional,
    Sequence,
)

from pandas.core.frame import DataFrame

from pandas._typing import FilePathOrBuffer

def read_spss(
    path: FilePathOrBuffer,
    usecols: Optional[Sequence[str]] = ...,
    convert_categoricals: bool = ...,
) -> DataFrame: ...
