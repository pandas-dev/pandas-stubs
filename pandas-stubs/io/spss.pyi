from typing import (
    Optional,
    Sequence,
)

from pandas.core.frame import DataFrame as DataFrame

from pandas._typing import FilePathOrBuffer as FilePathOrBuffer

def read_spss(
    path: FilePathOrBuffer,
    usecols: Optional[Sequence[str]] = ...,
    convert_categoricals: bool = ...,
) -> DataFrame: ...
