from typing import Sequence

from pandas.core.frame import DataFrame

from pandas._typing import FilePathOrBuffer

def read_spss(
    path: FilePathOrBuffer,
    usecols: Sequence[str] | None = ...,
    convert_categoricals: bool = ...,
) -> DataFrame: ...
