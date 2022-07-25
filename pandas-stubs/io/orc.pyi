from pandas.core.frame import DataFrame

from pandas._typing import FilePathOrBuffer

def read_orc(
    path: FilePathOrBuffer, columns: list[str] | None = ..., **kwargs
) -> DataFrame: ...
