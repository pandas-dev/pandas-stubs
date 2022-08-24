import pandas as pd

from pandas._typing import (
    CompressionOptions,
    FilePath,
    Label,
    ReadBuffer,
)

from pandas.io.sas.sasreader import ReaderBase

class XportReader(ReaderBase):
    def __init__(
        self,
        filepath_or_buffer: FilePath | ReadBuffer[bytes],
        index: Label = ...,
        encoding: str | None = ...,
        chunksize: int | None = ...,
        compression: CompressionOptions = ...,
    ) -> None: ...
    def close(self) -> None: ...
    def __next__(self) -> pd.DataFrame: ...
    def read(self, nrows: int | None = ...) -> pd.DataFrame: ...
