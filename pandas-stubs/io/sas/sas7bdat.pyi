import numpy as np
import pandas as pd
from pandas import DataFrame

from pandas._typing import (
    CompressionOptions as CompressionOptions,
    FilePath as FilePath,
    Label,
    ReadBuffer,
)

from pandas.io.sas.sasreader import ReaderBase

class SAS7BDATReader(ReaderBase):
    def __init__(
        self,
        path_or_buf: FilePath | ReadBuffer[bytes],
        index: Label = ...,
        convert_dates: bool = ...,
        blank_missing: bool = ...,
        chunksize: int | None = ...,
        encoding: str | None = ...,
        convert_text: bool = ...,
        convert_header_text: bool = ...,
        compression: CompressionOptions = ...,
    ) -> None: ...
    def close(self) -> None: ...
    def __next__(self) -> DataFrame: ...
    def read(self, nrows: int | None = ...) -> DataFrame: ...
