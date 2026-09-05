from pandas.core.frame import DataFrame
from typing_extensions import override

from pandas.io.sas.sasreader import SASReader

class XportReader(SASReader):
    @override
    def close(self) -> None: ...
    @override
    def __next__(self) -> DataFrame: ...
    @override
    def read(self, nrows: int | None = None) -> DataFrame: ...
