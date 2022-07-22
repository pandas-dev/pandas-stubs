from pandas.core.frame import DataFrame

from pandas._typing import ReadBuffer
from pandas.compat._optional import (
    import_optional_dependency as import_optional_dependency,
)

from pandas.core.dtypes.inference import is_integer as is_integer

from pandas.io.parsers.base_parser import ParserBase

class ArrowParserWrapper(ParserBase):
    kwds = ...  # Incomplete
    src = ...  # Incomplete
    def __init__(self, src: ReadBuffer[bytes], **kwds) -> None: ...
    def read(self) -> DataFrame: ...
