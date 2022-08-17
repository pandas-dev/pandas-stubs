from typing import Any

from pandas.core.frame import DataFrame

def read_clipboard(sep: str = ..., **kwargs: Any) -> DataFrame: ...
def to_clipboard(
    obj, excel: bool = ..., sep: str | None = ..., **kwargs: Any
) -> None: ...
