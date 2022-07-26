from typing import Any

from pandas.core.frame import DataFrame

def convert_to_line_delimits(s: Any): ...
def nested_to_record(
    ds: Any,
    prefix: str = ...,
    sep: str = ...,
    level: int = ...,
    max_level: int | None = ...,
) -> Any: ...
def json_normalize(
    data: dict | list[dict],
    record_path: str | list | None = ...,
    meta: str | list[str | list[str]] | None = ...,
    meta_prefix: str | None = ...,
    record_prefix: str | None = ...,
    errors: str = ...,
    sep: str = ...,
    max_level: int | None = ...,
) -> DataFrame: ...
