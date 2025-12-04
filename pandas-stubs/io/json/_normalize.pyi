from collections.abc import (
    Mapping,
    Sequence,
)
from typing import Any

from pandas import DataFrame

from pandas._typing import IgnoreRaise

def json_normalize(
    data: Mapping[str, Any] | Sequence[Mapping[str, Any]],
    record_path: str | list[str] | None = None,
    meta: str | list[str | list[str]] | None = None,
    meta_prefix: str | None = None,
    record_prefix: str | None = None,
    errors: IgnoreRaise = "raise",
    sep: str = ".",
    max_level: int | None = None,
) -> DataFrame: ...
