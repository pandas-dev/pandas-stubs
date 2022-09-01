from typing import Any

from pandas._typing import (
    ArrayLike as ArrayLike,
    DtypeArg as DtypeArg,
    Scalar as Scalar,
)

class ParserBase:
    names: Any  # Incomplete
    orig_names: Any  # Incomplete
    prefix: Any  # Incomplete
    index_col: Any  # Incomplete
    unnamed_cols: Any  # Incomplete
    index_names: Any  # Incomplete
    col_names: Any  # Incomplete
    parse_dates: Any  # Incomplete
    date_parser: Any  # Incomplete
    dayfirst: Any  # Incomplete
    keep_date_col: Any  # Incomplete
    na_values: Any  # Incomplete
    na_fvalues: Any  # Incomplete
    na_filter: Any  # Incomplete
    keep_default_na: Any  # Incomplete
    dtype: Any  # Incomplete
    converters: Any  # Incomplete
    true_values: Any  # Incomplete
    false_values: Any  # Incomplete
    mangle_dupe_cols: Any  # Incomplete
    infer_datetime_format: Any  # Incomplete
    cache_dates: Any  # Incomplete
    header: Any  # Incomplete
    on_bad_lines: Any  # Incomplete
    def __init__(self, kwds) -> None: ...
    def close(self) -> None: ...
