from enum import Enum

class ParserBase:
    class BadLineHandleMethod(Enum):
        ERROR: int
        WARN: int
        SKIP: int
    names = ...  # Incomplete
    orig_names = ...  # Incomplete
    prefix = ...  # Incomplete
    index_col = ...  # Incomplete
    unnamed_cols = ...  # Incomplete
    index_names = ...  # Incomplete
    col_names = ...  # Incomplete
    parse_dates = ...  # Incomplete
    date_parser = ...  # Incomplete
    dayfirst = ...  # Incomplete
    keep_date_col = ...  # Incomplete
    na_values = ...  # Incomplete
    na_fvalues = ...  # Incomplete
    na_filter = ...  # Incomplete
    keep_default_na = ...  # Incomplete
    dtype = ...  # Incomplete
    converters = ...  # Incomplete
    true_values = ...  # Incomplete
    false_values = ...  # Incomplete
    mangle_dupe_cols = ...  # Incomplete
    infer_datetime_format = ...  # Incomplete
    cache_dates = ...  # Incomplete
    header = ...  # Incomplete
    on_bad_lines = ...  # Incomplete
    def __init__(self, kwds) -> None: ...
    def close(self) -> None: ...

parser_defaults = ...  # Incomplete

def is_index_col(col) -> bool: ...
