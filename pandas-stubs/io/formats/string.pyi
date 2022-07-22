from pandas.io.formats.format import DataFrameFormatter
from pandas.io.formats.printing import pprint_thing as pprint_thing

class StringFormatter:
    fmt = ...  # Incomplete
    adj = ...  # Incomplete
    frame = ...  # Incomplete
    line_width = ...  # Incomplete
    def __init__(
        self, fmt: DataFrameFormatter, line_width: int | None = ...
    ) -> None: ...
    def to_string(self) -> str: ...
