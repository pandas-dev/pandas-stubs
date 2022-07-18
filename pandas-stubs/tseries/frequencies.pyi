from __future__ import annotations

from pandas.tseries.offsets import DateOffset as DateOffset

def get_period_alias(offset_str: str) -> str | None: ...
def to_offset(freq) -> DateOffset | None: ...
def get_offset(name: str) -> DateOffset: ...
def infer_freq(index, warn: bool = ...) -> str | None: ...
