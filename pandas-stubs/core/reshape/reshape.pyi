from __future__ import annotations

import numpy as np
from pandas.core.frame import DataFrame

class _Unstacker:
    values = ...
    value_columns = ...
    fill_value = ...
    constructor = ...
    index = ...
    level = ...
    lift = ...
    new_index_levels = ...
    new_index_names = ...
    removed_name = ...
    removed_level = ...
    removed_level_full = ...
    def __init__(
        self,
        values: np.ndarray,
        index,
        level=...,
        value_columns=...,
        fill_value=...,
        constructor=...,
    ) -> None: ...
    def get_result(self): ...
    def get_new_values(self): ...
    def get_new_columns(self): ...
    def get_new_index(self): ...

def unstack(obj, level, fill_value=...): ...
def stack(frame, level: int = ..., dropna: bool = ...): ...
def stack_multiple(frame, level, dropna: bool = ...): ...
def get_dummies(
    data,
    prefix=...,
    prefix_sep=...,
    dummy_na=...,
    columns=...,
    sparse=...,
    drop_first=...,
    dtype=...,
) -> DataFrame: ...
