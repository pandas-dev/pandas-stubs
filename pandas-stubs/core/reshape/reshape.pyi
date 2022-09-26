import numpy as np
from pandas import (
    DataFrame,
    Series,
)

def unstack(obj: Series | DataFrame, level, fill_value: object | None = ...): ...
def stack(frame: DataFrame, level: int = ..., dropna: bool = ...): ...
def stack_multiple(frame, level, dropna: bool = ...): ...
