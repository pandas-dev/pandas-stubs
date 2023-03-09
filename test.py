from __future__ import annotations

import pandas as pd


def some_function(param: pd.Series[str]) -> pd.Series[float]:
    return param.astype(float)


some_value = pd.Series([1.0, 1.0, 1.0], dtype=float)
some_function(some_value)