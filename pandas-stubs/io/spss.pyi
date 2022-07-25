from __future__ import annotations

from typing import Sequence

from pandas.core.frame import DataFrame

from pandas._typing import FilePath

def read_spss(
    path: FilePath,
    usecols: Sequence[str] | None = ...,
    convert_categoricals: bool = ...,
) -> DataFrame: ...
