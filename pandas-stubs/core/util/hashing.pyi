import numpy as np
from pandas import (
    DataFrame,
    Index,
    Series,
)

from pandas._typing import (
    ArrayLike,
    npt,
)

def hash_pandas_object(
    obj: Index | Series | DataFrame,
    index: bool = True,
    encoding: str = "utf8",
    hash_key: str | None = ...,
    categorize: bool = True,
) -> Series: ...
def hash_array(
    vals: ArrayLike,
    encoding: str = "utf8",
    hash_key: str = ...,
    categorize: bool = True,
) -> npt.NDArray[np.uint64]: ...
