from collections.abc import (
    Iterator,
)
from typing import (
    Generic,
)

import numpy as np

from pandas._typing import (
    AxisInt,
    NDFrameT,
    npt,
)

class DataSplitter(Generic[NDFrameT]):
    data: NDFrameT
    labels: npt.NDArray[np.intp]
    ngroups: int
    axis: AxisInt
    def __iter__(self) -> Iterator[NDFrameT]: ...
