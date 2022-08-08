# from pandas.core.dtypes.common import is_list_like as is_list_like, is_scalar as is_scalar
import dataclasses
from typing import Hashable

@dataclasses.dataclass(order=True, frozen=True)
class OutputKey:
    label: Hashable
    position: int
