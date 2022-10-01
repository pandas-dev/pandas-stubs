from typing import (
    Any,
    Literal,
    Mapping,
)

from pandas import (
    DataFrame,
    Series,
)
from pandas.core.computation.ops import BinOp

from pandas._typing import (
    Scalar,
    npt,
)

def eval(
    expr: str | BinOp,
    parser: Literal["pandas", "python"] = ...,
    engine: Literal["python", "numexpr"] | None = ...,
    # Keyword only due to omitted deprecated argument
    *,
    local_dict: dict[str, Any] | None = ...,
    global_dict: dict[str, Any] | None = ...,
    resolvers: list[Mapping] | None = ...,
    level: int = ...,
    target: object | None = ...,
    inplace: bool = ...,
) -> npt.NDArray | Scalar | DataFrame | Series | None: ...
