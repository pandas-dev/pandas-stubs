from __future__ import annotations

from typing import (
    Any,
    Optional,
    Set,
    Tuple,
)

ARITHMETIC_BINOPS: set[str] = ...
COMPARISON_BINOPS: set[str] = ...

def get_op_result_name(left: Any, right: Any): ...
def maybe_upcast_for_op(obj: Any, shape: tuple[int, ...]) -> Any: ...
def fill_binop(left: Any, right: Any, fill_value: Any): ...
def dispatch_to_series(
    left: Any,
    right: Any,
    func: Any,
    str_rep: Any | None = ...,
    axis: Any | None = ...,
): ...
