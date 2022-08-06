from typing import (
    Any,
    Callable,
)

from pandas._typing import Scalar

def generate_numba_apply_func(
    args: tuple,
    kwargs: dict[str, Any],
    func: Callable[..., Scalar],
    engine_kwargs: dict[str, bool] | None,
): ...
