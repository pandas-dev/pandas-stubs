from collections.abc import (
    Callable,
    Mapping,
)
from typing import Any

def deprecate(
    name: str,
    alternative: Callable[..., Any],
    version: str,
    alt_name: str | None = ...,
    klass: type[Warning] | None = ...,
    stacklevel: int = ...,
    msg: str | None = ...,
) -> Callable[..., Any]: ...
def deprecate_kwarg(
    old_arg_name: str,
    new_arg_name: str | None,
    mapping: Mapping[Any, Any] | Callable[[Any], Any] | None = ...,
    stacklevel: int = ...,
) -> Callable[..., Any]: ...
def rewrite_axis_style_signature(
    name: str, extra_params: list[tuple[str, Any]]
) -> Callable[..., Any]: ...
def indent(text: str | None, indents: int = ...) -> str: ...
