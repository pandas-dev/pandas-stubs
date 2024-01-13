from collections.abc import (
    Callable,
    Mapping,
)
from typing import Any

from pandas._libs.properties import cache_readonly as cache_readonly
from pandas._typing import (
    F,
    T,
)

__all__ = [
    "Appender",
    "cache_readonly",
    "deprecate",
    "deprecate_kwarg",
    "deprecate_nonkeyword_arguments",
    "doc",
    "future_version_msg",
    "Substitution",
]

def deprecate(
    name: str,
    alternative: Callable[..., Any],
    version: str,
    alt_name: str | None = ...,
    klass: type[Warning] | None = ...,
    stacklevel: int = ...,
    msg: str | None = ...,
) -> Callable[[F], F]: ...
def deprecate_kwarg(
    old_arg_name: str,
    new_arg_name: str | None,
    mapping: Mapping[Any, Any] | Callable[[Any], Any] | None = ...,
    stacklevel: int = ...,
) -> Callable[[F], F]: ...
def future_version_msg(version: str | None) -> str: ...
def deprecate_nonkeyword_arguments(
    version: str | None,
    allowed_args: list[str] | None = ...,
    name: str | None = ...,
) -> Callable[[F], F]: ...
def doc(*docstrings: str | Callable | None, **params) -> Callable[[F], F]: ...

class Substitution:
    params: Any
    def __init__(self, *args, **kwargs) -> None: ...
    def __call__(self, func: F) -> F: ...
    def update(self, *args, **kwargs) -> None: ...

class Appender:
    addendum: str | None
    join: str

    def __init__(
        self, addendum: str | None, join: str = ..., indents: int = ...
    ) -> None: ...
    def __call__(self, func: T) -> T: ...

def indent(text: str | None, indents: int = ...) -> str: ...
