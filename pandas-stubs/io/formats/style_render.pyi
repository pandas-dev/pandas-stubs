from typing import (
    Any,
    Callable,
    Literal,
    Optional,
    Sequence,
    TypedDict,
    Union,
)

import jinja2
from pandas import Index

from pandas._typing import (
    HashableT,
    Level,
)

BaseFormatter = Union[str, Callable[[object], str]]
ExtFormatter = Union[BaseFormatter, dict[Any, Optional[BaseFormatter]]]
CSSPair = tuple[str, Union[str, float]]
CSSList = list[CSSPair]
CSSProperties = Union[str, CSSList]

class CSSDict(TypedDict):
    selector: str
    props: CSSProperties

CSSStyles = list[CSSDict]
Subset = Union[slice, list[HashableT], Index]

class StylerRenderer:
    loader: jinja2.loaders.BaseLoader
    env: jinja2.environment.Environment
    template_html: jinja2.environment.Template
    template_html_table: jinja2.environment.Template
    template_html_style: jinja2.environment.Template
    template_latex: jinja2.environment.Template
    def format(
        self,
        formatter: ExtFormatter | None = ...,
        subset: Subset | None = ...,
        na_rep: str | None = ...,
        precision: int | None = ...,
        decimal: str = ...,
        thousands: str | None = ...,
        escape: str | None = ...,
        hyperlinks: Literal["html", "latex"] | None = ...,
    ) -> StylerRenderer: ...
    def format_index(
        self,
        formatter: ExtFormatter | None = ...,
        axis: int | Literal["index", "columns"] = ...,
        level: Level | list[Level] | None = ...,
        na_rep: str | None = ...,
        precision: int | None = ...,
        decimal: str = ...,
        thousands: str | None = ...,
        escape: str | None = ...,
        hyperlinks: Literal["html", "latex"] | None = ...,
    ) -> StylerRenderer: ...
