from collections.abc import (
    Callable,
    Sequence,
)
from typing import (
    Any,
    Literal,
    TypedDict,
)

from jinja2.environment import (
    Environment,
    Template,
)
from jinja2.loaders import PackageLoader
from pandas import Index
from pandas.core.indexing import _IndexSlice
from typing_extensions import (
    Self,
    TypeAlias,
)

from pandas._typing import (
    Axis,
    HashableT,
    Level,
)

BaseFormatter: TypeAlias = str | Callable[[object], str]
ExtFormatter: TypeAlias = BaseFormatter | dict[Any, BaseFormatter | None]
CSSPair: TypeAlias = tuple[str, str | float]
CSSList: TypeAlias = list[CSSPair]
CSSProperties: TypeAlias = str | CSSList

class CSSDict(TypedDict):
    selector: str
    props: CSSProperties

class StyleExportDict(TypedDict, total=False):
    apply: Any
    table_attributes: Any
    table_styles: Any
    hide_index: bool
    hide_columns: bool
    hide_index_names: bool
    hide_column_names: bool
    css: dict[str, str | int]

CSSStyles: TypeAlias = list[CSSDict]
Subset: TypeAlias = _IndexSlice | slice | tuple[slice, ...] | list[HashableT] | Index

class StylerRenderer:
    loader: PackageLoader
    env: Environment
    template_html: Template
    template_html_table: Template
    template_html_style: Template
    template_latex: Template
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
    ) -> Self: ...
    def format_index(
        self,
        formatter: ExtFormatter | None = ...,
        axis: Axis = ...,
        level: Level | list[Level] | None = ...,
        na_rep: str | None = ...,
        precision: int | None = ...,
        decimal: str = ...,
        thousands: str | None = ...,
        escape: str | None = ...,
        hyperlinks: Literal["html", "latex"] | None = ...,
    ) -> Self: ...
    def relabel_index(
        self,
        labels: Sequence[str] | Index,
        axis: Axis = ...,
        level: Level | list[Level] | None = ...,
    ) -> Self: ...
    @property
    def columns(self) -> Index: ...
    @property
    def index(self) -> Index: ...
