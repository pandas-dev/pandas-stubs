from typing import (
    Any,
    Callable,
    Optional,
    Sequence,
    TypedDict,
)

from pandas import (
    DataFrame,
    Index,
    Series,
)

from pandas._typing import Level

BaseFormatter = str | Callable
ExtFormatter = BaseFormatter | dict[Any, Optional[BaseFormatter]]
CSSPair = tuple[str, str | int | float]
CSSlist = list[CSSPair]
CSSProperties = str | CSSlist

class CSSdict(TypedDict):
    selector: str
    props: CSSProperties

CSSStyles = list[CSSdict]
Subset = slice | Sequence | Index

class StylerRenderer:
    loader = ...  # Incomplete
    env = ...  # Incomplete
    template_html = ...  # Incomplete
    template_html_table = ...  # Incomplete
    template_html_style = ...  # Incomplete
    template_latex = ...  # Incomplete
    template_string = ...  # Incomplete
    data = ...  # Incomplete
    index = ...  # Incomplete
    columns = ...  # Incomplete
    uuid = ...  # Incomplete
    uuid_len = ...  # Incomplete
    table_styles = ...  # Incomplete
    table_attributes = ...  # Incomplete
    caption = ...  # Incomplete
    cell_ids = ...  # Incomplete
    css = ...  # Incomplete
    concatenated = ...  # Incomplete
    hide_index_names: bool
    hide_column_names: bool
    hide_index_ = ...  # Incomplete
    hide_columns_ = ...  # Incomplete
    hidden_rows = ...  # Incomplete
    hidden_columns = ...  # Incomplete
    ctx = ...  # Incomplete
    ctx_index = ...  # Incomplete
    ctx_columns = ...  # Incomplete
    cell_context = ...  # Incomplete
    tooltips = ...  # Incomplete
    def __init__(
        self,
        data: DataFrame | Series,
        uuid: str | None = ...,
        uuid_len: int = ...,
        table_styles: CSSStyles | None = ...,
        table_attributes: str | None = ...,
        caption: str | tuple | None = ...,
        cell_ids: bool = ...,
        precision: int | None = ...,
    ): ...
    def format(
        self,
        formatter: ExtFormatter | None = ...,
        subset: Subset | None = ...,
        na_rep: str | None = ...,
        precision: int | None = ...,
        decimal: str = ...,
        thousands: str | None = ...,
        escape: str | None = ...,
        hyperlinks: str | None = ...,
    ) -> StylerRenderer: ...
    def format_index(
        self,
        formatter: ExtFormatter | None = ...,
        axis: int | str = ...,
        level: Level | list[Level] | None = ...,
        na_rep: str | None = ...,
        precision: int | None = ...,
        decimal: str = ...,
        thousands: str | None = ...,
        escape: str | None = ...,
        hyperlinks: str | None = ...,
    ) -> StylerRenderer: ...

def format_table_styles(styles: CSSStyles) -> CSSStyles: ...
def non_reducing_slice(slice_: Subset): ...
def maybe_convert_css_to_tuples(style: CSSProperties) -> CSSlist: ...
def refactor_levels(level: Level | list[Level] | None, obj: Index) -> list[int]: ...

class Tooltips:
    class_name = ...  # Incomplete
    class_properties = ...  # Incomplete
    tt_data = ...  # Incomplete
    table_styles = ...  # Incomplete
    def __init__(
        self,
        css_props: CSSProperties = ...,
        css_name: str = ...,
        tooltips: DataFrame = ...,
    ) -> None: ...
