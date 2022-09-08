from typing import (
    Any,
    Callable,
    Hashable,
    Sequence,
)

from pandas.core.frame import DataFrame
from pandas.core.series import Series

from pandas._typing import (
    Axis,
    FilePath,
    IndexLabel,
    Level,
    Scalar,
    WriteBuffer,
)

from pandas.io.formats.style_render import (
    CSSProperties,
    CSSStyles,
    ExtFormatter,
    StylerRenderer,
    Subset,
)

class Styler(StylerRenderer):
    def __init__(
        self,
        data: DataFrame | Series,
        precision: int | None = ...,
        table_styles: CSSStyles | None = ...,
        uuid: str | None = ...,
        caption: str | tuple | None = ...,
        table_attributes: str | None = ...,
        cell_ids: bool = ...,
        na_rep: str | None = ...,
        uuid_len: int = ...,
        decimal: str | None = ...,
        thousands: str | None = ...,
        escape: str | None = ...,
        formatter: ExtFormatter | None = ...,
    ) -> None: ...
    # def render(self,sparse_index: bool | None = ...,sparse_columns: bool | None = ...,**kwargs,) -> str: ...
    def set_tooltips(
        self,
        ttips: DataFrame,
        props: CSSProperties | None = ...,
        css_class: str | None = ...,
    ) -> Styler: ...
    def to_excel(
        self,
        excel_writer,
        sheet_name: str = ...,
        na_rep: str = ...,
        float_format: str | None = ...,
        columns: Sequence[Hashable] | None = ...,
        header: Sequence[Hashable] | bool = ...,
        index: bool = ...,
        index_label: IndexLabel | None = ...,
        startrow: int = ...,
        startcol: int = ...,
        engine: str | None = ...,
        merge_cells: bool = ...,
        encoding: str | None = ...,
        inf_rep: str = ...,
        verbose: bool = ...,
        freeze_panes: tuple[int, int] | None = ...,
    ) -> None: ...
    def to_latex(
        self,
        buf: FilePath | WriteBuffer[str] | None = ...,
        *,
        column_format: str | None = ...,
        position: str | None = ...,
        position_float: str | None = ...,
        hrules: bool | None = ...,
        clines: str | None = ...,
        label: str | None = ...,
        caption: str | tuple | None = ...,
        sparse_index: bool | None = ...,
        sparse_columns: bool | None = ...,
        multirow_align: str | None = ...,
        multicol_align: str | None = ...,
        siunitx: bool = ...,
        environment: str | None = ...,
        encoding: str | None = ...,
        convert_css: bool = ...,
    ): ...
    def to_html(
        self,
        buf: FilePath | WriteBuffer[str] | None = ...,
        *,
        table_uuid: str | None = ...,
        table_attributes: str | None = ...,
        sparse_index: bool | None = ...,
        sparse_columns: bool | None = ...,
        bold_headers: bool = ...,
        caption: str | None = ...,
        max_rows: int | None = ...,
        max_columns: int | None = ...,
        encoding: str | None = ...,
        doctype_html: bool = ...,
        exclude_styles: bool = ...,
        **kwargs,
    ): ...
    def set_td_classes(self, classes: DataFrame) -> Styler: ...
    def __copy__(self) -> Styler: ...
    def __deepcopy__(self, memo) -> Styler: ...
    def clear(self) -> None: ...
    def apply(
        self,
        func: Callable,
        axis: Axis | None = ...,
        subset: Subset | None = ...,
        **kwargs,
    ) -> Styler: ...
    def apply_index(
        self,
        func: Callable,
        axis: int | str = ...,
        level: Level | list[Level] | None = ...,
        **kwargs,
    ) -> Styler: ...
    def applymap_index(
        self,
        func: Callable,
        axis: int | str = ...,
        level: Level | list[Level] | None = ...,
        **kwargs,
    ) -> Styler: ...
    def applymap(
        self, func: Callable, subset: Subset | None = ..., **kwargs
    ) -> Styler: ...
    # def where(self, cond: Callable, value: str, other: str | None = ..., subset: Subset | None = ..., **kwargs) -> Styler: ...
    # def set_precision(self, precision: int) -> StylerRenderer: ...
    def set_table_attributes(self, attributes: str) -> Styler: ...
    def export(self) -> dict[str, Any]: ...
    def use(self, styles: dict[str, Any]) -> Styler: ...
    uuid: Any  # Incomplete
    def set_uuid(self, uuid: str) -> Styler: ...
    caption: Any  # Incomplete
    # def set_caption(self, caption: str | tuple) -> Styler: ...
    def set_sticky(
        self,
        axis: Axis = ...,
        pixel_size: int | None = ...,
        levels: Level | list[Level] | None = ...,
    ) -> Styler: ...
    def set_table_styles(
        self,
        table_styles: dict[Any, CSSStyles] | CSSStyles | None = ...,
        axis: int = ...,
        overwrite: bool = ...,
        css_class_names: dict[str, str] | None = ...,
    ) -> Styler: ...
    # def set_na_rep(self, na_rep: str) -> StylerRenderer: ...
    # def hide_index(self, subset: Subset | None = ..., level: Level | list[Level] | None = ..., names: bool = ...,) -> Styler: ...
    def hide_columns(
        self,
        subset: Subset | None = ...,
        level: Level | list[Level] | None = ...,
        names: bool = ...,
    ) -> Styler: ...
    def hide(
        self,
        subset: Subset | None = ...,
        axis: Axis = ...,
        level: Level | list[Level] | None = ...,
        names: bool = ...,
    ) -> Styler: ...
    def background_gradient(
        self,
        cmap: str = ...,
        low: float = ...,
        high: float = ...,
        axis: Axis | None = ...,
        subset: Subset | None = ...,
        text_color_threshold: float = ...,
        vmin: float | None = ...,
        vmax: float | None = ...,
        gmap: Sequence | None = ...,
    ) -> Styler: ...
    def text_gradient(
        self,
        cmap: str = ...,
        low: float = ...,
        high: float = ...,
        axis: Axis | None = ...,
        subset: Subset | None = ...,
        vmin: float | None = ...,
        vmax: float | None = ...,
        gmap: Sequence | None = ...,
    ) -> Styler: ...
    def set_properties(self, subset: Subset | None = ..., **kwargs) -> Styler: ...
    def bar(
        self,
        subset: Subset | None = ...,
        axis: Axis | None = ...,
        *,
        color: str | list | tuple | None = ...,
        cmap: Any | None = ...,
        width: float = ...,
        height: float = ...,
        align: str | float | Callable = ...,
        vmin: float | None = ...,
        vmax: float | None = ...,
        props: str = ...,
    ) -> Styler: ...
    def highlight_null(
        self,
        null_color: str = ...,
        subset: Subset | None = ...,
        props: str | None = ...,
    ) -> Styler: ...
    def highlight_max(
        self,
        subset: Subset | None = ...,
        color: str = ...,
        axis: Axis | None = ...,
        props: str | None = ...,
    ) -> Styler: ...
    def highlight_min(
        self,
        subset: Subset | None = ...,
        color: str = ...,
        axis: Axis | None = ...,
        props: str | None = ...,
    ) -> Styler: ...
    def highlight_between(
        self,
        subset: Subset | None = ...,
        color: str = ...,
        axis: Axis | None = ...,
        left: Scalar | Sequence | None = ...,
        right: Scalar | Sequence | None = ...,
        inclusive: str = ...,
        props: str | None = ...,
    ) -> Styler: ...
    def highlight_quantile(
        self,
        subset: Subset | None = ...,
        color: str = ...,
        axis: Axis | None = ...,
        q_left: float = ...,
        q_right: float = ...,
        interpolation: str = ...,
        inclusive: str = ...,
        props: str | None = ...,
    ) -> Styler: ...
    @classmethod
    def from_custom_template(
        cls,
        searchpath,
        html_table: str | None = ...,
        html_style: str | None = ...,
    ): ...
    def pipe(self, func: Callable, *args, **kwargs): ...
