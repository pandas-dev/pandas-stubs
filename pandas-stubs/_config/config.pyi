from collections.abc import (
    Callable,
    Iterable,
)
from contextlib import ContextDecorator
from typing import (
    Any,
    Literal,
    overload,
    type_check_only,
)

def get_option(pat: str) -> Any: ...
def set_option(pat: str, val: object) -> None: ...
def reset_option(pat: str) -> None: ...
@overload
def describe_option(pat: str, _print_desc: Literal[False]) -> str: ...
@overload
def describe_option(pat: str, _print_desc: Literal[True] = True) -> None: ...

class DictWrapper:
    def __init__(self, d: dict[str, Any], prefix: str = "") -> None: ...
    def __setattr__(
        self, key: str, val: str | bool | int | DictWrapper | None
    ) -> None: ...
    def __getattr__(self, key: str) -> str | bool | int | DictWrapper | None: ...
    def __dir__(self) -> Iterable[str]: ...

@type_check_only
class Compute(DictWrapper):
    use_bottleneck: bool
    use_numba: bool
    use_numexpr: bool

@type_check_only
class DisplayHTML(DictWrapper):
    border: int
    table_schema: bool
    use_mathjax: bool

@type_check_only
class DisplayUnicode(DictWrapper):
    ambiguous_as_wide: bool
    east_asian_width: bool

@type_check_only
class Display(DictWrapper):
    chop_threshold: float | None
    colheader_justify: Literal["left", "right"]
    date_dayfirst: bool
    date_yearfirst: bool
    encoding: str
    expand_frame_repr: bool
    float_format: Callable[[float], str] | None
    html: DisplayHTML
    large_repr: Literal["truncate", "info"]
    max_categories: int
    max_columns: int | None
    max_colwidth: int | None
    max_dir_items: int | None
    max_info_columns: int
    max_info_rows: int
    max_rows: int | None
    max_seq_items: int | None
    memory_usage: bool | Literal["deep"] | None
    min_rows: int | None
    multi_sparse: bool
    notebook_repr_html: bool
    pprint_nest_depth: int
    precision: int
    show_dimensions: bool | Literal["truncate"]
    unicode: DisplayUnicode
    width: int

@type_check_only
class Future(DictWrapper):
    distiguish_nan_and_na: bool
    infer_string: bool
    no_silent_downcasting: bool
    python_scalars: bool

@type_check_only
class IOExcelODS(DictWrapper):
    reader: str
    writer: str

@type_check_only
class IOExcelXLS(DictWrapper):
    reader: str

@type_check_only
class IOExcelXLSB(DictWrapper):
    reader: str

@type_check_only
class IOExcelXLSM(DictWrapper):
    reader: str
    writer: str

@type_check_only
class IOExcelXLSX(DictWrapper):
    reader: str
    writer: str

@type_check_only
class IOExcel(DictWrapper):
    ods: IOExcelODS
    xls: DictWrapper
    xlsb: DictWrapper
    xlsm: DictWrapper
    xlsx: DictWrapper

@type_check_only
class IOHDF(DictWrapper):
    default_format: Literal["table", "fixed"] | None
    dropna_table: bool

@type_check_only
class IOParquet(DictWrapper):
    engine: str

@type_check_only
class IOSQL(DictWrapper):
    engine: str

@type_check_only
class IO(DictWrapper):
    excel: IOExcel
    hdf: IOHDF
    parquet: IOParquet
    sql: IOSQL

@type_check_only
class Mode(DictWrapper):
    chained_assignment: Literal["warn", "raise"] | None
    copy_on_write: bool
    performance_warnings: bool
    sim_interactive: bool
    string_storage: str

@type_check_only
class PlottingMatplotlib(DictWrapper):
    register_converters: str

@type_check_only
class Plotting(DictWrapper):
    backend: str
    matplotlib: PlottingMatplotlib

@type_check_only
class StylerFormat:
    decimal: str
    escape: str | None
    formatter: str | None
    na_rep: str | None
    precision: int
    thousands: str | None

@type_check_only
class StylerHTML:
    mathjax: bool

@type_check_only
class StylerLatex:
    environment: str | None
    hrules: bool
    multicol_align: str
    multirow_align: str

@type_check_only
class StylerRender:
    encoding: str
    max_columns: int | None
    max_elements: int
    max_rows: int | None
    repr: str

@type_check_only
class StylerSparse:
    columns: bool
    index: bool

@type_check_only
class Styler(DictWrapper):
    format: StylerFormat
    html: StylerHTML
    latex: StylerLatex
    render: StylerRender
    sparse: StylerSparse

@type_check_only
class Options(DictWrapper):
    compute: Compute
    display: Display
    future: Future
    io: IO
    mode: Mode
    plotting: Plotting
    styler: Styler

options: Options

class option_context(ContextDecorator):
    def __init__(self, /, pat: str, val: Any, *args: Any) -> None: ...
    def __enter__(self) -> None: ...
    def __exit__(self, *args: object) -> None: ...

class OptionError(AttributeError, KeyError): ...
