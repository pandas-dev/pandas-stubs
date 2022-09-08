from __future__ import annotations

import os
import pathlib
from typing import (
    TYPE_CHECKING,
    Type,
)

from jinja2.environment import (
    Environment,
    Template,
)
from jinja2.loaders import PackageLoader
import numpy as np
import numpy.typing as npt
from pandas import (
    DataFrame,
    Series,
)
from pandas._testing import ensure_clean
import pytest
from typing_extensions import assert_type

from tests import check

from pandas.io.formats.style import Styler

DF = DataFrame({"a": [1, 2, 3], "b": [3.14, 2.72, 1.61]})


PWD = pathlib.Path(os.path.split(os.path.abspath(__file__))[0])

if TYPE_CHECKING:
    from pandas.io.formats.style_render import StyleExportDict
else:
    StyleExportDict = object


@pytest.fixture(autouse=True)
def reset_style():
    DF.style.clear()


def test_apply():
    def f(s: Series) -> Series:
        return s

    check(assert_type(DF.style.apply(f), Styler), Styler)

    def h(df: DataFrame) -> DataFrame:
        return df

    check(assert_type(DF.style.apply(h, axis=None), Styler), Styler)


def test_apply_index() -> None:
    def f(s: Series) -> npt.NDArray[np.str_]:
        return np.asarray(s, dtype=np.str_)

    check(assert_type(DF.style.apply_index(f), Styler), Styler)

    def f1(s: Series) -> Series[str]:
        return Series(s, dtype=str)

    check(assert_type(DF.style.apply_index(f1), Styler), Styler)


def test_applymap() -> None:
    def g(o: object) -> str:
        return str(o)

    check(assert_type(DF.style.applymap(g), Styler), Styler)


def test_applymap_index() -> None:
    def g(o: object) -> str:
        return str(o)

    check(assert_type(DF.style.applymap_index(g), Styler), Styler)


def test_background_gradient() -> None:
    check(assert_type(DF.style.background_gradient(), Styler), Styler)


def test_bar() -> None:
    check(assert_type(DF.style.bar(), Styler), Styler)


def test_clear() -> None:
    check(assert_type(DF.style.clear(), None), type(None))


def test_env() -> None:
    check(assert_type(DF.style.env, Environment), Environment)


def test_format() -> None:
    check(assert_type(DF.style.format(), Styler), Styler)


def test_format_index() -> None:
    check(assert_type(DF.style.format_index(), Styler), Styler)


def test_from_custom_template() -> None:
    check(
        assert_type(
            Styler.from_custom_template(str(PWD / "data" / "myhtml_table.tpl")),
            Type[Styler],
        ),
        type(Styler),
    )


def test_hide() -> None:
    check(assert_type(DF.style.hide(), Styler), Styler)


def test_highlight_between() -> None:
    check(assert_type(DF.style.highlight_between(), Styler), Styler)


def test_highlight_max() -> None:
    check(assert_type(DF.style.highlight_max(), Styler), Styler)


def test_highlight_min() -> None:
    check(assert_type(DF.style.highlight_min(), Styler), Styler)


def test_highlight_null() -> None:
    check(assert_type(DF.style.highlight_null(), Styler), Styler)


def test_highlight_quantile() -> None:
    check(assert_type(DF.style.highlight_quantile(), Styler), Styler)


def test_loader() -> None:
    check(assert_type(DF.style.loader, PackageLoader), PackageLoader)


def test_pipe() -> None:
    def f(s: Styler) -> Styler:
        return s

    check(assert_type(DF.style.pipe(f).pipe(f), Styler), Styler)


def test_set() -> None:
    check(assert_type(DF.style.set_caption("caption"), Styler), Styler)
    check(
        assert_type(DF.style.set_properties(color="white", align="right"), Styler),
        Styler,
    )
    check(assert_type(DF.style.set_sticky("columns", pixel_size=50), Styler), Styler)
    check(
        assert_type(DF.style.set_table_attributes('class="pure-table"'), Styler), Styler
    )
    check(
        assert_type(
            DF.style.set_table_styles(
                [{"selector": "tr:hover", "props": [("background-color", "yellow")]}]
            ),
            Styler,
        ),
        Styler,
    )
    classes = DataFrame(
        [["min-val red", "blue"], ["red", None], ["nothing", "huh"]],
        index=DF.index,
        columns=DF.columns,
    )
    check(assert_type(DF.style.set_td_classes(classes), Styler), Styler)
    ttips = DataFrame(
        data=[["Min", ""], [np.nan, "Max"], [np.nan, "Other"]],
        columns=DF.columns,
        index=DF.index,
    )
    check(assert_type(DF.style.set_tooltips(ttips), Styler), Styler)
    check(assert_type(DF.style.set_uuid("r4nd0mc44r4c73r5"), Styler), Styler)


def test_styler_templates():
    check(assert_type(DF.style.template_html, Template), Template)
    check(assert_type(DF.style.template_html_style, Template), Template)
    check(assert_type(DF.style.template_html_table, Template), Template)
    check(assert_type(DF.style.template_latex, Template), Template)


def test_text_gradient() -> None:
    check(assert_type(DF.style.text_gradient(), Styler), Styler)


def test_to_excel() -> None:
    with ensure_clean("test.xlsx") as path:
        check(assert_type(DF.style.to_excel(path), None), type(None))


def test_to_html() -> None:
    check(assert_type(DF.style.to_html(), str), str)
    with ensure_clean("test.html") as path:
        check(assert_type(DF.style.to_html(path), None), type(None))


def test_to_latex() -> None:
    check(assert_type(DF.style.to_latex(), str), str)
    with ensure_clean("test.tex") as path:
        check(assert_type(DF.style.to_latex(path), None), type(None))


def test_export_use() -> None:
    exported = DF.style.export()
    check(assert_type(exported, StyleExportDict), dict)
    check(assert_type(DF.style.use(exported), Styler), Styler)


def test_subset() -> None:
    from pandas import IndexSlice

    check(assert_type(DF.style.highlight_min(subset=slice(1, 2)), Styler), Styler)
    check(assert_type(DF.style.highlight_min(subset=IndexSlice[1:2]), Styler), Styler)
    check(assert_type(DF.style.highlight_min(subset=[1]), Styler), Styler)
    check(assert_type(DF.style.highlight_min(subset=DF.columns[1:]), Styler), Styler)
