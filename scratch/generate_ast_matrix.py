#!/usr/bin/env python3
"""Generate the Series arithmetic type algebra matrix from AST introspection.

Parses pandas-stubs/core/series.pyi via the ast module, extracts all @overload
signatures for arithmetic dunders, and writes the markdown matrix to
docs/architecture/matrices/01-series-arithmetic-algebra-matrix.md.
"""

import ast
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent
STUBS_FILE = REPO_ROOT / "pandas-stubs" / "core" / "series.pyi"
MATRIX_FILE = (
    REPO_ROOT
    / "docs"
    / "architecture"
    / "matrices"
    / "01-series-arithmetic-algebra-matrix.md"
)


def unparse_type(node: ast.AST | None) -> str:
    if node is None:
        return "Any"
    try:
        return ast.unparse(node)
    except Exception:  # noqa: BLE001
        return "Unknown"


def generate_ast_matrix() -> None:
    logger.info("Parsing AST of %s...", STUBS_FILE)
    with STUBS_FILE.open(encoding="utf-8") as f:
        tree = ast.parse(f.read())

    series_class = None
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "Series":
            series_class = node
            break

    if not series_class:
        logger.error("Could not find class 'Series'")
        return

    dunders = [
        "__add__",
        "__sub__",
        "__mul__",
        "__truediv__",
        "__floordiv__",
        "__mod__",
        "__pow__",
        "__radd__",
        "__rsub__",
        "__rmul__",
        "__rtruediv__",
        "__rfloordiv__",
        "__rmod__",
        "__rpow__",
    ]

    matrix_rows = []

    for node in series_class.body:
        if isinstance(node, ast.FunctionDef) and node.name in dunders:
            is_overload = any(
                (isinstance(dec, ast.Name) and dec.id == "overload")
                for dec in node.decorator_list
            )

            if is_overload:
                args = node.args
                self_type = "Series[Any]"
                other_type = "Any"

                all_args = getattr(args, "posonlyargs", []) + args.args

                for arg in all_args:
                    if arg.arg == "self":
                        self_type = unparse_type(arg.annotation)
                    elif arg.arg == "other":
                        other_type = unparse_type(arg.annotation)

                return_type = unparse_type(node.returns)

                self_type = self_type.replace("|", "\\|")
                other_type = other_type.replace("|", "\\|")
                return_type = return_type.replace("|", "\\|")

                matrix_rows.append(
                    f"| `{self_type}` | `{node.name}` | `{other_type}` | `{return_type}` |"
                )

    matrix_rows.sort(key=lambda x: (x.split("|")[1], x.split("|")[2]))

    markdown = """\
# Series Arithmetic & Operator Type Algebra Matrix (AST-Generated)

## 1. Overview & Formal Typing Model

This matrix is **dynamically generated via AST introspection** of \
`pandas-stubs/core/series.pyi`. It represents the absolute source of truth \
for overload resolution in Series arithmetic, guaranteeing fidelity between \
the documentation and the codebase.

### Core TypeVar & Parameter Definitions
- `S1`: Generic element type of the left-hand Series.
- `SeriesDType`: Union of numeric, boolean, datetime, timedelta, period, \
interval, category, string, and extension types.

---

## 2. Binary Arithmetic Matrix

The following matrix defines the statically resolved return type based on \
exactly what is defined in the stubs:

| Left Operand (`self`) | Operator | Right Operand (`other`) | Resolved Return Type |
| :--- | :--- | :--- | :--- |
"""
    markdown += "\n".join(matrix_rows)
    markdown += "\n\n---\n"
    markdown += (
        "> **Note**: Overloads are evaluated by type checkers from top to "
        "bottom. The stubs use precise structural protocols and type "
        "restrictions (e.g. `Never` for invalid combinations) to enforce "
        "mathematical validity.\n"
    )

    with MATRIX_FILE.open("w", encoding="utf-8") as f:
        f.write(markdown)
    logger.info("  ✓ Written %s overloads to %s", len(matrix_rows), MATRIX_FILE.name)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    generate_ast_matrix()
