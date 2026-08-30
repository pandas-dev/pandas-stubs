#!/usr/bin/env python3
"""
CI Documentation Synchronization Checker.

Verifies that the Markdown Type Algebra Matrices match the actual `.pyi` stubs via AST parsing.
If out of sync, CI fails and prompts the user to run the AST generator.
"""

import ast
from pathlib import Path
import sys


def unparse_type(node: ast.AST | None) -> str:
    if node is None:
        return "Any"
    try:
        return ast.unparse(node)
    except Exception:  # noqa: BLE001
        return "Unknown"


_DUNDERS = [
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


def _matrix_rows(series_class: ast.ClassDef) -> list[str]:
    """Build the arithmetic matrix rows from the Series overloads."""
    rows = []
    for node in series_class.body:
        if not isinstance(node, ast.FunctionDef) or node.name not in _DUNDERS:
            continue
        is_overload = any(
            isinstance(dec, ast.Name) and dec.id == "overload"
            for dec in node.decorator_list
        )
        if not is_overload:
            continue
        self_type, other_type = "Series[Any]", "Any"
        all_args = getattr(node.args, "posonlyargs", []) + node.args.args
        for arg in all_args:
            if arg.arg == "self":
                self_type = unparse_type(arg.annotation)
            elif arg.arg == "other":
                other_type = unparse_type(arg.annotation)

        return_type = unparse_type(node.returns)
        self_type = self_type.replace("|", "\\|")
        other_type = other_type.replace("|", "\\|")
        return_type = return_type.replace("|", "\\|")

        rows.append(
            f"| `{self_type}` | `{node.name}` | `{other_type}` | `{return_type}` |"
        )
    rows.sort(key=lambda row: (row.split("|")[1], row.split("|")[2]))
    return rows


def check_series_arithmetic(wt_dir: Path) -> bool:
    stubs_file = wt_dir / "pandas-stubs" / "core" / "series.pyi"
    matrix_file = (
        wt_dir
        / "docs"
        / "architecture"
        / "matrices"
        / "01-series-arithmetic-algebra-matrix.md"
    )

    if not stubs_file.exists() or not matrix_file.exists():
        sys.stdout.write("Error: Missing required files for sync check.\n")
        return False

    with stubs_file.open(encoding="utf-8") as f:
        tree = ast.parse(f.read())

    series_class = next(
        (
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == "Series"
        ),
        None,
    )

    if series_class is None:
        sys.stdout.write("Error: Could not find class 'Series' in stubs.\n")
        return False

    expected_block = "\n".join(_matrix_rows(series_class))

    with matrix_file.open(encoding="utf-8") as f:
        actual_content = f.read()

    if expected_block not in actual_content:
        sys.stdout.write(
            "❌ ERROR: docs/architecture/matrices/01-series-arithmetic-algebra-matrix.md is out of sync with series.pyi!\n"
        )
        sys.stdout.write(
            "Please run `python3 scratch/generate_ast_matrix.py` to update the documentation.\n"
        )
        return False

    sys.stdout.write("✅ Documentation is in sync with AST.\n")
    return True


if __name__ == "__main__":
    repo_root = Path(__file__).parent.parent
    if not check_series_arithmetic(repo_root):
        sys.exit(1)
    sys.exit(0)
