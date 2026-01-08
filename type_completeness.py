"""type_completeness

Create a temporary virtual environment, install pandas, copy the local
stubs into the venv site-packages as the ``pandas`` package, run
``pyright --verifytypes pandas --ignoreexternal --outputjson`` and parse the
resulting JSON to compute type-completeness.
"""

from __future__ import annotations

from fnmatch import fnmatch
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from typing import Any
import venv

EXCLUDE = [
    # pandas distributes (untyped) tests with the package
    "*.tests.*",
    "*.conftest.*",
    # pandas.core is technically private, and anything considered public
    # is re-exported in other places. For example, `DataFrameGroupBy` is
    # re-exported in `pandas.api.typing`. The re-exports are available
    # under `'alternateNames'`, which we consider when excluding symbols.
    "pandas.core.*",
    # Not considered public
    # https://github.com/pandas-dev/pandas/blob/e87248e1a5d6d78a138039f2856a3aec6b9fef54/doc/source/reference/index.rst#L34
    "pandas.compat.*",
    # The only parts of `pandas.io` which appears in the API reference are:
    # - `pandas.io.json`
    # - `pandas.io.formats.style`
    # https://github.com/pandas-dev/pandas/blob/b8371f5e6f329bfe1b5f1e099e221c8219fc6bbd/doc/source/reference/io.rst
    # See also: https://github.com/pandas-dev/pandas/issues/27522#issuecomment-516360201
    "pandas.io.common.*",
    "pandas.io.parsers.*",
    "pandas.io.excel.*",
    "pandas.io.formats.csvs.*",
    "pandas.io.formats.excel.*",
    "pandas.io.formats.html.*",
    "pandas.io.formats.info.*",
    "pandas.io.formats.printing.*",
    "pandas.io.formats.string.*",
    "pandas.io.formats.xml.*",
]
THRESHOLD = 0.9


def venv_site_packages(venv_python: str) -> Path:
    """Return the site-packages directory for a given venv Python executable."""
    cmd = [
        venv_python,
        "-c",
        "import sysconfig, json; print(sysconfig.get_paths()['purelib'])",
    ]
    out = subprocess.check_output(cmd, text=True).strip()  # noqa: S603
    return Path(out)


def run_pyright(venv_path: str) -> dict[str, Any]:
    # change command based on platform?
    out = subprocess.run(  # noqa: S602
        f"source {venv_path}/bin/activate && pyright --verifytypes pandas --ignoreexternal --outputjson",
        check=False,
        shell=True,
        executable="/bin/bash",
        text=True,
        capture_output=True,
    ).stdout
    Path("out.json").write_text(out)
    return json.loads(out)


def parse_pyright_json(data: dict[str, Any]) -> float:
    symbols = data["typeCompleteness"]["symbols"]
    matched_symbols = [
        x
        for x in symbols
        if x["isExported"]
        # Keep symbols where there's any name which doesn't match any excluded patterns.
        and any(
            all(not fnmatch(name, pattern) for pattern in EXCLUDE)
            for name in [x["name"], *x.get("alternateNames", [])]
        )
    ]
    return sum(x["isTypeKnown"] for x in matched_symbols) / len(matched_symbols)


def main() -> int:
    tmpdir = Path(tempfile.mkdtemp(prefix="pandas-stubs-venv-"))
    venv_dir = tmpdir / "venv"
    try:
        builder = venv.EnvBuilder(with_pip=True)
        builder.create(str(venv_dir.absolute()))

        venv_python = str(venv_dir / "bin" / "python")

        subprocess.check_call(  # noqa: S603
            [venv_python, "-m", "pip", "install", "-U", "pip"]
        )
        subprocess.check_call(  # noqa: S603
            [venv_python, "-m", "pip", "install", "-U", "pyright", "pandas"]
        )

        site_packages = venv_site_packages(venv_python)

        # Copy stubs into site-packages/pandas.
        dest = site_packages / "pandas"
        pandas_dir = Path(site_packages / "pandas").parent
        tracked_files = subprocess.run(
            ["git", "ls-files"],  # noqa: S607
            check=False,
            capture_output=True,
            text=True,
        ).stdout.splitlines()
        for item in tracked_files:
            if not item.startswith("pandas-stubs"):
                continue
            s = item
            d = pandas_dir / item.replace("pandas-stubs", "pandas")
            shutil.copy2(s, d)

        # Pyright requires `py.typed` to exist.
        (dest / "py.typed").write_text("\n")

        sys.stdout.write("Running pyright --verifytypes (may take a while)...")
        out = run_pyright(str(venv_dir))

        completeness = parse_pyright_json(out)

        sys.stdout.write("--- Results ---")
        sys.stdout.write(f"Completeness: {completeness:.4%}")

        if completeness < THRESHOLD:
            sys.stdout.write(
                f"Completeness {completeness:.1%} below threshold {THRESHOLD:.1%}"
            )
            return 1
        sys.stdout.write(
            f"Completeness {completeness:.1%} at or above threshold {THRESHOLD:.1%}"
        )
        return 0

    finally:
        shutil.rmtree(tmpdir)


if __name__ == "__main__":
    raise SystemExit(main())
