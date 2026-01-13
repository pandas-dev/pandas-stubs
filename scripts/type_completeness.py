"""Ensure that pandas' public API is type-complete, using Pyright.

We run Pyright's `--verifytypes` to ensure that type-completeness is at 100%.

Rather than running the command as-is, we need to make some adjustments:

- Use `--ignoreexternal` to ignore untyped symbols in dependent libraries:
  https://github.com/microsoft/pyright/discussions/9911#discussioncomment-12192388.
- We exclude symbols which are technically public (accordinging to Pyright) but which
  aren't in pandas' documented API and not considered public by pandas. There is no
  CLI flag for this in Pyright, but we can parse the output json and exclude paths ourselves:
  https://github.com/microsoft/pyright/discussions/10614#discussioncomment-13543475.
- We create a temporary virtual environment with pandas installed in it, as Pyright
  needs that to run its `--verifytypes` command.
"""

from __future__ import annotations

from fnmatch import fnmatch
import json
import os
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
THRESHOLD = 1


def venv_site_packages(venv_python: str) -> Path:
    """Return the site-packages directory for a given venv Python executable."""
    cmd = [
        venv_python,
        "-c",
        "import sysconfig, json; print(sysconfig.get_paths()['purelib'])",
    ]
    out = subprocess.check_output(cmd, text=True).strip()
    return Path(out)


def run_pyright(venv_path: str) -> dict[str, Any]:
    env = os.environ.copy()
    venv = Path(venv_path)
    bin_dir = venv / ("Scripts" if sys.platform == "win32" else "bin")
    env["PATH"] = f"{bin_dir}{os.pathsep}{env['PATH']}"
    out = subprocess.run(
        [  # noqa: S607
            "pyright",
            "--verifytypes",
            "pandas",
            "--ignoreexternal",
            "--outputjson",
        ],
        check=False,
        env=env,
        text=True,
        capture_output=True,
    ).stdout
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

        subprocess.check_call([venv_python, "-m", "pip", "install", "-U", "pip"])
        subprocess.check_call(
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

        sys.stdout.write("Running pyright --verifytypes (may take a while)...\n")
        out = run_pyright(str(venv_dir))

        completeness = parse_pyright_json(out)

        sys.stdout.write("--- Results ---\n")
        sys.stdout.write(f"Completeness: {completeness:.4%}\n")

        if completeness < 1:
            sys.stdout.write(f"Completeness {completeness:.1%} below threshold 100%\n")
            return 1
        sys.stdout.write("Completeness is at 100% threshold\n")
        return 0

    finally:
        shutil.rmtree(tmpdir)


if __name__ == "__main__":
    raise SystemExit(main())
