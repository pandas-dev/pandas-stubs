"""type_completeness

Create a temporary virtual environment, install pandas, copy the local
stubs into the venv site-packages as the ``pandas`` package, run
``pyright --verifytypes pandas --ignoreexternal --outputjson`` and parse the
resulting JSON to compute type-completeness.

Usage:
        python type_completeness.py [--threshold 0.95] [--keep] [--verbose]

Notes:
- Requires a working Python with ``venv`` and network access to install
  packages via pip.
- Requires the ``pyright`` CLI to be available, or ``npx`` (node/npm) so we
  can run ``npx pyright``.
"""

from __future__ import annotations

import fnmatch
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
from typing import Any

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
    "pandas.compat.*",
]
THRESHOLD = 0.9


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
    out = subprocess.run(
        f"source {venv_path}/bin/activate && pyright --verifytypes pandas --ignoreexternal --outputjson",
        check=False,
        shell=True,
        executable="/bin/bash",
        text=True,
        capture_output=True,
    ).stdout
    with open("out.json", "w") as fd:
        fd.write(out)
    return json.loads(out)


def parse_pyright_json(data: dict[str, Any]) -> float:
    """Parse pyright JSON and return (total_exports, unknown_types).

    total_exports: count of symbols where isTypeExport is true
    unknown_types: count of those where isTypeKnown is false
    """
    symbols = data["typeCompleteness"]["symbols"]
    matched_symbols = [
        x
        for x in symbols
        if not any(fnmatch.fnmatch(x["name"], pattern) for pattern in EXCLUDE)
        and x["isExported"]
    ]
    covered = sum(x["isTypeKnown"] for x in matched_symbols) / len(matched_symbols)
    return covered


def main() -> int:
    tmpdir = Path(tempfile.mkdtemp(prefix="pandas-stubs-venv-"))
    venv_dir = tmpdir / "venv"
    try:
        # create venv
        import venv

        builder = venv.EnvBuilder(with_pip=True)
        builder.create(str(venv_dir.absolute()))

        venv_python = str(venv_dir / "bin" / "python")

        # install pandas into venv
        print("Installing latest pandas into venv...")
        subprocess.check_call(
            [venv_python, "-m", "pip", "install", "-U", "pip"]
        )  # upgrade pip first
        subprocess.check_call([venv_python, "-m", "pip", "install", "-U", "pyright"])
        subprocess.check_call(
            [
                venv_python,
                "-m",
                "pip",
                "install",
                "--pre",
                "--extra-index",
                "https://pypi.anaconda.org/scientific-python-nightly-wheels/simple",
                "pandas",
            ]
        )

        site_packages = venv_site_packages(venv_python)

        # copy stubs into site-packages/pandas
        dest = site_packages / "pandas"
        print(f"Copying stubs to {dest}")
        pandas_dir = Path(site_packages / "pandas").parent

        tracked_files = subprocess.run(
            ["git", "ls-files"], check=False, capture_output=True, text=True
        ).stdout.splitlines()
        for item in tracked_files:
            if not item.startswith("pandas-stubs"):
                continue
            s = item
            d = pandas_dir / item.replace("pandas-stubs", "pandas")
            shutil.copy2(s, d)

        # ensure py.typed exists
        (dest / "py.typed").write_text("\n")

        print("Running pyright --verifytypes (may take a while)...")
        out = run_pyright(str(venv_dir))

        completeness = parse_pyright_json(out)

        print("--- Results ---")
        print(f"Completeness: {completeness:.4%}")

        if completeness < THRESHOLD:
            print(f"Completeness {completeness:.1%} below threshold {THRESHOLD:.1%}")
            return 1
        print(f"Completeness {completeness:.1%} at or above threshold {THRESHOLD:.1%}")
        return 0

    finally:
        try:
            shutil.rmtree(tmpdir)
        except Exception:
            pass


if __name__ == "__main__":
    raise SystemExit(main())
