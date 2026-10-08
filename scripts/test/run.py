from __future__ import annotations

from pathlib import Path
import re
import subprocess
import sys
from typing import Final

_PYTHON_VERSION: Final = "{}.{}".format(*sys.version_info[:2])

_SRC: Final = {
    "mypy": ["mypy", "pandas-stubs", "tests", "--no-incremental", "--strict"],
    "pyright": ["pyright", "--warnings", "--pythonversion", _PYTHON_VERSION],
    "pyrefly": [
        "pyrefly",
        "check",
        "pandas-stubs",
        "tests",
        "--python-version",
        _PYTHON_VERSION,
        "--preset",
        "strict",
    ],
    "ty": [
        "ty",
        "check",
        "pandas-stubs",
        "tests",
        "--python-version",
        _PYTHON_VERSION,
    ],
}

_SRC_ALL: Final = {
    "pyrefly": [
        "pyrefly",
        "check",
        "pandas-stubs",
        "tests",
        "--python-version",
        _PYTHON_VERSION,
        "--preset",
        "all",
    ],
    "ty": [
        "ty",
        "check",
        "pandas-stubs",
        "tests",
        "--python-version",
        _PYTHON_VERSION,
        "--error",
        "all",
    ],
}

_DIST: Final = {
    "mypy": [
        "mypy",
        "tests",
        "--no-incremental",
        "--strict",
        "--python-version",
        _PYTHON_VERSION,
    ],
    "pyright": ["pyright", "tests", "--warnings", "--pythonversion", _PYTHON_VERSION],
    "pyrefly": ["pyrefly", "check", "tests", "--python-version", _PYTHON_VERSION],
    "ty": ["ty", "check", "tests", "--python-version", _PYTHON_VERSION],
}


def checker_src(checker: str, *, all_rules: bool = False) -> None:
    cmd = (_SRC_ALL if all_rules else _SRC)[checker]
    subprocess.run(cmd, check=True)


def checker_dist(checker: str) -> None:
    cmd = _DIST[checker]
    subprocess.run(cmd, check=True)


def pytest() -> None:
    cmd = ["pytest", "--cache-clear"]
    subprocess.run(cmd, check=True)


def style() -> None:
    cmd = ["pre-commit", "run", "--all-files", "--verbose"]
    subprocess.run(cmd, check=True)


def stubtest(allowlist: str = "", check_missing: bool = False) -> None:
    cmd = [
        sys.executable,
        "-m",
        "mypy.stubtest",
        "pandas",
        "--concise",
        "--mypy-config-file",
        "pyproject.toml",
    ]
    if not check_missing:
        cmd += ["--ignore-missing-stub"]
    if allowlist:
        cmd += ["--allowlist", allowlist]
    subprocess.run(cmd, check=True)


def build_dist() -> None:
    cmd = ["poetry", "build", "-f", "wheel"]
    subprocess.run(cmd, check=True)


def install_dist() -> None:
    path = max(Path("dist/").glob("pandas_stubs-*.whl"))
    cmd = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--upgrade",
        "--force-reinstall",
        str(path),
        "numpy-typing-compat",
    ]
    subprocess.run(cmd, check=True)
    subprocess.run([sys.executable, "-m", "pip", "check"], check=True)


def rename_src() -> None:
    if Path(r"pandas-stubs").exists():
        Path(r"pandas-stubs").rename("_pandas-stubs")
    else:
        raise FileNotFoundError("'pandas-stubs' folder does not exists.")


def uninstall_dist() -> None:
    cmd = [sys.executable, "-m", "pip", "uninstall", "-y", "pandas-stubs"]
    subprocess.run(cmd, check=True)


def restore_src() -> None:
    if Path(r"_pandas-stubs").exists():
        Path(r"_pandas-stubs").rename("pandas-stubs")
    else:
        raise FileNotFoundError("'_pandas-stubs' folder does not exists.")


def _get_version_from_pyproject(program: str) -> str:
    """Find version of a package from the pyproject.toml file."""
    text = Path("pyproject.toml").read_text()
    # handle <, >, ==, <=, >= cases
    match = re.search(rf'"{re.escape(program)}[=<>~!]+([^"]+)"', text)
    if match is None:
        raise KeyError(f"Could not find {program} in pyproject.toml")
    return match.group(1)


def install_floor(pkg: str) -> None:
    version = _get_version_from_pyproject(pkg)
    cmd = [sys.executable, "-m", "pip", "install", f"{pkg}=={version}"]
    subprocess.run(cmd, check=True)


def install_latest(pkg: str, *, extra_index_url: str = "") -> None:
    cmd = [sys.executable, "-m", "pip", "install", "--pre", "--upgrade"]
    if extra_index_url:
        cmd += ["--extra-index-url", extra_index_url]
    cmd.append(pkg)
    subprocess.run(cmd, check=True)


def nightly_mypy() -> None:
    cmd = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--upgrade",
        "--find-links",
        "https://github.com/mypyc/mypy_mypyc-wheels/releases/",
        "mypy",
    ]
    subprocess.run(cmd, check=True)

    # ignore unused ignore errors
    config_file = Path("pyproject.toml")
    config_file.write_text(
        config_file.read_text().replace(
            "warn_unused_ignores = true", "warn_unused_ignores = false"
        )
    )


def released_mypy() -> None:
    install_floor("mypy")

    # check for unused ignores again
    config_file = Path("pyproject.toml")
    config_file.write_text(
        config_file.read_text().replace(
            "warn_unused_ignores = false", "warn_unused_ignores = true"
        )
    )


def type_completeness() -> None:
    cmd = ["pyrefly", "coverage", "check", "--public-only"]
    subprocess.run(cmd, check=True)
