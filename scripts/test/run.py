from __future__ import annotations

from pathlib import Path
import re
import subprocess
import sys
from typing import Final

_PYTHON_VERSION: Final = "{}.{}".format(*sys.version_info[:2])

_CHECKER_COMMANDS: Final = {
    "mypy": [
        "mypy",
        "--no-incremental",
        "--strict",
        "--python-version",
        _PYTHON_VERSION,
    ],
    "pyright": [
        "pyright",
        "--warnings",
        "--pythonversion",
        _PYTHON_VERSION,
    ],
    "pyrefly": [
        "pyrefly",
        "check",
        "--python-version",
        _PYTHON_VERSION,
    ],
    "ty": ["ty", "check", "--python-version", _PYTHON_VERSION],
}


def _checker_command(
    checker: str, *, installed: bool, all_rules: bool = False
) -> list[str]:
    if all_rules and (installed or checker not in {"pyrefly", "ty"}):
        raise ValueError("All-rules checks require local stubs and pyrefly or ty")

    cmd = [*_CHECKER_COMMANDS[checker]]

    if checker == "pyrefly":
        cmd += ["--preset", "all" if all_rules else "strict"]
    elif checker == "ty" and all_rules:
        cmd += ["--error", "all"]

    if installed:
        cmd += ["tests"]
    elif checker != "pyright":
        # pyright results depend on file order (microsoft/pyright#11644),
        # so let it use the `include` order from pyproject.toml
        # it can be removed after pyright 1.1.414
        cmd += ["pandas-stubs", "tests"]
    return cmd


def checker_src(checker: str, *, all_rules: bool = False) -> None:
    cmd = _checker_command(checker, installed=False, all_rules=all_rules)
    subprocess.run(cmd, check=True)


def checker_dist(checker: str) -> None:
    cmd = _checker_command(checker, installed=True)
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
