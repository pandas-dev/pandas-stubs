from __future__ import annotations

from pathlib import Path
import subprocess
import sys


def mypy_src():
    cmd = ["mypy", "pandas-stubs", "tests", "--no-incremental"]
    subprocess.run(cmd, check=True)


def pyright_src():
    cmd = ["pyright"]
    subprocess.run(cmd, check=True)


def pytest(flags: tuple[str, ...] = ("-Werror",)):
    cmd = ["pytest", "--cache-clear", *flags]
    subprocess.run(cmd, check=True)


def style():
    cmd = ["pre-commit", "run", "--all-files", "--verbose"]
    subprocess.run(cmd, check=True)


def stubtest(allowlist: str = "", check_missing: bool = False):
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


def build_dist():
    cmd = ["poetry", "build", "-f", "wheel"]
    subprocess.run(cmd, check=True)


def install_dist():
    path = sorted(Path("dist/").glob("pandas_stubs-*.whl"))[-1]
    cmd = [sys.executable, "-m", "pip", "install", "--force-reinstall", str(path)]
    subprocess.run(cmd, check=True)


def rename_src():
    if Path(r"pandas-stubs").exists():
        Path(r"pandas-stubs").rename("_pandas-stubs")
    else:
        raise FileNotFoundError("'pandas-stubs' folder does not exists.")


def mypy_dist():
    cmd = ["mypy", "tests", "--no-incremental"]
    subprocess.run(cmd, check=True)


def pyright_dist():
    cmd = ["pyright", "tests"]
    subprocess.run(cmd, check=True)


def uninstall_dist():
    cmd = [sys.executable, "-m", "pip", "uninstall", "-y", "pandas-stubs"]
    subprocess.run(cmd, check=True)


def restore_src():
    if Path(r"_pandas-stubs").exists():
        Path(r"_pandas-stubs").rename("pandas-stubs")
    else:
        raise FileNotFoundError("'_pandas-stubs' folder does not exists.")


def nightly_pandas():
    cmd = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--pre",
        "--use-deprecated=legacy-resolver",
        "--upgrade",
        "--extra-index-url",
        "https://pypi.anaconda.org/scipy-wheels-nightly/simple",
        "pandas",
    ]
    subprocess.run(cmd, check=True)


def _get_version_from_pyproject(program: str) -> str:
    text = Path("pyproject.toml").read_text()
    version_line = next(
        line for line in text.splitlines() if line.startswith(f"{program} = ")
    )
    return version_line.split('"')[1]


def released_pandas():
    version = _get_version_from_pyproject("pandas")
    cmd = [sys.executable, "-m", "pip", "install", f"pandas=={version}"]
    subprocess.run(cmd, check=True)


def nightly_mypy():
    cmd = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--upgrade",
        "git+https://github.com/python/mypy.git",
    ]
    subprocess.run(cmd, check=True)

    # ignore unused ignore errors
    config_file = Path("pyproject.toml")
    config_file.write_text(
        config_file.read_text().replace(
            "warn_unused_ignores = true", "warn_unused_ignores = false"
        )
    )


def released_mypy():
    version = _get_version_from_pyproject("mypy")
    cmd = [sys.executable, "-m", "pip", "install", f"mypy=={version}"]
    subprocess.run(cmd, check=True)

    # check for unused ignores again
    config_file = Path("pyproject.toml")
    config_file.write_text(
        config_file.read_text().replace(
            "warn_unused_ignores = false", "warn_unused_ignores = true"
        )
    )
