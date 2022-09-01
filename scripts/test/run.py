from pathlib import Path
import subprocess
import sys


def mypy_src():
    cmd = ["mypy", "pandas-stubs", "tests", "--no-incremental"]
    subprocess.run(cmd, check=True)


def pyright_src():
    cmd = ["pyright"]
    subprocess.run(cmd, check=True)


def pytest():
    cmd = ["pytest", "--cache-clear"]
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
    cmd = ["python", "-m", "pip", "install", "--force-reinstall", str(path)]
    subprocess.run(cmd, check=True)
    cmd = ["python", "-m", "pip", "show", "pandas-stubs"]


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
    cmd = [sys.executable, "-m", "pip", "uninstall", "-y", "pandas"]
    subprocess.run(cmd, check=True)
    cmd = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "-i",
        "https://pypi.anaconda.org/scipy-wheels-nightly/simple",
        "pandas",
    ]
    subprocess.run(cmd, check=True)


def released_pandas():
    cmd = [sys.executable, "-m", "pip", "uninstall", "-y", "pandas"]
    subprocess.run(cmd, check=True)
    cmd = [sys.executable, "-m", "pip", "install", "pandas"]
    subprocess.run(cmd, check=True)
