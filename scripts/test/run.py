from pathlib import Path
import subprocess


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


def stubtest():
    cmd = [
        "python",
        "-m",
        "mypy.stubtest",
        "pandas",
        "--concise",
        "--ignore-missing-stub",
        "--mypy-config-file",
        "pyproject.toml",
        "--allowlist",
        "stubtest_allowlist",
    ]
    subprocess.run(cmd, check=True)


def build_dist():
    cmd = ["poetry", "build", "-f", "wheel"]
    subprocess.run(cmd, check=True)


def install_dist():
    path = sorted(Path("dist/").glob("pandas_stubs-*.whl"))[-1]
    cmd = ["pip", "install", "--force-reinstall", str(path)]
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
    cmd = ["pip", "uninstall", "-y", "pandas-stubs"]
    subprocess.run(cmd, check=True)


def restore_src():
    if Path(r"_pandas-stubs").exists():
        Path(r"_pandas-stubs").rename("pandas-stubs")
    else:
        raise FileNotFoundError("'_pandas-stubs' folder does not exists.")
