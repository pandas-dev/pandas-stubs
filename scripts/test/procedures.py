from pathlib import Path
import shutil
import subprocess

from scripts._job import (
    Step,
    run_job,
)


def run_mypy_src():
    cmd = ["mypy", "pandas-stubs", "tests", "--no-incremental"]
    subprocess.run(cmd, check=True)


def run_pyright_src():
    cmd = ["pyright"]
    subprocess.run(cmd, check=True)


def run_pytest_src():
    cmd = ["pytest"]
    subprocess.run(cmd, check=True)


def build_dist():
    cmd = ["poetry", "build", "-f", "wheel"]
    subprocess.run(cmd, check=True)


def install_dist():
    path = next(Path("dist/").glob("*.whl"))
    cmd = ["pip", "install", str(path)]
    subprocess.run(cmd, check=True)


def rename_src():
    if Path(r"pandas-stubs").exists():
        Path(r"pandas-stubs").rename('_pandas-stubs')


def run_mypy_dist():
    cmd = ["mypy", "tests", "--no-incremental"]
    subprocess.run(cmd, check=True)


def run_pyright_dist():
    cmd = ["pyright", "tests"]
    subprocess.run(cmd, check=True)


def uninstall_dist():
    cmd = ["pip", "uninstall", "-y", "pandas-stubs"]
    subprocess.run(cmd, check=True)


def restore_src():
    if Path(r"_pandas-stubs").exists():
        Path(r"_pandas-stubs").rename('pandas-stubs')


def clean_mypy_cache():
    if Path(".mypy_cache").exists():
        shutil.rmtree(".mypy_cache")


def clean_pytest_cache():
    if Path(".mypy_cache").exists():
        shutil.rmtree(".pytest_cache")


def create_new_venv():
    cmd = ["poetry", "remove", "python"]
    subprocess.run(cmd, check=True)

    cmd = ["poetry", "update", "-vvv"]
    subprocess.run(cmd, check=True)

    cmd = ["poetry", "shell"]
    subprocess.run(cmd, check=True)
