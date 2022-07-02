from pathlib import Path
import subprocess

CHECK_FOLDERS = ["pandas-stubs", "tests", "scripts"]


def check_black():
    cmd = ["black", "--check"] + CHECK_FOLDERS
    subprocess.run(cmd, check=True)


def check_isort():
    cmd = ["isort", "--check-only"] + CHECK_FOLDERS
    subprocess.run(cmd, check=True)


def pyupgrade_check():
    cmd = ["pyupgrade", "--py38-plus", "--keep-runtime-typing"]
    for folder in CHECK_FOLDERS:
        for py_file in Path(folder).glob("**/*.py*"):
            subprocess.run(cmd + [str(py_file)], check=True)


def run_format_black():
    cmd = ["black"] + CHECK_FOLDERS
    cmd = ["black"] + CHECK_FOLDERS
    subprocess.run(cmd, check=True)


def format_isort():
    cmd = ["isort"] + CHECK_FOLDERS
    subprocess.run(cmd, check=True)


def format_pyupgrade():
    cmd = [
        "pyupgrade",
        "--py38-plus",
        "--keep-runtime-typing",
        "--exit-zero-even-if-changed",
    ]
    for folder in CHECK_FOLDERS:
        for py_file in Path(folder).glob("**/*.py*"):
            subprocess.run(cmd + [str(py_file)], check=True)
