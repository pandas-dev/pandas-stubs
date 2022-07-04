from pathlib import Path
import subprocess

CHECK_FOLDERS = ["pandas-stubs", "tests", "scripts"]


def check_black():
    cmd = ["black", "--check"] + CHECK_FOLDERS
    subprocess.run(cmd, check=True)


def check_isort():
    cmd = ["isort", "--check-only"] + CHECK_FOLDERS
    subprocess.run(cmd, check=True)


def check_pyupgrade():
    cmd = ["pyupgrade", "--py38-plus", "--keep-runtime-typing"]
    success = True
    for folder in CHECK_FOLDERS:
        for py_file in Path(folder).glob("**/*.py*"):
            if py_file.suffix not in (".py", ".pyi"):
                continue
            try:
                subprocess.run(cmd + [str(py_file)], check=True)
            except subprocess.CalledProcessError:
                success = False
    if not success:
         subprocess.run(["git", "diff"], check=True)
         raise RuntimeError("pyupgrade failed")


def format_black():
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
            if py_file.suffix not in (".py", ".pyi"):
                continue
            subprocess.run(cmd + [str(py_file)], check=True)
