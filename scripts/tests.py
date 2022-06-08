import subprocess
from pathlib import Path


def __test_all():
    cmd = ["mypy", "pandas-stubs", "tests"]
    subprocess.run(cmd)

    cmd = ["pytest"]
    subprocess.run(cmd)

    cmd = ["pyright"]
    subprocess.run(cmd)


def install_wheel():
    cmd = ["poetry", "build", "-f", "wheel"]
    subprocess.run(cmd)

    path = next(Path("dist/").glob("*.whl"))
    cmd = ["pip", "install", str(path)]
    subprocess.run(cmd)


def remove_src_code():
    Path("pandas-stubs").unlink()


def __clean_env():
    cmd = ["pip", "uninstall", "-y", "pandas-stubs"]
    subprocess.run(cmd)

    cmd = ["poetry", "install"]
    subprocess.run(cmd)


def run_all():

    __test_all()

    install_wheel()

    cmd = ["pyright"]
    subprocess.run(cmd)

    cmd = ["mypy", "tests"]
    subprocess.run(cmd)

    __clean_env()
