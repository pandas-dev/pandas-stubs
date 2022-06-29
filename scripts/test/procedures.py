import shutil
import subprocess
from pathlib import Path


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


def add_last_changes():
    cmd = ["git", "add", "."]
    subprocess.run(cmd, check=True)


def commit_last_changes():
    cmd = ["git", "commit", "-am", "\"temp commit\""]
    subprocess.run(cmd, check=True)


def remove_src():
    shutil.rmtree(r"pandas-stubs")


def run_mypy_dist():
    cmd = ["mypy", "tests", "--no-incremental"]
    subprocess.run(cmd, check=True)


def run_pyright_dist():
    cmd = ["pyright", "tests"]
    subprocess.run(cmd, check=True)


def uninstall_dist():
    cmd = ["pip", "uninstall", "-y", "pandas-stubs"]
    subprocess.run(cmd, check=True)


def restore_last_changes():
    cmd = ["git", "show", "-s", "--format=%s"]
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    last_commit_name = process.communicate()[0]

    if last_commit_name == b'"temp commit"\n':
        cmd = ["git", "reset", "--soft", "HEAD~1"]
        subprocess.run(cmd, check=True)
    else:
        print("There is not temp commit to restore.")


def restore_src():
    cmd = ["git", "checkout", "HEAD", "pandas-stubs"]
    subprocess.run(cmd, check=True)


def clean_mypy_cache():
    if Path('.mypy_cache').exists():
        shutil.rmtree('.mypy_cache')


def clean_pytest_cache():
    if Path('.mypy_cache').exists():
        shutil.rmtree('.pytest_cache')


def create_new_venv():
    cmd = ["poetry", "remove", "python"]
    subprocess.run(cmd, check=True)

    cmd = ["poetry", "update", "-vvv"]
    subprocess.run(cmd, check=True)

    cmd = ["poetry", "shell"]
    subprocess.run(cmd, check=True)


if __name__ == '__main__':
    restore_last_changes()