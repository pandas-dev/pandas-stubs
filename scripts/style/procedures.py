import subprocess


def run_black_check():
    cmd = ["black", "--check", "pandas-stubs", "tests"]
    subprocess.run(cmd, check=True)


def run_isort_check():
    cmd = ["isort", "--check-only", "pandas-stubs", "tests"]
    subprocess.run(cmd, check=True)
