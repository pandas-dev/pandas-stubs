import subprocess


def run_black_check():
    cmd = ["black", "--check", "pandas-stubs", "tests", "scripts"]
    subprocess.run(cmd, check=True)


def run_isort_check():
    cmd = ["isort", "--check-only", "pandas-stubs", "tests", "scripts"]
    subprocess.run(cmd, check=True)


def run_format_black():
    cmd = ["black", "pandas-stubs", "tests", "scripts"]
    subprocess.run(cmd, check=True)


def run_format_isort():
    cmd = ["isort", "pandas-stubs", "tests", "scripts"]
    subprocess.run(cmd, check=True)
