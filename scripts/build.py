import subprocess


def run_build():
    # Update project version
    cmd = ["poetry-dynamic-versioning"]
    subprocess.run(cmd, check=True)

    cmd = ["poetry", "build"]
    subprocess.run(cmd, check=True)
