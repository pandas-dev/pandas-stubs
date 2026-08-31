#!/usr/bin/env python3
"""
Architecture Quality Assurance (AQA) - Subsystem Validator.

Prevents AI hallucinations in Subsystem Guides by enforcing:
1. Every subsystem guide MUST map to at least one real physical file in the repository.
2. Any referenced `pandas-stubs/...` path must actually exist on disk.
"""

from pathlib import Path
import re
import sys


def validate_subsystem(filepath: Path, repo_root: Path) -> bool:
    content = filepath.read_text(encoding="utf-8")

    # Extract all file paths that look like pandas-stubs/...
    paths = re.findall(r"pandas-stubs/[\w/.-]+\.pyi?", content)
    unique_paths = list(set(paths))

    if not unique_paths:
        sys.stdout.write(
            f"❌ [AQA Failed] {filepath.name}: No physical modules referenced. Subsystem guides must map to real code.\n"
        )
        return False

    failed = False
    for path in unique_paths:
        full_path = repo_root / path
        if not full_path.exists():
            sys.stdout.write(
                f"❌ [AQA Failed] {filepath.name} references a HALLUCINATED module: {path}\n"
            )
            failed = True

    return not failed


def main() -> None:
    repo_root = Path(__file__).parent.parent
    subs_dir = repo_root / "docs" / "architecture" / "subsystems"

    failed = False
    for sub in subs_dir.glob("*.md"):
        if not validate_subsystem(sub, repo_root):
            failed = True

    if failed:
        sys.stdout.write(
            "\n❌ Architecture Quality Assurance (AQA) Pipeline Failed. Phantom modules detected.\n"
        )
        sys.exit(1)

    sys.stdout.write(
        "\n✅ AQA Passed: All Subsystem guides map to physical, existing modules.\n"
    )
    sys.exit(0)


if __name__ == "__main__":
    main()
