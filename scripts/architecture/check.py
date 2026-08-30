"""Offline integrity checks for generated architecture documentation."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
import sys
from typing import TYPE_CHECKING

from generate import (
    DATA,
    ROOT,
    check as check_generated,
    read_jsonl,
)

if TYPE_CHECKING:
    from pathlib import Path

GIT = shutil.which("git") or "git"
_MERGE_REV_LIST_MIN_FIELDS = 3


def fail(message: str) -> None:
    sys.stderr.write(f"architecture check failed: {message}\n")
    raise SystemExit(1)


def normalized_snapshot_bytes(path: Path) -> bytes:
    """Return text snapshot bytes in the canonical LF representation."""
    return path.read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")


def verify_manifest() -> dict[str, object]:
    manifest = json.loads((DATA / "manifest.json").read_text(encoding="utf-8"))
    for name, expected in manifest["sha256"].items():
        actual = hashlib.sha256(normalized_snapshot_bytes(DATA / name)).hexdigest()
        if actual != expected:
            fail(f"snapshot hash mismatch for {name}")
    return manifest


def git(*args: str) -> str:
    return subprocess.check_output([GIT, *args], cwd=ROOT, text=True).strip()


def verify_references(manifest: dict[str, object]) -> None:
    prs = {
        record["number"]: record for record in read_jsonl(DATA / "pull_requests.jsonl")
    }
    base = str(manifest["base_commit"])
    try:
        git("cat-file", "-e", f"{base}^{{commit}}")
    except subprocess.CalledProcessError:
        fail("recorded base commit is unavailable in this checkout")
    for path in (ROOT / "docs" / "architecture" / "generated").glob("**/*.md"):
        for number in re.findall(
            r"pandas-dev/pandas-stubs#(\d+)", path.read_text(encoding="utf-8")
        ):
            record = prs.get(int(number))
            if record is None:
                fail(f"unknown PR #{number} in {path.relative_to(ROOT)}")
            if record["state"] != "merged":
                fail(f"PR #{number} is not merged")
            commit = str(record["merge_commit"])
            try:
                parents = git("rev-list", "--parents", "-n", "1", commit)
            except subprocess.CalledProcessError:
                fail(f"merge commit for PR #{number} is unavailable in this checkout")
            if len(parents.split()) < _MERGE_REV_LIST_MIN_FIELDS:
                fail(f"PR #{number} does not identify a merge commit")
            if subprocess.run(
                [GIT, "merge-base", "--is-ancestor", commit, base],
                cwd=ROOT,
                check=False,
            ).returncode:
                fail(
                    f"merge commit for PR #{number} is not reachable from the recorded base"
                )


def verify_links() -> None:
    for path in (ROOT / "docs" / "architecture").glob("**/*.md"):
        for target in re.findall(
            r"\[[^]]*\]\(([^)#]+)(?:#[^)]*)?\)", path.read_text(encoding="utf-8")
        ):
            if "://" in target or target.startswith("mailto:"):
                continue
            if not (path.parent / target).resolve().exists():
                fail(f"broken link {target!r} in {path.relative_to(ROOT)}")


def verify_adr_frontmatter() -> None:
    for path in (ROOT / "docs" / "architecture" / "decisions").glob("*.md"):
        content = path.read_text(encoding="utf-8")
        if not content.startswith("---\n"):
            fail(f"missing frontmatter in {path.relative_to(ROOT)}")
        frontmatter = content.split("---", 2)[1]
        for key in ("status:", "date:", "deciders:"):
            if key not in frontmatter:
                fail(f"missing {key[:-1]} in {path.relative_to(ROOT)}")


def main() -> None:
    manifest = verify_manifest()
    verify_references(manifest)
    verify_links()
    verify_adr_frontmatter()
    if not check_generated():
        raise SystemExit(1)
    sys.stdout.write("architecture documentation is valid\n")


if __name__ == "__main__":
    main()
