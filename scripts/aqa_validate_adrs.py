#!/usr/bin/env python3
"""Compatibility entrypoint for the strict offline architecture checker."""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys

if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    raise SystemExit(
        subprocess.call([sys.executable, "scripts/architecture/check.py"], cwd=root)
    )
