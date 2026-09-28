#!/usr/bin/env python3
"""Script entry point; CI and the guide run `python scripts/check_operand_hierarchy.py`."""

import sys

from operand_hierarchy.check import main

if __name__ == "__main__":
    sys.exit(main())
