"""
Legacy entrypoint preserved for reference.
Note: The canonical entrypoint is now src/main.py.
"""

from __future__ import annotations

import runpy

if __name__ == "__main__":
    # Execute the old root main script as-is to preserve behavior
    runpy.run_module("src.main", run_name="__main__")
