"""
Thin entrypoint that forwards execution to src/main.py.

Run: python ./main.py
"""
from __future__ import annotations
import importlib


def main() -> int:
    mod = importlib.import_module("src.main")
    if hasattr(mod, "main") and callable(getattr(mod, "main")):
        return int(mod.main())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
