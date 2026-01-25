"""Remove all __pycache__ directories under the **current** working directory."""

import shutil
from pathlib import Path


def main() -> int:
    root = Path.cwd()
    removed = 0
    for path in root.rglob("__pycache__"):
        if path.is_dir():
            shutil.rmtree(path, ignore_errors=True)
            removed += 1
    print(f"Removed {removed} __pycache__ director{'y' if removed == 1 else 'ies'} under {root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
