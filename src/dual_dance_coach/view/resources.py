"""View-level helpers for resolving bundled resources (images, etc.).

These helpers are intentionally simple and avoid Qt's .qrc system so the app
can be run both from source and from an installed package.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path


@lru_cache(maxsize=None)
def resolve_blob_path(filename: str) -> Path:
    """Resolve a file under the project's ``blob/`` folder.

    Search order:
    1) Current working directory: ``./blob/<filename>``
    2) Repo root when running from source: ``<repo>/blob/<filename>``
    3) Installed package layouts: ``.../site-packages/blob/<filename>`` (best-effort)

    Returns the first existing path; otherwise returns the first candidate.
    """

    here = Path(__file__).resolve()

    candidates = [
        Path.cwd() / "blob" / filename,
        # src/dual_dance_coach/view/resources.py -> repo root is parents[3]
        here.parents[3] / "blob" / filename,
        # src/dual_dance_coach/view/resources.py -> src root is parents[2]
        here.parents[2] / "blob" / filename,
    ]

    for p in candidates:
        if p.exists():
            return p

    return candidates[0]
