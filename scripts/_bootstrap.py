from __future__ import annotations

import sys
from pathlib import Path


def bootstrap_src_path() -> None:
    src_path = Path(__file__).resolve().parents[1] / "src"
    src = str(src_path)
    if src not in sys.path:
        sys.path.insert(0, src)
