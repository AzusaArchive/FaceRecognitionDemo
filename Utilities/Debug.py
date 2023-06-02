from __future__ import annotations

from datetime import datetime
from typing import Any

debug = True

debugEnabledModule = ["App","DetectionInfo"]


def Log(source: str, *obj: Any):
    if debug and source in debugEnabledModule:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] {source}: ", end="")
        print(*obj)
