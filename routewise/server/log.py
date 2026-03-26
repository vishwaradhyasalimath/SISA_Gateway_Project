"""
Lightweight request logger.

Writes every event to a JSONL file (one JSON object per line) and keeps
the last 2000 entries in memory for the dashboard. The in-memory store
is a simple list with a manual size cap — no external dependencies.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

_LOG_PATH = Path(os.getenv("LOG_FILE", "logs/requests.jsonl"))
_buffer: list[dict] = []
_MAX = 2000


def append(record: dict[str, Any]):
    record.setdefault("ts", time.time())
    _buffer.append(record)
    if len(_buffer) > _MAX:
        del _buffer[0]
    _LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(_LOG_PATH, "a") as fh:
            fh.write(json.dumps(record) + "\n")
    except OSError:
        pass  # logging should never crash the server


def get_recent(n: int = 200) -> list[dict]:
    return list(reversed(_buffer[-n:]))


def get_all() -> list[dict]:
    return list(reversed(_buffer))
