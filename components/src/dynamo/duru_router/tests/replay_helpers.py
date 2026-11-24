"""Test-only replay of recorded inputs; production accounting only reads IPC."""

import json
import os
from pathlib import Path
from unittest.mock import patch

from dynamo.duru_router import scoring


class Tail:
    def __init__(self, path):
        self.path = Path(path)
        self.offset = 0
        self.sequence = 0
        self.watermark = 0.0
        self.closed = False

    def read(self):
        if not self.path.exists():
            return []
        assert self.path.stat().st_size >= self.offset, "Journal truncated"
        result = []
        with self.path.open("rb") as f:
            f.seek(self.offset)
            while True:
                pos = f.tell()
                line = f.readline()
                if not line.endswith(b"\n"):
                    f.seek(pos)
                    break
                if self.closed:
                    raise ValueError(f"Event after journal footer: {self.path}")
                r = json.loads(line)
                self.closed = r["event"] == "footer"
                self.sequence += 1
                assert r["sequence"] == self.sequence, (
                    "Journal gap",
                    self.path,
                    self.sequence,
                )
                self.watermark = max(self.watermark, r["time"])
                result.append(r)
            self.offset = f.tell()
        return result

    def close(self):
        pass


def replay_accounting(config, stop=None, *, publish=None):
    with (
        patch.object(scoring, "MemoryTail", Tail),
        patch.dict(os.environ, DURU_ACCOUNTING_TRANSPORT="memory"),
    ):
        scoring.run(config, stop, publish=publish)
