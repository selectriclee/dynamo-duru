# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared CPU fixtures and recorded-event replay helpers."""

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from dynamo.duru_router import scoring
from dynamo.duru_router.calibration.plan import METRIC


@pytest.fixture(autouse=True)
def memory_mode(tmp_path, monkeypatch):
    monkeypatch.setenv("DURU_ACCOUNTING_TRANSPORT", "memory")
    monkeypatch.delenv("DURU_RECORD_EVENTS", raising=False)
    with tempfile.TemporaryDirectory(prefix="duru-test-") as directory:
        monkeypatch.setenv("DURU_IPC_DIR", directory)
        yield


def configuration(tmp_path):
    calibration = tmp_path / "calibration.json"
    calibration.write_text(
        json.dumps(
            dict(
                schema_version=1,
                timing_metric=METRIC,
                contract={"model": "cpu"},
                max_prefill=8,
                prefill=dict(P=[1, 8], D=[0, 8], wall_ms=[[10, 10], [10, 10]]),
                decode=dict(grid=[dict(K=1, wall_ms=10), dict(K=16, wall_ms=10)]),
            )
        )
    )
    cfg = dict(
        ingress=str(tmp_path / "ingress/duru-ingress.jsonl"),
        ledgers=[str(tmp_path / "worker/compact-work.jsonl")],
        scores=str(tmp_path / "scores.json"),
        capacity=1,
        calibration=str(calibration),
        runtime_contract={"model": "cpu"},
        audit_dir=str(tmp_path / "audit"),
    )
    path = tmp_path / "config.json"
    path.write_text(json.dumps(cfg))
    return path, cfg


class Tail:
    def __init__(self, path):
        self.path = Path(path)
        self.offset = 0
        self.sequence = 0
        self.received_through = 0.0
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
                self.received_through = max(self.received_through, r["time"])
                result.append(r)
            self.offset = f.tell()
        return result

    def close(self):
        pass


def replay_accounting(config, stop=None, *, router_publish=None):
    with (
        patch.object(scoring, "EventStreamReader", Tail),
        patch.dict(os.environ, DURU_ACCOUNTING_TRANSPORT="memory"),
    ):
        scoring.run(config, stop, router_publish=router_publish)
