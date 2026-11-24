# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared CPU accounting configuration and isolated IPC environment."""

import json
import tempfile

import pytest

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
