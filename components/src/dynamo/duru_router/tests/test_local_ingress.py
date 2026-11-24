# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Router events stay local without changing ordering or overflow semantics."""

import pickle
import threading
from pathlib import Path

import pytest

from dynamo.duru_router import scoring, transport
from dynamo.duru_router.transport import InProcessChannel

pytestmark = [pytest.mark.pre_merge, pytest.mark.unit, pytest.mark.gpu_0]


def test_local_ingress_needs_no_socket_thread_or_directory(tmp_path, monkeypatch):
    def unexpected(*args, **kwargs):
        pytest.fail("Local ingress attempted external I/O or started a thread")

    monkeypatch.delenv("DURU_RECORD_EVENTS", raising=False)
    monkeypatch.setattr(transport.socket, "socket", unexpected)
    monkeypatch.setattr(threading.Thread, "start", unexpected)
    monkeypatch.setattr(Path, "mkdir", unexpected)
    path = tmp_path / "unused"
    ingress = scoring.Ingress(path, in_process=True)
    owner = scoring.RouterAccounting("unused", ingress=ingress)
    reader = scoring.EventStreamReader(path, receiver=ingress.journal)
    try:
        ingress.emit(
            "request_arrived",
            1,
            request_id="q",
            program_id="p",
            root="r",
            agent="a",
            prompt_tokens=8,
        )
        ingress.emit("request_completed", 2, request_id="q")
        ingress.close()
        rows = reader.read()
        assert [row["event"] for row in rows] == [
            "request_arrived",
            "request_completed",
            "footer",
        ]
        assert [row["sequence"] for row in rows] == [1, 2, 3]
        assert reader.closed
        assert reader.read() == []
        assert not path.exists()
    finally:
        reader.close()
        owner.close()


def test_local_channel_snapshots_values_and_drains_after_close():
    channel = InProcessChannel()
    row = {"event": "request_arrived", "request_id": "before"}
    channel.submit("record", row)
    row["request_id"] = "after"
    channel.submit("record", {"event": "footer"})
    channel.close()
    with pytest.raises(RuntimeError, match="closed"):
        channel.submit("record", row)
    assert channel.read() == [
        {"event": "request_arrived", "request_id": "before"},
        {"event": "footer"},
    ]
    assert channel.pending_bytes == 0
    channel.close()


@pytest.mark.parametrize("limit", ["bytes", "records"])
def test_local_channel_overflow_preserves_accepted_records(limit):
    row = {"event": "u_clock", "time": 1}
    size = len(pickle.dumps(row, protocol=5))
    channel = InProcessChannel(
        capacity=2 * size if limit == "bytes" else transport.CAPACITY,
        max_records=2 if limit == "records" else 8192,
    )
    try:
        channel.submit("record", row)
        channel.submit("record", row)
        with pytest.raises(RuntimeError, match="overflow"):
            channel.submit("record", row)
        # Reading one record frees precisely its byte/record budget.
        assert channel.read(budget=1) == [row]
        assert channel.pending_bytes == size
        channel.submit("record", row)
        assert channel.read() == [row, row]
        assert channel.pending_bytes == 0
    finally:
        channel.close()


def test_local_channel_rejects_oversized_frames(monkeypatch):
    monkeypatch.setattr(transport, "MAX_FRAME", 8)
    channel = InProcessChannel()
    try:
        with pytest.raises(RuntimeError, match="frame exceeds memory limit"):
            channel.submit("record", {"payload": "x" * 32})
        assert channel.read() == []
        assert channel.pending_bytes == 0
    finally:
        channel.close()


def test_local_ingress_keeps_sequence_validation(tmp_path):
    channel = InProcessChannel()
    reader = scoring.EventStreamReader(tmp_path / "unused", receiver=channel)
    try:
        channel.submit("record", dict(sequence=2, event="footer", time=1))
        with pytest.raises(ValueError, match="sequence gap"):
            reader.read()
    finally:
        reader.close()


def test_local_channel_concurrent_producer_and_consumer():
    channel = InProcessChannel()
    started = threading.Event()
    first_read = threading.Event()
    errors = []

    def produce():
        try:
            for number in range(100):
                channel.submit("record", {"number": number})
                if number == 0:
                    started.set()
                    if not first_read.wait(5):
                        raise TimeoutError("Consumer did not read the first record")
        except Exception as exc:
            errors.append(exc)
        finally:
            channel.close()

    producer = threading.Thread(target=produce)
    producer.start()
    rows = []
    try:
        assert started.wait(5)
        rows.extend(channel.read())
    finally:
        first_read.set()
        producer.join(5)
    assert not producer.is_alive()
    assert not errors
    rows.extend(channel.read())
    assert [row["number"] for row in rows] == list(range(100))
    assert channel.pending_bytes == 0
