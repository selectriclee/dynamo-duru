"""Latest-score IPC, fail-closed freshness, and actual process separation."""

import json
import os
import socket
import struct
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

import pytest

from dynamo.duru_router.scoring import Reader
from dynamo.duru_router.transport import MAX_FRAME, ScorePublisher, channel_path

pytestmark = [
    pytest.mark.post_merge,
    pytest.mark.integration,
    pytest.mark.gpu_0,
    pytest.mark.timeout(30),
]


@pytest.fixture
def endpoint(monkeypatch):
    with tempfile.TemporaryDirectory(prefix="duru-score-test-") as directory:
        monkeypatch.setenv("DURU_IPC_DIR", directory)
        logical = Path(directory) / "scores"
        monkeypatch.setenv("DURU_SCORES", str(logical))
        yield logical, channel_path(logical)


def snapshot(value, at=None):
    return dict(
        run_id="test",
        observation_time=time.time() if at is None else at,
        requests={"r": dict(score=value, root="not-needed-by-worker")},
        roots={"r": {"U": value}},
    )


def wait(reader, condition):
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        current = reader.refresh()
        if condition(current):
            return current
        time.sleep(0.01)
    pytest.fail("Score IPC did not reach expected state")


def test_latest_only_projection_and_no_score_file(endpoint):
    logical, address = endpoint
    publisher = ScorePublisher(address)
    reader = Reader()
    try:
        publisher.publish(snapshot(1))
        first = wait(
            reader, lambda s: s.get("requests", {}).get("r", {}).get("score") == 1
        )
        assert first["requests"] == {"r": {"score": 1}}
        assert "roots" not in first
        publisher.publish(snapshot(2))
        with reader.pin(first):
            assert reader.refresh() is first
        second = wait(
            reader, lambda s: s.get("requests", {}).get("r", {}).get("score") == 2
        )
        assert first["requests"]["r"]["score"] == 1
        assert second is not first
        assert not logical.exists()
        publisher.publish({})
        wait(reader, lambda s: not s)
    finally:
        reader.close()
        publisher.close()
    assert not reader._subscriber._thread.is_alive()
    assert not publisher._thread.is_alive()


def test_missing_stale_future_and_disconnect_never_use_file(endpoint, monkeypatch):
    logical, address = endpoint
    logical.write_text(json.dumps(snapshot(999)))
    monkeypatch.setattr(
        Path, "read_text", lambda *a, **k: pytest.fail("Worker read a score file")
    )
    reader = Reader()
    publisher = None
    try:
        assert reader.refresh() == {}
        publisher = ScorePublisher(address)
        publisher.publish(snapshot(1, time.time() - 10))
        wait(reader, lambda s: reader.reason == "stale_snapshot")
        assert reader.snapshot["requests"]["r"]["score"] == 1
        publisher.publish(snapshot(2, time.time() + 10))
        wait(reader, lambda s: reader.reason == "future_snapshot")
        publisher.publish(snapshot(3))
        wait(reader, lambda s: bool(s))
        publisher.close()
        publisher = None
        wait(reader, lambda s: not s)
        assert reader.reason == "missing_snapshot"
    finally:
        reader.close()
        if publisher is not None:
            publisher.close()


def test_scores_reach_separate_worker_processes(endpoint):
    _, address = endpoint
    publisher = ScorePublisher(address)
    publisher.publish(snapshot(7))
    code = """import os,time,json
from dynamo.duru_router.scoring import Reader
r=Reader();deadline=time.monotonic()+5
try:
 while time.monotonic()<deadline:
  s=r.refresh()
  if s.get('requests',{}).get('r',{}).get('score')==7:
   print(json.dumps(dict(pid=os.getpid(),score=7)));break
  time.sleep(.01)
 else: raise RuntimeError('No scores')
finally: r.close()
"""
    processes = [
        subprocess.Popen(
            [sys.executable, "-c", code],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        for _ in range(3)
    ]
    try:
        for process in processes:
            output, error = process.communicate(timeout=10)
            assert process.returncode == 0, error
            result = json.loads(output)
            assert result["pid"] != os.getpid()
            assert result["score"] == 7
    finally:
        for process in processes:
            if process.poll() is None:
                process.kill()
                process.wait()
        publisher.close()


@pytest.mark.parametrize(
    "frame", [struct.pack("!I", MAX_FRAME + 1), struct.pack("!I", 100) + b"truncated"]
)
def test_bad_frame_fails_closed(endpoint, frame):
    _, address = endpoint
    listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    listener.bind(str(address))
    listener.listen(1)
    listener.settimeout(2)

    def serve():
        with listener:
            peer, _ = listener.accept()
            with peer:
                peer.sendall(frame)

    thread = threading.Thread(target=serve)
    thread.start()
    reader = Reader()
    try:
        wait(reader, lambda s: reader._subscriber.failures > 0)
        assert reader.refresh() == {}
    finally:
        reader.close()
        thread.join(3)
    assert not thread.is_alive()
