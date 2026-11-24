"""Router-local score access and lifecycle, without a scorer subprocess."""

import threading
import time
from pathlib import Path

import pytest

from dynamo.duru_router import scoring
from dynamo.duru_router.events import open_feed
from dynamo.duru_router.tests.helpers import configuration
from dynamo.duru_router.tests.helpers import memory_mode as memory_mode

pytestmark = [pytest.mark.pre_merge, pytest.mark.unit, pytest.mark.gpu_0]


def test_accounting_failure_invalidates_local_scores(monkeypatch):
    def fail(*args, **kwargs):
        raise ValueError("bad event")

    monkeypatch.setattr(scoring, "run", fail)
    owner = scoring.RouterAccounting("unused").start()
    owner._thread.join(2)
    with pytest.raises(RuntimeError, match="accounting failed"):
        owner.reader.refresh()
    assert owner._snapshot == {}
    with pytest.raises(RuntimeError, match="accounting failed"):
        owner.close()


@pytest.mark.parametrize(
    "record,diagnostics", [(False, False), (False, True), (True, False)]
)
@pytest.mark.parametrize("in_process", [False, True])
def test_router_owned_accounting_matches_worker_snapshot(
    tmp_path, monkeypatch, record, diagnostics, in_process
):
    path, cfg = configuration(tmp_path)
    monkeypatch.setenv("DURU_SCORES", cfg["scores"])
    monkeypatch.setenv("DURU_RECORD_EVENTS", "1" if record else "0")
    monkeypatch.setenv("DURU_DIAGNOSTICS", "1" if diagnostics else "0")
    now = time.time()
    step = dict(
        root="r",
        agent="a",
        client_request_id="q",
        accepted=True,
        phase="prefill",
        p=8,
        d=0,
        k=8,
        generated_tokens=1,
        scheduled_at=now,
        end=now + 0.001,
    )
    streams = [
        (
            cfg["ingress"],
            [
                dict(
                    event="request_arrived",
                    time=now - 0.01,
                    root="r",
                    agent="a",
                    request_id="q",
                    program_id="p",
                    prompt_tokens=8,
                ),
                dict(event="footer", time=now + 0.002),
            ],
        ),
        (
            cfg["ledgers"][0],
            [
                dict(event="accepted_output", time=now + 0.001, steps=[step]),
                dict(event="footer", time=now + 0.002),
            ],
        ),
    ]
    feeds = []
    ingress = None
    for logical, rows in streams:
        logical = Path(logical)
        if in_process and str(logical) == cfg["ingress"]:
            ingress = scoring.Ingress(logical.parent, in_process=True)
            feed = ingress.journal
        else:
            feed = open_feed(logical.parent, name=logical.stem)
        feeds.append(feed)
        for i, row in enumerate(rows, 1):
            feed.submit("record", dict(sequence=i, **row))
    owner = scoring.RouterAccounting(path, ingress=ingress).start()
    engine_reader = scoring.Reader()
    try:
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            snapshot = owner.reader.refresh()
            engine_snapshot = engine_reader.refresh()
            if snapshot.get("accepted_steps") == 1 and engine_snapshot.get("requests"):
                break
            time.sleep(0.01)
        else:
            pytest.fail("Router accounting failed to publish")
        assert not Path(cfg["scores"]).exists()
        assert engine_snapshot["requests"] == {
            rid: {"score": row["score"]} for rid, row in snapshot["requests"].items()
        }
        assert ("roots" in snapshot) is (record or diagnostics)
        assert ("source_sequences" in snapshot) is record
        assert ("source_received_through" in snapshot) is record
        if not record:
            assert not Path(cfg["audit_dir"]).exists()
            assert not list(tmp_path.rglob("*.jsonl"))
        assert owner.reader.snapshot_source is not None
        assert owner._thread is not threading.current_thread()
        # File corruption cannot change the router's direct-memory decision.
        monkeypatch.setattr(
            Path, "read_text", lambda *a, **k: pytest.fail("Unexpected file read")
        )
        assert owner.reader.refresh()["programs"]["p"]["U"] is not None
    finally:
        for feed in feeds:
            feed.close()
        owner.close()
        engine_reader.close()
    assert not owner._thread.is_alive()
    assert owner._snapshot == {}
