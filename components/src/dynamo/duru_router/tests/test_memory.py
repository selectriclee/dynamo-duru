"""Memory transport parity, bounded storage, and terminal lifecycle accounting."""

import json
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from dynamo.duru_router import scoring
from dynamo.duru_router.events import open_feed
from dynamo.duru_router.scoring import EventStreamReader, LiveU, run
from dynamo.duru_router.tests.helpers import configuration
from dynamo.duru_router.tests.helpers import memory_mode as memory_mode
from dynamo.duru_router.tests.replay_helpers import replay_accounting
from dynamo.duru_router.transport import Sender

pytestmark = [pytest.mark.pre_merge, pytest.mark.unit, pytest.mark.gpu_0]


def read_until(receiver, count):
    rows = []
    deadline = time.monotonic() + 5
    while len(rows) < count and time.monotonic() < deadline:
        rows.extend(receiver.read())
        time.sleep(0.001)
    assert len(rows) == count
    return rows


def test_bounded_queue_overflow_and_cross_process(tmp_path):
    path = tmp_path / "channel"
    sender = Sender(path, capacity=512)
    try:
        for i in range(3):
            sender.submit("record", dict(number=i, payload="x" * 80))
        with pytest.raises(RuntimeError, match="overflow"):
            for _ in range(10):
                sender.submit("record", dict(number=9, payload="x" * 80))
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "from dynamo.duru_router.transport import Receiver; import sys,time; "
                "r=Receiver(sys.argv[1]); time.sleep(.1); rows=r.read(); "
                'assert rows[0]["number"]==0; assert len(rows)>=3; r.close()',
                str(path),
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert result.returncode == 0, result.stderr
    finally:
        sender.close()
    assert not sender.thread.is_alive()
    assert sender.pending_bytes == 0


def test_default_feed_needs_no_journal_and_rejects_second_producer(tmp_path):
    feed = open_feed(tmp_path / "events", name="ingress")
    tail = EventStreamReader(tmp_path / "events/ingress.jsonl")
    try:
        feed.submit("record", dict(sequence=1, time=1, event="footer"))
        with pytest.raises(OSError):
            open_feed(tmp_path / "events", name="ingress")
        assert read_until(tail, 1) == [dict(sequence=1, time=1, event="footer")]
        assert tail.closed
        assert tail.read() == []
    finally:
        feed.close()
        tail.close()
    assert not (tmp_path / "events").exists()


def test_memory_sequence_gap_fails_closed(tmp_path):
    feed = open_feed(tmp_path, name="worker")
    feed.submit("record", dict(sequence=2, event="footer", time=2))
    tail = EventStreamReader(tmp_path / "worker.jsonl")
    try:
        with pytest.raises(ValueError, match="sequence gap"):
            read_until(tail, 1)
    finally:
        feed.close()
        tail.close()


def run_memory(path, *, ingress_source=None):
    stop = threading.Event()
    timer = threading.Timer(0.3, stop.set)
    timer.start()
    try:
        run(path, stop, ingress_source=ingress_source)
    finally:
        timer.cancel()
        timer.join()


def test_root_cleanup_retains_tool_gap_then_releases_all_history():
    book = LiveU(2, {})
    book.cost = lambda *args: 0.01
    for i in range(1000):
        root, agent = f"root-{i}", f"agent-{i}"
        t = i * 10
        book.submit("request", root, agent, 10, t)
        book.accepted(
            dict(
                root=root,
                agent=agent,
                accepted=True,
                phase="decode",
                p=1,
                d=10,
                k=11,
                generated_tokens=1,
            )
        )
        with pytest.raises(ValueError, match="outstanding requests"):
            book.finish_root(root, t)
        book.finish("request", t + 1)
        book.advance(t + 5)
        assert book.measured(root) == 100
        book.submit("return", root, agent, 11, t + 5)
        book.finish("return", t + 6)
        assert book.measured(root) == 200
        book.finish_root(root, t + 7)
        assert not any(
            (
                book.active,
                book.agent_root,
                book.agent_active_requests,
                book.root_active,
                book.work,
                book.entitlement,
                book.invalid,
            )
        )
    assert book.output_tokens == book.accepted_steps == 1000


@pytest.mark.parametrize("completed", [False, True])
@pytest.mark.parametrize("in_process", [False, True])
def test_identical_events_have_identical_u_in_ipc_and_recorded_replay(
    tmp_path, monkeypatch, completed, in_process
):
    path, cfg = configuration(tmp_path)
    monkeypatch.setenv("DURU_RECORD_EVENTS", "1")
    ingress = [
        dict(
            event="request_arrived",
            time=100,
            request_id="q",
            root="r",
            agent="a",
            program_id="p",
            prompt_tokens=8,
        ),
        dict(event="request_completed", time=101, request_id="q"),
        # Callback at 102 arrives after ingress completion at 101.
        dict(
            event="request_arrived",
            time=103,
            request_id="q2",
            root="r",
            agent="a",
            program_id="p",
            prompt_tokens=8,
        ),
        dict(
            event="request_arrived",
            time=103,
            request_id="child",
            root="r",
            agent="b",
            program_id="pb",
            prompt_tokens=8,
        ),
        dict(event="request_completed", time=104, request_id="q2"),
        dict(event="request_completed", time=105, request_id="child"),
    ]
    if completed:
        ingress += [
            dict(event="agent_end", time=106, program_id="p"),
            dict(event="agent_group_end", time=107, root="r"),
        ]
    ingress.append(dict(event="footer", time=108))
    step = dict(
        client_request_id="q",
        root="r",
        agent="a",
        scheduled_at=100.5,
        end=102,
        phase="prefill",
        p=8,
        d=0,
        k=8,
        accepted=True,
        generated_tokens=1,
    )
    worker = [
        dict(event="accepted_output", time=102, steps=[step]),
        dict(event="footer", time=109),
    ]
    if completed:
        ingress.insert(1, dict(event="request_dispatched", time=100.1, request_id="q"))
        # Final ingress can precede an engine callback. Keep the root until
        # its worker acknowledgement, including across replay.
        worker[-1:-1] = [
            dict(event="accepted_output", time=107.2, steps=[dict(step, end=107.2)]),
            dict(
                event="worker_request_finished",
                time=107.3,
                requests=[dict(client_request_id="q", root="r", agent="a")],
            ),
        ]
    feeds = []
    ingress_source = None
    for logical, rows in [(cfg["ingress"], ingress), (cfg["ledgers"][0], worker)]:
        logical = Path(logical)
        if in_process and str(logical) == cfg["ingress"]:
            local = scoring.Ingress(logical.parent, in_process=True)
            feed = ingress_source = local.journal
        else:
            feed = open_feed(logical.parent, name=logical.stem)
        feeds.append(feed)
        for i, row in enumerate(rows, 1):
            feed.submit("record", dict(row, sequence=i))
    snapshots = []
    stop = threading.Event()
    stop.set()
    for mode in ("memory", "journal"):
        monkeypatch.setenv("DURU_ACCOUNTING_TRANSPORT", mode)
        if mode == "memory":
            try:
                run_memory(path, ingress_source=ingress_source)
            finally:
                for feed in feeds:
                    feed.close()
        else:
            replay_accounting(path, stop)
        footer = json.loads(
            (Path(cfg["audit_dir"]) / "duru-control.jsonl").read_text().splitlines()[-1]
        )
        assert footer["complete"]
        snapshots.append(footer["last_snapshot"])
    for key in (
        "roots",
        "programs",
        "requests",
        "accounting_state",
        "accepted_steps",
        "output_tokens",
    ):
        assert snapshots[0][key] == snapshots[1][key]
    if completed:
        assert snapshots[0]["accepted_steps"] == 2
        assert not snapshots[0]["roots"]
        assert not any(snapshots[0]["accounting_state"].values())
    else:
        assert snapshots[0]["roots"]["r"]["U"] == pytest.approx(300)
        assert snapshots[0]["programs"]["p"]["kind"] == "idle_measured_only"


@pytest.mark.parametrize("producer", ["ingress", "local_ingress", "worker"])
def test_pending_limit_applies_with_missing_peer_watermark(
    tmp_path, monkeypatch, producer
):
    path, cfg = configuration(tmp_path)
    monkeypatch.setattr(scoring, "MAX_PENDING_EVENTS", 2)
    logical = Path(cfg["ingress"] if producer != "worker" else cfg["ledgers"][0])
    if producer == "local_ingress":
        feed = scoring.Ingress(logical.parent, in_process=True).journal
    else:
        feed = open_feed(logical.parent, name=logical.stem)
    try:
        for sequence in range(1, 4):
            row = (
                dict(
                    event="request_arrived",
                    request_id=str(sequence),
                    program_id="p",
                    root="r",
                    agent="a",
                    prompt_tokens=8,
                )
                if producer != "worker"
                else dict(event="accepted_output", steps=[dict(end=1)])
            )
            feed.submit("record", dict(sequence=sequence, time=1, **row))
        with pytest.raises(RuntimeError, match="pending-event limit"):
            run_memory(
                path, ingress_source=feed if producer == "local_ingress" else None
            )
    finally:
        feed.close()
