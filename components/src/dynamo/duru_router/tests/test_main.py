# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared request handling with the real Duru admission scheduler."""

import asyncio
import copy
import os
import struct
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from dynamo.duru_router import __main__ as main_module
from dynamo.duru_router.__main__ import DuruRouterHandler
from dynamo.duru_router.events import CompactFeed
from dynamo.duru_router.router import ThunderAgentConfig
from dynamo.duru_router.scoring import Ingress, RouterAccounting
from dynamo.duru_router.tests.helpers import configuration
from dynamo.duru_router.tests.helpers import memory_mode as memory_mode
from dynamo.thunderagent_router.tests.test_main import FakeKvRouter

pytestmark = [pytest.mark.pre_merge, pytest.mark.unit, pytest.mark.gpu_0]


class AccountingKvRouter(FakeKvRouter):
    def __init__(self, feed, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.feed = feed

    async def generate_from_request(self, request, **kwargs):
        stream = await super().generate_from_request(request, **kwargs)
        extra = request["extra_args"]["sampling_options"]["vllm_xargs"]
        identity = dict(
            client_request_id=extra["duru_request_id"],
            root=extra["duru_root_id"],
            agent=extra["duru_agent_id"],
        )
        now = time.time()
        self.feed.capture(
            "accepted_output",
            dict(
                steps=[
                    dict(
                        identity,
                        accepted=True,
                        scheduled_at=now,
                        end=now,
                        phase="prefill",
                        p=8,
                        d=0,
                        k=8,
                        generated_tokens=1,
                    )
                ]
            ),
        )
        self.feed.emit("request_end", requests=[identity])
        return stream


@pytest.fixture
async def handler(monkeypatch, tmp_path):
    path, cfg = configuration(tmp_path)
    monkeypatch.setenv("DURU_DIAGNOSTICS", "1")
    monkeypatch.delenv("DURU_ADMISSION_ALPHA", raising=False)
    monkeypatch.setenv("DURU_SCORES", cfg["scores"])
    ingress = Ingress(Path(cfg["ingress"]).parent)
    feed = CompactFeed(Path(cfg["ledgers"][0]).parent)
    owner = RouterAccounting(path).start()
    capacities = {(7, 2): 10, (7, 3): 10}
    capacity = SimpleNamespace(snapshot=lambda: capacities, live_worker_ids=lambda: {7})
    result = DuruRouterHandler(
        runtime=None,
        config=SimpleNamespace(namespace="test", endpoint="test.vllm.generate"),
        accounting=owner,
        ingress=ingress,
    )
    result._scheduler = result.scheduler_cls(
        capacity,
        ThunderAgentConfig(
            buffer_per_program=0,
            pause_threshold=1,
            resume_hysteresis=0,
            scheduler_interval_seconds=0.01,
        ),
    )
    result._scheduler.start()
    result._kv_router = AccountingKvRouter(feed)
    monitoring = asyncio.create_task(result.monitor())
    result.test_monitor = monitoring
    result.test_feed = feed
    try:
        yield result
    finally:
        monitoring.cancel()
        await asyncio.gather(monitoring, return_exceptions=True)
        await result.shutdown()
        results = await asyncio.gather(
            asyncio.to_thread(ingress.close),
            asyncio.to_thread(feed.close),
            return_exceptions=True,
        )
        if owner._error is not None:
            # Fault injection disconnects the producers; they may observe it
            # either on their next heartbeat or during their final flush.
            for result in results:
                assert result is None or isinstance(result, RuntimeError)
            with pytest.raises(RuntimeError, match="accounting failed"):
                await asyncio.to_thread(owner.close)
        else:
            assert results == [None, None]
            await asyncio.to_thread(owner.close)


async def snapshot_when(handler, predicate):
    async with asyncio.timeout(5):
        while True:
            snapshot = handler.accounting.snapshot()
            if snapshot and predicate(snapshot):
                return snapshot
            await asyncio.sleep(0.01)


async def generate(handler, pid, tokens=8, **kwargs):
    request = {
        "token_ids": struct.pack(f"{tokens}I", *range(tokens)),
        "agent_context": {"session_id": pid},
        "elapsed_since_arrival_ms": 12,
    }
    return [chunk async for chunk in handler.generate(request, **kwargs)]


async def test_dp_capacity_pins_and_packed_tokens(handler):
    handler._config = SimpleNamespace(namespace="test", endpoint="test.vllm.generate")
    context = object()
    await generate(handler, "first", context=context)
    assert handler._kv_router.sent["routing"]["backend_instance_id"] == 7
    assert handler._kv_router.sent["routing"]["dp_rank"] == 2
    assert handler._kv_router.sent["elapsed_since_arrival_ms"] == 12
    assert handler._kv_router.context is context
    await generate(handler, "second")
    assert handler._kv_router.sent["routing"]["dp_rank"] == 3
    status = await handler._scheduler.status_snapshot()
    assert {k: v["used"] for k, v in status["workers"].items()} == {"7:2": 8, "7:3": 8}
    status = [value async for value in handler.status()]
    metrics = [value async for value in handler.metrics()]
    assert status[0]["scheduler"]["programs_total"] == 2
    assert metrics[0]["counters"]["program_requests_total"] == 2


@pytest.mark.parametrize("terminal", [False, True])
async def test_stream_usage_keeps_duru_reservation_accurate(handler, terminal):
    chunks = [
        {"token_ids": [1], "completion_usage": {"completion_tokens": 1}},
        {"token_ids": [2, 3]},
    ]
    if terminal:
        chunks.append(
            {"finish_reason": "stop", "completion_usage": {"completion_tokens": 4}}
        )
    handler._kv_router = AccountingKvRouter(
        handler.test_feed,
        chunks,
        error=None if terminal else RuntimeError("disconnected"),
    )
    if terminal:
        await generate(handler, "stream", 4)
    else:
        with pytest.raises(RuntimeError, match="disconnected"):
            await generate(handler, "stream", 4)
    assert handler._scheduler._table.programs["stream"].token_total == (
        8 if terminal else 7
    )


@pytest.mark.parametrize("ending", ["cancel", "session_final"])
async def test_paused_request_never_dispatches_after_termination(
    handler, monkeypatch, ending
):
    scheduler = handler._scheduler
    entered = asyncio.Event()
    original = scheduler._wait_for_resume

    async def wait_for_resume(*args):
        entered.set()
        await original(*args)

    monkeypatch.setattr(scheduler, "_wait_for_resume", wait_for_resume)
    request = asyncio.create_task(generate(handler, "paused", 11))
    try:
        await asyncio.wait_for(entered.wait(), 1)
        assert handler._kv_router.sent is None
        if ending == "cancel":
            request.cancel()
        else:
            final = {"agent_context": {"session_id": "paused", "session_final": True}}
            assert [chunk async for chunk in handler.generate(final)] == []
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(request, 1)
        assert handler._kv_router.sent is None
        assert not scheduler._table.programs
        assert not scheduler._table.paused
        assert not scheduler._admission_gates
    finally:
        request.cancel()
        await asyncio.gather(request, return_exceptions=True)


def test_main_uses_duru_configuration_without_control_port(monkeypatch, tmp_path):
    monkeypatch.setenv("DURU_SCORES", str(tmp_path / "scores.json"))
    monkeypatch.setenv("DURU_ACCOUNTING_CONFIG", str(tmp_path / "scoring.json"))
    monkeypatch.delenv("DURU_ADMISSION_ALPHA", raising=False)
    monkeypatch.setattr(
        sys, "argv", ["duru", "--alpha", "0.7", "--endpoint", "test.vllm.generate"]
    )
    invocation = object()
    started = []
    monkeypatch.setattr(main_module, "worker", lambda: invocation)
    monkeypatch.setattr(main_module, "uvloop", SimpleNamespace(run=started.append))

    main_module.main()

    assert started == [invocation]
    assert os.environ["DURU_ADMISSION_ALPHA"] == "0.7"
    assert sys.argv == ["duru", "--endpoint", "test.vllm.generate"]


async def test_public_requests_join_worker_accounting_and_cleanup_family(handler):
    request = {
        "token_ids": [1, 2, 3],
        "agent_context": {"session_id": "parent"},
        "extra_args": {"sampling_options": {"vllm_xargs": {"custom": 1}}},
    }
    original = copy.deepcopy(request)
    assert [chunk async for chunk in handler.generate(request)] == []
    parent = handler._kv_router.sent["extra_args"]["sampling_options"]["vllm_xargs"]
    assert request == original
    assert parent["custom"] == 1
    assert [chunk async for chunk in handler.generate(request)] == []
    again = handler._kv_router.sent["extra_args"]["sampling_options"]["vllm_xargs"]
    assert again["duru_root_id"] == parent["duru_root_id"]
    assert again["duru_agent_id"] == parent["duru_agent_id"]
    assert again["duru_request_id"] != parent["duru_request_id"]
    child_request = dict(
        request, agent_context={"session_id": "child", "parent_session_id": "parent"}
    )
    assert [chunk async for chunk in handler.generate(child_request)] == []
    child = handler._kv_router.sent["extra_args"]["sampling_options"]["vllm_xargs"]
    assert child["duru_root_id"] == parent["duru_root_id"]
    assert child["duru_agent_id"] != parent["duru_agent_id"]
    assert child["duru_request_id"] != parent["duru_request_id"]
    snapshot = await snapshot_when(
        handler, lambda s: s["accepted_steps"] == 3 and not s["requests"]
    )
    root = parent["duru_root_id"]
    assert snapshot["roots"][root]["W"] == pytest.approx(0.03)
    assert snapshot["roots"][root]["U"] > 0
    await handler.end_session("parent")
    snapshot = await snapshot_when(handler, lambda s: "parent" not in s["programs"])
    assert root in snapshot["roots"]
    await handler.end_session("child")
    snapshot = await snapshot_when(handler, lambda s: not s["roots"])
    assert not any(snapshot["accounting_state"].values())
    assert not handler.sessions and not handler.roots


async def test_anonymous_request_releases_accounting_without_session_final(handler):
    assert [chunk async for chunk in handler.generate({"token_ids": [1, 2]})] == []
    snapshot = await snapshot_when(
        handler, lambda s: s["accepted_steps"] == 1 and not s["roots"]
    )
    assert not any(snapshot["accounting_state"].values())
    assert not handler.sessions


async def test_accounting_failure_cancels_queued_work_and_stops_service(
    handler, monkeypatch
):
    serving = asyncio.Event()
    stopped = asyncio.Event()

    async def serve_router(*args):
        serving.set()
        try:
            await asyncio.Event().wait()
        finally:
            stopped.set()

    # Replace only the endpoint server; the actual accounting monitor supervises it.
    handler.test_monitor.cancel()
    await asyncio.gather(handler.test_monitor, return_exceptions=True)
    monkeypatch.setattr(main_module, "serve_router", serve_router)
    service = asyncio.create_task(main_module.serve(None, None, handler))
    request = asyncio.create_task(generate(handler, "queued", 11))
    try:
        await asyncio.wait_for(serving.wait(), 2)
        await snapshot_when(handler, lambda s: "queued" in s["programs"])
        handler.test_feed.emit(
            "accepted_output",
            steps=[
                dict(
                    client_request_id="unjoined",
                    root="bad",
                    agent="bad",
                    end=time.time(),
                )
            ],
        )
        with pytest.raises(RuntimeError, match="accounting failed"):
            await asyncio.wait_for(service, 3)
        with pytest.raises(RuntimeError, match="accounting failed during request"):
            await asyncio.wait_for(request, 3)
        assert stopped.is_set()
        assert handler._kv_router.sent is None
        assert [s async for s in handler.status()][0]["status"] == "unhealthy"
        with pytest.raises(RuntimeError):
            await generate(handler, "new")
    finally:
        service.cancel()
        request.cancel()
        await asyncio.gather(service, request, return_exceptions=True)


async def test_intentional_shutdown_stops_monitor_without_failure(handler):
    await handler.shutdown()
    await asyncio.wait_for(handler.test_monitor, 1)
    assert handler.failure is None


@pytest.mark.parametrize("pulling", [False, True])
async def test_session_final_closes_stream_across_per_chunk_tasks(handler, pulling):
    entered = asyncio.Event()

    class ChunkRouter(AccountingKvRouter):
        async def generate_from_request(self, request, **kwargs):
            await super().generate_from_request(request, **kwargs)

            async def chunks():
                yield {"token_ids": [1]}
                entered.set()
                await asyncio.Event().wait()

            return chunks()

    handler._kv_router = ChunkRouter(handler.test_feed)
    stream = handler.generate(
        {"token_ids": [1], "agent_context": {"session_id": "stream"}}
    )
    first = asyncio.create_task(anext(stream))
    assert await asyncio.wait_for(first, 2) == {"token_ids": [1]}
    second = None
    if pulling:
        second = asyncio.create_task(anext(stream))
        await asyncio.wait_for(entered.wait(), 1)
    try:
        await asyncio.wait_for(handler.end_session("stream"), 2)
        if second is not None:
            with pytest.raises(asyncio.CancelledError):
                await second
        with pytest.raises(StopAsyncIteration):
            await anext(stream)
        assert not handler.sessions
        await snapshot_when(handler, lambda s: not s["roots"])
    finally:
        if second is not None:
            second.cancel()
            await asyncio.gather(second, return_exceptions=True)
        await stream.aclose()
