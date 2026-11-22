# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared request handling with the real Duru admission scheduler."""

import asyncio
import os
import struct
import sys
from types import SimpleNamespace

import pytest
from dynamo.duru_router import __main__ as main_module
from dynamo.duru_router import router as router_module
from dynamo.duru_router.__main__ import DuruRouterHandler
from dynamo.duru_router.router import DuruScheduler, ThunderAgentConfig
from dynamo.thunderagent_router.tests.test_main import FakeKvRouter

pytestmark = [pytest.mark.pre_merge, pytest.mark.unit, pytest.mark.gpu_0]


@pytest.fixture
def handler(monkeypatch):
    monkeypatch.setattr(router_module, "begin_decision", lambda *args: True)
    monkeypatch.setattr(router_module, "control_active", lambda: True)
    monkeypatch.setattr(
        router_module, "score_row", lambda pid: {"score": 0.1, "request_id": pid}
    )
    monkeypatch.delenv("DURU_ADMISSION_ALPHA", raising=False)
    monkeypatch.delenv("DURU_SCORES", raising=False)
    capacities = {(7, 2): 10, (7, 3): 10}
    capacity = SimpleNamespace(snapshot=lambda: capacities, live_worker_ids=lambda: {7})
    result = DuruRouterHandler(runtime=None, config=None)
    result._scheduler = DuruScheduler(
        capacity,
        ThunderAgentConfig(
            buffer_per_program=0, pause_threshold=1, resume_hysteresis=0
        ),
    )
    result._kv_router = FakeKvRouter()
    return result


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
    handler._kv_router = FakeKvRouter(
        chunks, error=None if terminal else RuntimeError("disconnected")
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
    monkeypatch.setenv("DURU_AUDIT_DIR", str(tmp_path / "audit"))
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
