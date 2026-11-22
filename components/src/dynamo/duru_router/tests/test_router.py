import asyncio
import json
import time
from unittest.mock import patch

import pytest
from dynamo.duru_router import router as router_module
from dynamo.duru_router.router import DuruScheduler, ThunderAgentConfig
from dynamo.duru_router.scoring import Reader
from dynamo.thunderagent_router.program_state import ProgramLifecycle, ProgramStatus


class Capacity:
    def __init__(self):
        self.caps = {(1, 0): 1000}

    def snapshot(self):
        return self.caps

    def live_worker_ids(self):
        return {key[0] for key in self.caps}


class RouterCase:
    def __init__(self, path):
        self.path = path
        self.scores, self.events, self.tasks = {}, [], []
        self.cap = Capacity()
        self.r = DuruScheduler(
            self.cap,
            ThunderAgentConfig(
                buffer_per_program=0,
                pause_threshold=1.0,
                resume_hysteresis=0.0,
            ),
        )

    def score(self, program, value):
        self.scores[program] = dict(
            score=value, U=value, arrival=len(self.scores), request_id=program
        )

    def save(self, age=0):
        self.path.write_text(
            json.dumps(dict(observation_time=time.time() - age, programs=self.scores))
        )

    def put(self, program, demand, score):
        self.score(program, score)
        obj = self.r._table.begin_request(program, demand)
        obj.lifecycle = ProgramLifecycle.PAUSED
        obj.waiting = asyncio.Event()
        self.r._table.paused[program] = None
        return obj

    async def resume(self, age=0):
        self.save(age)
        await self.r._greedy_resume(self.cap.caps)

    async def start(self, program, demand):
        self.save()
        checked = asyncio.Event()
        original = self.r._admission_ready

        def ready(program_id):
            checked.set()
            return original(program_id)

        with patch.object(self.r, "_admission_ready", ready):
            task = asyncio.create_task(self.r.before_request(program, demand))
            self.tasks.append(task)
            await asyncio.wait_for(checked.wait(), 1)
        return task

    @property
    def admitted(self):
        return [event["program"] for event in self.events]


@pytest.fixture
async def case(tmp_path, monkeypatch):
    case = RouterCase(tmp_path / "scores.json")
    monkeypatch.setenv("DURU_SCORES", str(case.path))
    monkeypatch.setenv("DURU_AUDIT_DIR", str(tmp_path))
    monkeypatch.delenv("DURU_ADMISSION_ALPHA", raising=False)
    for name, value in dict(
        reader=None, journal=None, sequence=0, decision_snapshot={}, last_defer={}
    ).items():
        monkeypatch.setattr(router_module, name, value)
    monkeypatch.setattr(
        router_module,
        "record_choice",
        lambda p, kind="resume", **kw: case.events.append(
            dict(program=p, event=kind, control=router_module.score_row(p), **kw)
        ),
    )
    try:
        yield case
    finally:
        for task in case.tasks:
            task.cancel()
        await asyncio.wait_for(asyncio.gather(*case.tasks, return_exceptions=True), 1)
        if router_module.journal is not None:
            router_module.journal.close()


async def test_reader_pins_one_decision_and_retains_stale_evidence(case):
    case.score("high", 1.4)
    case.save()
    reader = Reader()
    snapshot = reader.refresh(force=True)
    with patch(
        "dynamo.duru_router.scoring.time.time",
        return_value=snapshot["observation_time"] + 3,
    ):
        with reader.pin(snapshot):
            assert reader.refresh()["programs"]["high"]["score"] == 1.4
        assert reader.refresh() == {}
        assert reader.reason == "stale_snapshot"
        assert reader.snapshot["programs"]["high"]["score"] == 1.4


@pytest.mark.parametrize("cancel", [False, True], ids=["recover", "cancel"])
async def test_missing_initial_score_defers_without_leaking_program(case, cancel):
    task = await case.start("new", 100)
    assert not task.done()
    assert "new" not in case.r._table.programs
    if cancel:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, 1)
        assert "new" not in case.r._table.programs
    else:
        case.score("new", 0)
        case.save()
        assert (await asyncio.wait_for(task, 1)).assigned_replica_hint == (1, 0)
    assert not case.r._admission_gates


@pytest.mark.parametrize(
    "override", [False, True], ids=["capacity-fit", "alpha-override"]
)
async def test_stale_scores_block_admission_then_fresh_scores_restore_rank(
    case, monkeypatch, override
):
    resident = case.r._table.begin_request("resident", 900)
    resident.assigned_replica = (1, 0)
    case.score("low", 0.2)
    low = await case.start("low", 300)
    case.score("high", 1.4)
    high = await case.start("high", 1100 if override else 200)
    if override:
        monkeypatch.setenv("DURU_ADMISSION_ALPHA", "0.7")
    resident.token_total = 650
    case.save(age=3)
    await case.r._scheduler_tick()
    assert not high.done() and not low.done()
    assert not case.events
    case.save()
    await case.r._scheduler_tick()
    await asyncio.wait_for(high, 1)
    assert case.admitted == ["high"]
    assert case.events[0]["control"]["score"] == 1.4
    assert case.events[0]["event"] == ("u_threshold_resume" if override else "resume")
    assert not low.done()


@pytest.mark.parametrize(
    "demand,scores,forced", [(300, [0.1, 0.01, 0.2], 0), (1100, [0.9, 0.7, 1.3], 3)]
)
async def test_fit_and_override_admissions_follow_score_order(
    case, monkeypatch, demand, scores, forced
):
    monkeypatch.setenv("DURU_ADMISSION_ALPHA", "0.7")
    for program, score in zip(["middle", "low", "high"], scores):
        case.put(program, demand, score)
    await case.resume()
    assert case.admitted == ["high", "middle", "low"]
    assert case.r._stat_forced_resumes == forced


@pytest.mark.parametrize("fragmented", [False, True])
async def test_backfill_skips_nonfits_without_overcommitting(
    case, monkeypatch, fragmented
):
    monkeypatch.setenv("DURU_ADMISSION_ALPHA", "0.7")
    if fragmented:
        case.cap.caps = {(1, 0): 500, (2, 0): 500}
    for program, demand, score in [
        ("large", 1100, 0.6),
        ("medium", 800, 0.5),
        ("small", 300, 0.4),
        ("tiny", 150, 0.3),
    ]:
        case.put(program, demand, score)
    await case.resume()
    assert case.admitted == (["small", "tiny"] if fragmented else ["medium", "tiny"])
    assert "large" in case.r._table.paused
    assert all(
        case.r._worker_used(key) <= capacity for key, capacity in case.cap.caps.items()
    )
    for program in list(case.admitted):
        await case.r.end_program(program)
    case.cap.caps = {(1, 0): 1500}
    await case.resume()
    assert (
        "large" in case.admitted
    )  # A previous nonfit remains eligible on a later pass.


async def test_new_arrival_joins_existing_rank(case):
    case.put("older_high", 800, 4)
    case.score("new_low", 0.1)
    task = await case.start("new_low", 400)
    assert case.admitted == ["older_high"]
    assert not task.done()
    assert case.r._worker_used((1, 0)) == 800


async def test_capacity_pauses_only_at_safe_boundary_and_never_for_score_alone(case):
    case.score("busy", 0.001)
    case.save()
    await case.r.before_request("busy", 900)
    await case.r.after_request("busy", 900, 0)
    case.put("high", 400, 100)
    case.save()
    await case.r._scheduler_tick()
    busy = case.r._table.programs["busy"]
    assert busy.lifecycle == ProgramLifecycle.ACTIVE
    assert "high" in case.r._table.paused
    await case.r.before_request("busy", 900)
    case.cap.caps = {(1, 0): 800}
    await case.r._scheduler_tick()
    assert busy.lifecycle == ProgramLifecycle.ACTIVE
    assert not busy.marked_for_pause
    await case.r.after_request("busy", 900, 0)
    await case.r._scheduler_tick()
    assert busy.lifecycle == ProgramLifecycle.PAUSED
    assert case.r._worker_used((1, 0)) == 0


@pytest.mark.parametrize(
    "idle_score,pending_score,winner", [(0.7, 0.5, "idle"), (0.5, 0.7, "pending")]
)
async def test_idle_reservation_competes_by_measured_u(
    case, idle_score, pending_score, winner
):
    idle = case.put("idle", 600, idle_score)
    idle.status = ProgramStatus.ACTING
    case.scores["idle"].update(score=None, kind="idle_measured_only")
    case.put("pending", 600, pending_score)
    await case.resume()
    assert case.admitted == [winner]
    assert idle.status == ProgramStatus.ACTING
    assert case.r._worker_used((1, 0)) == 600


async def test_newcomer_preserves_resume_headroom_for_established_waiters(case):
    case.r._cfg.pause_threshold, case.r._cfg.resume_hysteresis = 0.95, 0.1
    new = case.put("new", 900, 0.1)
    new._u_new_admission = True
    await case.resume()
    assert case.admitted == ["new"]  # First admission can use the full budget.
    new.token_total = 800
    case.put("older", 100, 5)
    case.put("newcomer", 100, 0.1)._u_new_admission = True
    await case.resume()
    assert case.admitted == ["new"]
    new.token_total = 700
    await case.resume()
    assert case.admitted == ["new", "older"]


async def test_lookahead_crossing_alpha_unblocks_pending_request(case, monkeypatch):
    monkeypatch.setenv("DURU_ADMISSION_ALPHA", "0.7")
    case.score("child", 0.69)
    case.scores["child"].update(U=None, kind="zero_work_root_next_five")
    task = await case.start("child", 1100)
    assert not task.done()
    assert case.r._stat_forced_resumes == 0
    case.scores["child"]["score"] = 0.7
    await case.resume()
    assert (await asyncio.wait_for(task, 1)).assigned_replica_hint == (1, 0)
    assert case.r._stat_forced_resumes == 1
    assert case.events[-1]["event"] == "u_threshold_resume"


async def test_missing_scores_and_nonpending_requests_cannot_force_admission(
    case, monkeypatch
):
    monkeypatch.setenv("DURU_ADMISSION_ALPHA", "0.7")
    case.put("missing", 100, 1)
    del case.scores["missing"]
    case.put("no_request", 1100, 9)
    del case.scores["no_request"]["request_id"]
    case.put("idle", 1100, 3).status = ProgramStatus.ACTING
    case.put("valid", 100, 0.01)
    await case.resume()
    assert case.admitted == ["valid"]


@pytest.mark.parametrize("capacities", [{}, {(1, 0): 0}])
async def test_alpha_requires_a_live_positive_capacity_worker(
    case, monkeypatch, capacities
):
    monkeypatch.setenv("DURU_ADMISSION_ALPHA", "0.7")
    case.cap.caps = capacities
    high = case.put("high", 1100, 4)
    await case.resume()
    assert high.lifecycle == ProgramLifecycle.PAUSED
    assert not case.events


@pytest.mark.parametrize("alpha", [None, "0.7"])
async def test_cancelled_waiter_cannot_reappear_or_force_admit(
    case, monkeypatch, alpha
):
    if alpha:
        monkeypatch.setenv("DURU_ADMISSION_ALPHA", alpha)
    case.score("cancelled", 0.1)
    task = await case.start("cancelled", 1100)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, 1)
    case.scores["cancelled"]["score"] = 3
    await case.resume()
    assert not case.events
    assert not case.r._table.programs
    assert not case.r._admission_gates
    assert case.r._stat_forced_resumes == 0
    assert case.r._worker_used((1, 0)) == 0


async def test_repeated_cancellation_finishes_blocked_rollback(case):
    case.score("blocked", 0.1)
    task = await case.start("blocked", 1100)
    rollback_started = asyncio.Event()
    original = case.r._rollback_admission

    async def rollback(*args):
        rollback_started.set()
        await original(*args)

    with patch.object(case.r, "_rollback_admission", rollback):
        async with case.r._lock:
            task.cancel()
            await asyncio.wait_for(rollback_started.wait(), 1)
            task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, 1)
    assert not case.r._table.programs
    assert not case.r._admission_gates
    assert case.r._stat_admissions_cancelled == 1


async def test_old_waiter_cannot_remove_recreated_program(case):
    case.score("reused", 0.1)
    task = await case.start("reused", 1100)
    await case.r.end_program("reused")
    replacement = case.r._table.begin_request("reused", 100)
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, 1)
    assert case.r._table.programs["reused"] is replacement
    assert replacement.token_total == 100


async def test_concurrent_admission_and_end_release_state_for_next_wave(
    case, monkeypatch
):
    monkeypatch.setenv("DURU_ADMISSION_ALPHA", "0.7")
    for wave in range(2):
        programs = [f"root-{wave}::child-{child}" for child in range(3)]
        tasks = []
        for program in programs:
            case.score(program, 0.1)
            tasks.append(await case.start(program, 1100))
        assert all(not task.done() for task in tasks)
        for program in programs:
            case.scores[program]["score"] = 0.7
        await case.resume()
        await asyncio.wait_for(asyncio.gather(*tasks), 1)
        for program in programs:
            await case.r.after_request(program, 1100, 10)
            await case.r.end_program(program)
        assert not case.r._table.programs
        assert not case.r._table.paused
        assert not case.r._admission_gates
    assert case.r._stat_forced_resumes == 6
