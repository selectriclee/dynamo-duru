import asyncio
import json
import time
from unittest.mock import patch

import pytest

from dynamo.duru_router import router as router_module
from dynamo.duru_router.router import DuruScheduler, ThunderAgentConfig
from dynamo.duru_router.scoring import Reader
from dynamo.thunderagent_router.policy import ResumeChoice
from dynamo.thunderagent_router.program_state import ProgramLifecycle, ProgramStatus

pytestmark = [pytest.mark.pre_merge, pytest.mark.unit, pytest.mark.gpu_0]


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
            reader=Reader(
                snapshot_source=lambda: (
                    json.loads(self.path.read_text()) if self.path.exists() else {}
                )
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

    def used(self, replica):
        # Read the same occupancy buckets used by the shared scheduler.
        return self.r._replica_usage_locked(self.cap.caps)[replica].used


@pytest.fixture
async def case(tmp_path, monkeypatch):
    monkeypatch.setenv("DURU_SCORES", str(tmp_path / "scores.json"))
    case = RouterCase(tmp_path / "scores.json")
    monkeypatch.setenv("DURU_AUDIT_DIR", str(tmp_path))
    monkeypatch.delenv("DURU_ADMISSION_ALPHA", raising=False)
    monkeypatch.delenv("DURU_DIAGNOSTICS", raising=False)
    selected = None

    def observe_selection(method):
        def select(*args, **kwargs):
            nonlocal selected
            selected = method(*args, **kwargs)
            return selected

        return select

    original_resume = case.r._resume_program

    def observe_resume(program, replica):
        # Observe the host's real decisions rather than requiring production
        # audit logging to expose scheduling behavior to tests.
        assert selected is not None and selected.replica == replica
        case.events.append(
            dict(
                program=program.program_id,
                score=case.r._policy.score_row(program.program_id).get("score"),
                forced=selected.forced,
                remaining=selected.remaining,
            )
        )
        original_resume(program, replica)

    for method in ("_select_resume_locked", "_select_resume_override_locked"):
        monkeypatch.setattr(case.r, method, observe_selection(getattr(case.r, method)))
    monkeypatch.setattr(case.r, "_resume_program", observe_resume)
    try:
        yield case
    finally:
        for task in case.tasks:
            task.cancel()
        await asyncio.wait_for(asyncio.gather(*case.tasks, return_exceptions=True), 1)


@pytest.mark.parametrize("diagnostics", [False, True])
@pytest.mark.parametrize("demand,forced", [(100, False), (1200, True)])
async def test_optional_diagnostics_preserve_admission(
    case, monkeypatch, caplog, diagnostics, demand, forced
):
    monkeypatch.setenv("DURU_DIAGNOSTICS", "1" if diagnostics else "0")
    monkeypatch.setenv("DURU_ADMISSION_ALPHA", "1")
    caplog.set_level("INFO", logger="dynamo.duru_router.diagnostics")
    case.put("ready", demand, 2)
    with patch(
        "dynamo.duru_router.events.open_journal",
        side_effect=AssertionError("Admission must not open an audit journal"),
    ):
        await case.resume()
    assert case.admitted == ["ready"]
    assert case.events[0]["forced"] is forced
    records = [r for r in caplog.records if r.name.endswith(".diagnostics")]
    assert len(records) == int(diagnostics)
    if diagnostics:
        row = json.loads(records[0].getMessage().split(" ", 1)[1])
        assert row["forced"] is forced
        assert row["U"] == 2
        assert row["required"] == demand
        assert row["remaining_before"] == 1000
        assert row["worker"] == [1, 0]
        assert row["request_id"] == "ready"
    assert not hasattr(router_module, "journal")


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
    assert case.events[0]["score"] == 1.4
    assert case.events[0]["forced"] is override
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
    assert all(case.used(key) <= capacity for key, capacity in case.cap.caps.items())
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
    assert case.used((1, 0)) == 800


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
    assert case.used((1, 0)) == 0


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
    assert case.used((1, 0)) == 600


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
    assert case.events[-1]["forced"]


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
    assert case.used((1, 0)) == 0


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


async def test_resume_updates_weighted_reservations_between_policy_choices(case):
    case.r._cfg.acting_token_weight = 0.5
    case.r._cfg.buffer_per_program = 10
    case.put("idle", 600, 3).status = ProgramStatus.ACTING
    case.put("pending", 600, 2)
    case.put("tail", 100, 1)

    await case.resume()

    # Idle requires 610 to fit, but reserves 300 + 10 after admission. The
    # next choice must see that weighted reservation, then the pending 610.
    assert case.admitted == ["idle", "pending"]
    assert [event["remaining"] for event in case.events] == [1000, 690]
    assert case.used((1, 0)) == 920
    assert list(case.r._table.paused) == ["tail"]


@pytest.mark.parametrize("override", [False, True], ids=["fit", "alpha"])
async def test_policy_placement_uses_replica_ties_and_decayed_usage(
    case, monkeypatch, override
):
    # Same worker, reverse DP-rank insertion order: normal fits prefer the
    # smaller replica key, while the override uses current decayed occupancy.
    case.cap.caps = {(7, 1): 1000, (7, 0): 1000}
    if override:
        monkeypatch.setenv("DURU_ADMISSION_ALPHA", "1")
        monkeypatch.setattr(time, "monotonic", lambda: 100.0)
        idle = case.r._table.begin_request("idle", 900)
        idle.status = ProgramStatus.ACTING
        idle.acting_since = 90.0
        idle.assigned_replica = (7, 1)
        busy = case.r._table.begin_request("busy", 100)
        busy.assigned_replica = (7, 0)
    candidate = case.put("candidate", 1100 if override else 100, 2)

    await case.resume()

    assert candidate.assigned_replica == ((7, 1) if override else (7, 0))
    assert case.events[0]["forced"] is override
    assert case.events[0]["remaining"] == (100 if override else 1000)


async def test_duru_waits_for_policy_even_with_zero_native_timeout(case, monkeypatch):
    case.r._cfg.resume_timeout_seconds = 0
    case.score("waiting", 0.1)
    task = await case.start("waiting", 1100)
    assert not task.done()
    assert case.r._stat_forced_resumes == 0

    monkeypatch.setenv("DURU_ADMISSION_ALPHA", "1")
    case.scores["waiting"]["score"] = 1
    await case.resume()

    decision = await asyncio.wait_for(task, 1)
    assert decision.assigned_replica_hint == (1, 0)
    assert case.r._stat_forced_resumes == 1


@pytest.mark.parametrize("high_score", [0.9, 1.0])
async def test_highest_score_exception_precedes_lower_score_backfill(
    case, monkeypatch, high_score
):
    monkeypatch.setenv("DURU_ADMISSION_ALPHA", "1")
    resident = case.r._table.begin_request("resident", 600)
    resident.assigned_replica = (1, 0)
    case.put("high", 700, high_score)
    case.put("backfill", 200, 0.1)

    await case.resume()

    # There is no capacity-only pass over the queue. A high-score override
    # happens before a lower-ranked candidate can consume the remaining 400.
    assert case.admitted == (["high"] if high_score == 1.0 else ["backfill"])
    assert case.events[0]["forced"] is (high_score == 1.0)


@pytest.mark.parametrize("demand,forced", [(100, False), (101, True)])
async def test_host_checks_fit_before_alpha_and_preserves_replica_ties(
    case, monkeypatch, demand, forced
):
    monkeypatch.setenv("DURU_ADMISSION_ALPHA", "0.7")
    case.cap.caps = {(7, 1): 100, (7, 0): 100}
    program = case.put("high", demand, 0.7)

    if forced:
        await case.resume()
    else:
        # A fit must never consult the exception, even when score >= alpha.
        with patch.object(
            router_module.DuruResumePlan,
            "may_force_admit",
            side_effect=AssertionError("Capacity fit must precede alpha exception"),
        ):
            await case.resume()

    assert program.assigned_replica == ((7, 1) if forced else (7, 0))
    assert case.events[0]["forced"] is forced


@pytest.mark.parametrize("channel", [None, "", "   "])
def test_duru_requires_score_channel_before_startup(monkeypatch, channel):
    if channel is None:
        monkeypatch.delenv("DURU_SCORES", raising=False)
    else:
        monkeypatch.setenv("DURU_SCORES", channel)
    with pytest.raises(ValueError, match="DURU_SCORES.*IPC channel"):
        DuruScheduler(
            Capacity(), ThunderAgentConfig(), reader=Reader(snapshot_source=lambda: {})
        )
    with pytest.raises(ValueError, match="DURU_SCORES.*IPC channel"):
        Reader()


def test_policies_and_prepared_plans_keep_their_own_scores(monkeypatch, caplog):
    monkeypatch.setenv("DURU_SCORES", "test-channel")
    monkeypatch.setenv("DURU_DIAGNOSTICS", "1")
    caplog.set_level("INFO", logger="dynamo.duru_router.diagnostics")
    programs = [router_module.Program(pid) for pid in ("a", "b")]
    snapshots = [
        dict(
            run_id=str(index),
            observation_time=time.time() - index * 0.1,
            programs={
                pid: dict(score=score, U=score, request_id=f"{index}-{pid}")
                for pid, score in zip(("a", "b"), scores)
            },
        )
        for index, scores in enumerate(((2, 1), (1, 2)))
    ]
    readers = [Reader(snapshot_source=lambda s=s: s) for s in snapshots]
    schedulers = [
        DuruScheduler(Capacity(), ThunderAgentConfig(), reader=reader)
        for reader in readers
    ]
    first, second = [scheduler._policy for scheduler in schedulers]
    plan = first.prepare_resume(programs, ThunderAgentConfig())
    other = second.prepare_resume(programs, ThunderAgentConfig())
    assert [p.program_id for p in plan.programs] == ["a", "b"]
    assert [p.program_id for p in other.programs] == ["b", "a"]
    assert first.score_row("a")["score"] == 2
    assert second.score_row("a")["score"] == 1

    # A later decision must not change an already prepared plan's audit identity.
    readers[0].snapshot_source = lambda: snapshots[1]
    first.prepare_resume(programs, ThunderAgentConfig())
    plan.record_admission(programs[0], ResumeChoice((1, 0), 1000, 0.85), 0)
    row = json.loads(caplog.records[-1].getMessage().split(" ", 1)[1])
    assert row["run_id"] == "0"
    assert row["request_id"] == "0-a"
    assert row["observation_time"] == snapshots[0]["observation_time"]
    assert row["U"] == 2

    readers[0].snapshot_source = lambda: {}
    assert not first.admission_ready("a")
    assert second.admission_ready("a")
