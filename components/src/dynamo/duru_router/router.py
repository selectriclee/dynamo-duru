# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Highest-U admission with backfill and a single alpha exception.

Keep ThunderAgent's worker reservations and tool-phase memory-pressure pausing.
Account for real token counts from request usage, not a character estimate.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from dataclasses import dataclass
from typing import Optional

from dynamo.duru_router.admission import Candidate, DuruPolicy, Resource
from dynamo.duru_router.events import open_journal
from dynamo.duru_router.scoring import Reader
from dynamo.thunderagent_router.program_state import Program as BaseProgram
from dynamo.thunderagent_router.program_state import (
    ProgramLifecycle,
    ProgramStatus,
    ReplicaKey,
)
from dynamo.thunderagent_router.program_state import ProgramTable as BaseProgramTable
from dynamo.thunderagent_router.router import (
    ThunderAgentConfig as ThunderAgentConfig,
)
from dynamo.thunderagent_router.router import (
    ThunderAgentScheduler as BaseScheduler,
)

logger = logging.getLogger(__name__)

POLICY_VERSION = "duru_v1"
reader = None
journal = None
sequence = 0

decision_snapshot = {}
last_defer = {}


def begin_decision(program_id=None):
    global reader, decision_snapshot, journal, sequence
    if not os.environ.get("DURU_SCORES"):
        decision_snapshot = {}
        return True
    if reader is None:
        reader = Reader()
    decision_snapshot = reader.refresh(force=True)
    reason = reader.reason
    if decision_snapshot and program_id is not None:
        entry = decision_snapshot.get("programs", {}).get(program_id, {})
        if not entry or "request_id" not in entry:
            reason = "missing_program_score"
    ready = bool(decision_snapshot) and reason == "fresh"
    if not ready:
        now = time.monotonic()
        key = (reason, program_id)
        if now - last_defer.get(key, 0) >= 1:
            last_defer[key] = now
            if journal is None:
                journal = open_journal(
                    os.environ["DURU_AUDIT_DIR"], name="duru-router", fresh=False
                )
            sequence += 1
            journal.submit(
                "record",
                dict(
                    sequence=sequence,
                    event="u_control_deferred",
                    time=time.time(),
                    reason=reason,
                    program_id=program_id,
                    snapshot_age_s=reader.age,
                    observation_time=reader.snapshot.get("observation_time"),
                    action="defer_admission_no_native_fallback",
                    policy_version=POLICY_VERSION,
                ),
            )
    return ready


def score_row(program_id):
    return decision_snapshot.get("programs", {}).get(program_id, {})


def control_active():
    return bool(os.environ.get("DURU_SCORES"))


def record_choice(program_id, kind="resume", **extra):
    global journal, sequence
    if kind == "resume":
        assert score_row(program_id), "Admission without a known control row"
    if journal is None:
        journal = open_journal(
            os.environ["DURU_AUDIT_DIR"], name="duru-router", fresh=False
        )
    sequence += 1
    r = score_row(program_id)
    journal.submit(
        "record",
        dict(
            sequence=sequence,
            event=kind,
            time=time.time(),
            program_id=program_id,
            control=r,
            snapshot_observation_time=decision_snapshot.get("observation_time"),
            snapshot_age_s=time.time()
            - decision_snapshot.get("observation_time", time.time()),
            thresholds=dict(admission_alpha=admission_alpha()),
            policy_version=POLICY_VERSION,
            **extra,
        ),
    )


def admission_alpha():
    raw = os.environ.get("DURU_ADMISSION_ALPHA")
    if raw is None:
        return None
    value = float(raw)
    DuruPolicy(alpha=value)
    return value


@dataclass
class Program(BaseProgram):
    _u_new_admission: bool = False


class ProgramTable(BaseProgramTable):
    program_cls = Program


class DuruScheduler(BaseScheduler):
    program_table_cls = ProgramTable

    def _admission_ready(self, program_id: str) -> bool:
        return begin_decision(program_id)

    async def _wait_for_resume(
        self, program_id: str, wait_event: asyncio.Event
    ) -> None:
        await self._greedy_resume(self._capacity.snapshot())
        # Duru retries by score/capacity updates, never by a forced-resume timer.
        await wait_event.wait()

    def _check_admission_owner(self, program, admitted, admitted_epoch) -> None:
        if (
            program is not admitted
            or program is None
            or program.admission_epoch != admitted_epoch
        ):
            raise asyncio.CancelledError("Program terminated during admission")

    def _admit_locked(
        self,
        program_id: str,
        estimated_prompt_tokens: int,
    ) -> tuple[Optional[asyncio.Event], bool]:
        # Caller holds self._lock.
        was_new = program_id not in self._table.programs
        program = self._table.begin_request(program_id, estimated_prompt_tokens)
        if was_new:
            self._stat_programs_created += 1
            logger.info(
                "thunderagent.program created program=%s "
                "estimated_prompt_tokens=%d active=%d",
                program_id,
                estimated_prompt_tokens,
                len(self._table.programs),
            )
        if program.lifecycle == ProgramLifecycle.PAUSED:
            program.waiting = program.waiting or asyncio.Event()
            return program.waiting, True

        needs_assignment = program.assigned_replica is None
        stale_replacement = False
        live_worker_ids: set[int] = set()
        if program.assigned_replica is not None:
            live_worker_ids = self._capacity.live_worker_ids()
            if not live_worker_ids or program.assigned_replica[0] in live_worker_ids:
                return None, False
            stale_worker_id = program.assigned_replica
            program.assigned_replica = None
            needs_assignment = True
            stale_replacement = True
            logger.info(
                "thunderagent.worker stale_pin program=%s old_worker=%s "
                "available_workers=%s",
                program_id,
                stale_worker_id,
                sorted(live_worker_ids),
            )

        if not needs_assignment:
            return None, False

        capacities = self._capacity.snapshot()
        if stale_replacement:
            capacities = {
                worker_id: capacity
                for worker_id, capacity in capacities.items()
                if worker_id[0] in live_worker_ids
            }

        if not control_active():
            if not capacities:
                # Cold start: MDC hasn't published yet. Let the request flow
                # through with no pin; the chunk-loop callback will populate
                # ``assigned_replica`` once the engine picks a worker, and
                # subsequent turns get the sticky pin.
                return None, False
            worker_id = self._select_worker_for_admission_locked(
                capacities,
                program.token_total,
                queue_behind_paused=not stale_replacement,
            )
            if worker_id is not None:
                program.assigned_replica = worker_id
                self._stat_worker_assignments += 1
                return None, False
        program._u_new_admission = True
        # Queue every new candidate for the same ranked capacity pass.
        program.waiting = program.waiting or asyncio.Event()
        program.lifecycle = ProgramLifecycle.PAUSED
        self._table.paused[program_id] = None
        self._stat_pauses += 1
        logger.info(
            "thunderagent.program paused program=%s reason=admission_full "
            "tokens=%d paused=%d",
            program_id,
            program.token_total,
            len(self._table.paused),
        )
        return program.waiting, True

    async def after_request(
        self,
        program_id: str,
        prompt_tokens: int,
        completion_tokens: int,
    ) -> None:
        async with self._lock:
            program = self._table.end_request(
                program_id, prompt_tokens, completion_tokens
            )
            if program is None:
                return
            # Do not defer an U decision made while the request was executing.
            program.marked_for_pause = False
        await self._greedy_resume(self._capacity.snapshot())

    async def _scheduler_tick(self) -> None:
        capacities = self._capacity.snapshot()
        if not capacities:
            return
        # Upstream TA ordering: resume first, then pause -- a program paused
        # this tick can't resume until the next.
        self._apply_soft_demotes(capacities)
        await self._greedy_resume(capacities)
        await self._pause_until_safe(capacities)

    def _active_programs_for_worker(self, worker_id: ReplicaKey) -> list[Program]:
        return [
            p
            for p in self._table.programs.values()
            if p.lifecycle == ProgramLifecycle.ACTIVE
            and p.assigned_replica == worker_id
        ]

    def _worker_used(self, worker_id: ReplicaKey, *, decayed: bool = False) -> int:
        programs = self._active_programs_for_worker(worker_id)
        tokens = sum(self._program_tokens(p, decayed=decayed) for p in programs)
        return tokens + len(programs) * self._cfg.buffer_per_program

    def _select_worker_for_admission_locked(
        self,
        capacities: dict[ReplicaKey, int],
        estimated_tokens: int,
        *,
        queue_behind_paused: bool,
    ) -> Optional[ReplicaKey]:
        # Fairness: new programs queue behind any existing paused program.
        if queue_behind_paused and self._table.paused:
            return None
        buffer = self._cfg.buffer_per_program
        required = estimated_tokens + buffer
        candidates = [
            (w, self._worker_used(w))
            for w, c in capacities.items()
            if c - self._worker_used(w) >= required
        ]
        if not candidates:
            return None
        return min(candidates, key=lambda item: item[1])[0]

    def _apply_soft_demotes(self, capacities: dict[ReplicaKey, int]) -> None:
        soft_until = time.monotonic() + self._cfg.scheduler_interval_seconds * 1.5
        for worker_id, capacity in capacities.items():
            util = self._worker_used(worker_id) / capacity
            if not (
                self._cfg.soft_demote_threshold <= util < self._cfg.pause_threshold
            ):
                continue
            for program in self._active_programs_for_worker(worker_id):
                if (
                    not program.marked_for_pause
                    and program.soft_demoted_until < soft_until
                ):
                    program.soft_demoted_until = soft_until

    async def _pause_until_safe(self, capacities: dict[ReplicaKey, int]) -> None:
        threshold = self._cfg.pause_threshold
        pause_target = min(self._cfg.pause_target, threshold)

        for worker_id, capacity in capacities.items():
            # Hold the lock for the entire per-worker decision so the snapshot
            # of program state used by _smallest_candidates / _worker_used
            # cannot race with concurrent before_request admissions.
            async with self._lock:
                base_used = self._worker_used(worker_id)
                if base_used <= capacity * threshold:
                    continue

                target_limit = capacity * pause_target
                paused_this_tick = 0
                marked_this_tick = 0
                # Bound the inner loop by total program count so a candidate
                # transitioning out from under us can't spin the tick.
                for _ in range(len(self._table.programs) + 1):
                    if self._worker_used(worker_id) <= target_limit:
                        break
                    acting, reasoning = self._smallest_candidates(worker_id)
                    if acting is not None:
                        if self._pause_acting_locked(acting.program_id):
                            paused_this_tick += 1
                        continue
                    # Executing and already-dispatched requests are not victims.
                    break

                final_used = self._worker_used(worker_id)

            if paused_this_tick or marked_this_tick:
                logger.info(
                    "scheduler.tick worker=%s paused=%d marked=%d util=%.4f -> %.4f",
                    worker_id,
                    paused_this_tick,
                    marked_this_tick,
                    base_used / capacity,
                    final_used / capacity,
                )

    def _smallest_candidates(
        self, worker_id: ReplicaKey
    ) -> tuple[Optional[Program], Optional[Program]]:
        smallest_acting: Optional[Program] = None
        smallest_reasoning: Optional[Program] = None
        for program in self._table.programs.values():
            if program.assigned_replica != worker_id:
                continue
            if program.lifecycle != ProgramLifecycle.ACTIVE:
                continue
            if program.marked_for_pause:
                continue
            if program.status == ProgramStatus.ACTING:
                if (
                    smallest_acting is None
                    or program.token_total < smallest_acting.token_total
                ):
                    smallest_acting = program
            elif program.status == ProgramStatus.REASONING and (
                smallest_reasoning is None
                or program.token_total < smallest_reasoning.token_total
            ):
                smallest_reasoning = program
        return smallest_acting, smallest_reasoning

    async def _greedy_resume(self, capacities: dict[ReplicaKey, int]) -> None:
        if not self._table.paused:
            return

        async with self._lock:
            if not begin_decision():
                return
            paused_programs = [
                self._table.programs[pid]
                for pid in self._table.paused
                if pid in self._table.programs
            ]
            if not paused_programs:
                return

            def group_key(program: Program) -> int:
                if program.step_count <= 1:
                    return 1
                if program.status == ProgramStatus.REASONING:
                    return 0
                return 2

            paused_programs.sort(key=lambda p: (group_key(p), p.token_total))
            native_rank = {p.program_id: i for i, p in enumerate(paused_programs)}
            candidates = []
            for program in paused_programs:
                row = score_row(program.program_id)
                candidates.append(
                    Candidate(
                        identity=program.program_id,
                        demand=program.token_total + self._cfg.buffer_per_program,
                        score=row.get("score"),
                        measured_u=row.get("U"),
                        arrival=row.get("arrival", 0.0),
                        known=bool(row),
                        idle=program.status == ProgramStatus.ACTING,
                        pending_request=bool(row.get("request_id"))
                        and program.status == ProgramStatus.REASONING
                        and program.lifecycle == ProgramLifecycle.PAUSED,
                        first_admission=program._u_new_admission,
                    )
                )
            established = any(not c.first_admission for c in candidates)
            policy = DuruPolicy(
                alpha=admission_alpha(),
                resume_ceiling=self._cfg.pause_threshold - self._cfg.resume_hysteresis,
                enabled=control_active(),
            )
            tie_breaks = {key: -i for i, key in enumerate(sorted(capacities))}
            resumed_this_tick = 0
            for candidate in policy.order(candidates):
                program = self._table.programs[candidate.identity]
                resources = [
                    Resource(w, c, self._worker_used(w), tie_break=tie_breaks[w])
                    for w, c in capacities.items()
                    if c > 0
                ]
                program._u_new_admission = False
                choice = policy.select(
                    candidate, resources, established_waiters=established
                )
                if choice is None and policy.may_force_admit(candidate):
                    resources = [
                        Resource(
                            r.key,
                            r.capacity,
                            r.used,
                            r.tie_break,
                            self._worker_used(r.key, decayed=True),
                        )
                        for r in resources
                    ]
                    choice = policy.select_override(
                        candidate, resources, established_waiters=established
                    )
                if choice is None:
                    continue
                if os.environ.get("DURU_SCORES"):
                    record_choice(
                        candidate.identity,
                        kind="u_threshold_resume" if choice.forced else "resume",
                        effective_score=candidate.effective_score,
                        admission_path=(
                            "duru_alpha_capacity_override"
                            if choice.forced
                            else "duru_capacity_backfill"
                        ),
                        alpha=policy.alpha,
                        capacity_override=choice.forced,
                        native_queue_rank=native_rank[candidate.identity],
                        admission_rank=resumed_this_tick,
                        worker=choice.resource,
                        required=candidate.demand,
                        remaining_before=choice.remaining,
                        first_admission=candidate.first_admission,
                        admission_ceiling=choice.ceiling,
                        established_waiters=established,
                        program_status=program.status.value,
                    )
                self._resume_program(program, choice.resource)
                if choice.forced:
                    self._stat_forced_resumes += 1
                resumed_this_tick += 1

            if resumed_this_tick:
                logger.info(
                    "scheduler.tick resumed=%d still_paused=%d",
                    resumed_this_tick,
                    len(self._table.paused),
                )
