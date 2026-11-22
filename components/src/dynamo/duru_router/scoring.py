"""Calculate U and publish live scores from ingress and accepted-work journals.

A common journal watermark keeps E and W at the same observation time. Engine
heartbeats use a serialized node-local compact journal, outside the utility queue.
"""

import argparse
import heapq
import json
import logging
import math
import os
import signal
import threading
import time
import uuid
from collections import Counter
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path

from dynamo.duru_router.admission import priority_key as priority_key
from dynamo.duru_router.calibration.measure import load as load_calibration
from dynamo.duru_router.calibration.predict import predict
from dynamo.duru_router.events import open_journal

logger = logging.getLogger(__name__)

METRIC = "root_hierarchical_ps_accepted_work_v3"
EXTRAPOLATION_POLICY = "positive_finite_edge_slope_v1"


class LiveU:
    def __init__(self, capacity, calibration):
        assert isinstance(capacity, int) and capacity > 0
        self.capacity = capacity
        self.calibration = calibration
        self.prefill_budget = calibration.get("max_prefill", 8192)
        if type(self.prefill_budget) is not int or self.prefill_budget <= 0:
            raise ValueError("Calibration prefill budget must be a positive integer")
        self.active = {}
        self.agent_active_requests = Counter()
        # Distinct active agents per root, not the number of requests.
        self.root_active = Counter()
        self.entitlement = Counter()
        self.work = Counter()
        self.invalid = Counter()
        self.executed_agents = set()
        self.last = None
        self.agent_root = {}
        self.accepted_steps = 0
        self.output_tokens = 0
        # Counts predictor evaluations, including lookahead and cache re-evaluations;
        # it is not a count of accepted steps or affected roots.
        self.extrapolated_cost_evaluations = 0

    def advance(self, now):
        if self.last is not None:
            assert now >= self.last, (now, self.last)
            roots = [r for r, count in self.root_active.items() if count]
            share = self.capacity / len(roots) if roots else 0.0
            for root in roots:
                self.entitlement[root] += (now - self.last) * min(
                    self.root_active[root], share
                )
        self.last = now

    def submit(self, request_id, root, agent, prompt_tokens, now):
        self.advance(now)
        assert request_id not in self.active and prompt_tokens > 0
        assert self.agent_root.get(agent, root) == root
        self.agent_root[agent] = root
        self.active[request_id] = dict(
            root=root, agent=agent, prompt_tokens=prompt_tokens, arrival=now
        )
        if not self.agent_active_requests[agent]:
            self.root_active[root] += 1
        self.agent_active_requests[agent] += 1

    def finish(self, request_id, now):
        self.advance(now)
        req = self.active.pop(request_id, None)
        if req is not None:
            agent = req["agent"]
            self.agent_active_requests[agent] -= 1
            if not self.agent_active_requests[agent]:
                del self.agent_active_requests[agent]
                self.root_active[req["root"]] -= 1
                if not self.root_active[req["root"]]:
                    del self.root_active[req["root"]]

    @lru_cache(maxsize=32768)
    def cost(self, phase, p, d, k):
        ms, outside = predict(self.calibration[phase], dict(phase=phase, p=p, d=d, k=k))
        if not math.isfinite(ms) or ms <= 0:
            return None
        # predict continues the grid's edge slope, exactly as offline analysis.
        # Out-of-grid coordinates alone must not permanently invalidate a root.
        if outside:
            self.extrapolated_cost_evaluations += 1
        return ms / 1000.0

    def dispatched(self, agent):
        self.executed_agents.add(agent)

    def accepted(self, step):
        if not step["accepted"]:
            return
        root, agent = step["root"], step["agent"]
        self.dispatched(agent)
        cost = self.cost(step["phase"], step["p"], step["d"], step["k"])
        self.accepted_steps += 1
        self.output_tokens += step["generated_tokens"]
        if cost is None:
            self.invalid[root] += 1
        else:
            self.work[root] += cost

    def measured(self, root):
        if self.invalid[root] or self.work[root] <= 0:
            return None
        return self.entitlement[root] / self.work[root]

    def next_five(self, prompt_tokens, known_cached_tokens=0, budget=None):
        if budget is None:
            budget = self.prefill_budget
        # Predict five engine steps from the prompt and observed cache state.
        # Unknown cache reuse is conservatively zero.
        d = min(max(0, known_cached_tokens), prompt_tokens - 1)
        total = 0.0
        for _ in range(5):
            phase = "prefill" if d < prompt_tokens else "decode"
            p = min(budget, prompt_tokens - d) if phase == "prefill" else 1
            cost = self.cost(phase, p, d, d + p)
            if cost is None:
                return None
            total += cost
            d += p
        return total

    def score(self, request_id, known_cached_tokens=0):
        req = self.active[request_id]
        root = req["root"]
        measured = self.measured(root)
        if self.work[root] > 0 or self.invalid[root]:
            return dict(U=measured, score=measured, kind="measured", lookahead_s=None)
        predicted = self.next_five(req["prompt_tokens"], known_cached_tokens)
        value = None
        if predicted is not None and not self.invalid[root]:
            value = self.entitlement[root] / (self.work[root] + predicted)
        return dict(
            U=measured,
            score=value,
            kind="zero_work_root_next_five",
            lookahead_s=predicted,
        )


class Reader:
    def __init__(self):
        self.path = Path(os.environ["DURU_SCORES"])
        self.last = 0.0
        self.snapshot = {}
        self.reason = "missing_snapshot"
        self.age = None
        self.counters = {}
        self.pinned = None

    def pin(self, snapshot):
        @contextmanager
        def decision():
            previous = self.pinned
            self.pinned = snapshot
            try:
                yield
            finally:
                self.pinned = previous

        return decision()

    def refresh(self, force=False):
        if self.pinned is not None:
            return self.pinned
        if force or time.monotonic() - self.last >= 0.02:
            self.last = time.monotonic()
            try:
                self.snapshot = json.loads(self.path.read_text())
            except FileNotFoundError:
                self.snapshot = {}
        self.age = (
            time.time() - self.snapshot.get("observation_time", 0)
            if self.snapshot
            else None
        )
        if not self.snapshot:
            self.reason = "missing_snapshot"
        elif self.age < 0:
            self.reason = "future_snapshot"
        elif self.age > 2:
            self.reason = "stale_snapshot"
        else:
            self.reason = "fresh"
            return self.snapshot
        self.counters[self.reason] = self.counters.get(self.reason, 0) + 1
        return {}


class Tail:
    def __init__(self, path):
        self.path = Path(path)
        self.offset = 0
        self.sequence = 0
        self.watermark = 0.0
        self.closed = False

    def read(self):
        if not self.path.exists():
            return []
        assert self.path.stat().st_size >= self.offset, "Journal truncated"
        result = []
        with self.path.open("rb") as f:
            f.seek(self.offset)
            while True:
                pos = f.tell()
                line = f.readline()
                if not line.endswith(b"\n"):
                    f.seek(pos)
                    break
                if self.closed:
                    raise ValueError(f"Event after journal footer: {self.path}")
                r = json.loads(line)
                self.closed = r["event"] == "footer"
                self.sequence += 1
                assert r["sequence"] == self.sequence, (
                    "Journal gap",
                    self.path,
                    self.sequence,
                )
                self.watermark = max(self.watermark, r["time"])
                result.append(r)
            self.offset = f.tell()
        return result


def _program_id(fields):
    program_id = fields.get("program_id")
    if not isinstance(program_id, str) or not program_id:
        raise ValueError("Ingress send requires a nonempty program_id")
    return program_id


class Ingress:
    """Publish request lifecycle events with explicit router program identity."""

    def __init__(self, path):
        self.journal = open_journal(path, name="duru-ingress")
        self.sequence = 0

    def emit(self, event, timestamp, **fields):
        if event not in ("send", "end", "cancel", "error", "u_clock", "footer"):
            return
        if event == "send":
            _program_id(fields)
        keys = (
            ("request_id", "program_id", "root", "agent", "prompt_tokens")
            if event == "send"
            else ("request_id",)
        )
        self.sequence += 1
        self.journal.submit(
            "record",
            dict(
                sequence=self.sequence,
                event=event,
                time=timestamp,
                **{k: fields[k] for k in keys if k in fields},
            ),
        )

    def close(self):
        try:
            self.emit("footer", time.time())
        finally:
            self.journal.close()


def run(config, stop=None):
    cfg = json.loads(Path(config).read_text())
    stop = stop if stop is not None else threading.Event()
    score_path = Path(cfg["scores"])
    score_path.parent.mkdir(parents=True, exist_ok=True)
    score_path.unlink(missing_ok=True)
    if not cfg["ledgers"]:
        raise ValueError("At least one worker ledger is required")
    run_id = uuid.uuid4().hex
    book = LiveU(
        cfg["capacity"], load_calibration(cfg["calibration"], cfg["runtime_contract"])
    )
    sources = [Tail(Path(cfg["ingress"]))] + [Tail(p) for p in cfg["ledgers"]]
    audit = open_journal(
        cfg.get("audit_dir", score_path.parent), name="duru-control", fresh=False
    )
    sequence = 0
    heap = []
    serial = 0
    last_save = 0.0
    last_snapshot = None
    # Completed requests can still receive accepted-work callbacks.
    request_identities = {}
    request_programs = {}
    program_identities = {}
    stopped_cleanly = False
    try:
        while True:
            stopping = stop.is_set()
            for index, source in enumerate(sources):
                for row in source.read():
                    if index == 0:
                        if row["event"] in (
                            "send",
                            "end",
                            "cancel",
                            "error",
                        ):
                            serial += 1
                            heapq.heappush(heap, (row["time"], serial, "ingress", row))
                    elif row["event"] in ("scheduled", "accepted_output"):
                        for st in row["steps"]:
                            at = (
                                st["scheduled_at"]
                                if row["event"] == "scheduled"
                                else st["end"]
                            )
                            serial += 1
                            heapq.heappush(heap, (at, serial, row["event"], st))
            # Closed journals cannot contribute later events or hold back other sources.
            watermark = min(
                (s.watermark for s in sources if not s.closed),
                default=max(s.watermark for s in sources),
            )
            if watermark > 0:
                while heap and heap[0][0] <= watermark:
                    at, _, kind, r = heapq.heappop(heap)
                    assert book.last is None or at >= book.last, (
                        "Late compact event",
                        at,
                        book.last,
                        kind,
                    )
                    # Accepted steps do not change active roots or agents.
                    # Integrate E only on ingress transitions and at publication.
                    # This is the same integral without O(steps * active roots).
                    if kind == "ingress":
                        if r["event"] == "send":
                            request_id = r["request_id"]
                            if request_id in request_identities:
                                raise ValueError("Request ID reused: " + request_id)
                            program_id = _program_id(r)
                            identity = (r["root"], r["agent"])
                            if program_identities.get(program_id, identity) != identity:
                                raise ValueError(
                                    "Router program identity changed: " + program_id
                                )
                            program_identities[program_id] = identity
                            request_programs[r["request_id"]] = program_id
                            request_identities[request_id] = identity
                            book.submit(
                                r["request_id"],
                                r["root"],
                                r["agent"],
                                r["prompt_tokens"],
                                at,
                            )
                        else:
                            book.finish(r["request_id"], at)
                            request_programs.pop(r["request_id"], None)
                    else:
                        request_id = r["client_request_id"]
                        if request_id not in request_identities:
                            raise ValueError("Unjoined work: " + request_id)
                        if (r.get("root"), r.get("agent")) != request_identities[
                            request_id
                        ]:
                            raise ValueError(
                                "Worker request identity mismatch: " + request_id
                            )
                        if kind == "scheduled":
                            book.dispatched(r["agent"])
                        else:
                            book.accepted(r)
                book.advance(watermark)
                requests = {
                    rid: dict(book.score(rid), program_id=request_programs[rid], **r)
                    for rid, r in book.active.items()
                }
                programs = {
                    r["program_id"]: dict(r, request_id=rid)
                    for rid, r in requests.items()
                }
                for program_id, (root, agent) in program_identities.items():
                    programs.setdefault(
                        program_id,
                        dict(
                            program_id=program_id,
                            root=root,
                            agent=agent,
                            U=book.measured(root),
                            score=None,
                            kind="idle_measured_only",
                        ),
                    )
                snapshot = dict(
                    schema=1,
                    run_id=run_id,
                    metric=METRIC,
                    extrapolation_policy=EXTRAPOLATION_POLICY,
                    extrapolated_cost_evaluations=book.extrapolated_cost_evaluations,
                    entitlement_scope="hierarchical PS: root rate min(active agents, replicas / active roots)",
                    time=time.time(),
                    observation_time=watermark,
                    capacity=book.capacity,
                    live_accounting="compact_local_v1",
                    source_watermarks=[x.watermark for x in sources],
                    source_offsets=[x.offset for x in sources],
                    requests=requests,
                    programs=programs,
                    accepted_steps=book.accepted_steps,
                    output_tokens=book.output_tokens,
                    executed_agents=sorted(book.executed_agents),
                    roots={
                        r: dict(
                            E=book.entitlement[r],
                            W=book.work[r],
                            U=book.measured(r),
                            invalid_steps=book.invalid[r],
                            active_agents=book.root_active[r],
                        )
                        for r in book.agent_root.values()
                    },
                )
                raw = json.dumps(snapshot)
                tmp = score_path.with_suffix(".tmp")
                tmp.write_text(raw)
                tmp.replace(score_path)
                last_snapshot = snapshot
                if time.monotonic() - last_save >= 1:
                    sequence += 1
                    audit.submit(
                        "record", dict(sequence=sequence, event="snapshot", **snapshot)
                    )
                    last_save = time.monotonic()
            if stopping:
                stopped_cleanly = True
                break
            stop.wait(0.05)
    finally:
        try:
            if heap:
                logger.warning(
                    "Stopping with %d events beyond the common watermark", len(heap)
                )
            sequence += 1
            audit.submit(
                "record",
                dict(
                    sequence=sequence,
                    event="footer",
                    run_id=run_id,
                    time=time.time(),
                    last_snapshot=last_snapshot,
                    pending_events=len(heap),
                    inputs_closed=all(source.closed for source in sources),
                    complete=stopped_cleanly
                    and all(source.closed for source in sources)
                    and not heap,
                ),
            )
        finally:
            try:
                audit.close()
            finally:
                score_path.unlink(missing_ok=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Calculate and publish Duru U scores")
    parser.add_argument("config", help="JSON service configuration")
    args = parser.parse_args(argv)
    stop = threading.Event()

    def request_stop(signum, frame):
        stop.set()

    previous = {
        sig: signal.signal(sig, request_stop) for sig in (signal.SIGINT, signal.SIGTERM)
    }
    try:
        run(args.config, stop)
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)


if __name__ == "__main__":
    main()
