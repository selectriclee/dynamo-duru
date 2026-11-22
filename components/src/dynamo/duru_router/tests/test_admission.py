"""Exercise Duru without ThunderAgent state or engine dependencies."""

import subprocess
import sys

import pytest
from dynamo.duru_router.admission import Candidate, DuruPolicy, Resource


def test_core_imports_without_serving_dependencies():
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; from dynamo.duru_router.admission import DuruPolicy; "
            "from dynamo.duru_router.scoring import LiveU; "
            'assert not any(name.startswith(("dynamo.thunderagent_router", '
            '"vllm", "serving_measurement")) for name in sys.modules)',
        ],
        check=True,
        capture_output=True,
        timeout=10,
    )


def test_another_host_applies_ranked_decisions():
    policy = DuruPolicy(alpha=2.0)
    candidates = [
        Candidate("fits", 40, score=0.3, pending_request=True),
        Candidate("large", 110, score=0.9, pending_request=True),
        Candidate("highest", 80, score=3.0, pending_request=True),
    ]
    resources = [Resource("engine-A", 100, 0), Resource("engine-B", 50, 0)]
    admitted = []
    for candidate in policy.order(candidates):
        decision = policy.select(candidate, resources, established_waiters=True)
        if decision is None:
            decision = policy.select_override(
                candidate, resources, established_waiters=True
            )
        if decision is None:
            continue
        admitted.append((candidate.identity, decision.resource))
        resources = [
            r._replace(
                used=r.used + candidate.demand, used_decayed=r.used + candidate.demand
            )
            if r.key == decision.resource
            else r
            for r in resources
        ]
    assert admitted == [("highest", "engine-A"), ("fits", "engine-B")]


def test_idle_measurement_stable_ties_and_disabled_order():
    candidates = [
        Candidate("first", 1, score=0.2, arrival=2),
        Candidate("unknown", 1, score=9, known=False),
        Candidate("second", 1, score=0.2, arrival=2),
        Candidate("idle", 1, measured_u=0.5, idle=True),
    ]
    assert [c.identity for c in DuruPolicy().order(candidates)] == [
        "idle",
        "first",
        "second",
    ]
    disabled = DuruPolicy(alpha=1, enabled=False)
    assert disabled.order(candidates) == candidates
    assert not disabled.may_force_admit(candidates[1]._replace(pending_request=True))


def test_fit_ties_use_rank_but_override_ties_use_insertion_order():
    resources = [
        Resource("first", 100, 0, tie_break=-2),
        Resource("second", 100, 0, tie_break=-1),
    ]
    policy = DuruPolicy(alpha=0.7)
    candidate = Candidate("request", 100, score=0.7, pending_request=True)
    assert (
        policy.select(candidate, resources, established_waiters=True).resource
        == "second"
    )
    candidate = candidate._replace(demand=101)
    assert policy.select(candidate, resources, established_waiters=True) is None
    assert (
        policy.select_override(candidate, resources, established_waiters=True).resource
        == "first"
    )


@pytest.mark.parametrize("established", [False, True])
def test_established_waiter_reserves_resume_headroom(established):
    candidate = Candidate("new", 90, score=1, first_admission=True)
    decision = DuruPolicy(resume_ceiling=0.8).select(
        candidate, [Resource("worker", 100, 0)], established_waiters=established
    )
    assert (decision is None) == established


@pytest.mark.parametrize(
    "score,pending,capacity,forced",
    [
        (0.69, True, 100, False),
        (float("nan"), True, 100, False),
        (9, False, 100, False),
        (0.7, True, 0, False),
        (0.7, True, 100, True),
    ],
)
def test_alpha_requires_pending_request_and_positive_capacity(
    score, pending, capacity, forced
):
    candidate = Candidate("request", 200, score=score, pending_request=pending)
    decision = DuruPolicy(alpha=0.7).select_override(
        candidate, [Resource("worker", capacity, 0)], established_waiters=True
    )
    assert (decision is not None) == forced
    if forced:
        assert decision.forced
