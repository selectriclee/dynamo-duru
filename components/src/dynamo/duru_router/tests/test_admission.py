"""Exercise Duru without ThunderAgent state or engine dependencies."""

import pytest

from dynamo.duru_router.admission import Candidate, DuruPolicy

pytestmark = [pytest.mark.pre_merge, pytest.mark.unit, pytest.mark.gpu_0]


def test_order_does_not_prefilter_by_capacity():
    policy = DuruPolicy(alpha=2.0)
    candidates = [
        Candidate("fits", 40, score=0.3, pending_request=True),
        Candidate("large", 110, score=0.9, pending_request=True),
        Candidate("highest", 80, score=3.0, pending_request=True),
    ]
    ordered = policy.order(candidates)
    assert [c.identity for c in ordered] == ["highest", "large", "fits"]
    assert [policy.may_force_admit(c) for c in ordered] == [True, False, False]


def test_idle_measurement_and_stable_ties():
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


@pytest.mark.parametrize("established", [False, True])
def test_established_waiter_reserves_resume_headroom(established):
    candidate = Candidate("new", 90, score=1, first_admission=True)
    ceiling = DuruPolicy(resume_ceiling=0.8).ceiling(candidate, established)
    assert ceiling == (0.8 if established else 1.0)


@pytest.mark.parametrize(
    "score,pending,allowed",
    [
        (0.69, True, False),
        (float("nan"), True, False),
        (float("inf"), True, False),
        (-1, True, False),
        (9, False, False),
        (0.7, True, True),
    ],
)
def test_alpha_requires_pending_request_and_valid_score(score, pending, allowed):
    candidate = Candidate("request", 200, score=score, pending_request=pending)
    assert DuruPolicy(alpha=0.7).may_force_admit(candidate) is allowed
