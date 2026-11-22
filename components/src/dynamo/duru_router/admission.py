"""Duru decisions from explicit inputs, independent of routers and engine APIs.

Hosts own freshness, observations, resource accounting, and applying decisions.
Resource keys are opaque; tie_break supplies the host's deterministic fit order.
"""

from dataclasses import dataclass
from math import isfinite
from typing import Hashable, NamedTuple


def valid_score(value):
    return isinstance(value, (int, float)) and isfinite(value) and value >= 0


def priority_key(score, arrival, fallback_index):
    """Descending valid scores, then arrival; invalid scores keep input order."""
    if not valid_score(score):
        return (1, 0.0, 0.0, fallback_index)
    return (0, -score, arrival, fallback_index)


class Candidate(NamedTuple):
    identity: str
    demand: int
    score: float | None = None
    measured_u: float | None = None
    arrival: float = 0.0
    known: bool = True
    idle: bool = False
    pending_request: bool = False
    first_admission: bool = False

    @property
    def effective_score(self):
        # Idle agents have measured work, but no pending-request lookahead.
        return (
            self.measured_u if self.idle and not valid_score(self.score) else self.score
        )


class Resource(NamedTuple):
    key: Hashable
    capacity: int
    used: int
    tie_break: int = 0
    used_decayed: int | None = None


class Admission(NamedTuple):
    resource: Hashable
    remaining: int
    ceiling: float
    forced: bool = False


@dataclass(frozen=True, slots=True)
class DuruPolicy:
    alpha: float | None = None
    resume_ceiling: float = 1.0
    enabled: bool = True

    def __post_init__(self):
        if self.alpha is not None and (not isfinite(self.alpha) or self.alpha <= 0):
            raise ValueError("Admission alpha must be finite and positive")

    def order(self, candidates: list[Candidate]) -> list[Candidate]:
        if not self.enabled:
            return list(candidates)
        return [
            candidate
            for _, candidate in sorted(
                ((i, c) for i, c in enumerate(candidates) if c.known),
                key=lambda pair: priority_key(
                    pair[1].effective_score, pair[1].arrival, pair[0]
                ),
            )
        ]

    def ceiling(self, candidate: Candidate, established_waiters: bool) -> float:
        if candidate.first_admission and not established_waiters:
            return 1.0
        return max(0.0, min(1.0, self.resume_ceiling))

    def may_force_admit(self, candidate: Candidate) -> bool:
        return bool(
            self.enabled
            and self.alpha is not None
            and candidate.pending_request
            and valid_score(candidate.score)
            and candidate.score >= self.alpha
        )

    def select(
        self,
        candidate: Candidate,
        resources: list[Resource],
        *,
        established_waiters: bool,
    ) -> Admission | None:
        ceiling = self.ceiling(candidate, established_waiters)
        fits = [
            (r, int(r.capacity * ceiling) - r.used)
            for r in resources
            if r.capacity > 0 and int(r.capacity * ceiling) - r.used >= candidate.demand
        ]
        if not fits:
            return None
        resource, remaining = max(fits, key=lambda pair: (pair[1], pair[0].tie_break))
        return Admission(resource.key, remaining, ceiling)

    def select_override(
        self,
        candidate: Candidate,
        resources: list[Resource],
        *,
        established_waiters: bool,
    ) -> Admission | None:
        """Called only after fit fails; the host supplies current decayed usage."""
        if not self.may_force_admit(candidate):
            return None
        usable = [r for r in resources if r.capacity > 0]
        if not usable:
            return None
        resource = max(
            usable,
            key=lambda r: (
                r.capacity - (r.used if r.used_decayed is None else r.used_decayed)
            ),
        )
        ceiling = self.ceiling(candidate, established_waiters)
        return Admission(
            resource.key,
            int(resource.capacity * ceiling) - resource.used,
            ceiling,
            True,
        )
