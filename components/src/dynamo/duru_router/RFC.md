# RFC: Incremental Duru integration with ThunderAgent

Status: local draft for review; not filed as a DEP and not an approved interface.
Proposed area: router, with a separately reviewed vLLM adapter.

## Summary

Add Duru as an opt-in fairness policy that reuses ThunderAgent's scheduler.
ThunderAgent continues to own capacity checks, worker placement, reservations,
locking, pause/resume execution, and cancellation. Duru supplies queue order and
eligibility for a capacity exception. A router-owned accounting component computes
U from request lifecycles and accepted engine work. Engine processes receive the
latest request scores through local IPC.

The current branch is a working prototype spanning all of these concerns. This
proposal breaks it into small changes with independent tests and explicit
dependencies. The first upstream PR should contain only a behavior-preserving
ThunderAgent refactor. It should not introduce U accounting, IPC, or Duru flags.

## Motivation

Agent requests alternate between generation and tool execution. Serving individual
requests in arrival order does not directly account for service received across
an agent's lifetime or across children of the same root. Duru uses accumulated
entitlement and accepted work to prioritize underserved roots.

The review problem is separate from the scheduling problem: the prototype mixes
shared scheduler changes, fairness mathematics, engine instrumentation, transport,
calibration, and deployment wiring. Reviewers should be able to accept or reject
each mechanism without reviewing the entire implementation at once.

## Proposal

### Responsibilities

| Component | Owns | Does not own |
| --- | --- | --- |
| ThunderAgent scheduler | Capacity tests, placement, reservations, lifecycle transitions and lock discipline | U calculation or socket transport |
| Duru policy | Candidate ordering, admission ceiling and exception eligibility | Direct mutation of worker reservations |
| U accounting | E/W state, accepted-work costs, lifecycle cleanup and fresh snapshots | Native scheduling operations |
| Event transport | Bounded, ordered delivery of captured measurements | Fairness decisions |
| Score transport | Latest request scores and observation time for engine processes | U computation or event history |
| Diagnostics | Optional counters, logs and recordings | Input to scheduling decisions |

The intended data flow is:

```text
request lifecycle + accepted engine work
                    |
             router accounting
                    |
          immutable score snapshot
             /             \
    direct memory       latest-score IPC
         |                    |
  router policy       engine waiting order
```

Event delivery and score delivery have different contracts. Every accepted-work
event must be accounted for in order. Score consumers may skip intermediate
snapshots and keep only the newest one. Combining these into one generic queue
would obscure this distinction.

### Accounting contract

For root `r`, `U_r = E_r / W_r`. Entitlement accumulates while that root has
outstanding requests. With `G` replica equivalents, `N` active roots, and `A_r`
active agents within a root, its entitlement rate is `min(A_r, G/N)`.
`W_r` accumulates calibrated costs of accepted engine work. Generated-token
counts are throughput diagnostics, not the work-cost definition.

The zero-work case uses the existing finite lookahead estimate until measured
work is available. Invalid costs must not produce an admissible score. Calibration
must identify the model and serving configuration it describes.

A common producer watermark aligns entitlement and accepted work. A request
ending does not mean its root is finished: tool gaps retain accumulated E/W, and
late engine callbacks still need request identity. Explicit terminal root events
release retained identities and accounting state after all work is accounted for.

### Scheduling contract

Candidates are ordered by Duru priority. For each candidate, ThunderAgent first
tries normal capacity admission. If that fails, the policy may authorize the
alpha exception; ThunderAgent performs the resulting placement and state changes.
If neither path admits the candidate, the scheduler considers later candidates
for capacity backfill.

There is no global pass that forces every eligible candidate before trying normal
admission. The normal-then-exception check happens per candidate in priority order.

The prototype also changes initial-admission ranking, retries when request state
changes, timeout forcing, handling of unplaced returning programs, and deferred
reasoning-phase pauses. These are observable behavior changes. They must be
reviewed explicitly, not hidden inside a queue-order refactor.

### Runtime contract

The router owns accounting lifecycle and reads immutable snapshots directly.
Workers refresh a compact score snapshot on a background thread; their scheduler
callbacks only read memory. Stale, missing, or future-dated snapshots defer new
admission rather than silently falling back to a different policy.

The initial implementation is same-node and targets vLLM 0.33.0 asynchronous,
non-speculative scheduling. It requires explicit request/program/root identity
correlation and producer completion notifications. Cross-node transport, scorer
recovery, speculative decoding, and other engine adapters are separate proposals.

The public handler owns lifecycle publication and identity injection. Session IDs
retain accounting across turns; child sessions join an already registered parent.
Session-final notifications end router reservations. Accounting retains a finished
root until all descendants have ended and dispatched requests have worker terminal
acknowledgements, so late accepted callbacks are counted before cleanup.

Both ingress and worker events consume the same bounded pending-event budget,
including when a producer stops advancing its watermark. The serving task and
accounting monitor share a lifetime: an accounting failure rejects new work,
cancels waiting/in-flight requests, and stops the service endpoints.

## Small PR sequence

Each row is a separate review unit, not a promise to merge every proposed
abstraction. Dependencies indicate code requirements; implementation can proceed
in parallel where they permit. Each PR gets its own work issue when filed, with
`Part of #<DEP>` linking the umbrella proposal. No placeholder issue number should
be submitted as a real reference.

| PR | Single problem and boundary | Dependencies | Acceptance evidence |
| --- | --- | --- | --- |
| 1 | Extract native resume planning/selection from `thunderagent_router/router.py` into the smallest policy interface. Default implementation exactly preserves native behavior. | None | Native tests plus deterministic before/after traces for queue order, capacity, placement, timeout, tool return and cancellation. No Duru imports. |
| 2 | Add pure entitlement and accepted-work accounting, including terminal cleanup. Extract `LiveU` from the runtime module. Inject the cost function. | None | Hand-computed E/W/U cases, fanout, tool gaps, invalid costs, late work and root cleanup. No transport or engine imports. |
| 3 | Add calibration representation and cost prediction. | None | Contract validation, interpolation, boundaries, positive finite costs and out-of-grid behavior. No scheduler changes. |
| 4 | Add calibration planning/measurement tooling. Keep experiment orchestration outside the runtime package. | 3 | Reproducible point generation and a fake measurement adapter; separately recorded hardware qualification. |
| 5 | Define typed lifecycle/work observations and implement bounded event delivery. | None | Sequence/watermark ordering, overflow, disconnect, producer closure and bounded buffers. No U-based decisions. |
| 6 | Capture accepted vLLM work and stable identities using the event contract. | 5 | Callback tests and v0.33.0 integration evidence; native scheduling order unchanged when only observing. |
| 7 | Own live accounting in the router and join lifecycle/work observations. | 2, 3, 5 | Watermark alignment, invalid joins, failure propagation, cleanup and replay parity. No file-based scorer or standalone scorer process. |
| 8 | Deliver latest scores to engine processes; move the reader/freshness interface beside score transport. | 5, 7 | Multi-process delivery, immutable decision snapshots, stale/missing scores, reconnect and shutdown. No score-file fallback. |
| 9 | Add pure Duru ordering and alpha eligibility rules. | None | Score order, tie breaks, first admission, idle programs, pending requests and alpha boundaries. No reservation mutations. |
| 10 | Add only the scheduler lifecycle extension points Duru needs, with native defaults preserved. Keep each independently useful hook as a separate commit. | 1 | Explicit tests for each opt-in behavior and unchanged native defaults. Review whether each hook is necessary. |
| 11 | Connect Duru policy and fresh router scores to ThunderAgent admission. | 7, 9, 10 | Normal fit before exception, highest-U ordering, backfill, cancellation races and stale-score deferral. |
| 12 | Add engine waiting-queue ordering as an opt-in adapter. | 6, 8 | Fresh known requests only, decision consistency, native mode parity and asynchronous scheduler tests. |
| 13 | Wire the public entrypoint and configuration for the first supported topology. | 11, 12 | One complete serving smoke test, validated calibration contract and clean startup/shutdown. |
| 14 | Add optional diagnostics, recordings, documentation and a reproducible evaluation recipe. Separate diagnostics from the recipe if the diff grows. | 13 | Logging-off behavior, bounded live state, log overhead measurement and valid matched benchmark results. |

PRs 5 and 6 should remain small enough to review independently: one establishes
delivery semantics, the other establishes what engine callbacks mean. PRs 11 and
12 likewise separate router admission from engine scheduling. A runtime-only
change should not require reviewers to understand the benchmark harness.

The existing prototype must be split into this sequence on a clean branch; these
PRs do not exist yet. A single large feature commit followed by cleanup commits
does not provide the intended review boundaries. Reconstruct the series against
the agreed upstream base, and run each PR's tests at its own tip.

## Immediate refactor scope

The current cleanup removes file-based live scoring and its standalone CLI.
Recorded-input replay remains a test helper, and optional diagnostic recording
remains available. This is one change, separate from renaming modules, moving the
score reader, extracting the pure accounting model, or redesigning policy hooks.

The follow-up cleanup reuses ThunderAgent's CLI configuration, removes unused
scheduled-work notifications, and gives each policy an explicit local score
reader. Accepted-work accounting, completion acknowledgements, and the three Duru
behavior overrides remain unchanged. Detailed root snapshots are optional.
CPU replay of saved engine events checks accounting and policy decisions across
the cleanup; it does not establish new GPU throughput measurements.

Extracting pure accounting or moving `Reader` alongside score transport remain
separate changes requiring their own tests and review diffs.

## Requirements and validation

- Native ThunderAgent behavior MUST remain unchanged with Duru disabled.
- ThunderAgent MUST own capacity checks and state transitions under its lock.
- Live accounting MUST NOT depend on JSONL recordings or a score file.
- Accepted-work delivery MUST fail visibly on gaps or overflow; dropping work
  would change U. Score delivery MAY discard intermediate snapshots.
- Storage MUST have explicit caps, and root completion MUST release retained
  state. Total accounting state scales with outstanding roots and identities;
  it is not constant memory independent of workload concurrency.
- Diagnostic output MUST be optional and MUST NOT determine scheduling outcomes.
- Validation MUST distinguish mathematical tests, transport/replay parity, engine
  compatibility, and end-to-end performance. Passing one does not establish another.
- The experiment MUST reuse unchanged baselines and calibration only with recorded
  provenance and matching contracts. A runtime refactor need not restart an
  already-running experiment that uses immutable source.

A preliminary matched TraceLab run with 500 completed roots per method measured
p99 U of 1.759 for native ThunderAgent and 0.859 for Duru, with throughput of 3,270
and 2,995 tokens/s respectively. This is a fairness improvement with an 8.4%
throughput cost for this single point, not evidence of universal improvement.
Live accounting passed comparison with recorded events across 2,136 snapshots.
Broader workload results and measurements with diagnostics disabled remain needed
before making performance claims about the final upstream implementation.

## Alternatives

- Keep a separate Duru scheduler: avoids shared hooks but duplicates capacity and
  lifecycle logic, increasing the chance that fixes diverge.
- Keep a standalone scorer and file snapshots: makes offline inspection easy but
  adds a runtime process, file I/O and a second live-accounting path.
- Merge all transport into scoring: reduces file count but mixes accounting
  correctness with socket lifecycle and two distinct delivery contracts.
- Change engine order only: smaller integration, but cannot prioritize programs
  waiting for admission at the router.

## Open review questions

1. Which policy/lifecycle hooks belong in ThunderAgent, and which should stay in
   the Duru adapter? In particular, review timeout and deferred-pause behavior.
2. Is same-node IPC an acceptable initial scope, or is cross-node delivery a
   requirement before upstream integration?
3. What existing request lifecycle interface should provide program/root identity
   and terminal notifications outside the experimental replay harness?
4. Should engine score distribution use an existing runtime channel rather than
   a dedicated Unix socket? Resolve ownership and boundedness before replacing it.
5. What benchmark coverage and acceptable throughput tradeoff are required before
   enabling or recommending this policy?

## References

- Local implementation: [README](README.md), [accounting](scoring.py),
  [admission rules](admission.py), [router adapter](router.py),
  [engine adapter](worker.py), [event and score IPC](transport.py).
- Repository proposal format: `.github/ISSUE_TEMPLATE/dep.yml`.
- Maintainer feedback supplied by the author: break the work into the smallest
  practical mergeable changes and file an RFC. No discussion URL was supplied.
- Local experiment provenance: campaign 539776, attempt 032; TraceLab native
  baseline reused from attempt 031. Results are preliminary and not a substitute
  for testing the eventual upstream PR series.
