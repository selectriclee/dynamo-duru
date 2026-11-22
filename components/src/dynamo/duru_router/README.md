# Duru router

Duru provides fairness accounting, calibrated work prediction, and score-based
admission. This package integrates that policy with Dynamo and ThunderAgent.

## Reuse the policy

`admission.py` and the calibration predictor have no serving dependencies.
`scoring.py` combines U calculation, journal consumption, and snapshot reading; it
uses the journal transport without importing ThunderAgent, vLLM, or the runtime. `DuruPolicy` accepts plain
`Candidate` and `Resource` values and returns an `Admission`; resource identifiers
can be strings, replica pairs, or another host's keys. The host applies decisions.

The ThunderAgent adapter subclasses the existing scheduler. It reuses admission
serialization, cancellation rollback, token accounting, background task lifecycle,
pressure pausing, program-state transactions, request streaming, and per-replica
capacity discovery. Pins contain both worker ID and DP rank. Adapter hooks retain
Duru's fresh-score waits and alpha-based admission. The default ThunderAgent policy
still uses its existing timeout and replica rules.

Another host supplies fresh scores, candidate demand/activity, and current capacity
and usage. It calls `order`, then `select` for each candidate; after a fit failure,
`select_override` applies the alpha rule. Update resource usage after each admission.
Clear `first_admission` after each attempted candidate, including a failed fit.
Compute `established_waiters` from the entire paused queue before filtering unknown
scores; this preserves headroom for existing waiters. Only eligible paused inference
requests set `pending_request=True`; idle agents cannot force admission.
The host owns cancellation, child dependencies, observation, and enforcement. Backend
calibration measurements and engine queue adapters remain backend-specific.

## The rule

1. Consider router admissions in decreasing U order. At the worker, order requests
   by decreasing U within each native waiting queue. vLLM retains precedence for
   requests already holding KV blocks and its pause handling.
2. The router may try another candidate when one cannot fit its admission budget.
   The worker uses native vLLM allocation behavior: if allocation fails, stop
   admission for that step. There is no worker backfill or vLLM source patch.
3. If normal router admission fails and the pending request has **U >= alpha**,
   admit it to the least-loaded eligible worker despite the router's budget.

A non-fitting high-U request can hold up smaller requests at the worker.

The worker still requires real KV space before executing a request. The exception
does not guarantee immediate GPU execution. Running requests keep the usual FCFS
behavior. There are no beta/gamma rules or fairness-based preemptions.

ThunderAgent still pauses agents in their tool phase when a worker is over its
memory budget, starting with smaller working sets. Its normal admission and resume
limits remain. Each session has its own reservation; U accounting is shared by the
root and its descendants when the ingress supplies their relationship. A parent
waiting for children retains its reservation until normal pressure pausing or session
termination releases it. Parent/child waits do not automatically release reservations or guarantee progress. These memory rules are not
additional U thresholds. Duru resumes by score and capacity; ThunderAgent's
30-minute forced-admission timer is off.

For a root with no accepted completed work, a pending request may use the predicted
cost of its next five engine steps for lookahead. Use only the submitted prompt,
observed cache state and matched calibration; unknown cache reuse is a miss.
After the first accepted root/descendant work, every child uses measured root U,
even if that child is new. First dispatch alone does not end lookahead.
Missing, invalid or stale accounting is not zero work and cannot trigger an exception.

Live accepted-work costs and next-five-step lookahead use the calibration grid's
edge-slope continuation outside its measured range: bilinear prefill and linear
decode. Accept only finite, positive predictions;
out-of-grid coordinates alone do not invalidate U. Snapshots label the policy
`positive_finite_edge_slope_v1` and count extrapolated predictor evaluations
(including lookahead and cache re-evaluations, not accepted steps or roots).
This does not certify calibration accuracy outside the measured grid.

## U accounting

U uses hierarchical processor sharing. At time t,
N is the number of roots with outstanding requests and k is the number of distinct
active agents of a root. Each agent's reference rate is `min(1, G/(N*k))`.
Equivalently, the root accrues entitlement at `min(k, G/N)`; divide its integral
by the root's summed calibrated accepted work. G counts matching TP replicas.

Requests waiting at the router or worker count as outstanding. Union overlapping
requests within one agent; do not count tool/child-join gaps with no request.
Divide the root share among active agents BEFORE applying the per-agent cap.
Unused shares are not redistributed, and total reference rate never exceeds G.

With G=2 and two roots, a two-child root gets total rate1 and a single-agent root
gets1. With G=4, they get2 and1 respectively. Fanout can use more of a root's
share but cannot raise its G/N ceiling.

Metric ID: `root_hierarchical_ps_accepted_work_v3`. U is normalized slowdown
relative to hierarchical processor sharing over outstanding requests. Tool time
and other gaps without outstanding requests do not accrue entitlement.

## Where to change the code

| File | Purpose |
| --- | --- |
| `admission.py` | Portable score ordering, capacity choice, and alpha admission decisions |
| `calibration/` | Graph-aware points, repeat measurements, shared cost predictor |
| `scoring.py` | Calculate U from events, publish scores, and read snapshots for the router/worker |
| `events.py` | Record the compact worker event feed and preserve accounting timestamps |
| `router.py` | Apply admission through ThunderAgent, check score freshness, and log decisions |
| `worker.py` | Highest-U waiting order, fresh-score guard, and accepted-work observation |
| `args.py` | Router CLI configuration |
| `__main__.py` | Shared ThunderAgent service startup and alpha checks |

## Use it

This adapter targets **vLLM 0.33.0** with its native asynchronous scheduler.
Current-runtime worker startup and GPU serving still need an integration check.
Calibration planning, measurement, and prediction are included in
[calibration/](calibration/README.md); the measurement CLI requires a serving
adapter supplied by the caller.

Select `dynamo.duru_router.worker.DuruScheduler` as the vLLM scheduler class.
Give each worker its own node-local `DURU_WORK_DIR`. All workers and the router
read the score snapshot at `DURU_SCORES`. The scorer must be able to read every
worker's event journal; this file-based transport assumes locally accessible
journals and snapshots. No separate measurement package is required.

The router entrypoint requires alpha explicitly:

```bash
export DURU_SCORES=/local/session/scores.json
export DURU_AUDIT_DIR=/local/session/router
python3 -m dynamo.duru_router --alpha 0.7 \
  --endpoint dynamo.vllm.generate --router-block-size 64
```

Supply the actual endpoint and block size for the deployment. The service uses
ThunderAgent's existing generate, status, and metrics endpoints. Worker metrics
use `"worker_id:dp_rank"` keys; `workers_total` counts replicas. Session lifecycle
is ACTIVE, PAUSED, or TERMINATED.

Connect `scoring.Ingress(directory).emit` to request send/end/cancel/error events.
For `send`, provide `request_id`, `program_id`, `root`, `agent`, `prompt_tokens`,
and the event timestamp. `program_id` must exactly match the router request's
`session_id`; its spelling is independent of the accounting agent ID.
Publish idle `u_clock` events too. The ingress writer creates `duru-ingress.jsonl`;
call `Ingress.close()` after requests drain. Each worker writes
`compact-work.jsonl` in its `DURU_WORK_DIR`.

Use these vLLM `extra_args` to correlate worker observations with ingress:

| Field | Value | Default |
| --- | --- | --- |
| `duru_request_id` | Ingress `request_id` | Native vLLM request ID |
| `duru_root_id` | Ingress `root` | Correlation request ID |
| `duru_agent_id` | Ingress `agent` | Root ID |

Set them explicitly when the serving path changes request IDs or several agents
share a root. Reusing one program ID for different root/agent identities is an
error. Request IDs must be unique within an accounting session, and worker
root/agent IDs must match ingress. Identity mapping does not change the hierarchical U calculation.

Start the score service with a JSON configuration:

```bash
python3 -m dynamo.duru_router.scoring /path/to/scoring.json
```

```json
{
  "ingress": "/local/session/ingress/duru-ingress.jsonl",
  "ledgers": ["/local/session/worker-0/compact-work.jsonl"],
  "scores": "/local/session/scores.json",
  "audit_dir": "/local/session/audit",
  "capacity": 1,
  "calibration": "/local/calibration/calibration.json",
  "runtime_contract": {"model": "your-model"}
}
```

Set `runtime_contract` to the actual serving contract recorded in the calibration
artifact. `capacity` counts TP replicas, not GPUs. `audit_dir` is optional and
defaults to the score directory. Run one score-service process per score path.

The service reads every configured journal from the beginning. It waits for all
inputs to supply an accounting timestamp before publishing; missing or stale
scores defer admission. Restarting the scorer reconstructs U from retained
journals. Each audit run has its own `run_id`.

SIGINT and SIGTERM request shutdown. The service reads available records one
final time, flushes its audit, and removes its score snapshot. For complete final
accounting, drain requests and close the ingress and worker journals before
signalling the service. If inputs remain open or events cannot yet be ordered,
the audit footer reports `complete: false`. A crash leaves scores to expire.

Producer journals are immutable streams. Producers refuse an existing journal
path. Start a new accounting session with fresh producer directories when
producers restart, and update the service's input paths. Restarting only the
scorer reuses the existing inputs. Event queue overflow and writer failures stop
accounting instead of silently dropping work; flushes do not provide power-loss
durability.

## Tests

Run the CPU tests in the Dynamo development environment:

```bash
PYTHONPATH=components/src python3 -m pytest components/src/dynamo/duru_router/tests -q
```

Tests cover U calculation, admission order, alpha boundaries, stale scores,
idle reservations, cancellation, worker callback accounting, and event transport.
`test_scoring.py` includes process lifecycle tests for startup, replay, signals,
and shutdown. Native ThunderAgent tests exercise the shared machinery. Backend
stand-ins in CPU tests do not establish compatibility with a running vLLM worker.
