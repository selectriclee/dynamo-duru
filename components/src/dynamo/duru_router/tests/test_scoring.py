"""U calculation and score-service lifecycle regressions."""

import json
import os
import signal
import subprocess
import sys
import threading
import time
import unittest
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

import pytest
from dynamo.duru_router.calibration.plan import METRIC
from dynamo.duru_router.scoring import Ingress, LiveU, priority_key, run


def work_step(*, p=100, d=0, root="root", agent="agent", **fields):
    return dict(
        accepted=True,
        root=root,
        agent=agent,
        phase="prefill",
        p=p,
        d=d,
        k=p + d,
        generated_tokens=1,
        **fields,
    )


@pytest.mark.pre_merge
@pytest.mark.unit
@pytest.mark.gpu_0
class TestUCalculation(unittest.TestCase):
    def book(self, capacity=4):
        b = LiveU(capacity, {})
        b.cost = lambda phase, p, d, k: 2.0 if phase == "prefill" else 1.0
        return b

    def test_children_split_root_ps_share_and_tool_gaps(self):
        b = self.book(1)
        b.submit("a", "root", "parent", 100, 0)
        b.submit("b", "root", "child", 100, 0)
        b.submit("c", "other", "other", 100, 0)
        b.finish("a", 2)
        b.finish("b", 4)
        b.finish("c", 6)
        b.advance(100)
        self.assertEqual(dict(b.entitlement), {"root": 2.0, "other": 4.0})

    def test_parallel_children_sum_time_and_work(self):
        b = self.book(2)
        b.cost = lambda *args: 10.0
        for agent in ("child1", "child2"):
            b.submit(agent, "root", agent, 100, 0)
            b.accepted(work_step(agent=agent))
        b.advance(10)
        self.assertEqual(b.entitlement["root"], 20.0)
        self.assertEqual(b.work["root"], 20.0)
        self.assertEqual(b.measured("root"), 1.0)

    def test_same_agent_overlap_duplicate_finish_and_return(self):
        b = self.book(1)
        b.submit("a", "root", "agent", 100, 0)
        b.submit("b", "root", "agent", 100, 2)
        b.finish("a", 4)
        b.finish("a", 5)  # Duplicate completion must not release b.
        b.advance(6)  # Cutoff while b remains outstanding.
        self.assertEqual(b.entitlement["root"], 6.0)
        self.assertEqual(b.root_active["root"], 1)
        b.finish("b", 8)
        b.advance(20)  # Tool gap.
        self.assertEqual(b.entitlement["root"], 8.0)
        self.assertFalse(b.root_active)
        b.submit("c", "root", "agent", 100, 20)
        b.finish("c", 22)
        self.assertEqual(b.entitlement["root"], 10.0)

    def test_multiple_requests_do_not_change_root_ps_population(self):
        b = self.book(1)
        b.submit("a", "r1", "a", 100, 0)
        b.submit("b", "r1", "b", 100, 0)
        b.submit("c", "r2", "c", 100, 0)
        b.advance(10)
        self.assertEqual(b.entitlement["r1"], 5.0)
        self.assertEqual(b.entitlement["r2"], 5.0)

    def test_unused_root_share_is_not_redistributed(self):
        b = self.book(4)
        b.submit("a", "X", "a", 100, 0)
        b.submit("b", "X", "b", 100, 0)
        b.submit("c", "Y", "c", 100, 0)
        b.advance(10)
        self.assertEqual(dict(b.entitlement), {"X": 20.0, "Y": 10.0})
        b.submit("d", "X", "d", 100, 10)
        b.advance(20)
        self.assertEqual(dict(b.entitlement), {"X": 40.0, "Y": 20.0})

    def test_single_root_fanout_capped_at_replica_count(self):
        b = self.book(2)
        for i in range(5):
            b.submit(str(i), "root", str(i), 100, 0)
        b.advance(10)
        self.assertEqual(b.entitlement["root"], 20.0)

    def test_zero_work_root_keeps_prediction_until_acceptance(self):
        b = self.book()
        b.submit("a", "root", "agent", 100, 0)
        b.advance(6)
        self.assertIsNone(b.score("a")["U"])
        self.assertEqual(b.score("a")["lookahead_s"], 6.0)
        self.assertEqual(b.score("a")["score"], 1.0)
        b.dispatched("agent")
        self.assertEqual(b.score("a")["score"], 1.0)
        self.assertEqual(b.score("a")["kind"], "zero_work_root_next_five")

    def test_new_child_uses_root_measured_work(self):
        b = self.book()
        b.submit("a", "root", "parent", 100, 0)
        b.accepted(work_step(agent="parent"))
        b.finish("a", 2)
        b.submit("b", "root", "child", 100, 3)
        b.advance(7)
        self.assertEqual(b.score("b")["score"], 3)
        self.assertIsNone(b.score("b")["lookahead_s"])
        self.assertEqual(b.score("b")["kind"], "measured")
        self.assertEqual(b.score("b")["U"], 3)

    def test_stale_outputs_are_not_work(self):
        b = self.book()
        b.accepted(dict(accepted=False))
        self.assertEqual(b.work, {})
        self.assertFalse(b.executed_agents)

    def test_valid_subunit_scores_and_invalid_fallback(self):
        values = [None, 0.2, float("nan"), 0.8, 0.8, 0.0]
        order = sorted(range(len(values)), key=lambda i: priority_key(values[i], i, i))
        self.assertEqual(order, [3, 4, 1, 5, 0, 2])

    def test_unknown_calibration_disables_score(self):
        b = self.book()
        b.cost = lambda *args: None
        b.submit("a", "root", "agent", 100, 0)
        b.advance(1)
        self.assertIsNone(b.score("a")["score"])

    def grid_book(self):
        # Prefill = 10 + 2P + D milliseconds; decode = 10 + K milliseconds.
        return LiveU(
            1,
            {
                "prefill": {"P": [1, 3], "D": [0, 4], "wall_ms": [[12, 16], [16, 20]]},
                "decode": {"grid": [{"K": 1, "wall_ms": 11}, {"K": 5, "wall_ms": 15}]},
            },
        )

    def test_grid_edges_and_extrapolated_prefill_decode(self):
        b = self.grid_book()
        for p, d in ((1, 0), (2, 2), (3, 4), (5, 2), (2, 8), (5, 8)):
            self.assertAlmostEqual(
                b.cost("prefill", p, d, p + d), (10 + 2 * p + d) / 1000
            )
        for k in (0, 1, 3, 5, 9):
            self.assertAlmostEqual(b.cost("decode", 1, k - 1, k), (10 + k) / 1000)
        self.assertEqual(b.extrapolated_cost_evaluations, 5)
        b.cost("decode", 1, 8, 9)  # Cached lookup, no new predictor evaluation.
        self.assertEqual(b.extrapolated_cost_evaluations, 5)

    def test_extrapolation_keeps_accepted_root_u_defined(self):
        b = self.grid_book()
        b.submit("req", "root", "agent", 5, 0)
        b.advance(2)
        b.accepted(work_step(p=5, d=8))
        self.assertEqual(b.invalid["root"], 0)
        self.assertAlmostEqual(b.work["root"], 0.028)
        self.assertAlmostEqual(b.score("req")["U"], 2 / 0.028)

    def test_extrapolated_next_five_lookahead(self):
        b = self.grid_book()
        b.submit("req", "root", "agent", 5, 0)
        b.advance(1)
        # P5/D0 prefill (20 ms), followed by K6..9 decode (16..19 ms).
        self.assertAlmostEqual(b.next_five(5), 0.090)
        self.assertAlmostEqual(b.score("req")["score"], 1 / 0.090)
        self.assertIsNone(b.score("req")["U"])

    def test_nonpositive_and_nonfinite_predictions_still_invalid(self):
        for outside in (False, True):
            for ms in (0, -1, float("nan"), float("inf"), -float("inf")):
                with self.subTest(outside=outside, ms=ms):
                    b = self.grid_book()
                    with patch(
                        LiveU.__module__ + ".predict", return_value=(ms, outside)
                    ):
                        b.submit("req", "root", "agent", 5, 0)
                        b.accepted(work_step(p=5, d=8))
                        self.assertEqual(b.invalid["root"], 1)
                        self.assertIsNone(b.measured("root"))
                        self.assertIsNone(b.score("req")["score"])
                        self.assertEqual(b.extrapolated_cost_evaluations, 0)

    def test_lookahead_chunk_shape(self):
        b = self.book()
        seen = []
        b.cost = lambda *args: seen.append(args) or 1.0
        self.assertEqual(b.next_five(20000), 5.0)
        self.assertEqual(
            seen,
            [
                ("prefill", 8192, 0, 8192),
                ("prefill", 8192, 8192, 16384),
                ("prefill", 3616, 16384, 20000),
                ("decode", 1, 20000, 20001),
                ("decode", 1, 20001, 20002),
            ],
        )


def write_rows(path, rows):
    path.write_text(
        "".join(
            json.dumps(dict(row, sequence=i)) + "\n" for i, row in enumerate(rows, 1)
        )
    )


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def send_record(
    time=100,
    request_id="req",
    program_id="native/session",
    *,
    root="root",
    agent="agent",
):
    return dict(
        time=time,
        event="send",
        request_id=request_id,
        program_id=program_id,
        prompt_tokens=8,
        root=root,
        agent=agent,
    )


@pytest.fixture
def config(tmp_path):
    calibration = tmp_path / "calibration.json"
    calibration.write_text(
        json.dumps(
            dict(
                schema_version=1,
                timing_metric=METRIC,
                contract={"model": "cpu"},
                max_prefill=8,
                prefill=dict(P=[1, 8], D=[0, 8], wall_ms=[[10, 10], [10, 10]]),
                decode=dict(grid=[dict(K=1, wall_ms=10), dict(K=16, wall_ms=10)]),
            )
        )
    )
    cfg = dict(
        ingress=str(tmp_path / "ingress.jsonl"),
        ledgers=[str(tmp_path / "worker.jsonl")],
        scores=str(tmp_path / "output" / "scores.json"),
        capacity=1,
        calibration=str(calibration),
        runtime_contract={"model": "cpu"},
        audit_dir=str(tmp_path / "audit"),
    )
    path = tmp_path / "config.json"
    path.write_text(json.dumps(cfg))
    return path, cfg


@pytest.fixture
def records(config):
    _, cfg = config
    ingress = [
        send_record(),
        dict(time=100.4, event="end", request_id="req"),
        dict(time=100.5, event="footer"),
    ]
    step = work_step(p=8, client_request_id="req", scheduled_at=100.1, end=100.2)
    worker = [
        dict(time=100.1, event="scheduled", steps=[dict(step)]),
        dict(time=100.2, event="accepted_output", steps=[dict(step)]),
        # Closed workers must not hold back later ingress events.
        dict(time=100.3, event="footer"),
    ]
    write_rows(Path(cfg["ingress"]), ingress)
    write_rows(Path(cfg["ledgers"][0]), worker)
    return ingress, worker


@contextmanager
def service(config_path):
    with subprocess.Popen(
        [sys.executable, "-m", "dynamo.duru_router.scoring", str(config_path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    ) as process:
        try:
            yield process
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.communicate(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.communicate(timeout=5)


def wait_until(process, condition):
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        if process.poll() is not None:
            _, error = process.communicate(timeout=1)
            pytest.fail(f"Scoring exited early: {error}")
        result = condition()
        if result:
            return result
        time.sleep(0.01)
    pytest.fail("Timed out waiting for scoring service")


def audit_rows(cfg):
    return read_rows(Path(cfg["audit_dir"]) / "duru-control.jsonl")


def run_once(config):
    path, cfg = config
    stop = threading.Event()
    stop.set()
    run(path, stop)
    return audit_rows(cfg)[-1]


@pytest.mark.post_merge
@pytest.mark.integration
@pytest.mark.gpu_0
@pytest.mark.timeout(30)
class TestScoringService:
    @pytest.mark.parametrize("sig", [signal.SIGINT, signal.SIGTERM])
    def test_process_group_signal_flushes_and_restart_replays(
        self, config, records, sig
    ):
        path, cfg = config
        score = Path(cfg["scores"])
        run_ids = set()
        for _ in range(2):
            with service(path) as process:
                wait_until(process, score.exists)
                snapshot = json.loads(score.read_text())
                assert snapshot["accepted_steps"] == 1
                assert snapshot["roots"]["root"]["U"] == pytest.approx(40)
                assert not snapshot["requests"]
                assert set(snapshot["programs"]) == {"native/session"}
                assert (
                    snapshot["programs"]["native/session"]["kind"]
                    == "idle_measured_only"
                )
                run_ids.add(snapshot["run_id"])
                os.killpg(process.pid, sig)
                _, error = process.communicate(timeout=5)
                assert process.returncode == 0, error
            assert not score.exists()
            footer = audit_rows(cfg)[-1]
            assert footer["event"] == "footer"
            assert footer["pending_events"] == 0
            assert footer["complete"]
            assert footer["last_snapshot"]["roots"] == snapshot["roots"]
        assert len(run_ids) == 2
        assert {row["run_id"] for row in audit_rows(cfg)} == run_ids

    def test_shutdown_while_inputs_missing_removes_old_snapshot(self, config):
        path, cfg = config
        score = Path(cfg["scores"])
        score.parent.mkdir()
        score.write_text("old snapshot")
        with service(path) as process:
            wait_until(
                process, lambda: not score.exists() and Path(cfg["audit_dir"]).exists()
            )
            os.killpg(process.pid, signal.SIGTERM)
            _, error = process.communicate(timeout=5)
            assert process.returncode == 0, error
        footer = audit_rows(cfg)[-1]
        assert footer["last_snapshot"] is None
        assert not footer["complete"]
        assert not score.exists()

    @pytest.mark.parametrize("closed", [False, True])
    def test_final_pass_respects_input_watermarks(self, config, records, closed):
        if not closed:
            Path(config[1]["ledgers"][0]).write_text("")
        footer = run_once(config)
        assert footer["complete"] is closed
        assert footer["inputs_closed"] is closed
        assert footer["pending_events"] == (0 if closed else 2)
        if closed:
            assert footer["last_snapshot"]["accepted_steps"] == 1

    def test_invalid_worker_configuration_invalidates_old_scores(self, config):
        path, cfg = config
        score = Path(cfg["scores"])
        score.parent.mkdir()
        score.write_text("old snapshot")
        cfg["ledgers"] = []
        path.write_text(json.dumps(cfg))
        with pytest.raises(ValueError, match="worker ledger"):
            run(path)
        assert not score.exists()

    def test_partial_footer_is_incomplete_until_newline_arrives(self, config, records):
        worker = Path(config[1]["ledgers"][0])
        worker.write_bytes(worker.read_bytes().rstrip(b"\n"))
        assert not run_once(config)["complete"]
        with worker.open("ab") as stream:
            stream.write(b"\n")
        footer = run_once(config)
        assert footer["complete"]
        assert footer["last_snapshot"]["accepted_steps"] == 1

    def test_journal_gap_never_reports_complete_or_leaves_scores(self, config, records):
        cfg = config[1]
        Path(cfg["ledgers"][0]).write_text(
            json.dumps(dict(sequence=2, event="footer", time=101)) + "\n"
        )
        with pytest.raises(AssertionError, match="Journal gap"):
            run_once(config)
        assert not audit_rows(cfg)[-1]["complete"]
        assert not Path(cfg["scores"]).exists()

    @pytest.mark.parametrize(
        "case, message",
        [
            ("missing_program", "program_id"),
            ("changed_program", "program identity changed"),
            ("reuse_active", "Request ID reused"),
            ("reuse_finished", "Request ID reused"),
        ],
    )
    def test_invalid_ingress_identity_invalidates_scores(
        self, config, records, case, message
    ):
        cfg = config[1]
        ingress, _ = records
        if case == "missing_program":
            ingress[0].pop("program_id")
        elif case == "changed_program":
            ingress.insert(
                -1, send_record(100.45, "next", root="other-root", agent="other-agent")
            )
        else:
            arrival = 100.15 if case == "reuse_active" else 100.45
            ingress.append(send_record(arrival))
            ingress.sort(key=lambda row: row["time"])
        write_rows(Path(cfg["ingress"]), ingress)
        with pytest.raises(ValueError, match=message):
            run_once(config)
        assert not audit_rows(cfg)[-1]["complete"]
        assert not Path(cfg["scores"]).exists()

    @pytest.mark.parametrize("event_index", [0, 1], ids=["scheduled", "accepted"])
    @pytest.mark.parametrize("field", ["client_request_id", "root", "agent"])
    def test_worker_identity_must_join_ingress(
        self, config, records, event_index, field
    ):
        cfg = config[1]
        _, worker = records
        worker[event_index]["steps"][0][field] = "unknown"
        write_rows(Path(cfg["ledgers"][0]), worker)
        message = (
            "Unjoined work"
            if field == "client_request_id"
            else "Worker request identity mismatch"
        )
        with pytest.raises(ValueError, match=message):
            run_once(config)
        assert not audit_rows(cfg)[-1]["complete"]
        assert not Path(cfg["scores"]).exists()

    def test_explicit_program_ids_preserve_idle_and_overlapping_agent_requests(
        self, config, records
    ):
        cfg = config[1]
        ingress, _ = records
        ingress[-1:-1] = [
            send_record(100.45, "next", "other-session"),
            send_record(100.46, "parallel", "parallel-session"),
        ]
        write_rows(Path(cfg["ingress"]), ingress)
        snapshot = run_once(config)["last_snapshot"]
        assert set(snapshot["programs"]) == {
            "native/session",
            "other-session",
            "parallel-session",
        }
        assert snapshot["programs"]["native/session"]["kind"] == "idle_measured_only"
        for request, program in [
            ("next", "other-session"),
            ("parallel", "parallel-session"),
        ]:
            assert snapshot["programs"][program]["request_id"] == request
            assert snapshot["requests"][request]["program_id"] == program
        assert snapshot["roots"]["root"]["active_agents"] == 1
        assert snapshot["roots"]["root"]["E"] == pytest.approx(0.45)

    def test_accepted_work_after_ingress_end_retains_request_identity(
        self, config, records
    ):
        ingress, _ = records
        ingress[1]["time"] = 100.15
        write_rows(Path(config[1]["ingress"]), ingress)
        footer = run_once(config)
        assert footer["complete"]
        snapshot = footer["last_snapshot"]
        assert snapshot["accepted_steps"] == 1
        assert not snapshot["requests"]
        assert snapshot["roots"]["root"]["E"] == pytest.approx(0.15)
        assert snapshot["roots"]["root"]["W"] == pytest.approx(0.01)
        assert snapshot["programs"]["native/session"]["U"] == pytest.approx(15)

    @pytest.mark.parametrize("bad_id", [None, "", 7])
    def test_ingress_requires_program_id_and_rejects_second_producer(
        self, tmp_path, bad_id
    ):
        ingress = Ingress(tmp_path)
        fields = send_record()
        fields.pop("event")
        fields.pop("time")
        try:
            with pytest.raises(ValueError, match="program_id"):
                ingress.emit("send", 1, **dict(fields, program_id=bad_id))
            ingress.emit("send", 2, **fields, unrelated="ignored")
        finally:
            ingress.close()
        journal = tmp_path / "duru-ingress.jsonl"
        original = journal.read_bytes()
        rows = read_rows(journal)
        assert rows[0]["sequence"] == 1
        assert rows[0]["program_id"] == "native/session"
        assert "unrelated" not in rows[0]
        assert [row["event"] for row in rows] == ["send", "footer"]
        with pytest.raises(FileExistsError):
            Ingress(tmp_path)
        assert journal.read_bytes() == original

    def test_ingress_closes_transport_when_footer_emission_fails(self, tmp_path):
        ingress = Ingress(tmp_path)
        with patch.object(
            ingress.journal, "close", wraps=ingress.journal.close
        ) as close:
            with patch.object(
                ingress, "emit", side_effect=RuntimeError("footer rejected")
            ):
                with pytest.raises(RuntimeError, match="footer rejected"):
                    ingress.close()
            close.assert_called_once()
