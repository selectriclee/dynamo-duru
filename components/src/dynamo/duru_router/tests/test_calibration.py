"""CPU regressions for graph-aware calibration; no accuracy-validation campaign."""

import json
import tempfile
import unittest
from functools import partial
from pathlib import Path

import pytest

from dynamo.duru_router.calibration.measure import load, measure
from dynamo.duru_router.calibration.plan import METRIC, design
from dynamo.duru_router.calibration.predict import predict, predict_prefill_table
from dynamo.duru_router.scoring import LiveU

pytestmark = [pytest.mark.pre_merge, pytest.mark.unit, pytest.mark.gpu_0]


class CalibrationTest(unittest.TestCase):
    def test_graph_neighbors_and_eager_transition(self):
        plan = design(
            32768,
            64,
            max_prefill=1024,
            capture_sizes=[64, 128, 256],
            prefill_graphs=True,
        )
        for boundary in (64, 128, 256):
            self.assertTrue({boundary - 1, boundary, boundary + 1} <= set(plan["P"]))
        self.assertEqual(plan["P_discontinuities"], [64, 128, 256])
        self.assertTrue(
            all(p + d + 1 <= plan["cap"] for p in plan["P"] for d in plan["D"])
        )
        self.assertTrue(all(d % 64 == 0 for d in plan["D"]))
        self.assertTrue(all(case["split"] == "train" for case in plan["cases"]))
        self.assertNotIn("validation", plan)

    def test_decode_only_graphs_do_not_create_prefill_boundaries(self):
        graph = design(
            32768, 64, max_prefill=1024, capture_sizes=[73, 149], prefill_graphs=True
        )
        decode_only = design(32768, 64, max_prefill=1024, capture_sizes=[73, 149])
        self.assertEqual(decode_only["P_discontinuities"], [])
        self.assertEqual(graph["K"], decode_only["K"])
        with self.assertRaises(ValueError):
            design(32768, 64, prefill_graphs=True)

    def test_live_predictor_keeps_graph_jump(self):
        table = dict(
            P=[1, 128, 512],
            D=[0, 16],
            P_discontinuities=[256],
            wall_ms=[[1, 1], [2, 2], [100, 100]],
        )
        for p, expected in ((256, 2), (257, 100)):
            self.assertEqual(predict_prefill_table(table, p, 0), expected)
            self.assertEqual(
                predict(table, dict(phase="prefill", p=p, d=0)), (expected, False)
            )

    def test_live_outer_edge_extrapolation_preserved(self):
        table = dict(P=[1, 2], D=[0, 1], wall_ms=[[1, 2], [2, 3]])
        value, outside = predict(table, dict(phase="prefill", p=3, d=2))
        self.assertEqual(value, 5)
        self.assertTrue(outside)
        with self.assertRaises(ValueError):
            predict_prefill_table(table, 3, 2)

    def test_lookahead_uses_calibrated_chunk_budget(self):
        book = LiveU(1, {"max_prefill": 4})
        coordinates = []

        def cost(phase, p, d, k):
            coordinates.append((phase, p, d, k))
            return 1.0

        book.cost = cost
        self.assertEqual(book.next_five(10), 5.0)
        self.assertEqual([row[1] for row in coordinates], [4, 4, 2, 1, 1])

    def test_measure_resume_and_load_without_accuracy_gate(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            ledger = root / "accepted.jsonl"
            ledger.touch()
            plan = dict(
                timing_metric=METRIC,
                cap=1024,
                max_prefill=4,
                P=[1, 4],
                D=[0, 16],
                K=[32, 128],
                P_discontinuities=[],
                repeats=1,
                warmups=1,
                decode_tokens=16,
                decode_warmup_tokens=8,
                cases=[
                    *[dict(phase="prefill", P=p, D=d) for d in (0, 16) for p in (1, 4)],
                    *[dict(phase="decode", K=k) for k in (32, 128)],
                ],
            )
            contract = dict(timing_metric=METRIC, model="cpu-fixture")
            primed = set()
            clock = 0.0

            def submit(prompt, count, rid, salt):
                nonlocal clock
                cached = (
                    int(salt.rsplit("D", 1)[1])
                    if "-D" in salt and salt in primed
                    else 0
                )
                primed.add(salt)
                steps = []
                for i in range(count):
                    p, d = (
                        (len(prompt) - cached, cached)
                        if i == 0
                        else (1, len(prompt) + i - 1)
                    )
                    begin = clock
                    clock += 0.001 * (2 + p + 0.001 * d) if i == 0 else 0.001
                    steps.append(
                        dict(
                            client_request_id=rid,
                            accepted=True,
                            p=p,
                            d=d,
                            k=p + d,
                            generated_tokens=1,
                            scheduled_at=begin,
                            end=clock,
                        )
                    )
                with ledger.open("a") as output:
                    output.write(
                        json.dumps(dict(event="accepted_output", steps=steps)) + "\n"
                    )
                return dict(output_tokens=count, cached_tokens=cached)

            run = partial(
                measure,
                root / "cal",
                contract,
                submit=submit,
                drain=lambda: None,
                ledger=ledger,
                reset=primed.clear,
            )
            result = run(plan)
            self.assertNotIn("quality_passed", result)
            self.assertNotIn("validation", result)
            self.assertAlmostEqual(result["prefill"]["wall_ms"][0][0], 3)
            loaded = load(root / "cal/calibration.json", contract)
            self.assertEqual(loaded, result)
            legacy_path = root / "legacy.json"
            legacy_path.write_text(json.dumps(dict(result, measurement_only=True)))
            legacy = load(legacy_path, contract)
            self.assertEqual(legacy["prefill"], loaded["prefill"])
            self.assertEqual(legacy["decode"], loaded["decode"])
            self.assertEqual(
                run(plan, resume=True),
                result,
            )
            with self.assertRaises(ValueError):
                load(root / "cal/calibration.json", dict(contract, model="other"))
            with self.assertRaises(ValueError):
                run(dict(plan, repeats=2), resume=True)
            legacy_plan = dict(
                plan,
                measurement_only=True,
                point_selection="graph_neighbors_and_geometric_context_v1",
            )
            legacy_run = partial(measure, root / "legacy-cal", contract, **run.keywords)
            legacy_result = legacy_run(legacy_plan)
            self.assertEqual(legacy_run(legacy_plan, resume=True), legacy_result)
            result["prefill"]["wall_ms"][0][0] = 0
            (root / "bad.json").write_text(json.dumps(result))
            with self.assertRaises(ValueError):
                load(root / "bad.json", contract)


if __name__ == "__main__":
    unittest.main()
