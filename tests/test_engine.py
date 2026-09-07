from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from precure.cli import DEFAULT_DATA, main
from precure.engine import fixed_cps_search, prefpo_cps_search
from precure.io import load_campaigns, load_personas
from precure.mock_backend import MockBackend


class EngineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.campaign = load_campaigns(DEFAULT_DATA / "task_c_facts.json")[0]
        cls.personas = load_personas(DEFAULT_DATA / "task_c_personas.jsonl")[
            cls.campaign.campaign_id
        ]

    def test_fixed_search_selects_valid_candidate(self) -> None:
        result = fixed_cps_search(
            MockBackend(), self.campaign, self.personas,
            iterations=5, samples_per_persona=2, seed=20260822,
        )
        self.assertTrue(result.selected.fidelity.passed)
        self.assertGreater(result.selected.cps, result.trajectory[0].cps)
        self.assertEqual(result.selected.expression.offer, self.campaign.expression.offer)

    def test_prefpo_keeps_original_in_pool(self) -> None:
        result = prefpo_cps_search(
            MockBackend(), self.campaign, self.personas,
            iterations=8, samples_per_persona=1, seed=20260822,
        )
        self.assertEqual(result.trajectory[0].expression, self.campaign.expression)
        self.assertTrue(result.selected.fidelity.passed)

    def test_cli_writes_auditable_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            main([
                "--backend", "mock", "--limit-campaigns", "1",
                "--iterations", "2", "--samples-per-persona", "1",
                "--output-dir", str(output),
            ])
            self.assertTrue((output / "summary.csv").is_file())
            self.assertTrue((output / "trajectories.jsonl").is_file())
            self.assertTrue((output / "run_config.json").is_file())


class OptimizerTests(unittest.TestCase):
    setUpClass = classmethod(EngineTests.setUpClass.__func__)
    def test_history_alternates_recent_and_best(self):
        from precure.engine import ipc_history_window
        history = [{"score": score} for score in [9, 8, 1, 2, 3, 0]]
        self.assertEqual(ipc_history_window(history[:5], 2), history[3:5])
        self.assertEqual(ipc_history_window(history, 2), [history[1], history[0]])

    def test_new_searches_are_deterministic_and_keep_facts(self):
        from precure.engine import ipc_cps_search, gepa_cps_search
        import importlib.util
        searches = [ipc_cps_search]
        if importlib.util.find_spec("gepa"):
            searches.append(gepa_cps_search)
        for search in searches:
            with self.subTest(method=search.__name__):
                kwargs = dict(iterations=4, samples_per_persona=2, seed=73)
                first = search(MockBackend(), self.campaign, self.personas, **kwargs)
                second = search(MockBackend(), self.campaign, self.personas, **kwargs)
                self.assertEqual(first, second)
                self.assertGreater(len(first.trajectory), 1)
                self.assertEqual(first.trajectory[0].expression, self.campaign.expression)
                self.assertTrue(first.selected.fidelity.passed)
                for row in first.trajectory:
                    self.assertEqual(row.expression.offer, self.campaign.expression.offer)
                    self.assertEqual(row.expression.period, self.campaign.expression.period)
                    self.assertEqual(row.expression.eligibility, self.campaign.expression.eligibility)
                self.assertEqual(first.selected.cps,
                                 max(row.cps for row in first.trajectory if row.cps is not None))

    def test_ipc_passes_analysis_and_history_to_proposal(self):
        from precure.engine import ipc_cps_search
        class RecordingBackend(MockBackend):
            prompts = []
            def optimizer_text(self, campaign, prompt, **kwargs):
                self.prompts.append((kwargs["purpose"], prompt))
                return super().optimizer_text(campaign, prompt, **kwargs)
        backend = RecordingBackend()
        ipc_cps_search(backend, self.campaign, self.personas, iterations=3)
        self.assertEqual(backend.prompts[0][0], "ipc_analysis")
        proposal = backend.prompts[1][1]
        self.assertIn("Reduce response burden", proposal)
        self.assertIn('"score":', proposal)
        self.assertIn("responses", backend.prompts[0][1])

    def test_invalid_proposals_cannot_win(self):
        from precure.engine import ipc_cps_search, gepa_cps_search
        import importlib.util
        class InvalidBackend(MockBackend):
            def optimizer_text(self, campaign, prompt, **kwargs):
                return campaign.source_schema["forbidden_claims"][0] + "を教えてください。"
        searches = [ipc_cps_search]
        if importlib.util.find_spec("gepa"):
            searches.append(gepa_cps_search)
        for search in searches:
            result = search(InvalidBackend(), self.campaign, self.personas, iterations=3)
            self.assertEqual(result.selected_iteration, 0)
            self.assertTrue(any(not row.fidelity.passed for row in result.trajectory))

    def test_new_cli_methods(self):
        import importlib.util
        import json
        methods = ["ipc_cps"]
        if importlib.util.find_spec("gepa"):
            methods.append("gepa_cps")
        for method in methods:
            with tempfile.TemporaryDirectory() as directory:
                main(["--method", method, "--limit-campaigns", "1", "--iterations", "3",
                      "--samples-per-persona", "1", "--output-dir", directory])
                config = json.loads((Path(directory)/"run_config.json").read_text())
                self.assertEqual(config["method"], method)
                self.assertIn("optimizer_prompt_version", config)


if __name__ == "__main__":
    unittest.main()
