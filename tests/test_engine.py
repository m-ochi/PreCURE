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


if __name__ == "__main__":
    unittest.main()
