from __future__ import annotations

from dataclasses import replace
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

from precure.cli import DEFAULT_DATA, main
from precure.engine import evaluate_expression, fixed_cps_search, prefpo_cps_search
from precure.io import load_campaigns, load_personas
from precure.mock_backend import MockBackend
from precure.models import AxisScores
from precure.protocol import ProtocolConfig, run_protocol, paired_adjusted_gain


class ProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.campaign = load_campaigns(DEFAULT_DATA / "task_c_facts.json")[0]
        cls.personas = load_personas(DEFAULT_DATA / "task_c_personas.jsonl")[cls.campaign.campaign_id][:1]

    def test_equal_budget_and_three_disjoint_paired_stages(self):
        methods = ["sequential_cps", "ipc_cps", "prefpo_cps"]
        if importlib.util.find_spec("gepa"):
            methods.append("gepa_cps")
        all_originals = []
        for method in methods:
            with self.subTest(method=method):
                result = run_protocol(MockBackend(), self.campaign, self.personas,
                                      method=method, seed=11,
                                      config=ProtocolConfig(candidate_budget=5, proposal_ceiling=20))
                self.assertEqual(result["status"], "complete")
                self.assertEqual(result["candidate_count"], 5)
                self.assertEqual(len({json.dumps(row["expression"], sort_keys=True)
                                      for row in result["screening"]}), 5)
                screen_keys = result["screening"][0]["sample_seeds"]
                for row in result["screening"]:
                    self.assertEqual(row["sample_seeds"], screen_keys)
                    self.assertEqual(len(row["responses"]), 5)
                self.assertEqual(len(result["confirmation"]), 4)
                confirmation_keys = result["confirmation"][0]["sample_seeds"]
                for row in result["confirmation"]:
                    self.assertEqual(row["sample_seeds"], confirmation_keys)
                    self.assertEqual(len(row["responses"]), 18)
                final = result["final_original"]
                self.assertEqual(final["sample_seeds"], result["final_selected"]["sample_seeds"])
                self.assertFalse(set(screen_keys) & set(confirmation_keys))
                self.assertFalse(set(final["sample_seeds"]) & set(confirmation_keys + screen_keys))
                all_originals.append(final)
        self.assertTrue(all(row == all_originals[0] for row in all_originals))

    def test_duplicates_invalid_and_empty_do_not_consume_budget(self):
        class RetryBackend(MockBackend):
            def optimizer_text(self, campaign, prompt, **kwargs):
                attempt = kwargs["iteration"]
                if attempt == 0:
                    return campaign.expression.creative_hint
                if attempt == 1:
                    return campaign.source_schema["forbidden_claims"][0]
                if attempt == 2:
                    return "  "
                return super().optimizer_text(campaign, prompt, **kwargs)
        result = run_protocol(RetryBackend(), self.campaign, self.personas,
                              method="sequential_cps", seed=12,
                              config=ProtocolConfig(candidate_budget=3, proposal_ceiling=8))
        self.assertEqual(result["status"], "complete")
        self.assertEqual(result["proposal_attempts"], 5)
        self.assertEqual([event["status"] for event in result["attempts"]],
                         ["duplicate", "invalid", "invalid", "unique_valid", "unique_valid"])
        self.assertIn("Recent rejected proposals", result["optimizer_trace"][-1]["prompt"])

    def test_incomplete_search_has_no_confirmation_or_endpoint(self):
        class DuplicateBackend(MockBackend):
            def optimizer_text(self, campaign, prompt, **kwargs):
                return campaign.expression.creative_hint
        result = run_protocol(DuplicateBackend(), self.campaign, self.personas,
                              method="sequential_cps", seed=12,
                              config=ProtocolConfig(candidate_budget=45, proposal_ceiling=3))
        self.assertEqual(result["status"], "incomplete_candidate_budget")
        self.assertEqual(result["proposal_attempts"], 3)
        self.assertEqual(result["candidate_count"], 1)
        self.assertNotIn("confirmation", result)
        self.assertNotIn("final_selected", result)

    def test_lcb_uses_paired_variance_and_rejects_mismatched_keys(self):
        original = evaluate_expression(MockBackend(), self.campaign, self.campaign.expression,
                                       self.personas, samples_per_persona=18, seed=13)
        # A positive mean with high uncertainty must not pass the LCB rule.
        original = replace(original, response_scores=(AxisScores(.5, .5, .5, .5),) * 18)
        candidate = replace(original, response_scores=tuple(
            AxisScores(value, value, value, value) for value in [.1, .92] * 9))
        comparison = paired_adjusted_gain(candidate, original, 1.699)
        self.assertGreater(comparison["mean_gain"], 0)
        self.assertLess(comparison["adjusted_gain"], 0)
        with self.assertRaises(ValueError):
            paired_adjusted_gain(replace(candidate, sample_seeds=tuple(range(18))), original, 1.699)

    def test_original_wins_confirmation_ties(self):
        class ConstantBackend(MockBackend):
            def score_response(self, *args, **kwargs):
                return AxisScores(.5, .5, .5, .5)
            def score_risk(self, *args, **kwargs):
                from precure.models import RiskScores
                return RiskScores(0, 0)
        result = run_protocol(ConstantBackend(), self.campaign, self.personas,
                              method="prefpo_cps", seed=14,
                              config=ProtocolConfig(candidate_budget=4))
        self.assertEqual(result["selected_index"], 0)
        self.assertEqual(result["delta_cps"], 0)

    def test_final_original_is_generated_once_across_methods(self):
        class RecordingBackend(MockBackend):
            def __init__(self):
                self.original_generation_seeds = []
            def generate_response(self, campaign, expression, *args, **kwargs):
                if expression == campaign.expression:
                    self.original_generation_seeds.append(kwargs["seed"])
                return super().generate_response(campaign, expression, *args, **kwargs)
        backend = RecordingBackend()
        cache = {}
        results = [run_protocol(backend, self.campaign, self.personas,
                                method=method, seed=15, config=ProtocolConfig(candidate_budget=2),
                                final_original_cache=cache)
                   for method in ("sequential_cps", "ipc_cps")]
        self.assertEqual(results[0]["final_original"], results[1]["final_original"])
        final_seeds = results[0]["final_original"]["sample_seeds"]
        self.assertTrue(all(backend.original_generation_seeds.count(seed) == 1 for seed in final_seeds))
        original = results[0]["screening"][0]
        self.assertEqual(len(original["responses"]), 5)
        self.assertEqual(len(set(original["responses"])), 1)
        self.assertEqual(len(original["response_scores"]), 5)

    def test_demo_searches_use_common_rollout_keys(self):
        for search in (fixed_cps_search, prefpo_cps_search):
            result = search(MockBackend(), self.campaign, self.personas, iterations=4)
            valid = [row for row in result.trajectory if row.fidelity.passed]
            self.assertTrue(all(row.sample_seeds == valid[0].sample_seeds for row in valid))

    def test_cli_preserves_provenance_and_reports_incomplete(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(SystemExit) as caught:
                main(["--protocol", "equal_candidates", "--method", "sequential_cps",
                      "--limit-campaigns", "1", "--proposal-ceiling", "0",
                      "--run-id", "test-only", "--output-dir", directory])
            self.assertEqual(caught.exception.code, 2)
            config = json.loads((Path(directory) / "run_config.json").read_text())
            self.assertEqual(config["status"], "incomplete_candidate_budget")
            self.assertEqual(config["run_id"], "test-only")
            self.assertEqual(config["parameters"]["screening_samples"], 5)
            self.assertIn("src/precure/protocol.py", config["source_sha256"])
            self.assertTrue((Path(directory) / "summary.csv").is_file())

    def test_campaign_inference_exact_test_and_holm(self):
        from precure.analysis import summarize_comparison
        results = []
        for campaign in ("one", "two", "three"):
            for method in ("sequential_cps", "ipc_cps"):
                results.append({"campaign_id": campaign, "method": method, "status": "complete",
                                "final_original": {"cps": .2, "q": .3, "risks": {"c_b": .1, "c_f": 0}},
                                "final_selected": {"cps": .3, "q": .4, "risks": {"c_b": .1, "c_f": 0}}})
        analysis = summarize_comparison(results, ("sequential_cps", "ipc_cps"), resamples=50)
        row = analysis["methods"]["sequential_cps"]
        self.assertEqual(row["p_exact_two_sided"], .25)
        self.assertEqual(row["p_holm"], .5)
        self.assertAlmostEqual(row["components"]["cps"]["ci95"][0], .1)
        with self.assertRaises(ValueError):
            summarize_comparison(results[:-1], ("sequential_cps", "ipc_cps"))
        results[0]["status"] = "incomplete_candidate_budget"
        with self.assertRaises(ValueError):
            summarize_comparison(results, ("sequential_cps", "ipc_cps"))

    def test_sequential_demo_retains_current_on_invalid_proposal(self):
        class InvalidRefiner(MockBackend):
            def refine(self, campaign, current, **kwargs):
                return campaign.source_schema["forbidden_claims"][0]
        result = fixed_cps_search(InvalidRefiner(), self.campaign, self.personas, iterations=3)
        self.assertTrue(all(row.expression == self.campaign.expression for row in result.trajectory))


if __name__ == "__main__":
    unittest.main()
