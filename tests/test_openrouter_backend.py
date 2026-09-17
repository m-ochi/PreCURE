from __future__ import annotations

import unittest

from precure.models import AxisScores, Campaign, Evaluation, Expression, FidelityReport, RiskScores
from precure.openrouter_backend import OpenRouterBackend


class CapturingBackend(OpenRouterBackend):
    """Captures the prompt instead of making a network call."""

    def __init__(self):
        super().__init__("test/model", api_key="x")
        self.last_prompt: str | None = None

    def _complete_json(self, prompt, schema_name, schema, *, seed, temperature, max_tokens):
        self.last_prompt = prompt
        return {"text": "書き直した文言"}


def _expression(hint: str = "元の呼びかけ") -> Expression:
    return Expression(offer="抽選で1名様", period="今月末まで", eligibility="フォローして返信", creative_hint=hint)


def _campaign(expression: Expression) -> Campaign:
    return Campaign("c1", "テストブランド", "テスト商品", "食品", expression, {})


class RefineFeedbackTests(unittest.TestCase):
    def test_refine_omits_rejection_block_when_gate_passed(self) -> None:
        backend = CapturingBackend()
        expr = _expression()
        passing = Evaluation(
            expr, AxisScores(0.5, 0.5, 0.5, 0.5), RiskScores(0.1, 0.1, "ok"),
            0.5, 0.5, FidelityReport(True, ()), (),
        )
        backend.refine(campaign=_campaign(expr), current=passing, iteration=0, seed=1)
        self.assertNotIn("却下されました", backend.last_prompt)

    def test_refine_includes_gate_violations_when_rejected(self) -> None:
        backend = CapturingBackend()
        expr = _expression("フォローしてハッシュタグを付けてください")
        rejected = Evaluation(
            expr, AxisScores(0.0, 0.0, 0.0, 0.0), RiskScores(1.0, 1.0, "rejected"),
            0.0, None,
            FidelityReport(False, ("risk:restates_procedural_content",)), (),
        )
        backend.refine(campaign=_campaign(expr), current=rejected, iteration=0, seed=1)
        self.assertIn("却下されました", backend.last_prompt)
        self.assertIn("risk:restates_procedural_content", backend.last_prompt)


if __name__ == "__main__":
    unittest.main()
