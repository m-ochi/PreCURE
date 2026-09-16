from __future__ import annotations

import unittest

from precure.fidelity import validate_expression
from precure.models import Campaign, Expression


def campaign() -> Campaign:
    return Campaign(
        "c1",
        "Fictional Brand",
        "テスト飲料",
        "beverage",
        Expression("抽選で1枚", "9月1日まで", "#テスト で返信", "感想を教えてください"),
        {
            "required_terms": ["テスト飲料"],
            "required_hashtags": ["#テスト"],
            "forbidden_claims": ["効果を保証"],
        },
    )


class FidelityTests(unittest.TestCase):
    def test_creative_only_rewrite_passes(self) -> None:
        base = campaign()
        report = validate_expression(
            base, base.expression.with_creative_hint("飲みたい場面を教えてください")
        )
        self.assertTrue(report.passed)

    def test_protected_field_change_fails(self) -> None:
        base = campaign()
        candidate = Expression("全員に2枚", base.expression.period, base.expression.eligibility, "感想")
        report = validate_expression(base, candidate)
        self.assertFalse(report.passed)
        self.assertTrue(any("offer:numeric_fact_changed" in value for value in report.violations))

    def test_reworded_fact_preserving_offer_passes(self) -> None:
        base = campaign()
        candidate = Expression(
            "抽選で1枚プレゼント", base.expression.period, base.expression.eligibility, "感想"
        )
        report = validate_expression(base, candidate)
        self.assertTrue(report.passed)

    def test_entry_action_change_fails(self) -> None:
        base = campaign()
        candidate = Expression(
            base.expression.offer, base.expression.period, "#テスト をつけて投稿", base.expression.creative_hint
        )
        report = validate_expression(base, candidate)
        self.assertFalse(report.passed)
        self.assertTrue(any("entry_action_changed" in value for value in report.violations))

    def test_encoded_forbidden_claim_fails(self) -> None:
        base = campaign()
        candidate = base.expression.with_creative_hint("効果を保証する理由を書いてください")
        report = validate_expression(base, candidate)
        self.assertFalse(report.passed)
        self.assertTrue(any("forbidden_claim_added" in value for value in report.violations))


if __name__ == "__main__":
    unittest.main()
