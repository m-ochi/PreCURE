from __future__ import annotations

import unittest

from precure.models import AxisScores
from precure.objective import calibrated_target, campaign_proposal_score, quality_term


class ObjectiveTests(unittest.TestCase):
    def test_paper_weights(self) -> None:
        scores = AxisScores(0.6, 0.6, 0.6, 0.6)
        self.assertAlmostEqual(quality_term(scores), 0.618)
        self.assertAlmostEqual(campaign_proposal_score(0.6, 0.3, 0.2), 0.566)

    def test_task_b_calibration(self) -> None:
        mu = AxisScores(0.2, 0.2, 0.2, 0.2)
        nu = AxisScores(0.6, 0.6, 0.6, 0.6)
        target = calibrated_target(mu, nu, 0.25)
        self.assertEqual(target, AxisScores(0.5, 0.5, 0.5, 0.5))

    def test_calibration_rejects_invalid_eta(self) -> None:
        scores = AxisScores(0.2, 0.2, 0.2, 0.2)
        with self.assertRaises(ValueError):
            calibrated_target(scores, scores, 1.1)


if __name__ == "__main__":
    unittest.main()
