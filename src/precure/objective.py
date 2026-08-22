from __future__ import annotations

from .models import AxisScores


AXES = ("a", "d_cov", "d_spec", "d_rea")
DEFAULT_OMEGA = (0.505, 0.225, 0.150, 0.150)


def quality_term(
    scores: AxisScores,
    omega: tuple[float, float, float, float] = DEFAULT_OMEGA,
) -> float:
    return sum(weight * getattr(scores, axis) for axis, weight in zip(AXES, omega))


def campaign_proposal_score(
    q_value: float,
    c_b: float,
    c_f: float,
    lambda_burden: float = 0.10,
    lambda_fidelity: float = 0.02,
) -> float:
    return q_value - lambda_burden * c_b - lambda_fidelity * c_f


def calibrated_target(mu: AxisScores, nu: AxisScores, eta: float = 0.25) -> AxisScores:
    if not 0.0 <= eta <= 1.0:
        raise ValueError("eta must be in [0, 1]")
    return AxisScores(**{
        axis: round(eta * getattr(mu, axis) + (1.0 - eta) * getattr(nu, axis), 4)
        for axis in AXES
    })
