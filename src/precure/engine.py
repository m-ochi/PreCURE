from __future__ import annotations

import hashlib
from dataclasses import dataclass

from .backend import Backend
from .fidelity import validate_expression
from .models import AxisScores, Campaign, Evaluation, Expression, Persona, RiskScores
from .objective import calibrated_target, campaign_proposal_score, quality_term


def stable_seed(base: int, *parts: object) -> int:
    key = ":".join([str(base), *(str(part) for part in parts)])
    return base + int(hashlib.sha256(key.encode("utf-8")).hexdigest()[:8], 16)


@dataclass(frozen=True)
class SearchResult:
    method: str
    campaign_id: str
    selected_iteration: int
    trajectory: tuple[Evaluation, ...]

    @property
    def selected(self) -> Evaluation:
        return self.trajectory[self.selected_iteration]


def evaluate_expression(
    backend: Backend,
    campaign: Campaign,
    expression: Expression,
    personas: list[Persona],
    *,
    samples_per_persona: int,
    seed: int,
    eta: float = 0.25,
) -> Evaluation:
    fidelity = validate_expression(campaign, expression, creative_only=True)
    if not fidelity.passed:
        return Evaluation(
            expression,
            AxisScores(0.0, 0.0, 0.0, 0.0),
            RiskScores(1.0, 1.0, "hard-fidelity failure"),
            0.0,
            None,
            fidelity,
            (),
        )

    risks = backend.score_risk(
        campaign, expression, seed=stable_seed(seed, "risk")
    )
    responses: list[str] = []
    scores: list[AxisScores] = []
    for p_index, persona in enumerate(personas):
        mu = backend.score_persona(
            persona, seed=stable_seed(seed, p_index, "persona")
        )
        nu = backend.score_campaign(
            campaign,
            expression,
            persona,
            seed=stable_seed(seed, p_index, "campaign"),
        )
        target = calibrated_target(mu, nu, eta)
        for sample in range(samples_per_persona):
            item_seed = stable_seed(seed, p_index, sample)
            response = backend.generate_response(
                campaign, expression, persona, target, seed=item_seed
            )
            responses.append(response)
            scores.append(
                backend.score_response(campaign, response, seed=item_seed)
            )
    if not scores:
        raise ValueError("at least one persona and one sample are required")
    mean = AxisScores(**{
        axis: sum(getattr(row, axis) for row in scores) / len(scores)
        for axis in ("a", "d_cov", "d_spec", "d_rea")
    })
    q_value = quality_term(mean)
    cps = campaign_proposal_score(q_value, risks.c_b, risks.c_f)
    return Evaluation(
        expression, mean, risks, q_value, cps, fidelity, tuple(responses)
    )


def _select_best(trajectory: list[Evaluation]) -> int:
    valid = [
        (index, row) for index, row in enumerate(trajectory)
        if row.fidelity.passed and row.cps is not None
    ]
    if not valid:
        return 0
    # Strict greater-than preserves Original on ties.
    best_index, best = valid[0]
    for index, row in valid[1:]:
        if float(row.cps) > float(best.cps):
            best_index, best = index, row
    return best_index


def fixed_cps_search(
    backend: Backend,
    campaign: Campaign,
    personas: list[Persona],
    *,
    iterations: int = 5,
    samples_per_persona: int = 3,
    seed: int = 42,
) -> SearchResult:
    if iterations < 1:
        raise ValueError("iterations must be positive")
    trajectory: list[Evaluation] = []
    expression = campaign.expression
    for iteration in range(iterations):
        evaluated = evaluate_expression(
            backend,
            campaign,
            expression,
            personas,
            samples_per_persona=samples_per_persona,
            seed=stable_seed(seed, campaign.campaign_id, iteration),
        )
        trajectory.append(evaluated)
        if iteration + 1 < iterations:
            creative_hint = backend.refine(
                campaign,
                evaluated,
                iteration=iteration,
                seed=stable_seed(seed, campaign.campaign_id, iteration, "refine"),
            )
            expression = campaign.expression.with_creative_hint(creative_hint)
    return SearchResult(
        "fixed_cps", campaign.campaign_id, _select_best(trajectory), tuple(trajectory)
    )


def prefpo_cps_search(
    backend: Backend,
    campaign: Campaign,
    personas: list[Persona],
    *,
    iterations: int = 15,
    samples_per_persona: int = 3,
    seed: int = 42,
) -> SearchResult:
    """Small pairwise-pool PrefPO-CPS adaptation used by the public runner."""

    if iterations < 1:
        raise ValueError("iterations must be positive")
    original = evaluate_expression(
        backend,
        campaign,
        campaign.expression,
        personas,
        samples_per_persona=samples_per_persona,
        seed=stable_seed(seed, campaign.campaign_id, 0),
    )
    trajectory = [original]
    seen = {campaign.expression.creative_hint}
    for iteration in range(iterations - 1):
        preferred = trajectory[_select_best(trajectory)]
        nonpreferred = trajectory[-1]
        hint = backend.refine(
            campaign,
            nonpreferred,
            preferred=preferred,
            iteration=iteration,
            seed=stable_seed(seed, campaign.campaign_id, iteration, "prefpo"),
        )
        if hint in seen:
            continue
        seen.add(hint)
        candidate = campaign.expression.with_creative_hint(hint)
        trajectory.append(evaluate_expression(
            backend,
            campaign,
            candidate,
            personas,
            samples_per_persona=samples_per_persona,
            seed=stable_seed(seed, campaign.campaign_id, len(trajectory)),
        ))
    return SearchResult(
        "prefpo_cps", campaign.campaign_id, _select_best(trajectory), tuple(trajectory)
    )
