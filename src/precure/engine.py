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
    optimizer_trace: tuple[dict, ...] = ()

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


OPTIMIZER_PROMPT_VERSION = "public-cps-gepa-ipc-v1"


def ipc_history_window(history: list[dict], history_length: int = 5) -> list[dict]:
    """Use recent history, switching to highest-CPS history every third step."""
    if history_length < 1:
        raise ValueError("history_length must be positive")
    if len(history) % 3:
        return history[-history_length:]
    return sorted(history, key=lambda row: row["score"])[-history_length:]


def _optimizer_context(campaign: Campaign) -> str:
    import json
    return (
        "Optimize only the Japanese creative_hint. Preserve offer, period and eligibility. "
        "Maximize CPS = Q - 0.10 C_B - 0.02 C_F; reduce burden without inventing facts. "
        "Do not add identifying persona details. Return only the new creative_hint.\n"
        + json.dumps({"original": campaign.expression.to_dict(),
                      "source_schema": campaign.source_schema}, ensure_ascii=False)
    )


def ipc_cps_search(
    backend: Backend, campaign: Campaign, personas: list[Persona], *,
    iterations: int = 15, samples_per_persona: int = 3, seed: int = 42,
    restarts: int = 1, history_length: int = 5, patience: int = 3,
) -> SearchResult:
    """IPC-CPS: error analysis, alternating history, restarts and cached screening."""
    import json
    if min(iterations, samples_per_persona, restarts, history_length) < 1 or patience < 0:
        raise ValueError("budgets must be positive and patience nonnegative")
    trajectory: list[Evaluation] = []
    cache: dict[str, Evaluation] = {}
    trace: list[dict] = []

    def screen(hint: str) -> Evaluation:
        if hint not in cache:
            row = evaluate_expression(
                backend, campaign, campaign.expression.with_creative_hint(hint), personas,
                samples_per_persona=samples_per_persona,
                seed=stable_seed(seed, campaign.campaign_id, "screen"),
            )
            cache[hint] = row
            trajectory.append(row)
        return cache[hint]

    original = screen(campaign.expression.creative_hint)
    for restart in range(restarts):
        current = original
        history: list[dict] = []
        best, stalled = float("-inf"), 0
        for iteration in range(iterations):
            analysis = backend.optimizer_text(
                campaign, "Analyze response failures, low-scoring axes, burden and factual drift.\n"
                + json.dumps(current.to_dict(), ensure_ascii=False),
                purpose="ipc_analysis", iteration=iteration,
                seed=stable_seed(seed, campaign.campaign_id, restart, iteration, "analysis"),
            )
            score = current.cps if current.cps is not None else float("-inf")
            trace.append({"restart": restart, "iteration": iteration,
                          "analysis": analysis, "current": current.to_dict()})
            history.append({"iteration": iteration, "score": score,
                            "expression": current.expression.creative_hint, "analysis": analysis})
            if score > best:
                best, stalled = score, 0
            else:
                stalled += 1
            if iteration + 1 == iterations or stalled > patience:
                break
            hint = backend.optimizer_text(
                campaign, _optimizer_context(campaign) + "\nHistory:\n"
                + json.dumps(ipc_history_window(history, history_length), ensure_ascii=False)
                + "\nCurrent error analysis:\n" + analysis,
                purpose="ipc_proposal", iteration=iteration,
                seed=stable_seed(seed, campaign.campaign_id, restart, iteration, "proposal"),
            ).strip()
            trace[-1]["proposed_creative_hint"] = hint
            if hint:
                proposed = screen(hint)
                if proposed.fidelity.passed:
                    current = proposed
    return SearchResult("ipc_cps", campaign.campaign_id, _select_best(trajectory),
                        tuple(trajectory), tuple(trace))


def gepa_cps_search(
    backend: Backend, campaign: Campaign, personas: list[Persona], *,
    iterations: int = 15, samples_per_persona: int = 3, seed: int = 42,
    restarts: int = 1, reflection_minibatch_size: int = 3,
) -> SearchResult:
    """GEPA 0.1.1 Pareto search over fixed independent persona/sample rollouts."""
    if min(iterations, samples_per_persona, restarts, reflection_minibatch_size) < 1 or not personas:
        raise ValueError("budgets and persona count must be positive")
    try:
        from gepa.optimize_anything import optimize_anything, GEPAConfig, EngineConfig, ReflectionConfig
    except ImportError as exc:
        raise RuntimeError("GEPA-CPS requires pip install -e '.[gepa]'") from exc
    examples = [(p, sample) for p in range(len(personas)) for sample in range(samples_per_persona)]
    cache: dict[tuple[str, int, int], Evaluation] = {}
    hints = [campaign.expression.creative_hint]
    trace: list[dict] = []

    def evaluator(candidate, example):
        hint = candidate.strip()
        if not hint:
            return 0.0, {"Feedback": "Empty creative_hint is ineligible."}
        if hint not in hints:
            hints.append(hint)
        p, sample = example
        key = (hint, p, sample)
        if key not in cache:
            cache[key] = evaluate_expression(
                backend, campaign, campaign.expression.with_creative_hint(hint), [personas[p]],
                samples_per_persona=1,
                seed=stable_seed(seed, campaign.campaign_id, "screen", p, sample),
            )
        row = cache[key]
        # Positive affine normalization preserves CPS ordering; invalid candidates get zero.
        score = (row.cps + 0.12) / 1.15 if row.cps is not None else 0.0
        return score, row.to_dict()

    for restart in range(restarts):
        calls = 0
        def reflect(prompt):
            nonlocal calls
            import json
            text = prompt if isinstance(prompt, str) else json.dumps(prompt, ensure_ascii=False)
            hint = backend.optimizer_text(
                campaign, text + "\n" + _optimizer_context(campaign),
                purpose="gepa_reflection", iteration=calls,
                seed=stable_seed(seed, campaign.campaign_id, restart, calls, "reflection"),
            )
            trace.append({"restart": restart, "iteration": calls,
                          "reflection_prompt": text, "proposed_creative_hint": hint})
            calls += 1
            return "```\n" + hint + "\n```"
        if iterations > 1:
            optimize_anything(
                seed_candidate=campaign.expression.creative_hint, evaluator=evaluator,
                dataset=examples, objective="Maximize source-faithful Campaign Proposal Score.",
                background=_optimizer_context(campaign),
                config=GEPAConfig(
                    engine=EngineConfig(seed=stable_seed(seed, restart),
                        max_candidate_proposals=iterations - 1,
                        max_metric_calls=len(examples) * (2 * iterations + 1),
                        candidate_selection_strategy="pareto", parallel=False,
                        use_cloudpickle=False, cache_evaluation=True),
                    reflection=ReflectionConfig(reflection_lm=reflect,
                        reflection_minibatch_size=min(reflection_minibatch_size, len(examples)),
                        skip_perfect_score=False), merge=None,
                ),
            )
    trajectory = []
    for hint in hints:
        for example in examples:
            evaluator(hint, example)
        rows = [cache[(hint, *example)] for example in examples]
        first = rows[0]
        if not first.fidelity.passed:
            trajectory.append(first)
            continue
        axes = first.axes.mean_with([row.axes for row in rows[1:]])
        risks = RiskScores(sum(row.risks.c_b for row in rows)/len(rows),
                           sum(row.risks.c_f for row in rows)/len(rows), first.risks.rationale)
        q = quality_term(axes)
        trajectory.append(Evaluation(first.expression, axes, risks, q,
            campaign_proposal_score(q, risks.c_b, risks.c_f), first.fidelity,
            tuple(response for row in rows for response in row.responses)))
    return SearchResult("gepa_cps", campaign.campaign_id, _select_best(trajectory),
                        tuple(trajectory), tuple(trace))
