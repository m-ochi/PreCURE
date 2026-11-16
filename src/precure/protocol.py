"""Publication-safe equal-candidate search and independent endpoint evaluation.

This implements the paper's experimental design on the public input contract;
it does not redistribute historical private prompts, inputs or provider outputs.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
import json
import math
from statistics import mean, stdev
import time

from .backend import Backend
from .engine import (evaluate_expression, stable_seed, _optimizer_context,
                     ipc_history_window)
from .fidelity import validate_expression
from .models import Campaign, Evaluation, Expression, Persona


PROTOCOL_VERSION = "public-equal-candidates-v1"
METHODS = ("sequential_cps", "ipc_cps", "gepa_cps", "prefpo_cps")


def canonical_expression(expression: Expression) -> str:
    # Match the paper runner: sorted field JSON, preserving internal whitespace.
    return json.dumps(expression.to_dict(), ensure_ascii=False, sort_keys=True)


@dataclass(frozen=True)
class ProtocolConfig:
    candidate_budget: int = 45
    proposal_ceiling: int = 360
    screening_samples: int = 5
    confirmation_samples: int = 18
    final_samples: int = 18
    top_revisions: int = 3
    selection_kappa: float = 1.699
    history_length: int = 5
    reflection_minibatch_size: int = 3

    def __post_init__(self):
        if min(self.candidate_budget, self.screening_samples, self.final_samples,
               self.top_revisions, self.history_length, self.reflection_minibatch_size) < 1:
            raise ValueError("protocol budgets must be positive")
        if self.confirmation_samples < 2 or self.proposal_ceiling < 0:
            raise ValueError("confirmation needs >=2 samples; proposal ceiling must be nonnegative")
        if not math.isfinite(self.selection_kappa) or self.selection_kappa < 0:
            raise ValueError("selection kappa must be finite and nonnegative")


def paired_adjusted_gain(candidate: Evaluation, original: Evaluation, kappa: float) -> dict:
    if (not candidate.fidelity.passed or not original.fidelity.passed
            or not candidate.sample_seeds or candidate.sample_seeds != original.sample_seeds
            or len(candidate.sample_cps) != len(candidate.sample_seeds)
            or len(original.sample_cps) != len(original.sample_seeds)):
        raise ValueError("confirmation requires valid evaluations with identical sample keys")
    differences = [a - b for a, b in zip(candidate.sample_cps, original.sample_cps)]
    if len(differences) < 2 or len(differences) != len(original.sample_seeds):
        raise ValueError("paired confirmation requires at least two scored samples")
    gain = mean(differences)
    standard_error = stdev(differences) / math.sqrt(len(differences))
    return {"mean_gain": gain, "standard_error": standard_error,
            "adjusted_gain": gain - kappa * standard_error}


class CandidateArchive:
    def __init__(self, backend, campaign, personas, config, seed):
        self.backend, self.campaign, self.personas = backend, campaign, personas
        self.config, self.seed = config, seed
        self.rows: list[Evaluation] = []
        self.by_key: dict[str, Evaluation] = {}
        self.attempts: list[dict] = []
        self.trace: list[dict] = []
        original = self.screen(campaign.expression.creative_hint)
        if original is None:
            raise ValueError("Original must pass the hard-fidelity gate")

    @property
    def complete(self):
        return len(self.rows) == self.config.candidate_budget

    def screen(self, hint: str, *, proposal: bool = False):
        event = {"attempt": len(self.attempts) + 1, "creative_hint": hint}
        if proposal:
            self.attempts.append(event)
        if not hint.strip():
            event.update(status="invalid", violations=["empty creative_hint"])
            return None
        expression = self.campaign.expression.with_creative_hint(hint)
        key = canonical_expression(expression)
        if key in self.by_key:
            event["status"] = "duplicate"
            return self.by_key[key]
        gate = validate_expression(self.campaign, expression, creative_only=True)
        if not gate.passed:
            event.update(status="invalid", violations=list(gate.violations))
            return None
        if self.complete:
            raise RuntimeError("candidate budget already reached")
        row = evaluate_expression(
            self.backend, self.campaign, expression, self.personas,
            samples_per_persona=self.config.screening_samples,
            seed=stable_seed(self.seed, self.campaign.campaign_id, "screen"),
        )
        if not row.fidelity.passed or row.cps is None:
            event.update(status="invalid", violations=list(row.fidelity.violations))
            return None
        self.rows.append(row)
        self.by_key[key] = row
        event.update(status="unique_valid", candidate_index=len(self.rows) - 1)
        return row

    def feedback(self):
        return (
            "\nAlready evaluated invitation archive (do not repeat these):\n"
            + json.dumps([row.expression.creative_hint for row in self.rows], ensure_ascii=False)
            + "\nProduce a materially distinct invitation design. Duplicate and invalid "
              "proposals do not count toward the candidate budget.\nRecent rejected proposals:\n"
            + json.dumps(self.attempts[-5:], ensure_ascii=False)
        )


def _search(archive: CandidateArchive, method: str):
    if archive.complete:
        return
    backend, campaign = archive.backend, archive.campaign
    config = archive.config
    if method == "gepa_cps":
        _gepa_search(archive)
        return
    current = archive.rows[0]
    history = []
    for attempt in range(config.proposal_ceiling):
        if archive.complete:
            break
        context = _optimizer_context(campaign, tuple(archive.personas)) + archive.feedback()
        purpose = method + "_proposal"
        proposal_seed = stable_seed(archive.seed, campaign.campaign_id, method, attempt, "proposal")
        trace = {"attempt": attempt + 1, "method": method}
        if method == "sequential_cps":
            prompt = context + "\nRevise the current invitation using its replies and CPS:\n"
            prompt += json.dumps(current.to_dict(), ensure_ascii=False)
        elif method == "ipc_cps":
            analysis = backend.optimizer_text(
                campaign, "Analyze reply failures, four-axis scores, burden and fidelity:\n"
                + json.dumps(current.to_dict(), ensure_ascii=False),
                purpose="ipc_analysis", iteration=attempt,
                seed=stable_seed(proposal_seed, "analysis"),
            )
            history.append({"score": current.cps, "expression": current.expression.creative_hint,
                            "analysis": analysis})
            trace["analysis"] = analysis
            prompt = context + "\nUse this error analysis and candidate history:\n"
            prompt += json.dumps(ipc_history_window(history, config.history_length), ensure_ascii=False)
        else:
            # Pairwise pool: contrast the best and a rotating distinct pool member,
            # then revise the nonpreferred member. Original wins score ties.
            ranked = sorted(archive.rows, key=lambda row: row.cps, reverse=True)
            preferred = ranked[0]
            nonpreferred = ranked[1 + attempt % (len(ranked) - 1)] if len(ranked) > 1 else preferred
            prompt = context + "\nCompare CPS components and replies; revise the nonpreferred invitation:\n"
            prompt += json.dumps({"preferred": preferred.to_dict(),
                                  "nonpreferred": nonpreferred.to_dict()}, ensure_ascii=False)
        trace["prompt"] = prompt
        hint = backend.optimizer_text(campaign, prompt, purpose=purpose,
                                      iteration=attempt, seed=proposal_seed).strip()
        trace["proposed_creative_hint"] = hint
        archive.trace.append(trace)
        row = archive.screen(hint, proposal=True)
        if row is not None:
            current = row


def _gepa_search(archive):
    try:
        from gepa.optimize_anything import optimize_anything, GEPAConfig, EngineConfig, ReflectionConfig
    except ImportError as exc:
        raise RuntimeError("GEPA-CPS requires pip install -e '.[gepa]'") from exc
    if archive.config.proposal_ceiling == 0:
        return
    config, campaign = archive.config, archive.campaign
    # Every proposal is fully screened under the shared seeds. GEPA still uses
    # its upstream Pareto parent selection and reflective minibatch mechanism.
    examples = list(range(config.screening_samples))

    def evaluator(candidate, example):
        row = archive.screen(candidate)
        if row is None:
            return 0.0, {"Feedback": "Invalid invitation; preserve fixed facts."}
        return (row.sample_cps[example] + .12) / 1.15, row.to_dict()

    def reflect(prompt):
        text = prompt if isinstance(prompt, str) else json.dumps(prompt, ensure_ascii=False)
        text += "\n" + _optimizer_context(campaign, tuple(archive.personas)) + archive.feedback()
        attempt = len(archive.attempts)
        hint = archive.backend.optimizer_text(
            campaign, text, purpose="gepa_reflection", iteration=attempt,
            seed=stable_seed(archive.seed, campaign.campaign_id, "gepa_cps", attempt, "proposal"),
        ).strip()
        archive.trace.append({"attempt": attempt + 1, "reflection_prompt": text,
                              "proposed_creative_hint": hint})
        archive.screen(hint, proposal=True)
        return "```\n" + hint + "\n```"

    optimize_anything(
        seed_candidate=campaign.expression.creative_hint, evaluator=evaluator,
        dataset=examples, objective="Maximize source-faithful Campaign Proposal Score.",
        background=_optimizer_context(campaign, tuple(archive.personas)),
        config=GEPAConfig(
            engine=EngineConfig(
                seed=stable_seed(archive.seed, campaign.campaign_id, "gepa"),
                max_candidate_proposals=config.proposal_ceiling,
                max_metric_calls=len(examples) * (2 * config.proposal_ceiling + 1),
                candidate_selection_strategy="pareto", parallel=False,
                use_cloudpickle=False, cache_evaluation=True),
            reflection=ReflectionConfig(
                reflection_lm=reflect,
                reflection_minibatch_size=min(config.reflection_minibatch_size, len(examples)),
                skip_perfect_score=False), merge=None,
            stop_callbacks=lambda state: archive.complete or len(archive.attempts) >= config.proposal_ceiling),
    )


def run_protocol(backend: Backend, campaign: Campaign, personas: list[Persona], *,
                 method: str, seed: int, config: ProtocolConfig | None = None,
                 final_original_cache: dict | None = None) -> dict:
    config = config or ProtocolConfig()
    if method not in METHODS:
        raise ValueError(f"unknown equal-candidate method: {method}")
    if len(personas) != 1:
        raise ValueError("the paper's equal-candidate design requires one fixed persona per campaign")
    started = time.monotonic()
    before_calls = getattr(backend, "request_count", 0)
    before_cost = getattr(backend, "cost_usd", 0.0)
    before_prompt_tokens = getattr(backend, "prompt_tokens", 0)
    before_completion_tokens = getattr(backend, "completion_tokens", 0)
    archive = CandidateArchive(backend, campaign, personas, config, seed)
    _search(archive, method)
    result = {
        "protocol_version": PROTOCOL_VERSION, "config": asdict(config),
        "method": method, "campaign_id": campaign.campaign_id,
        "status": "complete" if archive.complete else "incomplete_candidate_budget",
        "candidate_count": len(archive.rows), "proposal_attempts": len(archive.attempts),
        "attempts": archive.attempts, "optimizer_trace": archive.trace,
        "screening": [row.to_dict() for row in archive.rows],
        "checkpoints": {str(count): max(row.cps for row in archive.rows[:count])
                        for count in (5, 10, 20, 45) if count <= len(archive.rows)},
        "stage_seeds": {stage: stable_seed(seed, campaign.campaign_id, stage)
                        for stage in ("screen", "confirmation", "final")},
        "search_usage": {"requests": getattr(backend, "request_count", 0) - before_calls,
                         "cost_usd": getattr(backend, "cost_usd", 0.0) - before_cost,
                         "prompt_tokens": getattr(backend, "prompt_tokens", 0) - before_prompt_tokens,
                         "completion_tokens": getattr(backend, "completion_tokens", 0) - before_completion_tokens,
                         "wall_seconds": time.monotonic() - started},
    }
    if archive.complete:
        # Screening alone determines the shortlist; final results cannot influence selection.
        indices = [0] + sorted(range(1, len(archive.rows)),
                               key=lambda i: archive.rows[i].cps, reverse=True)[:config.top_revisions]
        confirmed = [evaluate_expression(
            backend, campaign, archive.rows[index].expression, personas,
            samples_per_persona=config.confirmation_samples,
            seed=result["stage_seeds"]["confirmation"],
        ) for index in indices]
        selected, best_adjusted = 0, 0.0
        comparisons = []
        for position, row in enumerate(confirmed[1:], 1):
            comparison = paired_adjusted_gain(row, confirmed[0], config.selection_kappa)
            comparisons.append({"candidate_index": indices[position], **comparison})
            if comparison["adjusted_gain"] > best_adjusted:
                selected, best_adjusted = position, comparison["adjusted_gain"]
        selected_index = indices[selected]
        baseline_key = (campaign.campaign_id, result["stage_seeds"]["final"], config.final_samples,
                        personas[0].persona_id, personas[0].text, canonical_expression(campaign.expression))
        if final_original_cache is not None and baseline_key in final_original_cache:
            original_final = final_original_cache[baseline_key]
        else:
            original_final = evaluate_expression(
                backend, campaign, campaign.expression, personas,
                samples_per_persona=config.final_samples, seed=result["stage_seeds"]["final"],
            )
            if final_original_cache is not None:
                final_original_cache[baseline_key] = original_final
        selected_final = original_final if selected_index == 0 else evaluate_expression(
            backend, campaign, archive.rows[selected_index].expression, personas,
            samples_per_persona=config.final_samples, seed=result["stage_seeds"]["final"],
        )
        result.update(
            confirmation=[{"candidate_index": index, **row.to_dict()}
                          for index, row in zip(indices, confirmed)],
            confirmation_comparisons=comparisons, selected_index=selected_index,
            selected_expression=archive.rows[selected_index].expression.to_dict(),
            final_original=original_final.to_dict(), final_selected=selected_final.to_dict(),
            delta_cps=selected_final.cps - original_final.cps,
        )
    result["total_usage"] = {"requests": getattr(backend, "request_count", 0) - before_calls,
                             "cost_usd": getattr(backend, "cost_usd", 0.0) - before_cost,
                             "prompt_tokens": getattr(backend, "prompt_tokens", 0) - before_prompt_tokens,
                             "completion_tokens": getattr(backend, "completion_tokens", 0) - before_completion_tokens,
                             "wall_seconds": time.monotonic() - started}
    return result
