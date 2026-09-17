from __future__ import annotations

from typing import Protocol

from .models import AxisScores, Campaign, Evaluation, Expression, Persona, RiskScores


class Backend(Protocol):
    name: str

    def score_persona(self, persona: Persona, *, seed: int) -> AxisScores: ...

    def score_campaign(
        self, campaign: Campaign, expression: Expression, persona: Persona, *, seed: int
    ) -> AxisScores: ...

    def generate_response(
        self,
        campaign: Campaign,
        expression: Expression,
        persona: Persona,
        target: AxisScores,
        *,
        seed: int,
    ) -> str: ...

    def score_response(
        self, campaign: Campaign, response: str, *, seed: int
    ) -> AxisScores: ...

    def score_risk(
        self, campaign: Campaign, expression: Expression, *, seed: int
    ) -> RiskScores: ...

    def check_restatement(
        self, campaign: Campaign, expression: Expression, *, seed: int
    ) -> bool: ...

    def refine(
        self,
        campaign: Campaign,
        current: Evaluation,
        *,
        iteration: int,
        seed: int,
        preferred: Evaluation | None = None,
    ) -> str: ...

    def optimizer_text(
        self, campaign: Campaign, prompt: str, *, purpose: str, iteration: int, seed: int
    ) -> str: ...
