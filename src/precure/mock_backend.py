from __future__ import annotations

import hashlib
import re

from .models import AxisScores, Campaign, Evaluation, Expression, Persona, RiskScores


def _unit(key: str) -> float:
    return int(hashlib.sha256(key.encode("utf-8")).hexdigest()[:8], 16) / 0xFFFFFFFF


class MockBackend:
    """Deterministic offline backend for testing the complete control flow.

    It is deliberately transparent and is not a substitute for the LLM
    evaluations reported in the paper.
    """

    name = "mock-v1"

    def score_persona(self, persona: Persona, *, seed: int) -> AxisScores:
        base = 0.30 + 0.20 * _unit(f"{seed}:{persona.persona_id}")
        return AxisScores(base + 0.08, base, max(0.0, base - 0.04), base + 0.02)

    def score_campaign(
        self, campaign: Campaign, expression: Expression, persona: Persona, *, seed: int
    ) -> AxisScores:
        burden = self.score_risk(campaign, expression, seed=seed).c_b
        fit = 0.58 - 0.25 * burden + 0.08 * _unit(
            f"{campaign.campaign_id}:{persona.persona_id}:{seed}"
        )
        reason = 0.12 if "理由" in expression.creative_hint else 0.0
        return AxisScores(
            min(1.0, fit + 0.08),
            min(1.0, fit),
            min(1.0, fit + 0.04),
            min(1.0, fit + reason),
        )

    def generate_response(
        self,
        campaign: Campaign,
        expression: Expression,
        persona: Persona,
        target: AxisScores,
        *,
        seed: int,
    ) -> str:
        risk = self.score_risk(campaign, expression, seed=seed)
        if risk.c_b >= 0.65:
            return "条件が多く、今回は回答が難しいです。"
        contexts = campaign.source_schema.get("allowed_response_contexts") or ["試したい点"]
        context = str(contexts[int(_unit(f"{seed}:context") * len(contexts)) % len(contexts)])
        response = f"{campaign.product}を{context}で楽しんでみたいです"
        if "理由" in expression.creative_hint or target.d_rea >= 0.45:
            response += "。新しい味や香りを気軽に試せそうだからです"
        return response + "。"

    def score_response(
        self, campaign: Campaign, response: str, *, seed: int
    ) -> AxisScores:
        refusal = "回答が難しい" in response
        if refusal:
            return AxisScores(0.12, 0.08, 0.05, 0.08)
        product = campaign.product in response
        reason = any(token in response for token in ("から", "ので", "理由"))
        specific = any(token in response for token in ("味", "香り", "朝", "夜", "食事", "場面"))
        return AxisScores(
            0.56 + 0.05 * _unit(f"{seed}:affect"),
            0.62 if product else 0.35,
            0.62 if specific else 0.38,
            0.66 if reason else 0.25,
        )

    def score_risk(
        self, campaign: Campaign, expression: Expression, *, seed: int
    ) -> RiskScores:
        text = expression.creative_hint
        burden = 0.08
        burden += min(0.28, 0.055 * (text.count("、") + text.count("と")))
        burden += 0.18 if re.search(r"\d+字|\d+項目|\d+段階", text) else 0.0
        burden += 0.14 if "写真" in text else 0.0
        burden += 0.15 if "タグ" in text else 0.0
        burden += 0.14 if any(x in text for x in ("位置情報", "購入履歴")) else 0.0
        burden = min(1.0, burden)
        unsupported = any(x in text for x in (
            "必ず上がる", "必ず痩せる", "必ず回復", "必ず排出",
        ))
        fidelity = 0.82 if unsupported else 0.05
        return RiskScores(
            burden,
            fidelity,
            "deterministic burden markers and unsupported-guarantee markers",
        )

    def check_restatement(
        self, campaign: Campaign, expression: Expression, *, seed: int
    ) -> bool:
        text = expression.creative_hint
        return bool(re.search(r"#\S", text)) or any(x in text for x in (
            "フォロー", "ハッシュタグ", "引用ポスト", "引用リポスト", "リポスト",
            "リツイート", "締切", "締め切り", "応募方法", "参加条件",
        ))

    def refine(
        self,
        campaign: Campaign,
        current: Evaluation,
        *,
        iteration: int,
        seed: int,
        preferred: Evaluation | None = None,
        personas: tuple[Persona, ...] = (),
    ) -> str:
        contexts = campaign.source_schema.get("allowed_response_contexts") or ["期待すること"]
        first = str(contexts[iteration % len(contexts)])
        variants = (
            f"{first}を一つ、短く教えてください。",
            f"{first}と、その理由を一つ教えてください。",
            f"{first}について、いちばん楽しみな点を教えてください。",
            f"{first}または{contexts[-1]}を一つ、理由とともに教えてください。",
        )
        return variants[iteration % len(variants)]

    def optimizer_text(
        self, campaign: Campaign, prompt: str, *, purpose: str, iteration: int, seed: int
    ) -> str:
        if purpose == "ipc_analysis":
            return "Reduce response burden and ask for one concrete reason; preserve source facts."
        return self.refine(campaign, None, iteration=iteration, seed=seed)
