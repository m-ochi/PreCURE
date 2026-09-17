from __future__ import annotations

import json
import os
import random
import threading
import time
import urllib.error
import urllib.request
from typing import Any

from .models import AxisScores, Campaign, Evaluation, Expression, Persona, RiskScores


OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
AXIS_SCHEMA = {
    "type": "object",
    "properties": {
        "a": {"type": "number", "minimum": 0, "maximum": 1},
        "d_cov": {"type": "number", "minimum": 0, "maximum": 1},
        "d_spec": {"type": "number", "minimum": 0, "maximum": 1},
        "d_rea": {"type": "number", "minimum": 0, "maximum": 1},
    },
    "required": ["a", "d_cov", "d_spec", "d_rea"],
    "additionalProperties": False,
}
RISK_SCHEMA = {
    "type": "object",
    "properties": {
        "c_b": {"type": "number", "minimum": 0, "maximum": 1},
        "c_f": {"type": "number", "minimum": 0, "maximum": 1},
        "rationale": {"type": "string"},
    },
    "required": ["c_b", "c_f", "rationale"],
    "additionalProperties": False,
}
TEXT_SCHEMA = {
    "type": "object",
    "properties": {"text": {"type": "string"}},
    "required": ["text"],
    "additionalProperties": False,
}


class OpenRouterBackend:
    """Minimal provider client for paper-style LLM execution.

    Only the synthetic examples bundled with this package should be sent by the
    public demo. Do not pass private user histories or restricted UGC.
    """

    def __init__(
        self,
        model: str,
        *,
        api_key: str | None = None,
        timeout_seconds: int = 180,
        max_retries: int = 4,
        max_cost_usd: float = 1.0,
        min_request_interval_seconds: float = 0.5,
    ) -> None:
        self.model = model
        self.name = f"openrouter:{model}"
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY", "")
        if not self.api_key:
            raise RuntimeError("OPENROUTER_API_KEY is required for --backend openrouter")
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.max_cost_usd = max_cost_usd
        self.min_request_interval_seconds = min_request_interval_seconds
        self._last_request_at = 0.0
        self._throttle_lock = threading.Lock()
        self._stats_lock = threading.Lock()
        self.request_count = 0
        self.cost_usd = 0.0

    def _throttle(self) -> None:
        """Space consecutive requests out so upstream per-second caps aren't tripped.

        precure/engine.py evaluates candidates strictly sequentially, so this is
        normally the only pacing needed; the lock also covers callers (e.g.
        per-persona evaluation in the demo app) that share one backend instance
        across a thread pool."""
        with self._throttle_lock:
            elapsed = time.monotonic() - self._last_request_at
            wait = self.min_request_interval_seconds - elapsed
            if wait > 0:
                time.sleep(wait)
            self._last_request_at = time.monotonic()

    def _complete_json(
        self,
        prompt: str,
        schema_name: str,
        schema: dict[str, Any],
        *,
        seed: int,
        temperature: float,
        max_tokens: int,
    ) -> dict[str, Any]:
        if self.cost_usd >= self.max_cost_usd:
            raise RuntimeError(
                f"OpenRouter cost cap reached: ${self.cost_usd:.4f} >= "
                f"${self.max_cost_usd:.4f}"
            )
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "seed": seed % 2_147_483_648,
            "provider": {
                "data_collection": "deny",
                "zdr": True,
                "require_parameters": True,
            },
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": schema_name,
                    "strict": True,
                    "schema": schema,
                },
            },
        }
        request = urllib.request.Request(
            OPENROUTER_URL,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://github.com/m-ochi/PreCURE",
                "X-Title": "PreCURE reproducibility package",
            },
            method="POST",
        )
        last_error: Exception | None = None
        for attempt in range(self.max_retries):
            self._throttle()
            try:
                with urllib.request.urlopen(
                    request, timeout=self.timeout_seconds
                ) as response:
                    data = json.loads(response.read().decode("utf-8"))
                usage = data.get("usage") or {}
                cost = float(usage.get("cost", data.get("cost", 0)) or 0)
                with self._stats_lock:
                    self.request_count += 1
                    self.cost_usd += cost
                content = data["choices"][0]["message"]["content"]
                parsed = json.loads(content)
                if not isinstance(parsed, dict):
                    raise RuntimeError("structured response was not a JSON object")
                return parsed
            except (urllib.error.URLError, TimeoutError, KeyError, json.JSONDecodeError) as exc:
                last_error = exc
                if attempt + 1 < self.max_retries:
                    time.sleep(self._backoff_seconds(exc, attempt))
        raise RuntimeError(f"OpenRouter request failed: {last_error}")

    @staticmethod
    def _backoff_seconds(exc: Exception, attempt: int) -> float:
        """429s get a longer, Retry-After-aware wait; other errors keep the short backoff."""
        if isinstance(exc, urllib.error.HTTPError) and exc.code == 429:
            retry_after = exc.headers.get("Retry-After") if exc.headers else None
            if retry_after is not None:
                try:
                    return max(1.0, float(retry_after))
                except ValueError:
                    pass
            return min(30.0, 4 * 2**attempt + random.random())
        return min(8.0, 2**attempt + random.random())

    @staticmethod
    def _campaign_text(campaign: Campaign, expression: Expression) -> str:
        return (
            f"brand: {campaign.brand}\nproduct: {campaign.product}\n"
            f"category: {campaign.category}\noffer: {expression.offer}\n"
            f"period: {expression.period}\neligibility: {expression.eligibility}\n"
            f"editable participation expression: {expression.creative_hint}"
        )

    def _axes(self, prompt: str, *, seed: int) -> AxisScores:
        data = self._complete_json(
            prompt, "precure_axis_scores", AXIS_SCHEMA,
            seed=seed, temperature=0.0, max_tokens=300,
        )
        return AxisScores(**{key: float(data[key]) for key in AXIS_SCHEMA["required"]})

    def score_persona(self, persona: Persona, *, seed: int) -> AxisScores:
        return self._axes(
            "次の架空personaが一般に書く商品関連SNS投稿の傾向を4軸で推定してください。"
            "aは感情強度、d_covは内容範囲、d_specは具体性、d_reaは理由説明です。"
            "各値は0から1です。\n\npersona:\n" + persona.text,
            seed=seed,
        )

    def score_campaign(
        self, campaign: Campaign, expression: Expression, persona: Persona, *, seed: int
    ) -> AxisScores:
        return self._axes(
            "次の架空personaがこの架空キャンペーンへ反応する場合の投稿傾向を、"
            "a（感情強度）、d_cov（内容範囲）、d_spec（具体性）、d_rea（理由説明）の"
            "0から1で推定してください。\n\n"
            + self._campaign_text(campaign, expression)
            + "\n\npersona:\n"
            + persona.text,
            seed=seed,
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
        data = self._complete_json(
            "次の架空personaによる自然な日本語のキャンペーン反応UGCを1件だけ生成してください。"
            "15〜120文字を目安とし、氏名、勤務先、居住地などの識別情報は書かないでください。"
            "キャンペーンにない効果や事実を追加しないでください。\n\n"
            + self._campaign_text(campaign, expression)
            + f"\n\ntarget scores: {target}\n\npersona:\n{persona.text}",
            "precure_generated_response",
            TEXT_SCHEMA,
            seed=seed,
            temperature=0.8,
            max_tokens=300,
        )
        return str(data["text"]).strip()

    def score_response(
        self, campaign: Campaign, response: str, *, seed: int
    ) -> AxisScores:
        return self._axes(
            "次の架空キャンペーンへの日本語UGCを4軸で評価してください。"
            "aは感情強度、d_covは商品・キャンペーン関連内容の範囲、"
            "d_specは具体性、d_reaは明示的な理由説明です。各値は0から1です。\n\n"
            f"product: {campaign.product}\nUGC: {response}",
            seed=seed,
        )

    def score_risk(
        self, campaign: Campaign, expression: Expression, *, seed: int
    ) -> RiskScores:
        data = self._complete_json(
            "架空キャンペーンの参加表現を評価してください。C_Bは回答負担、C_Fは固定事実からの"
            "逸脱リスクで、各0から1です。多項目、長文、写真、位置情報、友人タグ、購入履歴は"
            "負担を上げます。根拠のない健康・性能保証は逸脱リスクを上げます。\n\n"
            + self._campaign_text(campaign, expression),
            "precure_risk_scores",
            RISK_SCHEMA,
            seed=seed,
            temperature=0.0,
            max_tokens=400,
        )
        return RiskScores(float(data["c_b"]), float(data["c_f"]), str(data["rationale"]))

    def refine(
        self,
        campaign: Campaign,
        current: Evaluation,
        *,
        iteration: int,
        seed: int,
        preferred: Evaluation | None = None,
    ) -> str:
        comparison = ""
        if preferred is not None:
            comparison = (
                "\n比較対象のより良い候補:\n"
                f"{preferred.expression.creative_hint}\nCPS={preferred.cps}\n"
            )
        data = self._complete_json(
            "固定情報を一切変更せず、editable participation expressionだけを日本語で書き直してください。"
            "offer・period・eligibilityの内容はcreative_hint内で繰り返し述べないでください"
            "（これらは投稿の別の部分として固定情報のまま提示されるため、creative_hintは参加を促す"
            "一つか二つの観点だけに集中してください）。"
            "特定personaの属性を公開文へ埋め込まず、一つか二つの答えやすい観点に絞り、"
            "自然で具体的なUGCを促してください。根拠のない効果を要求しないでください。\n\n"
            + self._campaign_text(campaign, current.expression)
            + f"\ncurrent scores: axes={current.axes}, Q={current.q}, C_B={current.risks.c_b}, "
            f"C_F={current.risks.c_f}, CPS={current.cps}\n"
            + comparison,
            "precure_refined_expression",
            TEXT_SCHEMA,
            seed=seed,
            temperature=0.4,
            max_tokens=300,
        )
        return str(data["text"]).strip()

    def optimizer_text(
        self, campaign: Campaign, prompt: str, *, purpose: str, iteration: int, seed: int
    ) -> str:
        data = self._complete_json(
            prompt + "\nReturn the requested analysis or creative_hint as the JSON text field.",
            "precure_optimizer_text", TEXT_SCHEMA, seed=seed,
            temperature=0.0 if purpose == "ipc_analysis" else 0.4,
            max_tokens=1600,
        )
        return str(data["text"]).strip()
