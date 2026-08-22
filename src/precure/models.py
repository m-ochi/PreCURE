from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class AxisScores:
    a: float
    d_cov: float
    d_spec: float
    d_rea: float

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            if not 0.0 <= float(value) <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")

    def mean_with(self, others: list["AxisScores"]) -> "AxisScores":
        rows = [self, *others]
        return AxisScores(**{
            field: sum(getattr(row, field) for row in rows) / len(rows)
            for field in ("a", "d_cov", "d_spec", "d_rea")
        })


@dataclass(frozen=True)
class RiskScores:
    c_b: float
    c_f: float
    rationale: str = ""

    def __post_init__(self) -> None:
        if not 0.0 <= self.c_b <= 1.0 or not 0.0 <= self.c_f <= 1.0:
            raise ValueError("risk scores must be in [0, 1]")


@dataclass(frozen=True)
class Expression:
    offer: str
    period: str
    eligibility: str
    creative_hint: str

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "Expression":
        fields = ("offer", "period", "eligibility", "creative_hint")
        missing = [field for field in fields if not str(value.get(field, "")).strip()]
        if missing:
            raise ValueError(f"expression has empty fields: {missing}")
        return cls(**{field: str(value[field]) for field in fields})

    def with_creative_hint(self, creative_hint: str) -> "Expression":
        if not creative_hint.strip():
            raise ValueError("creative_hint must not be empty")
        return Expression(self.offer, self.period, self.eligibility, creative_hint.strip())

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


@dataclass(frozen=True)
class Campaign:
    campaign_id: str
    brand: str
    product: str
    category: str
    expression: Expression
    source_schema: dict[str, Any]

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "Campaign":
        return cls(
            campaign_id=str(value["campaign_id"]),
            brand=str(value["brand"]),
            product=str(value["product"]),
            category=str(value.get("category", "")),
            expression=Expression.from_dict(value),
            source_schema=dict(value.get("source_schema") or {}),
        )


@dataclass(frozen=True)
class Persona:
    persona_id: str
    text: str


@dataclass(frozen=True)
class FidelityReport:
    passed: bool
    violations: tuple[str, ...]


@dataclass(frozen=True)
class Evaluation:
    expression: Expression
    axes: AxisScores
    risks: RiskScores
    q: float
    cps: float | None
    fidelity: FidelityReport
    responses: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "expression": self.expression.to_dict(),
            "axes": asdict(self.axes),
            "risks": asdict(self.risks),
            "q": self.q,
            "cps": self.cps,
            "fidelity": asdict(self.fidelity),
            "responses": list(self.responses),
        }
