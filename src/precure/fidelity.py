from __future__ import annotations

import re
import unicodedata

from .models import Campaign, Expression, FidelityReport


def _normalize(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def validate_expression(
    campaign: Campaign,
    candidate: Expression,
    *,
    creative_only: bool = True,
) -> FidelityReport:
    """Fail closed on protected fields and encoded source constraints.

    The public runner uses the conservative creative-only search space from the
    paper's PrefPO-CPS condition. Offer, period, and eligibility therefore have
    to remain byte-for-byte equivalent after Unicode/whitespace normalization.
    """

    violations: list[str] = []
    if creative_only:
        for field in ("offer", "period", "eligibility"):
            if _normalize(getattr(candidate, field)) != _normalize(
                getattr(campaign.expression, field)
            ):
                violations.append(f"{field}:protected_field_changed")

    context = _normalize(" ".join([
        campaign.product,
        candidate.offer,
        candidate.period,
        candidate.eligibility,
        candidate.creative_hint,
    ]))
    schema = campaign.source_schema
    for term in schema.get("required_terms", []):
        if _normalize(str(term)) not in context:
            violations.append(f"source_schema:required_term_missing:{term}")
    for term in schema.get("forbidden_claims", []):
        if _normalize(str(term)) in context:
            violations.append(f"source_schema:forbidden_claim_added:{term}")

    required_hashtags = {
        str(tag).casefold() for tag in schema.get("required_hashtags", [])
    }
    present_hashtags = {
        tag.casefold() for tag in re.findall(r"#[^\s#]+", context)
    }
    for tag in sorted(required_hashtags - present_hashtags):
        violations.append(f"source_schema:required_hashtag_missing:{tag}")

    return FidelityReport(not violations, tuple(violations))
