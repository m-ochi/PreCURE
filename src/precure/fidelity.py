from __future__ import annotations

import re
import unicodedata

from .models import Campaign, Expression, FidelityReport

_NUMBER_RE = re.compile(
    r"(?P<prefix>[$€£¥￥]\s*)?(?P<number>\d+(?:[.,]\d+)?)"
    r"(?:\s*(?P<unit>%|％|usd|eur|gbp|jpy|dollars?|euros?|pounds?|yen|days?|hours?|weeks?|months?|"
    r"円|名|人|個|本|枚|杯|種類|点|日|時間|週|週間|月|か月|ヶ月|年))?",
    re.IGNORECASE,
)
_MONTH_PATTERN = (
    r"jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|"
    r"jul(?:y)?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?"
)
_MONTH_DAY_RE = re.compile(
    rf"\b(?P<month>{_MONTH_PATTERN})\s+(?P<day>\d{{1,2}})(?:st|nd|rd|th)?"
    rf"(?:,?\s+(?P<year>\d{{4}}))?\b",
    re.IGNORECASE,
)
_DAY_MONTH_RE = re.compile(
    rf"\b(?P<day>\d{{1,2}})(?:st|nd|rd|th)?\s+(?P<month>{_MONTH_PATTERN})"
    rf"(?:,?\s+(?P<year>\d{{4}}))?\b",
    re.IGNORECASE,
)
_JA_DATE_RE = re.compile(
    r"(?:(?P<year>\d{4})年)?\s*(?P<month>\d{1,2})月\s*(?P<day>\d{1,2})日"
)
_NUMERIC_DATE_RE = re.compile(
    r"(?:(?P<year>\d{4})[-/])(?P<month>\d{1,2})[-/](?P<day>\d{1,2})"
)
_WORD_RE = re.compile(r"[a-z0-9]+(?:['’-][a-z0-9]+)?", re.IGNORECASE)

_PRODUCT_STOPWORDS = {
    "a", "an", "and", "the", "of", "for", "to", "your", "our", "one",
    "win", "giveaway", "choice", "available", "receive", "winner", "winners",
}
_NEGATIONS = {"no", "not", "never", "without", "excluding", "except"}
_JA_NEGATION_RE = re.compile(r"(?:ない|なし|不要|除く|以外|不可|禁止|限定しない)")
_ACTION_PATTERNS = {
    "follow": re.compile(r"\bfollow(?:ing|ed|s)?\b", re.IGNORECASE),
    "like": re.compile(r"\blik(?:e|es|ed|ing)\b", re.IGNORECASE),
    "repost": re.compile(r"\b(?:repost|re-post|retweet)(?:s|ed|ing)?\b", re.IGNORECASE),
    "comment": re.compile(r"\b(?:comment|reply)(?:s|ed|ing)?\b", re.IGNORECASE),
    "tag": re.compile(r"\btag(?:s|ged|ging)?\b", re.IGNORECASE),
    "share": re.compile(r"\bshare(?:s|d|ing)?\b", re.IGNORECASE),
    "subscribe": re.compile(r"\bsubscrib(?:e|es|ed|ing)\b", re.IGNORECASE),
    "join": re.compile(r"\bjoin(?:s|ed|ing)?\b", re.IGNORECASE),
    "purchase": re.compile(r"\b(?:purchase|buy)(?:s|ing|ought)?\b", re.IGNORECASE),
    "follow_ja": re.compile(r"フォロー"),
    "like_ja": re.compile(r"(?:いいね|イイネ)"),
    "repost_ja": re.compile(r"(?:リポスト|リツイート|再投稿)"),
    "reply_ja": re.compile(r"(?:返信|リプ(?:ライ)?|コメント)"),
    "quote_ja": re.compile(r"引用(?:投稿|ポスト|リポスト)?"),
    "hashtag_ja": re.compile(r"(?:ハッシュタグ|#[^\s#]+)"),
    "purchase_ja": re.compile(r"(?:購入|買って|買う|注文)"),
    "register_ja": re.compile(r"(?:登録|エントリー|応募)"),
    "mobile_order_ja": re.compile(r"モバイルオーダー"),
}


def _normalize(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text or "").casefold().split())


def _numbers(text: str) -> set[str]:
    currency = {
        "$": "usd", "usd": "usd", "dollar": "usd", "dollars": "usd",
        "€": "eur", "eur": "eur", "euro": "eur", "euros": "eur",
        "£": "gbp", "gbp": "gbp", "pound": "gbp", "pounds": "gbp",
        "¥": "jpy", "￥": "jpy", "jpy": "jpy", "yen": "jpy", "円": "jpy",
    }
    singular = {"days": "day", "hours": "hour", "weeks": "week", "months": "month"}
    facts = set()
    for match in _NUMBER_RE.finditer(text or ""):
        number = match.group("number").replace(",", "")
        if "." in number:
            number = number.rstrip("0").rstrip(".")
        prefix = (match.group("prefix") or "").strip()
        unit = (match.group("unit") or "").lower()
        unit = currency.get(prefix or unit, singular.get(unit, unit))
        unit = {"％": "%", "週間": "週", "ヶ月": "月", "か月": "月"}.get(unit, unit)
        facts.add(f"{number}:{unit}" if unit else number)
    return facts


def _dates(text: str) -> set[str]:
    months = {
        "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
        "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
    }
    facts = set()
    for pattern in (_MONTH_DAY_RE, _DAY_MONTH_RE):
        for match in pattern.finditer(text or ""):
            month = months[match.group("month")[:3].lower()]
            year = match.group("year") or "????"
            facts.add(f"{year}-{month:02d}-{int(match.group('day')):02d}")
    for pattern in (_JA_DATE_RE, _NUMERIC_DATE_RE):
        for match in pattern.finditer(text or ""):
            year = match.group("year") or "????"
            facts.add(
                f"{year}-{int(match.group('month')):02d}-{int(match.group('day')):02d}"
            )
    return facts


def _actions(text: str) -> set[str]:
    return {name for name, pattern in _ACTION_PATTERNS.items() if pattern.search(text or "")}


def _negations(text: str) -> set[str]:
    present = set(_WORD_RE.findall(_normalize(text))) & _NEGATIONS
    return {"negation"} if present or _JA_NEGATION_RE.search(text or "") else set()


def _protected_negations(field: str, text: str) -> set[str]:
    """Ignore metadata-like absence labels while preserving substantive negation."""
    normalized = _normalize(text)
    if field in {"offer", "period"} and "記載なし" in normalized:
        return set()
    return _negations(text)


def _product_anchors(product: str) -> set[str]:
    ascii_anchors = {
        word for word in _WORD_RE.findall(_normalize(product))
        if len(word) >= 3 and word not in _PRODUCT_STOPWORDS and not word.isdigit()
    }
    japanese_chunks = re.findall(r"[ぁ-んァ-ヶ一-龠ー]{2,}", _normalize(product))
    return ascii_anchors | set(japanese_chunks)


def _compare_set(field: str, code: str, original: set[str], candidate: set[str]) -> list[str]:
    if original == candidate:
        return []
    removed = sorted(original - candidate)
    added = sorted(candidate - original)
    detail_parts = []
    if removed:
        detail_parts.append(f"missing={removed}")
    if added:
        detail_parts.append(f"added={added}")
    detail = ", ".join(detail_parts)
    return [f"{field}:{code}:{detail}" if detail else f"{field}:{code}"]


def validate_campaign_fidelity(campaign: Campaign, candidate: Expression) -> FidelityReport:
    """Fail closed on drift in non-negotiable campaign facts.

    Rather than requiring the protected fields (`offer`, `period`,
    `eligibility`) to stay byte-for-byte identical, this compares the
    underlying facts they encode: monetary/count amounts, dates, negations,
    and the required entry action. A rewrite that keeps the same facts in
    different words still passes; one that silently drops a fact, changes an
    amount or deadline, or swaps the entry action does not. It also confirms
    the fixed product is still referenced after the creative rewrite.
    """

    violations: list[str] = []

    anchors = _product_anchors(campaign.product)
    candidate_context = _normalize(
        f"{campaign.product} {candidate.offer} {candidate.creative_hint}"
    )
    missing_anchors = sorted(anchor for anchor in anchors if anchor not in candidate_context)
    if missing_anchors:
        violations.append(f"product:product_anchor_missing:missing={missing_anchors}")

    for field in ("offer", "period", "eligibility"):
        original_text = getattr(campaign.expression, field)
        candidate_text = getattr(candidate, field)
        violations.extend(
            _compare_set(field, "numeric_fact_changed", _numbers(original_text), _numbers(candidate_text))
        )
        violations.extend(
            _compare_set(field, "date_fact_changed", _dates(original_text), _dates(candidate_text))
        )
        violations.extend(
            _compare_set(
                field, "negation_changed",
                _protected_negations(field, original_text),
                _protected_negations(field, candidate_text),
            )
        )

    violations.extend(
        _compare_set(
            "eligibility",
            "entry_action_changed",
            _actions(campaign.expression.eligibility),
            _actions(candidate.eligibility),
        )
    )

    return FidelityReport(not violations, tuple(violations))


def validate_source_schema(source_schema: dict, campaign: Campaign, candidate: Expression) -> FidelityReport:
    """Fail closed on required terms, forbidden claims, and required hashtags."""

    context = _normalize(
        " ".join([campaign.product, candidate.offer, candidate.period, candidate.eligibility, candidate.creative_hint])
    )
    violations: list[str] = []
    for term in source_schema.get("required_terms", []):
        if _normalize(str(term)) not in context:
            violations.append(f"source_schema:required_term_missing:{term}")
    for term in source_schema.get("forbidden_claims", []):
        if _normalize(str(term)) in context:
            violations.append(f"source_schema:forbidden_claim_added:{term}")

    missing_hashtags = sorted(
        tag for tag in source_schema.get("required_hashtags", [])
        if _normalize(str(tag)) not in context
    )
    if missing_hashtags:
        violations.append(f"source_schema:required_hashtag_missing:missing={missing_hashtags}")

    return FidelityReport(not violations, tuple(violations))


def validate_expression(
    campaign: Campaign,
    candidate: Expression,
    *,
    creative_only: bool = True,
) -> FidelityReport:
    """Hard-fidelity gate: deterministic, fail-closed, and non-negotiable.

    This mirrors the fact-level check used in the paper's full experiments
    (as opposed to a graded, LLM-judged score): candidate expressions are
    only eligible for selection if the protected facts in `offer`, `period`,
    and `eligibility` are unchanged and the source schema's required terms,
    hashtags, and forbidden claims are respected. `creative_only` reflects
    the public runner's conservative search space, where only `creative_hint`
    is proposed by the optimizer and `offer`/`period`/`eligibility` are
    otherwise held fixed by construction; the fact-level comparison here is
    an additional, independent check rather than the sole guarantee.
    """

    violations: list[str] = []
    if creative_only:
        violations.extend(validate_campaign_fidelity(campaign, candidate).violations)
    violations.extend(validate_source_schema(campaign.source_schema, campaign, candidate).violations)

    return FidelityReport(not violations, tuple(violations))
