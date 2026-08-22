from __future__ import annotations

import json
from pathlib import Path

from .models import Campaign, Persona


def load_campaigns(path: Path) -> list[Campaign]:
    value = json.loads(path.read_text(encoding="utf-8"))
    rows = value.get("campaigns") if isinstance(value, dict) else value
    if not isinstance(rows, list):
        raise ValueError("campaign file must contain a campaigns array")
    return [Campaign.from_dict(row) for row in rows]


def load_personas(path: Path) -> dict[str, list[Persona]]:
    result: dict[str, list[Persona]] = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            campaign_id = str(row["campaign_id"])
            result[campaign_id] = [
                Persona(
                    persona_id=str(item.get("source_row_hash") or f"{campaign_id}-{index}"),
                    text=str(item["persona_text"]),
                )
                for index, item in enumerate(row["personas"])
            ]
    return result
