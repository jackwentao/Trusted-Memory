"""Structured extraction using the configured Mem0 LLM."""

import json
from datetime import datetime, timedelta
from typing import Protocol


class JsonModel(Protocol):
    def generate_response(
        self, *, messages: list[dict], response_format: dict
    ) -> str: ...


ATTRIBUTES = ("residence.city", "employment.employer", "education.school")


def extract(model: JsonModel, text: str, anchor: datetime) -> list[dict]:
    """Require verbatim values and time expressions before publishing any facts."""
    if anchor.utcoffset() is None:
        raise ValueError("session_time needs timezone")
    prompt = (
        "Extract only clearly asserted states of the speaker. Return JSON with facts array. "
        f"attribute_key must be one of {ATTRIBUTES}. "
        "Each object has attribute_key, value (short exact substring naming the city, employer or school), "
        "time (exact temporal phrase from input, empty only for a clearly current statement). "
        "Omit unsupported, hypothetical, negated, ambiguous or superseded claims. "
        "Do not translate values or resolve dates yourself."
    )
    response = model.generate_response(
        messages=[
            {"role": "system", "content": prompt},
            {
                "role": "user",
                "content": json.dumps(
                    {"session_time": anchor.isoformat(), "message": text},
                    ensure_ascii=False,
                ),
            },
        ],
        response_format={"type": "json_object"},
    )
    rows = json.loads(response)["facts"]
    result = []
    for row in rows:
        attr, value, phrase = (row[k] for k in ("attribute_key", "value", "time"))
        if (
            attr not in ATTRIBUTES
            or not value
            or value not in text
            or phrase not in text
        ):
            raise ValueError("unsupported extraction")
        event, precision = resolve_time(phrase, anchor)
        result.append(
            {
                "attribute": attr,
                "value": value,
                "event": event,
                "precision": precision,
                "phrase": phrase,
            }
        )
    return result


def resolve_time(phrase: str, anchor: datetime) -> tuple[datetime, str]:
    """Normalize using local calendar boundaries; month granularity uses the lower bound."""
    day = anchor.replace(hour=0, minute=0, second=0, microsecond=0)
    match phrase.lower():
        case "" | "现在" | "目前" | "now" | "currently":
            return anchor, "observation"
        case "今天" | "today":
            return day, "day"
        case "昨天" | "yesterday":
            return day - timedelta(days=1), "day"
        case "上个月" | "上月" | "last month":
            return (day.replace(day=1) - timedelta(days=1)).replace(day=1), "month"
        case _:
            date = datetime.strptime(phrase, "%Y-%m-%d")
            return date.replace(tzinfo=anchor.tzinfo), "day"
