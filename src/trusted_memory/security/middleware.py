"""Deterministic pre-write gate. The caller supplies authenticated provenance."""

import json
import re
from collections.abc import Callable
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from trusted_memory.schemas import Action, Audit, Decision, Source

_PII = (
    ("SECRET", re.compile(r"(?:sk-|ghp_|github_pat_)[A-Za-z0-9_-]{4,}")),
    ("ID_CARD", re.compile(r"(?<!\d)\d{17}[\dXx](?!\d)")),
    ("BANK_CARD", re.compile(r"(?<!\d)\d(?:[ -]?\d){12,18}(?!\d)")),
    ("PHONE", re.compile(r"(?<!\d)(?:\+?86[ -]?)?1[3-9]\d[ -]?\d{4}[ -]?\d{4}(?!\d)")),
)
_DIRECTIVE = re.compile(
    r"(?:忽略|无视|覆盖)(?:之前|前面|此前|所有|系统|以上)(?:的)?(?:系统)?(?:规则|指令|要求)"
    r"|(?:ignore|override)\s+(?:all\s+)?(?:previous|prior|system)\s+(?:instructions|rules)",
    re.IGNORECASE,
)


class SecurityMiddleware:
    """Release content only after the audit sink accepts the decision."""

    def __init__(self, audit_sink: Callable[[Audit], None]) -> None:
        self._audit = audit_sink

    def inspect(self, text: str, *, source: Source) -> Decision:
        """Return sanitized approved content or a separately held candidate."""
        source = Source(source)
        reasons = []
        for category, pattern in _PII:
            text, count = pattern.subn(f"<{category}>", text)
            if count:
                reasons.append(category)
        if _DIRECTIVE.search(text):
            # Discussion is safe to retain only with the quoted directive removed.
            text = re.sub(
                r"“[^”]*”",
                lambda m: "<QUOTED_INSTRUCTION>" if _DIRECTIVE.search(m[0]) else m[0],
                text,
            )
            if _DIRECTIVE.search(text):
                action = Action.BLOCK
                reasons.append("instruction")
            else:
                reasons.append("quoted_instruction")
                action = Action.REDACT
        else:
            action = Action.REDACT if reasons else Action.ACCEPT
        if action != Action.BLOCK and source != Source.USER:
            action = Action.REVIEW
            reasons.append("unverified_source")
        audit = Audit(uuid4(), datetime.now(UTC), source, action, tuple(reasons))
        self._audit(audit)
        return Decision(
            audit,
            text if action in (Action.ACCEPT, Action.REDACT) else None,
            text if action == Action.REVIEW else None,
        )


class JsonAuditLog:
    """Append decisions to a UTF-8 JSON lines file without user content."""

    def __init__(self, path: Path) -> None:
        self.path = path

    def __call__(self, record: Audit) -> None:
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(asdict(record), default=str) + "\n")
