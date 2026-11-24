"""Shared types for time and security decisions."""

from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from typing import Any, Self
from uuid import UUID


class Source(StrEnum):
    USER = "user"
    TOOL = "tool"
    DOCUMENT = "document"


class Action(StrEnum):
    ACCEPT = "accept"
    REDACT = "redact"
    REVIEW = "review"
    BLOCK = "block"


class State(StrEnum):
    ACTIVE = "ACTIVE"
    SUPERSEDED = "SUPERSEDED"


@dataclass(frozen=True)
class Audit:
    id: UUID
    timestamp: datetime
    source: Source
    action: Action
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class Decision:
    audit: Audit
    approved_text: str | None
    candidate_text: str | None = None


@dataclass(frozen=True)
class Fact:
    """Stable application-issued ID and scope, independent of message text."""

    id: UUID
    owner: str
    attribute: str
    text: str
    source: Source
    event_time: datetime
    native: dict[str, Any] | None = None

    @classmethod
    def from_mem0_payload(
        cls,
        payload: dict[str, Any],
        *,
        owner: str | None = None,
        attribute: str | None = None,
        event_time: datetime | None = None,
        source: Source | None = None,
    ) -> Self:
        """Import a read result. Supply provenance and effective time if unknown."""
        native = deepcopy(payload)
        metadata = native.get("metadata")
        if metadata is not None and not isinstance(metadata, dict):
            raise ValueError("metadata must be an object or null")
        meta = metadata or {}
        identifier = UUID(native["id"])
        scope = owner or native.get("user_id") or meta.get("owner")
        if not scope:
            scope = next(
                (f"{k}:{native[k]}" for k in ("agent_id", "run_id") if native.get(k)),
                None,
            )
        event = (
            datetime.fromisoformat(meta["event_time"])
            if meta.get("event_time")
            else event_time
        )
        provenance = source or meta.get("source_type")
        if not scope or not event or event.utcoffset() is None or not provenance:
            raise ValueError(
                "owner, timezone-aware event_time and trusted source required"
            )
        if not isinstance(native["memory"], str):
            raise TypeError("memory must be text")
        if meta.get("status"):
            State(meta["status"])
        if meta.get("valid_until"):
            end = datetime.fromisoformat(meta["valid_until"])
            if end.utcoffset() is None or end <= event:
                raise ValueError("invalid validity interval")
        if meta.get("superseded_by"):
            UUID(meta["superseded_by"])
        return cls(
            identifier,
            scope,
            attribute or meta.get("attribute") or str(identifier),
            native["memory"],
            Source(provenance),
            event,
            native,
        )

    def to_mem0_payload(self) -> dict[str, Any]:
        """Export native fields and independent metadata for this fact."""
        result = (
            deepcopy(self.native)
            if self.native is not None
            else {"user_id": self.owner}
        )
        meta = dict(result.get("metadata") or {})
        meta.update(
            event_time=self.event_time.isoformat(),
            source_type=self.source.value,
            attribute=self.attribute,
            owner=self.owner,
        )
        meta.setdefault("status", State.ACTIVE.value)
        meta.setdefault("valid_until", None)
        meta.setdefault("superseded_by", None)
        result.update(id=str(self.id), memory=self.text, metadata=meta)
        return result


@dataclass(frozen=True)
class Version:
    fact: Fact
    ingest_time: datetime
    valid_to: datetime | None
    superseded_by: UUID | None
    state: State

    def reader_context(self) -> str:
        """Render the reconciled half-open interval alongside the retrieved fact."""
        end = self.valid_to.isoformat() + "（不含）" if self.valid_to else "无截止"
        label = {State.ACTIVE: "现行记录", State.SUPERSEDED: "历史记录"}[self.state]
        return (
            f"[event_time={self.fact.event_time.isoformat()}（含） | "
            f"valid_until={end} | {self.state.value} {label}] {self.fact.text}"
        )

    def to_mem0_payload(self) -> dict[str, Any]:
        """Export the effective chain after local version reconciliation."""
        payload = self.fact.to_mem0_payload()
        if self.fact.native is None:
            payload.update(
                created_at=self.ingest_time.isoformat(),
                updated_at=self.ingest_time.isoformat(),
            )
        payload["metadata"].update(
            valid_until=self.valid_to.isoformat() if self.valid_to else None,
            status=self.state.value,
            superseded_by=str(self.superseded_by) if self.superseded_by else None,
        )
        return payload
