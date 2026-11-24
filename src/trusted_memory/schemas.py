"""Shared types for time and security decisions."""

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
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


@dataclass(frozen=True)
class Version:
    fact: Fact
    ingest_time: datetime
    valid_to: datetime | None
    superseded_by: UUID | None
    state: State
