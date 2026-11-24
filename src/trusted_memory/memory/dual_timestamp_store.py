"""SQLite owns visibility. The vector sink must upsert by stable fact ID."""

import sqlite3
from collections.abc import Callable
from contextlib import closing
from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID

from trusted_memory.schemas import Fact, Source, State, Version


def _stamp(value: datetime) -> str:
    if value.utcoffset() is None:
        raise ValueError("timezone required")
    return value.astimezone(UTC).isoformat(timespec="microseconds")


class DualTimestampStore:
    """Publish a version only after an idempotent vector upsert succeeds."""

    def __init__(self, path: Path, upsert: Callable[[Fact], None]) -> None:
        self.path = path
        self.upsert = upsert
        with closing(sqlite3.connect(path)) as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS facts(
                    id TEXT PRIMARY KEY, owner TEXT NOT NULL, attribute TEXT NOT NULL,
                    text TEXT NOT NULL, source TEXT NOT NULL, event TEXT NOT NULL,
                    ingest TEXT NOT NULL, ready INTEGER NOT NULL DEFAULT 0,
                    UNIQUE(owner,attribute,event)
                );
                CREATE VIEW IF NOT EXISTS chain AS SELECT *,
                    lead(event) OVER timeline AS valid_to,
                    lead(id) OVER timeline AS successor
                    FROM facts WHERE ready=1
                    WINDOW timeline AS (PARTITION BY owner,attribute ORDER BY event);
            """)

    def write(self, fact: Fact) -> None:
        """Persist intent, then publish. Retrying the same fact resumes after failure."""
        key = str(fact.id)
        values = (
            fact.owner,
            fact.attribute,
            fact.text,
            Source(fact.source).value,
            _stamp(fact.event_time),
        )
        with closing(sqlite3.connect(self.path)) as db:
            with db:
                db.execute(
                    "INSERT INTO facts(id,owner,attribute,text,source,event,ingest) VALUES(?,?,?,?,?,?,?) ON CONFLICT(id) DO NOTHING",
                    (key, *values, _stamp(datetime.now(UTC))),
                )
                current = db.execute(
                    "SELECT owner,attribute,text,source,event,ready FROM facts WHERE id=?",
                    (key,),
                ).fetchone()
                if current[:-1] != values:
                    raise ValueError("fact ID reused with different content")
                if current[-1]:
                    return
            self.upsert(fact)
            with db:
                db.execute("UPDATE facts SET ready=1 WHERE id=?", (key,))

    def pending(self) -> list[Fact]:
        """Return durable retry inputs after restart."""
        with closing(sqlite3.connect(self.path)) as db:
            return [
                Fact(UUID(i), o, a, t, Source(s), datetime.fromisoformat(e))
                for i, o, a, t, s, e in db.execute(
                    "SELECT id,owner,attribute,text,source,event FROM facts WHERE ready=0"
                )
            ]

    def lookup(
        self, owner: str, attribute: str, as_of: datetime | None = None
    ) -> Version | None:
        """Select the version effective at a point in time, defaulting to now."""
        now = _stamp(as_of or datetime.now(UTC))
        with closing(sqlite3.connect(self.path)) as db:
            row = db.execute(
                "SELECT id,text,source,event,ingest,valid_to,successor FROM chain WHERE owner=? AND attribute=? AND event<=? AND (valid_to IS NULL OR valid_to>?)",
                (owner, attribute, now, now),
            ).fetchone()
        if row is None:
            return None
        i, t, s, e, ingest, end, nxt = row
        end_time = datetime.fromisoformat(end) if end else None
        active = datetime.fromisoformat(e) <= datetime.now(UTC) and (
            not end_time or datetime.now(UTC) < end_time
        )
        return Version(
            Fact(UUID(i), owner, attribute, t, Source(s), datetime.fromisoformat(e)),
            datetime.fromisoformat(ingest),
            end_time,
            UUID(nxt) if nxt else None,
            State.ACTIVE if active else State.SUPERSEDED,
        )
