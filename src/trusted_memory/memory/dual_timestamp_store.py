"""SQLite owns visibility. The vector sink must upsert by stable fact ID."""

import json
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

            # Additive migration preserves existing facts and interrupted writes.
            db.execute("BEGIN IMMEDIATE")
            if "native" not in {r[1] for r in db.execute("PRAGMA table_info(facts)")}:
                db.execute("ALTER TABLE facts ADD COLUMN native TEXT")
            db.commit()

    def write(self, fact: Fact) -> None:
        """Persist intent, then publish. Retrying the same fact resumes after failure."""
        key = str(fact.id)
        values = (
            fact.owner,
            fact.attribute,
            fact.text,
            Source(fact.source).value,
            _stamp(fact.event_time),
            json.dumps(fact.native, sort_keys=True, allow_nan=False)
            if fact.native is not None
            else None,
        )
        with closing(sqlite3.connect(self.path)) as db:
            with db:
                db.execute(
                    "INSERT INTO facts(id,owner,attribute,text,source,event,native,ingest) VALUES(?,?,?,?,?,?,?,?) ON CONFLICT(id) DO NOTHING",
                    (key, *values, _stamp(datetime.now(UTC))),
                )
                current = db.execute(
                    "SELECT owner,attribute,text,source,event,native,ready FROM facts WHERE id=?",
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
                Fact(
                    UUID(i),
                    o,
                    a,
                    t,
                    Source(s),
                    datetime.fromisoformat(e),
                    json.loads(native) if native else None,
                )
                for i, o, a, t, s, e, native in db.execute(
                    "SELECT id,owner,attribute,text,source,event,native FROM facts WHERE ready=0"
                )
            ]

    def lookup(
        self, owner: str, attribute: str, as_of: datetime | None = None
    ) -> Version | None:
        """Select the version effective at a point in time, defaulting to now."""
        now = _stamp(as_of or datetime.now(UTC))
        with closing(sqlite3.connect(self.path)) as db:
            row = db.execute(
                "SELECT id,text,source,event,ingest,valid_to,successor,native FROM chain WHERE owner=? AND attribute=? AND event<=? AND (valid_to IS NULL OR valid_to>?)",
                (owner, attribute, now, now),
            ).fetchone()
        if row is None:
            return None
        i, t, s, e, ingest, end, nxt, native = row
        end_time = datetime.fromisoformat(end) if end else None
        meta = (json.loads(native).get("metadata") or {}) if native else {}
        imported_end = (
            datetime.fromisoformat(meta["valid_until"])
            if meta.get("valid_until")
            else None
        )
        if imported_end and (end_time is None or imported_end < end_time):
            end_time, nxt = imported_end, meta.get("superseded_by")
        elif end_time is None:
            nxt = meta.get("superseded_by")
        if end_time and datetime.fromisoformat(now) >= end_time:
            return None
        if meta.get("status") == State.SUPERSEDED and not end_time and as_of is None:
            return None
        active = (
            meta.get("status") != State.SUPERSEDED
            and datetime.fromisoformat(e) <= datetime.now(UTC)
            and (not end_time or datetime.now(UTC) < end_time)
        )
        return Version(
            Fact(
                UUID(i),
                owner,
                attribute,
                t,
                Source(s),
                datetime.fromisoformat(e),
                json.loads(native) if native else None,
            ),
            datetime.fromisoformat(ingest),
            end_time,
            UUID(nxt) if nxt else None,
            State.ACTIVE if active else State.SUPERSEDED,
        )
