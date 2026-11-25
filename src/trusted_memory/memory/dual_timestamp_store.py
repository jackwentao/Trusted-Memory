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
            db.execute("BEGIN IMMEDIATE")
            cols = {r[1] for r in db.execute("PRAGMA table_info(facts)")}
            if cols and "is_exclusive" not in cols:
                db.execute("DROP VIEW IF EXISTS chain")
                db.execute("ALTER TABLE facts RENAME TO facts_legacy")
            db.execute("""CREATE TABLE IF NOT EXISTS facts(
                id TEXT PRIMARY KEY, owner TEXT NOT NULL, attribute TEXT NOT NULL,
                text TEXT NOT NULL, source TEXT NOT NULL, event TEXT NOT NULL,
                ingest TEXT NOT NULL, ready INTEGER NOT NULL DEFAULT 0,
                native TEXT, is_exclusive INTEGER NOT NULL CHECK(is_exclusive IN (0,1)))""")
            if cols and "is_exclusive" not in cols:
                native = "native" if "native" in cols else "NULL"
                db.execute(f"""INSERT INTO facts SELECT id,owner,attribute,text,source,event,
                           ingest,ready,{native},1 FROM facts_legacy""")
                db.execute("DROP TABLE facts_legacy")
            db.execute(
                "CREATE INDEX IF NOT EXISTS owner_key ON facts(owner, attribute)"
            )
            db.execute("""CREATE UNIQUE INDEX IF NOT EXISTS exclusive_event
                       ON facts(owner, attribute,event) WHERE is_exclusive=1""")
            db.execute("""CREATE VIEW IF NOT EXISTS chain AS SELECT *,
                CASE WHEN is_exclusive THEN lead(event) OVER timeline END AS valid_to,
                CASE WHEN is_exclusive THEN lead(id) OVER timeline END AS successor
                FROM facts WHERE ready=1 WINDOW timeline AS
                (PARTITION BY owner,attribute ORDER BY event,id)""")
            db.commit()

    def attributes(self, owner: str) -> dict[str, bool]:
        """Return the observed vocabulary for this owner alone."""
        with closing(sqlite3.connect(self.path)) as db:
            return {
                key: bool(mode)
                for key, mode in db.execute(
                    "SELECT DISTINCT attribute,is_exclusive FROM facts WHERE owner=? AND ready=1",
                    (owner,),
                )
            }

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
            int(fact.is_exclusive),
        )
        with closing(sqlite3.connect(self.path)) as db:
            with db:
                db.execute("BEGIN IMMEDIATE")
                if db.execute(
                    "SELECT 1 FROM facts WHERE owner=? AND attribute=? AND is_exclusive!=? LIMIT 1",
                    (fact.owner, fact.attribute, int(fact.is_exclusive)),
                ).fetchone():
                    raise ValueError("attribute cardinality conflict")
                db.execute(
                    "INSERT INTO facts(id,owner,attribute,text,source,event,native,is_exclusive,ingest) VALUES(?,?,?,?,?,?,?,?,?) ON CONFLICT(id) DO NOTHING",
                    (key, *values, _stamp(datetime.now(UTC))),
                )
                current = db.execute(
                    "SELECT owner,attribute,text,source,event,native,is_exclusive,ready FROM facts WHERE id=?",
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
                    bool(mode),
                )
                for i, o, a, t, s, e, native, mode in db.execute(
                    "SELECT id,owner,attribute,text,source,event,native,is_exclusive FROM facts WHERE ready=0"
                )
            ]

    def lookup(
        self, owner: str, attribute: str, as_of: datetime | None = None
    ) -> Version | None:
        """Select the version effective at a point in time, defaulting to now."""
        versions = self.effective(owner, as_of, attribute)
        if len(versions) > 1:
            raise ValueError("multiple effective values; use effective()")
        return versions[0] if versions else None

    def effective(
        self, owner: str, as_of: datetime | None = None, attribute: str | None = None
    ) -> list[Version]:
        """Read all eligible values, including concurrent observations of the same key."""
        now = _stamp(as_of or datetime.now(UTC))
        sql = """SELECT id,owner,attribute,text,source,event,ingest,valid_to,successor,native,is_exclusive
                 FROM chain WHERE owner=? AND event<=? AND (valid_to IS NULL OR valid_to>?)"""
        params = [owner, now, now]
        if attribute is not None:
            sql += " AND attribute=?"
            params.append(attribute)
        with closing(sqlite3.connect(self.path)) as db:
            return [
                v
                for row in db.execute(sql, params)
                if (v := self._version(row, now, as_of)) is not None
            ]

    @staticmethod
    def _version(row: tuple, now: str, as_of: datetime | None) -> Version | None:
        i, owner, attribute, t, s, e, ingest, end, nxt, native, mode = row
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
                bool(mode),
            ),
            datetime.fromisoformat(ingest),
            end_time,
            UUID(nxt) if nxt else None,
            State.ACTIVE if active else State.SUPERSEDED,
        )
