"""D/E experimental arms using Mem0's model, embedder and Qdrant backend."""

import json
import math
import re
import sqlite3
from contextlib import closing
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal
from uuid import NAMESPACE_URL, UUID, uuid5

from mem0 import Memory
from qdrant_client.models import Filter, HasIdCondition

from trusted_memory.llm.extraction import extract
from trusted_memory.memory.dual_timestamp_store import DualTimestampStore
from trusted_memory.schemas import Fact, Source, Version
from trusted_memory.security.middleware import JsonAuditLog, SecurityMiddleware


class GovernedMemory:
    """Use one persistent directory per experiment and arm."""

    def __init__(self, config: Path, root: Path, arm: Literal["D", "E"]) -> None:
        if arm not in ("D", "E"):
            raise ValueError("expected D or E")
        root = root / arm
        root.mkdir(parents=True, exist_ok=True)
        cfg = json.loads(config.read_text(encoding="utf8"))
        cfg["history_db_path"] = str(root / "history.sqlite")
        cfg["vector_store"]["config"]["path"] = str(root / "vectors")
        self._memory = Memory.from_config(cfg)
        self._store = DualTimestampStore(root / "facts.sqlite", self._index)
        self._gate = (
            SecurityMiddleware(JsonAuditLog(root / "audit.jsonl"))
            if arm == "E"
            else None
        )
        self._reviews = root / "review.sqlite"
        with closing(sqlite3.connect(self._reviews)) as db:
            db.execute(
                "CREATE TABLE IF NOT EXISTS candidates(id TEXT PRIMARY KEY, owner TEXT, body TEXT, anchor TEXT, source TEXT, confirmed INTEGER DEFAULT 0)"
            )

    def _index(self, fact: Fact) -> None:
        payload = fact.to_mem0_payload()
        payload["data"] = payload.pop("memory")
        embedding = self._memory.embedding_model.embed(fact.text, "add")
        self._memory.vector_store.insert(
            [embedding], payloads=[payload], ids=[str(fact.id)]
        )

    def remember(
        self, text: str, *, owner: str, source: Source, session_time: datetime
    ) -> str:
        """Return the disposition; review receipts include the candidate identifier."""
        if not owner or session_time.utcoffset() is None:
            raise ValueError("owner and aware session_time required")
        if self._gate:
            decision = self._gate.inspect(text, source=source)
            if decision.candidate_text is not None:
                with closing(sqlite3.connect(self._reviews)) as db, db:
                    db.execute(
                        "INSERT INTO candidates(id,owner,body,anchor,source) VALUES(?,?,?,?,?)",
                        (
                            str(decision.audit.id),
                            owner,
                            decision.candidate_text,
                            session_time.isoformat(),
                            source.value,
                        ),
                    )
                return "PENDING_REVIEW:" + str(decision.audit.id)
            if decision.approved_text is None:
                return "BLOCKED"
            text = decision.approved_text
        return self._publish(text, owner, source, session_time)

    def _publish(self, text: str, owner: str, source: Source, anchor: datetime) -> str:
        rows = extract(self._memory.llm, text, anchor)
        for row in rows:
            key = uuid5(
                NAMESPACE_URL,
                json.dumps(
                    [
                        owner,
                        row["attribute"],
                        row["value"],
                        row["event"].isoformat(),
                        source.value,
                    ]
                ),
            )
            fact = Fact(
                key,
                owner,
                row["attribute"],
                row["attribute"] + " " + row["value"],
                source,
                row["event"],
                {
                    "user_id": owner,
                    "metadata": {
                        "session_time": anchor.isoformat(),
                        "precision": row["precision"],
                        "time_expression": row["phrase"],
                        "evidence": text,
                        "security_screened": self._gate is not None,
                        "redaction_placeholders": sorted(
                            set(
                                re.findall(
                                    r"<(?:PHONE|ID_CARD|BANK_CARD|SECRET)>", text
                                )
                            )
                        )
                        if self._gate
                        else [],
                    },
                },
                row["is_exclusive"],
            )
            self._store.write(fact)
        return "SAVED" if rows else "NO_FACT"

    def confirm(self, owner: str, candidate: UUID) -> str:
        """Publish a sanitized candidate after explicit owner confirmation."""
        with closing(sqlite3.connect(self._reviews)) as db, db:
            row = db.execute(
                "SELECT body,anchor,source,confirmed FROM candidates WHERE owner=? AND id=?",
                (owner, str(candidate)),
            ).fetchone()
            if not row:
                raise KeyError("unknown candidate")
            if row[3]:
                return "CONFIRMED"
            outcome = self._publish(
                row[0], owner, Source(row[2]), datetime.fromisoformat(row[1])
            )
            if outcome == "SAVED":
                db.execute(
                    "UPDATE candidates SET confirmed=1 WHERE id=?", (str(candidate),)
                )
                return "CONFIRMED"
            return outcome

    def recover(self) -> None:
        """Replay pre-registered facts without another extraction."""
        for fact in self._store.pending():
            self._store.write(fact)

    def recall(
        self, query: str, *, owner: str, as_of: datetime | None = None, k: int = 5
    ) -> list[Version]:
        """Filter by owner and effective interval before ranking eligible points."""
        instant = as_of or datetime.now(UTC)
        versions = {str(v.fact.id): v for v in self._store.effective(owner, as_of)}
        # A month-level transition has no known exact day inside that month.
        versions = {
            key: version
            for key, version in versions.items()
            if not (
                (version.fact.native or {}).get("metadata", {}).get("precision")
                == "month"
                and instant.astimezone(
                    datetime.fromisoformat(
                        version.fact.native["metadata"]["session_time"]
                    ).tzinfo
                ).strftime("%Y-%m")
                == version.fact.event_time.astimezone(
                    datetime.fromisoformat(
                        version.fact.native["metadata"]["session_time"]
                    ).tzinfo
                ).strftime("%Y-%m")
            )
        }
        if not versions or k <= 0:
            return []
        store = self._memory.vector_store
        points = store.client.query_points(
            store.collection_name,
            query=self._memory.embedding_model.embed(query, "search"),
            query_filter=Filter(must=[HasIdCondition(has_id=list(versions))]),
            limit=len(versions),
        ).points

        def score(point):
            age = (
                instant - versions[str(point.id)].fact.event_time
            ).total_seconds() / 86400
            return point.score + 0.05 * math.exp(-age / 365)

        return [
            versions[str(p.id)] for p in sorted(points, key=score, reverse=True)[:k]
        ]

    def close(self) -> None:
        """Release the local database clients."""
        self._memory.close()
        self._memory.vector_store.client.close()
        telemetry = getattr(self._memory, "_telemetry_vector_store", None)
        if (
            telemetry is not None
            and telemetry.client is not self._memory.vector_store.client
        ):
            telemetry.client.close()
