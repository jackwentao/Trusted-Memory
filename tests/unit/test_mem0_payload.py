from datetime import UTC, datetime
from uuid import uuid4

import pytest

from trusted_memory.memory.dual_timestamp_store import DualTimestampStore
from trusted_memory.schemas import Fact, Source

START = datetime(2020, 1, 1, tzinfo=UTC)
END = datetime(2020, 3, 1, tzinfo=UTC)


@pytest.mark.parametrize("governed", [False, True])
def test_import_restart_export(tmp_path, governed):
    successor = str(uuid4())
    meta = {"nested": {"labels": ["demo"]}}
    if governed:
        meta.update(
            event_time=START.isoformat(),
            valid_until=END.isoformat(),
            source_type="document",
            status="SUPERSEDED",
            superseded_by=successor,
        )
    payload = {
        "id": str(uuid4()),
        "memory": "old address",
        "user_id": "u",
        "agent_id": "a",
        "run_id": "r",
        "created_at": "2023-01-01T00:00:00Z",
        "updated_at": "2024-01-01T00:00:00Z",
        "metadata": meta,
    }
    context = {} if governed else {"event_time": START, "source": Source.USER}
    fact = Fact.from_mem0_payload(payload, attribute="address", **context)
    path = tmp_path / "memory.db"
    DualTimestampStore(path, lambda _: None).write(fact)
    store = DualTimestampStore(path, lambda _: None)
    saved = store.lookup("u", "address", START).to_mem0_payload()
    assert {k: v for k, v in saved.items() if k != "metadata"} == {
        k: v for k, v in payload.items() if k != "metadata"
    }
    assert saved["metadata"]["nested"] == meta["nested"]
    assert saved["metadata"]["valid_until"] == (END.isoformat() if governed else None)
    assert saved["metadata"]["status"] == ("SUPERSEDED" if governed else "ACTIVE")
    assert saved["metadata"]["superseded_by"] == (successor if governed else None)
    assert saved["metadata"]["event_time"] == START.isoformat()
    assert saved["metadata"]["source_type"] == ("document" if governed else "user")
    if governed:
        assert store.lookup("u", "address", END) is None


def test_import_preserves_agent_scope_without_user():
    payload = {
        "id": str(uuid4()),
        "agent_id": "assistant",
        "memory": "hello",
        "metadata": None,
    }
    fact = Fact.from_mem0_payload(payload, source=Source.TOOL, event_time=START)
    exported = fact.to_mem0_payload()
    assert "user_id" not in exported and exported["agent_id"] == "assistant"
    assert payload["metadata"] is None
