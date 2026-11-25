import json
from dataclasses import replace
from datetime import UTC, datetime
from unittest.mock import Mock
from uuid import uuid4

import pytest

from trusted_memory.llm.extraction import extract
from trusted_memory.memory.dual_timestamp_store import DualTimestampStore
from trusted_memory.schemas import Fact, Source, State

DAY = datetime(2025, 2, 1, tzinfo=UTC)


@pytest.mark.parametrize(
    ("key", "exclusive"),
    [
        ("user.residence", True),
        ("user.phone", True),
        ("user.purchase", False),
        ("user.interest", False),
        ("user.daily_event", False),
    ],
)
def test_open_history(tmp_path, key, exclusive):
    store = DualTimestampStore(tmp_path / "facts", lambda _: None)
    first = Fact(
        uuid4(), "alice", key, "first", Source.USER, DAY, is_exclusive=exclusive
    )
    second = replace(
        first,
        id=uuid4(),
        text="second",
        event_time=DAY.replace(month=3) if exclusive else DAY,
    )
    for fact in (second, first, first):
        store.write(fact)
    current = store.effective("alice", attribute=key)
    assert {v.fact.text for v in current} == (
        {"second"} if exclusive else {"first", "second"}
    )
    assert all(
        v.state == State.ACTIVE
        and v.to_mem0_payload()["metadata"]["is_exclusive"] is exclusive
        for v in current
    )
    if exclusive:
        old = store.lookup("alice", key, DAY)
        assert old.superseded_by == second.id and old.state == State.SUPERSEDED
    assert not store.effective("bob")
    with pytest.raises(ValueError, match="cardinality"):
        store.write(replace(first, id=uuid4(), is_exclusive=not exclusive))


@pytest.mark.parametrize(
    ("date", "expected", "precision"),
    [
        ("2025-01", DAY.replace(month=1), "month"),
        ("2025-01-31", DAY.replace(month=1, day=31), "day"),
        (None, DAY, "observation"),
        ("2025-02-30", DAY, "observation"),
        ("2030-01", DAY, "observation"),
        ([], DAY, "observation"),
    ],
)
def test_paraphrase_and_date_guard(date, expected, precision):
    model = Mock()
    model.generate_response.return_value = json.dumps(
        {
            "facts": [
                {
                    "attribute_key": "user.purchase",
                    "value": "acquired a guitar",
                    "event_time": date,
                    "is_exclusive": True,
                },
                {"attribute_key": "bad path", "value": "bad"},
            ]
        }
    )
    [fact] = extract(model, "I bought an instrument.", DAY)
    assert (fact["value"], fact["event"], fact["precision"], fact["is_exclusive"]) == (
        "acquired a guitar",
        expected,
        precision,
        False,
    )
    model.generate_response.assert_called_once()


def test_profile_requires_explicit_boolean_true():
    model = Mock()
    model.generate_response.return_value = json.dumps(
        {
            "facts": [
                {
                    "attribute_key": "profile.residence",
                    "value": "Paris",
                    "is_exclusive": mode,
                }
                for mode in [True, False, "true", None]
            ]
        }
    )
    records = extract(model, "I moved to Paris", DAY)
    assert [r["is_exclusive"] for r in records] == [True, False, False, False]
    assert records[0]["attribute"] == "profile.residence"


def test_legacy_history_and_pending_writes_survive_upgrade(tmp_path):
    import sqlite3

    path = tmp_path / "old.sqlite"
    old, pending = uuid4(), uuid4()
    with sqlite3.connect(path) as db:
        db.execute("""CREATE TABLE facts(id TEXT PRIMARY KEY,owner TEXT,attribute TEXT,
            text TEXT,source TEXT,event TEXT,ingest TEXT,ready INTEGER,native TEXT,
            UNIQUE(owner,attribute,event))""")
        for i, ready, month in [(old, 1, 1), (pending, 0, 2)]:
            stamp = DAY.replace(month=month).isoformat(timespec="microseconds")
            db.execute(
                "INSERT INTO facts VALUES(?,?,?,?,?,?,?,?,?)",
                (
                    str(i),
                    "u",
                    "legacy.key",
                    str(month),
                    "user",
                    stamp,
                    stamp,
                    ready,
                    None,
                ),
            )
    store = DualTimestampStore(path, lambda _: None)
    assert store.lookup("u", "legacy.key").fact.id == old
    [retry] = store.pending()
    assert retry.id == pending and retry.is_exclusive
    store.write(retry)
    assert store.lookup("u", "legacy.key").fact.id == pending
    assert (
        store.lookup("u", "legacy.key", DAY.replace(month=1)).superseded_by == pending
    )
    assert store.attributes("u") == {"legacy.key": True}


def test_invalid_json_is_not_retried():
    model = Mock()
    model.generate_response.return_value = "{"
    with pytest.raises(ValueError, match="invalid extraction JSON"):
        extract(model, "Hello", DAY)
    model.generate_response.assert_called_once()


def test_snapshot_scope_and_exact_legacy_reuse(tmp_path):
    store = DualTimestampStore(tmp_path / "snapshot.db", lambda _: None)
    first = Fact(uuid4(), "a", "legacy.home", "legacy.home 北京", Source.USER, DAY)
    latest = replace(
        first, id=uuid4(), text="legacy.home 上海", event_time=DAY.replace(month=3)
    )
    for fact in (
        first,
        latest,
        replace(first, id=uuid4(), owner="b"),
        replace(first, id=uuid4(), attribute="user.pet", is_exclusive=False),
    ):
        store.write(fact)
    snapshot = store.active_exclusive_snapshot("a")
    assert snapshot == {"legacy.home": "上海"}
    model = Mock()
    model.generate_response.return_value = json.dumps(
        {
            "facts": [
                {
                    "attribute_key": key,
                    "value": val,
                    "is_exclusive": True,
                    "event_time": "2025-04-01",
                }
                for key, val in [("legacy.home", "广州"), ("profile.hometown", "上海")]
            ]
        }
    )
    result = extract(model, "我搬到了广州，老家在上海", DAY.replace(month=5), snapshot)
    messages = model.generate_response.call_args.kwargs["messages"]
    assert json.loads(messages[1]["content"])["active_exclusive_snapshot"] == snapshot
    for r in result:
        store.write(
            replace(
                first,
                id=uuid4(),
                attribute=r["attribute"],
                text=r["value"],
                event_time=r["event"],
                is_exclusive=r["is_exclusive"],
            )
        )
    assert store.active_exclusive_snapshot("a") == {
        "legacy.home": "广州",
        "profile.hometown": "上海",
    }
