import sqlite3
from dataclasses import replace
from datetime import UTC, datetime
from uuid import uuid4

import pytest

from trusted_memory.memory.dual_timestamp_store import DualTimestampStore
from trusted_memory.schemas import Fact, Source, State


def at(month):
    return datetime(2020, month, 1, tzinfo=UTC)


def city(month, text):
    return Fact(uuid4(), "user", "city", text, Source.USER, at(month))


def test_late_arrival_splices_history_and_preserves_current(tmp_path):
    store = DualTimestampStore(tmp_path / "m.db", lambda _: None)
    jan, mar, feb = city(1, "北京"), city(3, "上海"), city(2, "南京")
    for fact in [jan, mar, feb]:
        store.write(fact)
    first = store.lookup("user", "city", at(1))
    assert (first.fact, first.valid_to, first.superseded_by, first.state) == (
        jan,
        at(2),
        feb.id,
        State.SUPERSEDED,
    )
    middle = store.lookup("user", "city", at(2))
    assert middle.fact == feb and middle.superseded_by == mar.id
    latest = store.lookup("user", "city")
    assert latest.fact == mar and latest.state == State.ACTIVE
    assert middle.ingest_time >= latest.ingest_time
    assert store.lookup("other", "city") is None


def test_vector_ack_lost_then_restart_and_retry(tmp_path):
    path = tmp_path / "m.db"
    indexed = {}
    store = DualTimestampStore(path, lambda f: indexed.update({f.id: f}))
    old, new = city(1, "北京"), city(2, "上海")
    store.write(old)

    def lost_ack(fact):
        indexed[fact.id] = fact
        raise OSError

    with pytest.raises(OSError):
        DualTimestampStore(path, lost_ack).write(new)
    assert store.lookup("user", "city").fact == old
    assert store.pending() == [new]
    store.write(store.pending()[0])
    assert store.pending() == [] and store.lookup("user", "city").fact == new
    before = store.lookup("user", "city")
    DualTimestampStore(path, lost_ack).write(new)
    assert store.lookup("user", "city") == before and len(indexed) == 2


def test_conflicting_replays_preserve_history(tmp_path):
    store = DualTimestampStore(tmp_path / "m.db", lambda _: None)
    original = city(1, "北京")
    store.write(original)
    with pytest.raises(ValueError):
        store.write(replace(original, text="广州"))
    with pytest.raises(sqlite3.IntegrityError):
        store.write(city(1, "广州"))
    with pytest.raises(ValueError):
        store.write(replace(original, event_time=datetime(2020, 1, 1)))
    assert store.lookup("user", "city").fact == original
