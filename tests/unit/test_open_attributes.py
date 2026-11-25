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


def test_open_extraction_verifies_each_fact_independently():
    fact = {
        "attribute_key": "user.music.instrument",
        "is_exclusive": False,
        "value": "violin",
        "time": "yesterday",
    }
    model = Mock()
    model.generate_response.return_value = json.dumps(
        {
            "facts": [
                fact,
                dict(fact, value="piano"),
                dict(fact, is_exclusive="false"),
                dict(fact, time="tomorrow"),
                dict(fact, attribute_key="bad key"),
            ]
        }
    )
    result = extract(model, "I played violin yesterday", DAY)
    assert len(result) == 1 and result[0]["event"] == datetime(2025, 1, 31, tzinfo=UTC)
    assert result[0]["is_exclusive"] is False
