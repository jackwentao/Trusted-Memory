from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID

import pytest

from trusted_memory.memory.governed import GovernedMemory
from trusted_memory.schemas import Source


@pytest.mark.slow
def test_real_history_review_and_recovery(tmp_path, monkeypatch):
    cfg = Path("configs/exp/commandcode-bge-m3.yaml")

    def open_arm():
        return GovernedMemory(cfg, tmp_path, "E")

    anchor = datetime(2025, 4, 15, tzinfo=UTC)
    m = open_arm()

    def save(text, source=Source.USER):
        return m.remember(text, owner="demo", source=source, session_time=anchor)

    try:
        assert save("我上个月搬到上海。") == "SAVED"
        index = m._store.upsert

        def lose_ack(fact):
            index(fact)
            raise OSError("simulated lost acknowledgement")

        monkeypatch.setattr(m._store, "upsert", lose_ack)
        with pytest.raises(OSError, match="simulated"):
            save("我在2025-01-01搬到北京。")
        assert not m.recall(
            "住在哪", owner="demo", as_of=datetime(2025, 2, 1, tzinfo=UTC)
        )
    finally:
        m.close()
    m = open_arm()
    try:
        m.recover()

        def city(when=None):
            return m.recall("我住在哪里", owner="demo", as_of=when)[0].fact

        assert "北京" in city(datetime(2025, 2, 1, tzinfo=UTC)).text
        assert not m.recall(
            "住在哪", owner="demo", as_of=datetime(2025, 3, 15, tzinfo=UTC)
        )
        assert "上海" in city().text
        assert city().native["metadata"]["precision"] == "month"
        receipt = save("我今天搬到广州。电话13800001234。", Source.DOCUMENT)
        assert receipt.startswith("PENDING_REVIEW:")
        assert "上海" in city().text
        assert m.confirm("demo", UUID(receipt.split(":")[1])) == "CONFIRMED"
        assert "广州" in city().text
    finally:
        m.close()

    for path in (tmp_path / "E").rglob("*"):
        if path.is_file():
            assert b"13800001234" not in path.read_bytes()
    d = GovernedMemory(cfg, tmp_path, "D")
    try:
        assert (
            d.remember(
                "我今天搬到广州。",
                owner="demo",
                source=Source.DOCUMENT,
                session_time=anchor,
            )
            == "SAVED"
        )
        assert "广州" in d.recall("我的住所", owner="demo")[0].fact.text
    finally:
        d.close()


@pytest.mark.slow
def test_real_coexisting_purchases(tmp_path):
    memory = GovernedMemory(Path("configs/exp/commandcode-bge-m3.yaml"), tmp_path, "E")
    memory._memory.llm.config.temperature = 0
    anchor = datetime(2025, 4, 15, tzinfo=UTC)
    try:
        assert (
            memory.remember(
                "我今天买了一把吉他。",
                owner="shopper",
                source=Source.USER,
                session_time=anchor,
            )
            == "SAVED"
        )
        assert (
            memory.remember(
                "我今天还买了一架钢琴。",
                owner="shopper",
                source=Source.USER,
                session_time=anchor,
            )
            == "SAVED"
        )
        versions = memory.recall("今天买了什么？", owner="shopper", as_of=anchor)
        purchases = [
            v for v in versions if "吉他" in v.fact.text or "钢琴" in v.fact.text
        ]
        assert len({v.fact.attribute for v in purchases}) == 1
        assert any("吉他" in v.fact.text for v in purchases)
        assert any("钢琴" in v.fact.text for v in purchases)
        assert all(not v.fact.is_exclusive and v.valid_to is None for v in purchases)
    finally:
        memory.close()
