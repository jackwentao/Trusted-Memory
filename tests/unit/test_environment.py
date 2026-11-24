from importlib.metadata import version

from mem0 import Memory


def test_baseline_available() -> None:
    assert version("mem0ai") == "2.2.1"
    assert callable(Memory.add)
    assert callable(Memory.search)
