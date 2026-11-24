import json
from pathlib import Path
from uuid import uuid4

from mem0 import Memory

config = json.loads(Path("configs/exp/commandcode-bge-m3.yaml").read_text())
run_id = uuid4().hex
Path("data").mkdir(exist_ok=True)
memory = Memory.from_config(config)
added = memory.add(
    "我住在上海，喜欢面食。",
    user_id="smoke-user",
    run_id=run_id,
)
found = memory.search(
    "我住在哪里？",
    filters={"user_id": "smoke-user", "run_id": run_id},
    top_k=5,
)
result = {"config": config, "added": added, "found": found}
output = Path("eval/results/20261124-C-smoke-bge-m3") / run_id
output.mkdir(parents=True, exist_ok=True)
(output / "raw.json").write_text(
    json.dumps(result, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
)
print(
    json.dumps(
        {
            "saved_result_file": str(output / "raw.json"),
            "found_count": len(found["results"]),
        },
        ensure_ascii=True,
    )
)
