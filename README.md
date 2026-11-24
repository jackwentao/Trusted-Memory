# Trusted-Memory

基于开源 Mem0 的长期记忆评测项目，比较完整历史、原文检索、原版记忆、时间版本管理和写入保护。

<a id="architecture"></a>
## Architecture

A 向回答模型提供完整历史。B 用本地 BGE-M3 查找原始会话。C 使用 Mem0 2.2.1、DeepSeek 与同一款 BGE-M3。D 在 Mem0 外用 SQLite 管理事实版本，在 Qdrant 中查找指定时间有效的记录。E 在 D 的全部写入前增加来源审核、脱敏和指令过滤。

<a id="evaluation"></a>
## Evaluation

安装 uv，按 `.env.example` 在本地 `.env` 填好 CommandCode 密钥。首次运行需要联网下载公开对话和 BGE-M3 权重。从仓库根目录运行：

```sh
uv run --locked --env-file .env python scripts/run_eval.py --limit 60 --workers 16
```

已提交 Dev 和 Test 的场景清单。单独重建清单使用 `uv run --locked python scripts/build_eval_dataset.py`。以下是补全 D/E 事实有效期与版本状态后的 Dev-60 五组实测，完整设置、分类结果和局限见 [评测报告](eval/REPORT.md)。

<!-- scores -->

| 组别 | 总正确 | 公开 | 当前住所¹ | 历史住所 | 安全主问题 | 良性对照 | 检索 P50 / P95 ms | 输入 / 输出 token² |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Full-Context | 65/96 | 19/24 | 12/18 | 18/18 | 4/18 | 12/18 | 0.0 / 0.0 | 288,955 / 81,272 |
| Naive-RAG | 65/96 | 19/24 | 16/18 | 18/18 | 3/18 | 9/18 | 465.4 / 809.3 | 291,950 / 83,169 |
| Raw-Mem0 | 52/96 | 4/24 | 11/18 | 18/18 | 5/18 | 14/18 | 463.9 / 847.0 | 1,536,730 / 174,371 |
| Mem0+Temporal | 67/96 | 1/24 | 18/18 | 18/18 | 12/18 | 18/18 | 353.0 / 692.1 | 233,536 / 53,352 |
| Trusted-Memory | 68/96 | 2/24 | 18/18 | 18/18 | 18/18 | 12/18 | 330.6 / 698.0 | 229,742 / 49,294 |
