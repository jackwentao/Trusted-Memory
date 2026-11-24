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

已提交 Dev 和 Test 的场景清单。单独重建清单使用 `uv run --locked python scripts/build_eval_dataset.py`。以下是本仓库首轮开发集实测，完整设置、分类结果和局限见 [评测报告](eval/REPORT.md)。

<!-- scores -->

| 组别 | 总正确 | 公开 | 当前住所¹ | 历史住所 | 安全主问题 | 良性对照 | 检索 P95 ms | 输入 / 输出 token² |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Full-Context | 66/96 | 18/24 | 14/18 | 17/18 | 5/18 | 12/18 | 0.0 | 343,626 / 80,081 |
| Naive-RAG | 63/96 | 21/24 | 9/18 | 17/18 | 3/18 | 13/18 | 759.9 | 267,414 / 73,677 |
| Raw-Mem0 | 59/96 | 4/24 | 17/18 | 18/18 | 6/18 | 14/18 | 766.8 | 1,532,776 / 171,185 |
| Mem0+Temporal | 43/96 | 1/24 | 16/18 | 0/18 | 11/18 | 15/18 | 686.3 | 235,084 / 86,481 |
| Trusted-Memory | 39/96 | 2/24 | 16/18 | 0/18 | 12/18 | 9/18 | 637.2 | 232,267 / 83,869 |
