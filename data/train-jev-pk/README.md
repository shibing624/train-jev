# train-jev-pk

本项目组装的固定决策评测：**2,494 题、22 个任务**，覆盖公开金标、旧分类来源及本项目的独立合成规则。它对应 README 中原称“封存同题 PK”的结果；改名不改变题目、标签或分数。

| 部分 | 题数 | 任务数 | 原文发布情况 |
|---|---:|---:|---|
| 公开金标 | 594 | 4 | 已包含 |
| 原创精确规则 | 1,000 | 6 | 已包含 |
| 旧分类来源 | 900 | 12 | 仅发布索引与标签，原文自行取得 |

`rows.jsonl` 是全部 2,494 题的有序索引：`id`、`task`、`kind`、`question`、`options`、`target`、`group_id`、`input_sha256` 和 `source`。可分发的题另含 `state`。输入哈希只覆盖题型、state、question 和选项，不把标签或来源元数据送给模型。

## 使用

```bash
python scripts/pk_build.py --out data/pk
python scripts/prepare_eval_data.py --dataset data/train-jev-pk \
  --source-dir data/pk --out eval/train-jev-pk-inputs.jsonl
python scripts/evaluate_pk.py --dataset data/train-jev-pk --source-dir data/pk \
  --base Qwen/Qwen3.5-4B --adapter shibing624/train-jev-4b-lora --out eval/train-jev-pk-4b
```

恢复缺失输入需要来源目录下的 `{scenario}.jsonl` 文件（格式为 `text` / `label`，每个来源最多 600 行）。索引保留原行号、标签、完整模型输入哈希；任意不匹配都会报错，不用新抽样替代。只恢复输入，不重新拟合温度。

报告按 22 个任务等权计算宏平均准确率、NLL、Brier 和 ECE；不要与按行平均混淆。未恢复全部输入时，不输出整套分数。

## 来源与许可

公开金标保持原任务内容并统一为带类型问题：

| 来源 | 题数 | 许可 / 署名 |
|---|---:|---|
| [PAWS（Google Research）](https://huggingface.co/datasets/google-research-datasets/paws) | 150 | 上游宽松条款；保留 Google LLC 署名 |
| [AQuA-RAT（DeepMind）](https://huggingface.co/datasets/deepmind/aqua_rat) | 144 | Apache-2.0；来源与原题 ID 在索引中保留 |
| [QASC（AllenAI）](https://huggingface.co/datasets/allenai/qasc) | 150 | CC-BY-4.0，署名 Allen Institute for AI / QASC 作者 |
| [HelpSteer2（NVIDIA）](https://huggingface.co/datasets/nvidia/HelpSteer2) | 150 | CC-BY-4.0，署名 NVIDIA / HelpSteer2 作者 |
| train-jev 独立合成规则 | 1,000 | CC0-1.0，规则样本由本项目发布到公共领域 |

六类规则为路由、显式证据、指代、退款换算、日历和条件计数，反事实同组样本保留 `group_id`。规则标签经过两个独立实现校验。

旧分类来源：AG News、DAIR Emotion、BANKING77、MASSIVE en/zh、XNLI en/zh、TNEWS、SST-5、Yelp、BoolQ、Civil Comments。具体公开仓库和许可状态在每行 `source` 中；未核验许可的原文不包含在本仓库，不能因标签是公开的就认为原文可自由再分发。

部分题来自与新增训练相同的公开来源或规则族，但不是同一道题；它不是完全跨来源的泛化评测。所有题已用于模型比较，不应再用于选点、训练或温度拟合。统计口径与局限见 [结果文档](../../docs/results.md)。
