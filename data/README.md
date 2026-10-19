# train-jev 评测数据

这里发布本项目固定评测的索引、标签和许可已核验的输入。**这些是评测集，不要加入训练数据。**

| 数据集 | 规模 | 内容 |
|---|---:|---|
| [train-jev-pk](train-jev-pk/) | 2,494 题，22 个任务 | 本项目组装的公开金标、分类与原创规则评测 |
| [train-jev-classification-pk](train-jev-classification-pk/) | 1,300 题，13 个场景 | 13 个公开分类来源各 100 题 |

因部分来源的原文再分发许可或署名要求尚未核验，两个目录都包含完整索引和标签，但只包含部分原文。`manifest.json` 的 `available_inputs` / `withheld_inputs` 标明覆盖情况；缺少 `state` 的行需要自行取得上游数据后恢复。不能把已附原文的子集分数当成整套评测的分数。

```bash
# 从上游构建每个场景 600 行的来源文件，供索引恢复使用
python scripts/pk_build.py --out data/pk

# 核验并恢复完整输入（不访问网络、不重新抽样）
python scripts/prepare_eval_data.py --dataset data/train-jev-pk \
  --source-dir data/pk --out eval/train-jev-pk-inputs.jsonl

# 本地模型评测；需要 GPU，保持原问题和 score 格式
python scripts/evaluate_pk.py --dataset data/train-jev-pk --source-dir data/pk \
  --base Qwen/Qwen3.5-4B --adapter shibing624/train-jev-4b-lora --out eval/train-jev-pk-4b
```

恢复时核对题目顺序、原生标签和模型输入哈希，不会悄悄替换不匹配的数据。上游数据变化后应使用与索引匹配的快照，不能重新抽样代替。处理文件写到 `eval/`，不会被提交。

脚本、索引整理方式按仓库许可发布；附带的上游数据分别遵循其原许可，不由仓库 Apache-2.0 统一覆盖。具体来源与署名见各数据集 README。Decision Index 和 Kev transfer-v4 继续从各自的上游仓库获取，见 [评测文档](../docs/evaluation.md)。
