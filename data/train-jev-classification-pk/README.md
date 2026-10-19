# train-jev-classification-pk

固定的 **13 个场景 × 100 题，共 1,300 题**，对应 README 的“13 数据集分类 PK”。本项目组装和冻结评测输入，原始内容属于各上游来源。

发布完整题目索引、标签集合和可分发输入；目前附带 500 题原文，剩余 800 题需要自行取得来源文件恢复。`manifest.json` 列出覆盖与许可情况；改名不改变任何题目或分数。

## 数据格式与运行

`rows.jsonl` 每行包含固定的 `id`、任务名、问题、选项、target、来源行号和模型输入哈希；可分发的行另含 `state`。5 级评分场景依照原 PK 协议转为 `choice`，文本限制为前 4,000 个字符。

```bash
# 下载并构建来源文件，每个来源最多 600 行
python scripts/pk_build.py --out data/pk

# 从原行号恢复同一套题，检查标签和输入哈希
python scripts/prepare_eval_data.py --dataset data/train-jev-classification-pk \
  --source-dir data/pk --out eval/train-jev-classification-pk-inputs.jsonl

# 用发布模型的温度评测
python scripts/evaluate_pk.py --dataset data/train-jev-classification-pk --source-dir data/pk \
  --base Qwen/Qwen3.5-4B --adapter shibing624/train-jev-4b-lora --out eval/classification-pk-4b
```

如果上游快照变化，脚本会拒绝输入哈希不一致的行。不要换题、忽略缺失输入或把已附原文的子集分数称为 1,300 题结果。每个任务均为 100 题，因此宏平均准确率等于按行平均；历史分类 PK 的 ECE 是全部题目合并后计算的 10-bin ECE，查看报告的 `all_rows.ece10`，不要用各任务 ECE 的平均数替代。

## 来源与再分发

| 场景 | 来源 | 原文是否包含 / 许可 |
|---|---|---|
| ag_news | [AG News](https://huggingface.co/datasets/fancyzhx/ag_news) | 否，许可待核验 |
| dair_emotion | [DAIR Emotion](https://huggingface.co/datasets/dair-ai/emotion) | 否，上游标记 other |
| banking77 | [BANKING77（PolyAI）](https://huggingface.co/datasets/legacy-datasets/banking77) | 是，CC-BY-4.0，署名 PolyAI |
| massive_en / massive_zh | [MASSIVE（Amazon）](https://huggingface.co/datasets/mteb/amazon_massive_intent) | 是，Apache-2.0，署名 Amazon MASSIVE |
| xnli_en / xnli_zh | [XNLI](https://huggingface.co/datasets/facebook/xnli) | 否，许可与署名待核验 |
| tnews_zh | [CLUE TNEWS](https://huggingface.co/datasets/clue/clue) | 否，许可待核验 |
| sst5 | [SST-5](https://huggingface.co/datasets/SetFit/sst5) | 否，许可待核验 |
| yelp_stars | [Yelp Review Full](https://huggingface.co/datasets/Yelp/yelp_review_full) | 否，上游标记 other |
| boolq | [BoolQ](https://huggingface.co/datasets/google/boolq) | 否，CC-BY-SA-3.0 原文署名待核验 |
| prompt_injection | [Prompt Injections（deepset）](https://huggingface.co/datasets/deepset/prompt-injections) | 是，Apache-2.0，署名 deepset |
| toxicity | [Civil Comments](https://huggingface.co/datasets/google/civil_comments) | 是，CC0-1.0 |

原文的许可不由仓库代码许可覆盖。索引编排、标签映射和 JSONL 格式统一来自本项目；上游原文、任务和标注保留原作者署名及许可。这些题只用于评测，不用于训练或拟合温度。
